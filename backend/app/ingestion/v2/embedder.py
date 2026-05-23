"""jina-clip-v2 multimodal embedder via ONNX Runtime (CPU-friendly).

Jina ships pre-quantized ONNX variants on HF Hub:
  - model.onnx        (fp32, ~1.7 GB, needs external `model.onnx_data`)
  - model_fp16.onnx   (~850 MB)
  - model_int8.onnx   (~440 MB)

Choose via `EMBEDDING_QUANT` env var. We download the chosen file once
into the HF cache and serve via `onnxruntime.InferenceSession` on CPU.

The model is a single ONNX graph with both branches:
  Inputs:  input_ids[B, T], pixel_values[B, 3, 512, 512]
  Outputs: l2norm_text_embeddings[B, 1024], l2norm_image_embeddings[B, 1024]

For text-only inference we still must supply pixel_values, so we feed
zero tensors for the unused branch and read only the requested output.
"""
from __future__ import annotations

import logging
import threading
from io import BytesIO
from pathlib import Path

import numpy as np

from ...config import get_settings


log = logging.getLogger(__name__)


_VARIANT_FILES = {
    "fp32":  ("onnx/model.onnx",      "onnx/model.onnx_data"),
    "fp16":  ("onnx/model_fp16.onnx", None),
    "int8":  ("onnx/model_int8.onnx", None),
}

# Constants from the ONNX graph (inspected once with onnx.load).
_IMAGE_SIZE = 512
_EMBED_DIM = 1024

# Internal batching caps — the jina-clip-v2 graph runs both towers on every
# call, so a single MatMul in the vision branch wants ~1.5 GB at batch=20.
# Process in small batches to keep peak RAM well under 1 GB on CPU.
_TEXT_BATCH = 8
_IMAGE_BATCH = 4


class JinaV4Embedder:
    """Single-instance jina-clip-v2 embedder. Name kept for caller compatibility."""

    _instance: "JinaV4Embedder | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model_v2
        self.quant = (settings.embedding_quant or "fp32").lower()
        if self.quant not in _VARIANT_FILES:
            log.warning("Unknown EMBEDDING_QUANT=%s; defaulting to fp32", self.quant)
            self.quant = "fp32"
        self._loaded = False
        self._load_lock = threading.Lock()
        self._session = None
        self._tokenizer = None
        self._processor = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    @classmethod
    def get(cls) -> "JinaV4Embedder":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return

            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import AutoTokenizer, AutoImageProcessor

            log.info("Loading %s (quant=%s) via ONNX Runtime CPU...", self.model_name, self.quant)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self._processor = AutoImageProcessor.from_pretrained(self.model_name, trust_remote_code=True)

            model_file, data_file = _VARIANT_FILES[self.quant]
            onnx_path = Path(hf_hub_download(repo_id=self.model_name, filename=model_file))
            if data_file is not None:
                # fp32 stores weights externally; hf_hub_download places it
                # next to model.onnx and onnxruntime picks it up automatically.
                hf_hub_download(repo_id=self.model_name, filename=data_file)

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"],
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self._output_names = [o.name for o in self._session.get_outputs()]
            self._loaded = True
            log.info("Embedder ready. inputs=%s outputs=%s", self._input_names, self._output_names)

    # ---------- text ----------
    def _embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        enc = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int64)
        batch = input_ids.shape[0]
        feed = {
            "input_ids": input_ids,
            # Model graph requires BOTH inputs even for text-only; feed zeros.
            "pixel_values": np.zeros((batch, 3, _IMAGE_SIZE, _IMAGE_SIZE), dtype=np.float32),
        }
        out = self._session.run(["l2norm_text_embeddings"], feed)
        return np.asarray(out[0], dtype=np.float32).tolist()

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        results: list[list[float]] = []
        for i in range(0, len(texts), _TEXT_BATCH):
            results.extend(self._embed_text_batch(texts[i:i + _TEXT_BATCH]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text([text])[0]

    # ---------- image ----------
    def _embed_image_batch(self, pil_imgs: list) -> list[list[float]]:
        proc = self._processor(images=pil_imgs, return_tensors="np")
        pixel_values = proc["pixel_values"]
        # jina-clip-v2 ships a timm-backed processor that returns torch.Tensor
        # regardless of return_tensors. Coerce to numpy.
        if hasattr(pixel_values, "detach"):
            pixel_values = pixel_values.detach().cpu().numpy()
        pixel_values = np.asarray(pixel_values, dtype=np.float32)
        batch = pixel_values.shape[0]
        pad_id = int(getattr(self._tokenizer, "pad_token_id", 0) or 0)
        feed = {
            "input_ids": np.full((batch, 1), pad_id, dtype=np.int64),
            "pixel_values": pixel_values,
        }
        out = self._session.run(["l2norm_image_embeddings"], feed)
        return np.asarray(out[0], dtype=np.float32).tolist()

    def embed_image(self, images: list[bytes]) -> list[list[float]]:
        """Embed image bytes. Skips bad/unreadable images by emitting a zero
        vector at that position so caller's index alignment is preserved.
        """
        if not images:
            return []
        self._ensure_loaded()
        from PIL import Image
        pils: list = []
        ok_mask: list[bool] = []
        for b in images:
            try:
                pils.append(Image.open(BytesIO(b)).convert("RGB"))
                ok_mask.append(True)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping unreadable image: %s", e)
                pils.append(None)
                ok_mask.append(False)
        # Embed only the good ones, batched.
        good_pils = [p for p, ok in zip(pils, ok_mask) if ok]
        good_vecs: list[list[float]] = []
        for i in range(0, len(good_pils), _IMAGE_BATCH):
            good_vecs.extend(self._embed_image_batch(good_pils[i:i + _IMAGE_BATCH]))
        # Re-thread results, putting zero-vec placeholders for the skipped ones.
        out: list[list[float]] = []
        gi = 0
        zero = [0.0] * _EMBED_DIM
        for ok in ok_mask:
            if ok:
                out.append(good_vecs[gi])
                gi += 1
            else:
                out.append(zero)
        return out

    # ---------- Chroma EmbeddingFunction surface ----------
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self.embed_text(list(input or []))

    def name(self) -> str:
        return f"jina-clip-v2:{self.quant}"

    @property
    def dim(self) -> int:
        return _EMBED_DIM
