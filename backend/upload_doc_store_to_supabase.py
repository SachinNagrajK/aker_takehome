"""One-shot uploader: backend/doc_store/* → Supabase Storage `doc-store` bucket.

Walks the local doc store and uploads every file under its existing
`{property_code}/{sha256}.{ext}` key. Idempotent (x-upsert: true) and
parallelised with a thread pool because each upload is a single HTTP call.

Run once after Step 3 is wired in; safe to re-run.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from app.config import get_settings
from app.ingestion.v2.doc_store import _sniff_ext, _EXT_TO_MIME, _storage_endpoint


def upload_one(key: str, data: bytes, headers: dict) -> tuple[str, int, str | None]:
    url = _storage_endpoint(key, public=False)
    try:
        # Re-derive content-type from extension on disk; fallback to sniffed.
        ext = key.rsplit(".", 1)[-1].lower()
        local_headers = dict(headers)
        local_headers["Content-Type"] = _EXT_TO_MIME.get(ext, "application/octet-stream")
        r = requests.post(url, headers=local_headers, data=data, timeout=60)
        if r.status_code >= 400:
            return key, r.status_code, r.text[:200]
        return key, r.status_code, None
    except Exception as e:  # noqa: BLE001
        return key, -1, str(e)


def main() -> None:
    s = get_settings()
    if not s.supabase_url or not s.supabase_service_role_key:
        print("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")
        sys.exit(1)

    root = Path(s.doc_store_dir)
    if not root.exists():
        print(f"Doc store not found at {root}")
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {s.supabase_service_role_key}",
        "x-upsert": "true",
        "cache-control": "public, max-age=31536000, immutable",
    }

    files = [p for p in root.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    print(
        f"Uploading {len(files)} files / {total_bytes/1_048_576:.1f} MB to "
        f"bucket {s.supabase_storage_bucket} at {s.supabase_url}"
    )

    t0 = time.time()
    ok = 0
    failed: list[tuple[str, int, str]] = []
    done = 0

    def job(p: Path):
        rel = p.relative_to(root).as_posix()
        return upload_one(rel, p.read_bytes(), headers)

    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = [ex.submit(job, p) for p in files]
        for fut in as_completed(futures):
            key, status, err = fut.result()
            done += 1
            if err is None:
                ok += 1
            else:
                failed.append((key, status, err))
            if done % 50 == 0 or done == len(files):
                rate = done / max(1, time.time() - t0)
                print(f"  {done}/{len(files)}  ok={ok}  failed={len(failed)}  "
                      f"({rate:.1f}/s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. ok={ok}  failed={len(failed)}")
    if failed:
        print("\nFailures (first 10):")
        for key, status, err in failed[:10]:
            print(f"  [{status}] {key}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
