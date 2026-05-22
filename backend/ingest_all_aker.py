"""One-shot ingestion of all Aker portfolio properties found in MySQL."""
import sys, time, json, logging, os
sys.path.insert(0, '.')

logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')

from app.ingestion.v2.pipeline import ingest_urls

SEEDS: dict[str, tuple[str, list[str]]] = {
    "138r": ("Everbend Tarrytown", [
        "https://everbendny.com/",
        "https://everbendny.com/amenities/",
        "https://everbendny.com/floorplans/",
        "https://everbendny.com/gallery/",
    ]),
    "139r": ("The Mill Greenwich", [
        "https://themillgreenwich.com/",
        "https://themillgreenwich.com/amenities/",
        "https://themillgreenwich.com/floor-plans/",
        "https://themillgreenwich.com/residences/",
    ]),
    "153r": ("Abbot Mill", [
        "https://abbotmill.com/",
        "https://abbotmill.com/amenities/",
        "https://abbotmill.com/floor-plans/",
        "https://abbotmill.com/gallery/",
    ]),
    "175r": ("Kinwood Apartments", [
        "https://kinwoodny.com/",
        "https://kinwoodny.com/amenities/",
        "https://kinwoodny.com/floorplans/",
        "https://kinwoodny.com/gallery/",
    ]),
    "176r": ("The Alexander", [
        "https://alexanderalbany.com/",
        "https://alexanderalbany.com/amenities/",
        "https://alexanderalbany.com/floorplans/",
        "https://alexanderalbany.com/gallery/",
    ]),
    "183r": ("Luckey Platt", [
        "https://luckeyplatt.com/",
        "https://luckeyplatt.com/amenities/",
        "https://luckeyplatt.com/floorplans/",
        "https://luckeyplatt.com/gallery/",
    ]),
    "184r": ("Lakeshore Preserve", [
        "https://livelakeshorepreserve.com/",
        "https://livelakeshorepreserve.com/amenities/",
        "https://livelakeshorepreserve.com/floorplans/",
        "https://livelakeshorepreserve.com/gallery/",
        "https://livelakeshorepreserve.com/neighborhood/",
    ]),
    "185r": ("Waterfront at the Strand", [
        "https://livewaterfrontstrand.com/",
        "https://livewaterfrontstrand.com/amenities/",
        "https://livewaterfrontstrand.com/floorplans/",
        "https://livewaterfrontstrand.com/gallery/",
        "https://livewaterfrontstrand.com/neighborhood/",
    ]),
    "462a": ("Stony Run", [
        "https://stonyrunstockade.com/",
        "https://stonyrunstockade.com/amenities/",
        "https://stonyrunstockade.com/floorplans/",
        "https://stonyrunstockade.com/gallery/",
    ]),
}

results = {}
overall_start = time.time()
for code, (name, urls) in SEEDS.items():
    print(f"\n=== {code} {name} ({len(urls)} URLs) ===", flush=True)
    t0 = time.time()
    try:
        r = ingest_urls(code, urls, replace=True)
    except Exception as e:
        r = {"property_code": code, "error": str(e), "type": type(e).__name__}
    r["elapsed_s"] = round(time.time() - t0, 1)
    results[code] = r
    print(json.dumps(r, indent=2, default=str), flush=True)

print(f"\n=== TOTAL TIME: {round(time.time()-overall_start, 1)}s ===")
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'errors'} for k, v in results.items()}, indent=2, default=str))
