"""
EcoLens Step 16 -- derive REALM / BIOME / ECOREGION for every location (Phase 3, C7)

C7 asks whether retrieval is consistent ACROSS GEOGRAPHIC REGIONS rather than only
between nearby ecosystems. Answering it needs a region label per location, and
07_evaluate_retrieval.py currently contains no notion of region at all.

WHY DERIVE RATHER THAN HAND-TYPE
--------------------------------
config.PATCH_LOCATIONS carries a hand-written `realm` on the 50 Phase 1 entries and
nothing on the original 81, and a hand-maintained field can silently drift from the
coordinate it describes -- which is exactly how 13 mislocated coordinates arose
(implementation.md 3.1). RESOLVE Ecoregions 2017 is already on disk and carries
REALM, BIOME_NAME, ECO_NAME and NNH_NAME per polygon, so every location gets a label
derived from its actual coordinate.

Biogeographic realm is also a BETTER axis for this question than "continent": it is
the standard unit for exactly the "geographically distinct landscapes" the research
question names, and it splits places continents do not (Nearctic vs Neotropic runs
through Mexico, not along a coastline).

Writes metadata/location_geography.json  {base_id: {realm, biome, ecoregion, nnh}}.
Locations whose coordinate falls outside every polygon get None -- never a guess.

Run:
    python 16_derive_location_geography.py
"""

import json
import os

from config import PATCH_LOCATIONS, METADATA_DIR, ECOREGIONS_PATH

OUT_PATH = f"{METADATA_DIR}/location_geography.json"


def main():
    import geopandas as gpd
    from shapely.geometry import Point

    if not os.path.exists(ECOREGIONS_PATH):
        raise SystemExit(f"{ECOREGIONS_PATH} not found -- run download_reference_data.py")

    print(f"loading {ECOREGIONS_PATH} ...")
    gdf = gpd.read_file(ECOREGIONS_PATH)
    print(f"  {len(gdf)} polygons")

    out, missing = {}, []
    for loc in PATCH_LOCATIONS:
        hit = gdf[gdf.contains(Point(loc["lon"], loc["lat"]))]
        if hit.empty:
            out[loc["id"]] = {"realm": None, "biome": None, "ecoregion": None,
                              "nnh": None, "ecosystem": loc["ecosystem"]}
            missing.append(loc["id"])
            continue
        r = hit.iloc[0]
        out[loc["id"]] = {
            "realm": str(r["REALM"]),
            "biome": str(r["BIOME_NAME"]),
            "ecoregion": str(r["ECO_NAME"]),
            "nnh": str(r["NNH_NAME"]),
            "ecosystem": loc["ecosystem"],
        }

    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    from collections import Counter
    realms = Counter(v["realm"] or "(none)" for v in out.values())
    print(f"\nwrote {OUT_PATH}: {len(out)} locations")
    print(f"no polygon at the coordinate: {len(missing)} {missing if missing else ''}")
    print("\nrealm distribution (all configured locations):")
    for k, n in realms.most_common():
        print(f"  {k:<16} {n}")

    print("\nrealm x ecosystem (acquired locations only):")
    cat_path = f"{METADATA_DIR}/catalog.json"
    if os.path.exists(cat_path):
        with open(cat_path, encoding="utf-8") as f:
            acquired = {e.get("base_id", e["id"].rsplit("_p", 1)[0]) for e in json.load(f)}
        pairs = Counter((out[i]["realm"] or "(none)", out[i]["ecosystem"])
                        for i in out if i in acquired)
        realms_seen = sorted({p[0] for p in pairs})
        ecos = sorted({p[1] for p in pairs})
        hdr = f"  {'realm':<16}" + "".join(f"{e[:9]:>10}" for e in ecos)
        print(hdr)
        for rm in realms_seen:
            row = f"  {rm:<16}" + "".join(f"{pairs.get((rm, e), 0):>10}" for e in ecos)
            print(row)
        print("\n  NOTE for C7: a leave-one-realm-out protocol is only meaningful where a "
              "realm holds enough locations to leave out. Read the row totals before "
              "designing the folds.")


if __name__ == "__main__":
    main()
