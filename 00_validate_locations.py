"""
EcoLens -- Step 00: Location and patch quality control

---------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
---------------------------------------------------------------------
Faculty review comment: "Provide a detailed explanation of image
preprocessing, patch generation, spatial resolution, temporal selection,
cloud filtering, and quality control to improve reproducibility."

The pipeline had no quality control at all. Nothing ever asked whether a
downloaded patch CONTAINED what its label claimed, and the failures that
produced stayed invisible for months:

  * three patches (agri_015, mangrove_001, mangrove_015) were entirely
    nodata -- all-zero arrays -- yet were embedded, indexed and retrieved
    like any other. Two all-zero patches match each other at cosine
    0.9998, which is how "Shark Bay Mangroves" came to be the top analog
    for "Hokkaido Potato Farms" in the retrieval case studies.
  * three "mangrove" locations sat offshore and were 96-100% open water,
    so their embeddings describe the sea, not a mangrove.
  * seven coordinates pointed somewhere other than the place named --
    "Salonga National Park, DRC" was 690 km from Salonga.
  * ten configured locations had no patch at all, because step 01 printed
    one warning per failure and no summary at the end.

Run this before trusting any number the pipeline produces.

    python 00_validate_locations.py
    python 00_validate_locations.py --purge   (delete unusable patches)

Exits non-zero when ERROR-level problems are found, so it can gate a
pipeline run from a script or CI job.

---------------------------------------------------------------------
WHAT IT CHECKS
---------------------------------------------------------------------
Without any reference data (always available):
    MISSING    configured location has no patch on disk
    STALE      patch was downloaded from coordinates config no longer uses
    EMPTY      patch is entirely nodata
    WATER      patch is mostly open water
    DUPLICATE  two locations closer than DUPLICATE_KM but labelled as
               different ecosystems, which makes those two classes
               impossible to separate and unfairly punishes the metrics

Additionally, once geo_data/ has been downloaded:
    MISLABEL   RESOLVE biome at the coordinate contradicts the declared
               ecosystem
    ELEVATION  mangrove well above sea level, or wetland improbably high
"""

import argparse
import json
import os
import sys
from importlib import import_module

import numpy as np

from config import (
    PATCH_LOCATIONS, PATCHES_DIR, ACQUISITION_MANIFEST_PATH,
)

acquire = import_module("01_acquire_patches")

# Two locations closer than this but carrying different ecosystem labels
# may be the same ecosystem split across two classes, which makes those
# classes impossible to separate and unfairly punishes the metrics.
#
# Calibrated against the real failure this check exists to catch: the
# Sundarbans appeared three times, 77-83 km apart, as one mangrove and two
# wetlands. A 25 km threshold would have missed it entirely. This is an
# advisory WARN, not an error -- genuinely different ecosystems can sit
# this close (Daintree rainforest is 82 km from Great Barrier Reef
# mangroves), so each hit needs a human look rather than an auto-fix.
DUPLICATE_KM = 100.0

# Biome keywords consistent with each declared ecosystem. Agricultural and
# urban_green are deliberately unconstrained -- farmland and city parks
# legitimately occur in any biome, so there is nothing to assert.
EXPECTED_BIOME = {
    "forest":   ["Forest", "Taiga", "Mangrove"],
    "mangrove": ["Mangrove", "Forest"],
    "wetland":  ["Forest", "Grassland", "Savanna", "Flooded", "Wetland", "Tundra"],
}

SEVERITY = {
    "EMPTY": "ERROR", "WATER": "ERROR", "STALE": "ERROR", "MISLABEL": "ERROR",
    "MISSING": "WARN", "DUPLICATE": "WARN", "ELEVATION": "WARN",
}

ORDER = {"EMPTY": 0, "WATER": 1, "STALE": 2, "MISLABEL": 3,
         "ELEVATION": 4, "MISSING": 5, "DUPLICATE": 6}


def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_manifest():
    if not os.path.exists(ACQUISITION_MANIFEST_PATH):
        return {}
    try:
        with open(ACQUISITION_MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def try_load_geo():
    """geo_lookups needs geo_data/; return None rather than failing hard."""
    try:
        gl = import_module("geo_lookups")
    except Exception:
        return None
    try:
        probe = gl.get_physical_descriptors(0.0, 0.0, use_cache=True)
    except Exception:
        return None
    if probe.get("ecoregion_source") in (None, "unavailable"):
        return None
    return gl


def check_patch(loc, manifest, findings, unusable):
    pid = loc["id"]
    path = os.path.join(PATCHES_DIR, pid + ".npy")

    if not os.path.exists(path):
        findings.append(("MISSING", pid, loc["name"], "no patch on disk"))
        return

    rec = manifest.get(pid)
    if rec is not None:
        moved = (abs(rec.get("lon", 1e9) - loc["lon"]) > 1e-6
                 or abs(rec.get("lat", 1e9) - loc["lat"]) > 1e-6)
        if moved:
            findings.append((
                "STALE", pid, loc["name"],
                "downloaded at ({:.4f}, {:.4f}), config now says ({:.4f}, {:.4f})".format(
                    rec["lon"], rec["lat"], loc["lon"], loc["lat"])))
            unusable.append(path)

    try:
        quality = acquire.assess_patch_quality(np.load(path))
    except Exception as exc:
        findings.append(("EMPTY", pid, loc["name"], "unreadable: {}".format(exc)))
        unusable.append(path)
        return

    if quality["nodata_fraction"] >= 0.999:
        findings.append(("EMPTY", pid, loc["name"],
                         "patch is entirely nodata (all-zero array)"))
        unusable.append(path)
    elif not quality["ok"]:
        findings.append(("WATER", pid, loc["name"], quality["reason"]))
        unusable.append(path)


def check_geography(loc, gl, findings):
    pid = loc["id"]
    desc = gl.get_physical_descriptors(loc["lon"], loc["lat"], use_cache=True)
    biome = desc.get("biome")
    elevation = desc.get("elevation_m")

    expected = EXPECTED_BIOME.get(loc["ecosystem"])
    if expected and biome and not any(k.lower() in biome.lower() for k in expected):
        findings.append(("MISLABEL", pid, loc["name"],
                         "declared {}, biome at this point is {}".format(
                             loc["ecosystem"], biome)))

    if isinstance(elevation, (int, float)):
        if loc["ecosystem"] == "mangrove" and elevation > 20:
            findings.append(("ELEVATION", pid, loc["name"],
                             "mangrove at {:.0f} m -- mangroves are intertidal".format(
                                 elevation)))
        elif loc["ecosystem"] == "wetland" and elevation > 500:
            findings.append(("ELEVATION", pid, loc["name"],
                             "wetland at {:.0f} m -- verify this is correct".format(
                                 elevation)))


def check_duplicates(findings):
    for i, a in enumerate(PATCH_LOCATIONS):
        for b in PATCH_LOCATIONS[i + 1:]:
            if a["ecosystem"] == b["ecosystem"]:
                continue
            km = haversine_km(a["lon"], a["lat"], b["lon"], b["lat"])
            if km < DUPLICATE_KM:
                findings.append((
                    "DUPLICATE", a["id"], a["name"],
                    "{:.1f} km from {} ({}), but labelled {} -- these two "
                    "classes cannot be separated".format(
                        km, b["id"], b["ecosystem"], a["ecosystem"])))


def main(purge=False):
    manifest = load_manifest()
    gl = try_load_geo()

    print("=" * 92)
    print("EcoLens location and patch quality control")
    print("=" * 92)
    print("  locations configured : {}".format(len(PATCH_LOCATIONS)))
    if manifest:
        print("  provenance manifest  : {} entries".format(len(manifest)))
    else:
        print("  provenance manifest  : absent "
              "(run 01_acquire_patches.py to create it)")
    if gl:
        print("  reference data       : available")
    else:
        print("  reference data       : absent -- "
              "MISLABEL and ELEVATION checks skipped")
    print()

    findings = []
    unusable = []

    for loc in PATCH_LOCATIONS:
        check_patch(loc, manifest, findings, unusable)
        if gl is not None:
            check_geography(loc, gl, findings)
    check_duplicates(findings)

    if not findings:
        print("No problems found. Every configured location has a patch that "
              "passes quality control.")
        return 0

    errors = [f for f in findings if SEVERITY[f[0]] == "ERROR"]
    warns = [f for f in findings if SEVERITY[f[0]] == "WARN"]

    print("{} finding(s): {} error, {} warning".format(
        len(findings), len(errors), len(warns)))
    print()
    print("{:9} {:10} {:17} {:32} {}".format(
        "severity", "check", "id", "name", "detail"))
    print("-" * 130)
    for kind, pid, name, detail in sorted(findings, key=lambda f: (ORDER[f[0]], f[1])):
        print("{:9} {:10} {:17} {:32} {}".format(
            SEVERITY[kind], kind, pid, name[:32], detail))

    unusable = sorted(set(unusable))
    if unusable:
        print()
        print("{} patch file(s) are unusable and should be re-acquired.".format(
            len(unusable)))
        if purge:
            for path in unusable:
                os.remove(path)
                print("  removed {}".format(path))
            print()
            print("Re-run 01_acquire_patches.py to fetch them at the "
                  "corrected coordinates.")
        else:
            print("Re-run with --purge to delete them, then run "
                  "01_acquire_patches.py.")

    return 1 if errors else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate configured locations and downloaded patches.")
    parser.add_argument("--purge", action="store_true",
                        help="Delete patch files that are empty, mostly water, or "
                             "stale, so 01_acquire_patches.py re-acquires them.")
    args = parser.parse_args()
    sys.exit(main(purge=args.purge))
