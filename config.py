"""
EcoLens Phase 1 — Configuration

Defines the 10 proof-of-concept patch locations across different
ecosystem categories, plus shared constants used by every script
in the pipeline.

Pick real coordinates for ecosystems you can verify on Google Maps
satellite view first — this avoids wasting a STAC query on a patch
that turns out to be the wrong land cover type.
"""

import sys


# ---------------------------------------------------------------
# Windows console encoding
# ---------------------------------------------------------------
# This pipeline was developed on Linux, where sys.stdout defaults to
# UTF-8. On Windows the default is the ANSI code page (cp1252 here),
# and several scripts print characters outside it -- em dashes in
# banners (01, run_pipeline) and the degree sign in 07b's temperature
# output. Printing those to a cp1252 stream raises UnicodeEncodeError.
#
# Interactive consoles happen to survive this (Python writes them via
# WriteConsoleW), which is what makes the bug so easy to miss: it only
# appears once output is REDIRECTED -- `python 07b_... > log.txt`, a
# CI job, or any wrapper that captures stdout. run_pipeline.py invokes
# every stage as a subprocess, so this matters for the full run too.
#
# Every script in the pipeline imports config, so reconfiguring here
# covers all of them from one place. Guarded on platform and on the
# attribute's existence so it's a no-op everywhere else.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8")
            except (ValueError, OSError):
                pass  # already detached or not reconfigurable; not fatal


PATCH_SIZE_PX = 224          # Standard ViT-style input size
PATCH_SIZE_M = 2240          # 224 px * 10m/px native Sentinel-2 resolution

# Prithvi-100M (HLS-trained) expects these 6 bands in this order:
# Blue, Green, Red, NIR (Narrow), SWIR 1, SWIR 2
# Sentinel-2 band codes for each:
PRITHVI_BANDS = {
    "blue":   "B02",   # 10m native
    "green":  "B03",   # 10m native
    "red":    "B04",   # 10m native
    "nir":    "B8A",   # 20m native — narrow NIR, matches HLS better than B08
    "swir1":  "B11",   # 20m native
    "swir2":  "B12",   # 20m native
}

# ---------------------------------------------------------------
# STAC search parameters
# ---------------------------------------------------------------

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
AWS_STAC_URL = "https://earth-search.aws.element84.com/v1"

# Acquisition window. Widened from "2024-01-01/2024-06-30" to the full year:
# perpetually-cloudy equatorial sites (Singapore, Gulf of Guayaquil) had no
# scene at all under MAX_CLOUD_COVER in a six-month window, and a location
# absent from the catalog biases results more than one acquired in a different
# month does. Patches already on disk are NOT re-acquired by this change --
# staleness keys on coordinates, not on the search window -- so a re-run only
# re-searches the locations that previously failed. That does mean the catalog
# spans a wider range of acquisition dates than the original design; report the
# actual per-patch dates from catalog.json rather than quoting this constant.
SEARCH_DATE_RANGE = "2024-01-01/2024-12-31"
MAX_CLOUD_COVER = 15  # percent

# ---------------------------------------------------------------
# Patch quality control (01_acquire_patches.py)
# ---------------------------------------------------------------
# A coordinate can be perfectly cloud-free and still be useless: if it
# sits offshore, the 2.24 km patch is open water and the embedding
# describes the sea rather than the ecosystem the label claims. This
# was a real failure in this catalog -- three "mangrove" locations
# scored 96-100% water and still fed every retrieval result, because
# nothing in the pipeline ever looked at patch CONTENT.
#
# These thresholds make that failure loud instead of silent. NDWI
# ( = (green - nir) / (green + nir) ) marks water; a patch above
# MAX_WATER_FRACTION, or with too many nodata pixels, is rejected and
# reported. Mangroves and wetlands are legitimately wet, so the bar is
# set at "almost entirely water", not "any water".
MAX_WATER_FRACTION = 0.80     # reject if >80% of pixels are water
MAX_NODATA_FRACTION = 0.20    # reject if >20% of pixels are nodata/zero

# How many scenes 01 will try per location before giving up.
#
# A STAC search matches any scene whose footprint INTERSECTS the bbox, and
# Sentinel-2 tiles carry wide nodata margins -- so the least-cloudy match can
# still clip the point and yield an all-zero patch. Cloud cover is a property
# of the scene; whether the scene covers this particular point only becomes
# knowable after reading the window. Trying the next-clearest scene is the
# only way to tell those apart, and it absorbs transient COG read errors too.
#
# 6 is a compromise: enough to get past a couple of clipping tiles and a flaky
# read, few enough that a genuinely offshore coordinate fails quickly rather
# than burning a dozen downloads to confirm the sea is still the sea.
MAX_SCENE_ATTEMPTS = 6

# STAC search retry. Planetary Computer is behind Azure Front Door, which
# intermittently returns 502 OriginConnectionAborted; without a retry a single
# blip aborts an entire 80-location acquisition run. Backoff doubles each try
# (5s, 10s, 20s, 40s) so a briefly-overloaded origin gets room to recover.
STAC_SEARCH_RETRIES = 5
STAC_RETRY_BACKOFF_S = 5

# ---------------------------------------------------------------
# Growing-season preference (01_acquire_patches.py)
# ---------------------------------------------------------------
# Picking the least-cloudy scene sounds neutral and is not: outside the
# tropics the clearest skies are in WINTER, when deciduous canopies are bare
# and ground may be snow-covered. Measured on this catalog before the fix,
# 44 of 79 patches came from Jan-Mar, and Jiuzhaigou -- a montane deciduous
# forest -- was sampled on 9 January and returned 0.1% forest cover.
#
# Left uncorrected this is a confound, not just noise: embeddings would encode
# acquisition month alongside ecosystem type, and retrieval would partly match
# season. 01 therefore ranks candidate scenes by growing season first and
# cloud second, within the set that already passes MAX_CLOUD_COVER.
#
# Tropical sites are exempt: |lat| < TROPICS_LAT has no dormant season worth
# avoiding, and applying a penalty there would discard usable scenes for
# nothing. The month sets are deliberately broad -- this is a preference
# ordering, not a phenological model.
TROPICS_LAT = 23.5
NORTHERN_GROWING_MONTHS = {5, 6, 7, 8, 9}
SOUTHERN_GROWING_MONTHS = {11, 12, 1, 2, 3}

# Sentinel-2 BOA_ADD_OFFSET (ESA processing baseline 04.00+, 2022-01-25 onward).
# L2A DN values must be shifted by this before scaling: reflectance =
# (DN + BOA_ADD_OFFSET) / 10000. Omitting it does not cancel out of normalized
# indices -- it survives in the denominator -- and compressed every NDVI in
# this project toward zero (Daintree read 0.57 instead of 0.91). 01 applies it
# at extraction so patches/ holds corrected DN and every downstream consumer
# is right without needing to know about it.
BOA_ADD_OFFSET = -1000

# Minimum vegetation for a patch labelled "forest" or "mangrove".
#
# Calibrated against this catalog's healthy examples -- Hinchinbrook 0.91 NDVI
# / 100% forest cover, Matang 0.90/77%, Bialowieza 0.88/100%, Jiuzhaigou
# 0.52/59%, Pichavaram 0.48/54% -- against the failures, which sit far below:
# Bhitarkanika 0.30/10% (mudflat and shrimp ponds) and the old Red Sea patch
# at 0.00/0% (bare desert).
#
# 01 uses these to reject a scene and try the next one, which fixes the common
# case where the patch is fine but the SCENE is hazy: Niger Delta measured NDVI
# 0.14 under haze in a genuinely dense mangrove system. 00 uses the same
# thresholds as an ERROR-level gate on whatever survived.
#
# Only forest and mangrove are constrained. Wetlands range from open water to
# dense reedbed, cropland is bare between harvest and planting, and urban_green
# is mostly built surface by area -- a vegetation floor would reject correct
# data for all three.
MIN_VEG_NDVI = 0.35
MIN_VEG_FOREST_PCT = 20.0
VEGETATED_ECOSYSTEMS = ("forest", "mangrove", "boreal")
# "boreal" added 3 Sep with the Phase 1 categories. Boreal forest IS forest, so a
# boreal patch with no vegetation in the growing season is as wrong as a forest one,
# and both 01's rejection gate and 00's CONTENT ERROR should catch it.
#
# The other four new categories are DELIBERATELY excluded. Tundra, shrubland,
# grassland and savanna legitimately sit below MIN_VEG_NDVI -- a desert with NDVI
# 0.1 is a correct desert, not a failed download -- so applying the vegetation
# floor to them would reject exactly the ecosystems Phase 1 set out to add. That
# is the §9 rule about never relaxing a threshold, read in the other direction:
# don't apply a threshold to a class it was never meant to describe.
#
# The gap this used to leave is now closed by ECOSYSTEM_NDVI_BANDS below.

# ---------------------------------------------------------------
# Per-category expected NDVI band -- the CONTENT check for categories that
# MIN_VEG_NDVI cannot describe.
# ---------------------------------------------------------------
# MIN_VEG_NDVI is a single global floor, which only makes sense for classes that
# are supposed to be green. It cannot express "a desert should be sparse" or
# "tundra is somewhere in between", so the Phase 1 categories initially had no
# content check at all beyond water/nodata.
#
# That gap produced real failures on the first Phase 1 acquisition (3 Sep), all of
# which passed every existing gate:
#   tundra_007  West Greenland   NDVI -0.122, 79.3% water  -> icecap/sea, not tundra
#                                (79.3% squeaked under the 80% water limit)
#   tundra_010  Kolyma delta     NDVI -0.366, 68.2% water  -> water/ice
#   boreal_009  Komi, May scene  NDVI  0.214               -> pre-greenup snow
#
# A BAND (min, max) catches all three: negative NDVI is never tundra, and a boreal
# conifer canopy in the growing season is never 0.21. The bands are deliberately
# WIDE -- they are meant to catch "this is the wrong surface entirely", not to
# police natural variation. Per §9 these are content checks, never to be relaxed to
# admit a patch that failed: move the coordinate or record the loss.
#
# DELIBERATELY EXCLUDES forest / mangrove / boreal. Those are in
# VEGETATED_ECOSYSTEMS and already governed by the stricter-in-the-right-way
# rule above, which rejects only when NDVI **and** forest cover are both low.
# An NDVI-only band would be blunter and would discard patches that rule
# correctly keeps: forest_002 (Amazon) reads NDVI 0.257 but 27.8% forest cover,
# and mangrove_006 reads 0.299 / 37.3% -- both genuine, both would have been
# thrown away. Meanwhile boreal_009 (NDVI 0.214, forest cover 3.5%) fails the
# combined rule on its own merits, which is the correct outcome.
ECOSYSTEM_NDVI_BANDS = {
    "savanna":      (0.15, 0.90),   # spans wet and dry season
    "grassland":    (0.05, 0.90),   # includes arid steppe (Patagonia reads ~0.06)
    "tundra":       (0.05, 0.85),   # low but POSITIVE; negative means water/ice
    "shrubland":    (-0.05, 0.45),  # sparse by definition; low NDVI is correct here
    "wetland":      (-0.60, 0.90),  # Camargue lagoon reads -0.54 and is genuine
    "agricultural": (-0.10, 0.95),  # bare field to closed canopy across the season
    "urban_green":  (0.05, 0.90),
}

# Haze proxy: median blue reflectance above this is REPORTED (never rejected)
# by 00_validate_locations.py. See assess_patch_quality()'s docstring for why
# this is not an automatic rejection -- bright deserts, snow and urban cores
# occupy the same range as genuine haze.
HAZE_BLUE_REFLECTANCE = 0.10


# ---------------------------------------------------------------
# 10 proof-of-concept patches across 5 ecosystem categories
# (lon, lat) — chosen as well-known, verifiable examples
# ---------------------------------------------------------------

PATCH_LOCATIONS = [
    # Forest ecosystems (15)
    {"id": "forest_001", "ecosystem": "forest", "lon": 76.6320, "lat": 9.4981,
     "name": "Periyar forest, Kerala, India", "protected_area": True, "climatic_region": "Tropical Monsoon"},
    {"id": "forest_002", "ecosystem": "forest", "lon": -60.0261, "lat": -3.1019,
     "name": "Amazon rainforest, Brazil", "protected_area": False, "climatic_region": "Tropical Rainforest"},
    {"id": "forest_003", "ecosystem": "forest", "lon": 8.2250, "lat": 48.0500,
     "name": "Black Forest, Germany", "protected_area": True, "climatic_region": "Marine West Coast"},
    {"id": "forest_004", "ecosystem": "forest", "lon": -124.0072, "lat": 41.2764,
     "name": "Redwood National Park, California, USA", "protected_area": True, "climatic_region": "Temperate Rainforest"},
    {"id": "forest_005", "ecosystem": "forest", "lon": 23.8611, "lat": 52.7439,
     "name": "Bialowieza Forest, Poland", "protected_area": True, "climatic_region": "Humid Continental"},
    {"id": "forest_006", "ecosystem": "forest", "lon": 92.5000, "lat": 56.5000,
     "name": "Siberian Boreal Forest, Russia", "protected_area": False, "climatic_region": "Boreal"},
    {"id": "forest_007", "ecosystem": "forest", "lon": -123.6000, "lat": 47.6000,
     "name": "Olympic National Forest, Washington, USA", "protected_area": True, "climatic_region": "Temperate Rainforest"},
    {"id": "forest_008", "ecosystem": "forest", "lon": 153.0720, "lat": -25.7500,
     "name": "Great Sandy National Park, Australia", "protected_area": True, "climatic_region": "Subtropical"},
    {"id": "forest_009", "ecosystem": "forest", "lon": 145.4180, "lat": -16.1700,
     "name": "Daintree Rainforest, Australia", "protected_area": True, "climatic_region": "Tropical Rainforest"},
    {"id": "forest_010", "ecosystem": "forest", "lon": 11.5000, "lat": -0.5000,
     "name": "Congo Basin Forest, Gabon", "protected_area": False, "climatic_region": "Tropical Rainforest"},
    # CORRECTED 37.3000, -0.1500 -> 37.1500, -0.2000: was the 4596 m summit, above the treeline; moved to the SW montane forest belt
    {"id": "forest_011", "ecosystem": "forest", "lon": 37.1500, "lat": -0.2000,
     "name": "Mount Kenya Forest, Kenya", "protected_area": True, "climatic_region": "Montane Forest"},
    {"id": "forest_012", "ecosystem": "forest", "lon": 130.5000, "lat": 30.3500,
     "name": "Yakushima Forest, Japan", "protected_area": True, "climatic_region": "Warm Temperate"},
    {"id": "forest_013", "ecosystem": "forest", "lon": -73.2500, "lat": -39.8000,
     "name": "Valdivian Rainforest, Chile", "protected_area": True, "climatic_region": "Temperate Rainforest"},
    # CORRECTED 103.9184, 33.2613 -> 103.9000, 33.1200: patch measured 3.7% forest cover, 32% bare and mean NDVI 0.31 -- the point sat in the upper
    #   valley above the treeline, which is why RESOLVE reports Montane Grasslands & Shrublands here. Moved south into the lower forested valley.
    {"id": "forest_014", "ecosystem": "forest", "lon": 103.9000, "lat": 33.1200,
     "name": "Jiuzhaigou Forest, China", "protected_area": True, "climatic_region": "Montane Deciduous"},
    {"id": "forest_015", "ecosystem": "forest", "lon": -1.0730, "lat": 53.2030,
     "name": "Sherwood Forest, UK", "protected_area": True, "climatic_region": "Temperate Deciduous"},
    # CORRECTED -105.0000, 40.0000 -> -105.6000, 40.3500: was shortgrass prairie 58 km E of the park at 1590 m; moved into Moraine Park
    {"id": "forest_016", "ecosystem": "forest", "lon": -105.6000, "lat": 40.3500,
     "name": "Rocky Mountain National Park, USA", "protected_area": True, "climatic_region": "Montane Forest"},
    # CORRECTED 14.5000, -2.5000 -> 20.7000, -2.2000: was 690 km W of Salonga in forest-savanna mosaic; moved to the park core
    {"id": "forest_017", "ecosystem": "forest", "lon": 20.7000, "lat": -2.2000,
     "name": "Salonga National Park, DRC", "protected_area": True, "climatic_region": "Tropical Rainforest"},

    # Wetlands (15)
    {"id": "wetland_001", "ecosystem": "wetland", "lon": 81.8500, "lat": 26.7500,
     "name": "Dudwa wetlands, Uttar Pradesh, India", "protected_area": True, "climatic_region": "Humid Subtropical"},
    # CORRECTED -81.3930, 25.8650 -> -80.3500, 26.5000: the patch itself was fine (72% vegetation cover, NDVI 0.51) but it was the WRONG ECOSYSTEM --
    #   the coordinate sat in the Gulf-coast mangrove belt, which RESOLVE confirms (biome "Mangroves") and which put it 89 km from mangrove_002 with a
    #   conflicting label. Moved north-east to the Loxahatchee sawgrass marsh: genuine freshwater Everglades, and >150 km from mangrove_002, which also
    #   clears the DUPLICATE warning. This is the failure the label check exists to catch -- a good patch filed under the wrong class teaches the
    #   retrieval model that mangrove and wetland are the same thing.
    {"id": "wetland_002", "ecosystem": "wetland", "lon": -80.3500, "lat": 26.5000,
     "name": "Loxahatchee Everglades Marsh, Florida, USA", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
    {"id": "wetland_003", "ecosystem": "wetland", "lon": -56.2500, "lat": -18.0000,
     "name": "Pantanal Wetland, Brazil", "protected_area": True, "climatic_region": "Tropical Savanna"},
    {"id": "wetland_004", "ecosystem": "wetland", "lon": 22.5000, "lat": -19.5000,
     "name": "Okavango Delta, Botswana", "protected_area": True, "climatic_region": "Semi-Arid"},
    {"id": "wetland_005", "ecosystem": "wetland", "lon": 132.5000, "lat": -12.5000,
     "name": "Kakadu Wetlands, Australia", "protected_area": True, "climatic_region": "Tropical Monsoon"},
    # CORRECTED 89.5000, 22.3000 -> 91.0500, 25.1000: sat inside the Sundarbans mangrove biome alongside mangrove_001, making the wetland-vs-mangrove distinction unlearnable; moved to a true freshwater haor
    {"id": "wetland_006", "ecosystem": "wetland", "lon": 91.0500, "lat": 25.1000,
     "name": "Tanguar Haor Freshwater Wetland, Bangladesh", "protected_area": True, "climatic_region": "Tropical Swamps"},
    {"id": "wetland_007", "ecosystem": "wetland", "lon": 29.5000, "lat": 45.2500,
     "name": "Danube Delta, Romania", "protected_area": True, "climatic_region": "Temperate Wetland"},
    {"id": "wetland_008", "ecosystem": "wetland", "lon": 4.5000, "lat": 43.5300,
     "name": "Camargue Wetland, France", "protected_area": True, "climatic_region": "Mediterranean"},
    {"id": "wetland_009", "ecosystem": "wetland", "lon": 35.6000, "lat": 33.1000,
     "name": "Hula Valley Wetlands, Israel", "protected_area": True, "climatic_region": "Mediterranean"},
    {"id": "wetland_010", "ecosystem": "wetland", "lon": -68.0000, "lat": 7.5000,
     "name": "Llanos Swamps, Venezuela", "protected_area": False, "climatic_region": "Tropical Savanna"},
    # CORRECTED 48.0000, 46.0000 -> 48.7000, 45.8500: measured 4.9% vegetation and only 8.4% water with 34% bare -- the round-number coordinate landed
    #   on the Caspian lowland semi-desert north-west of the delta, not in it, which is why RESOLVE reports Deserts & Xeric Shrublands. Moved
    #   south-east into the wet delta proper below Astrakhan.
    {"id": "wetland_011", "ecosystem": "wetland", "lon": 48.7000, "lat": 45.8500,
     "name": "Volga Delta, Russia", "protected_area": True, "climatic_region": "Temperate Wetland"},
    # CORRECTED (1) 89.6500, 22.0500 -> 85.3500, 19.7000: duplicated the Sundarbans delta already covered by mangrove_001; moved to Chilika lagoon
    # CORRECTED (2) 85.3500, 19.7000 -> 93.8000, 24.4800: Chilika is a LAGOON -- all 6 candidate scenes measured 96-99% open water, so the coordinate
    #   was describing the lagoon accurately and the ecosystem label wrongly. Moved to Loktak Lake, whose phumdi (floating vegetation mat) cover makes
    #   it a vegetated freshwater wetland rather than open water. VERIFY the water fraction reported by 01 on the next run.
    {"id": "wetland_012", "ecosystem": "wetland", "lon": 93.8000, "lat": 24.4800,
     "name": "Loktak Lake Wetland, India", "protected_area": True, "climatic_region": "Tropical Monsoon"},
    {"id": "wetland_013", "ecosystem": "wetland", "lon": -111.5000, "lat": 58.7500,
     "name": "Peace-Athabasca Delta, Canada", "protected_area": True, "climatic_region": "Subarctic"},
    {"id": "wetland_014", "ecosystem": "wetland", "lon": 47.0000, "lat": 31.0000,
     "name": "Mesopotamian Marshes, Iraq", "protected_area": False, "climatic_region": "Arid Marshland"},
    # CORRECTED 6.2000, 53.4500 -> 5.8000, 53.3800: was open sea in the Wadden tidal basin (100% water across every candidate scene). The Dutch salt
    #   marshes sit landward of the dike, not in the basin; moved to Noord-Friesland Buitendijks, the largest of them. NOTE: this is an intertidal
    #   system, so water fraction depends on the tide at acquisition -- a rejection here may be a tide artefact rather than a bad coordinate.
    {"id": "wetland_015", "ecosystem": "wetland", "lon": 5.8000, "lat": 53.3800,
     "name": "Noord-Friesland Buitendijks Salt Marsh, Netherlands", "protected_area": True, "climatic_region": "Temperate Coastal"},
    {"id": "wetland_016", "ecosystem": "wetland", "lon": -91.0000, "lat": 30.0000,
     "name": "Atchafalaya Basin, USA", "protected_area": True, "climatic_region": "Humid Subtropical"},
    # CORRECTED -60.0000, -30.0000 -> -57.2000, -28.5000: was Humid Chaco 300 km SW; moved to Laguna Ibera
    {"id": "wetland_017", "ecosystem": "wetland", "lon": -57.2000, "lat": -28.5000,
     "name": "Ibera Wetlands, Argentina", "protected_area": True, "climatic_region": "Humid Subtropical"},

    # Mangroves (15)
    {"id": "mangrove_001", "ecosystem": "mangrove", "lon": 88.8500, "lat": 21.9500,
     "name": "Sundarbans, West Bengal, India", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
    # CORRECTED -80.1500, 25.3500 -> -80.9000, 25.2000: patch measured 96% open water; moved to the Snake Bight mangrove fringe
    {"id": "mangrove_002", "ecosystem": "mangrove", "lon": -80.9000, "lat": 25.2000,
     "name": "Florida Bay mangroves, USA", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
    {"id": "mangrove_003", "ecosystem": "mangrove", "lon": 6.0000, "lat": 4.5000,
     "name": "Niger Delta Mangroves, Nigeria", "protected_area": False, "climatic_region": "Tropical Coastal"},
    # CORRECTED -80.0000, -2.7500 -> -79.7000, -2.4300: was inland dry forest at 33 m; moved to Churute mangrove reserve in the estuary
    # CLOUD OVERRIDE: the coordinate is correct -- the sky is the problem. No Sentinel-2 scene covering this point fell under the global 15% cloud
    #   threshold in ANY month of 2024. Rather than relax MAX_CLOUD_COVER for all 81 locations to rescue one, this entry carries its own limit, so the
    #   exception is visible in the catalog and reportable in the methods section instead of hidden in a global constant.
    #   CAVEAT: scene-level cloud cover is not patch-level cloud cover -- a 40%-cloudy scene often has a clear 2.24 km window, but assess_patch_quality()
    #   checks water and nodata, NOT cloud, so a cloudy patch would pass QC silently. Inspect this patch before trusting it.
    {"id": "mangrove_004", "ecosystem": "mangrove", "lon": -79.7000, "lat": -2.4300,
     "name": "Gulf of Guayaquil Mangroves, Ecuador", "protected_area": True, "climatic_region": "Tropical Coastal",
     "max_cloud_cover": 40},
    {"id": "mangrove_005", "ecosystem": "mangrove", "lon": 79.7820, "lat": 11.4310,
     "name": "Pichavaram Mangroves, India", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED 86.8500, 20.6500 -> 86.9200, 20.7200: measured 1.8% forest cover and 77% bare/built with mean NDVI 0.26 -- sparse ground west of the
    #   mangrove block, not the mangrove itself (Odisha's coast is not xeric; the Deserts & Xeric Shrublands biome reading came from the point being
    #   off the vegetated delta). Moved east into the Bhitarkanika core.
    # CORRECTED (2) 86.9200, 20.7200 -> 86.9700, 20.7100: still 71% bare with median NDVI 0.30 and only 10% forest cover -- the western approach to
    #   Bhitarkanika is mudflat and shrimp aquaculture, not mangrove. Real mangroves here measure NDVI 0.90 (Matang) to 0.48 (Pichavaram). Moved east
    #   into the park's dense core. THIRD AND FINAL ATTEMPT: if the CONTENT check rejects this too, delete this entry rather than guessing again --
    #   16/17 mangroves with one honest gap beats a mudflat patch labelled "mangrove" in every retrieval result.
    {"id": "mangrove_006", "ecosystem": "mangrove", "lon": 86.9700, "lat": 20.7100,
     "name": "Bhitarkanika Mangroves, India", "protected_area": True, "climatic_region": "Tropical Monsoon"},
    # CORRECTED 106.7750, -6.1100 -> 108.8000, -7.7000: Muara Angke reserve is ~25 ha, far smaller than a 2.24 km patch, so 72% of it was water; swapped for Segara Anakan, a mangrove system larger than the patch footprint
    {"id": "mangrove_007", "ecosystem": "mangrove", "lon": 108.8000, "lat": -7.7000,
     "name": "Segara Anakan Mangroves, Indonesia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED 44.2500, -16.0000 -> 44.4500, -19.6500: patch measured 100% open water (Mozambique Channel); moved to the Tsiribihina delta
    {"id": "mangrove_008", "ecosystem": "mangrove", "lon": 44.4500, "lat": -19.6500,
     "name": "Madagascar Mangroves, Madagascar", "protected_area": True, "climatic_region": "Tropical Dry"},
    # CORRECTED 34.6000, 25.0000 -> 35.0200, 24.2800: was 539 m up in the Red Sea Hills; moved to the Hamata coastal mangroves
    # REPLACED Red Sea Mangroves, Egypt (35.0200, 24.2800) -> Hara Mangroves, Qeshm, Iran (55.7500, 26.8500).
    #   The Egyptian patch measured 99.1% bare, 0% forest cover and had NO pixel above NDVI 0.2 at all -- pure desert rock. This is the same defect
    #   class as the all-zero patches: a member of the mangrove class containing no mangrove, which corrupts every mangrove retrieval it appears in.
    #   Red Sea mangrove stands are narrow fringes a few hundred metres across, structurally too small to fill a 2.24 km patch however the coordinate
    #   is nudged -- the Key West / Muara Angke problem again. Swapped for the Hara protected area on Qeshm, a large mangrove forest in an equally
    #   arid climate, so the "Arid Coastal" slot keeps its ecological role with a system big enough to sample.
    {"id": "mangrove_009", "ecosystem": "mangrove", "lon": 55.7500, "lat": 26.8500,
     "name": "Hara Mangroves, Qeshm, Iran", "protected_area": True, "climatic_region": "Arid Coastal"},
    # CORRECTED 145.2500, -15.4500 -> 146.2000, -18.3200: "Great Barrier Reef mangroves" named a reef system, not a mangrove one -- the coordinate sat
    #   offshore and measured 96% water. Moved to the Hinchinbrook Channel / Missionary Bay system, the largest continuous mangrove forest in Australia
    #   and comfortably larger than a 2.24 km patch.
    {"id": "mangrove_010", "ecosystem": "mangrove", "lon": 146.2000, "lat": -18.3200,
     "name": "Hinchinbrook Channel Mangroves, Australia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_011", "ecosystem": "mangrove", "lon": -61.4500, "lat": 10.5800,
     "name": "Caroni Swamp Mangroves, Trinidad", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_012", "ecosystem": "mangrove", "lon": -87.5000, "lat": 20.1500,
     "name": "Yucatan Peninsula Mangroves, Mexico", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED -16.2000, 11.8000 -> -16.0500, 12.3000: patch measured 100% open water (offshore Atlantic); moved to the Rio Cacheu estuary
    {"id": "mangrove_013", "ecosystem": "mangrove", "lon": -16.0500, "lat": 12.3000,
     "name": "Guinea-Bissau Mangroves, Guinea-Bissau", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_014", "ecosystem": "mangrove", "lon": 100.6200, "lat": 4.8500,
     "name": "Matang Mangrove Forest, Malaysia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED 113.6800, -25.8000 -> 114.3000, -22.3500: every candidate scene came from tile T49JGM and every one returned an entirely-nodata patch,
    #   i.e. the point sits on that tile's empty margin and the search bbox never reaches a neighbouring tile. Shark Bay's mangrove stands are also
    #   small relative to a 2.24 km patch. Moved to Exmouth Gulf, whose eastern shore carries an extensive stand. This id was one of the three all-zero
    #   patches that used to match each other at cosine 0.9998 and dominate the retrieval results.
    {"id": "mangrove_015", "ecosystem": "mangrove", "lon": 114.3000, "lat": -22.3500,
     "name": "Exmouth Gulf Mangroves, Australia", "protected_area": False, "climatic_region": "Semi-Arid Coastal"},
    {"id": "mangrove_016", "ecosystem": "mangrove", "lon": -81.0000, "lat": 24.5000,
     "name": "Key West Mangroves, USA", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED (1) 47.5000, -15.5000 -> 47.1500, -15.4500: was inland at 84 m; moved into Mahajamba Bay proper
    # CORRECTED (2) 47.1500, -15.4500 -> 47.0800, -15.5300: overshot into open bay -- best of 6 scenes was 81% water against an 80% limit, i.e. the
    #   patch was mostly sea with a mangrove fringe. Nudged south-west onto the bay's western mangrove margin. Resisting the temptation to raise
    #   MAX_WATER_FRACTION to 0.85 to admit the old patch: that would re-open the exact hole the QC gate was added to close.
    {"id": "mangrove_017", "ecosystem": "mangrove", "lon": 47.0800, "lat": -15.5300,
     "name": "Mahajamba Bay Mangroves, Madagascar", "protected_area": True, "climatic_region": "Tropical Dry"},

    # Agricultural landscapes (15)
    {"id": "agri_001", "ecosystem": "agricultural", "lon": 75.7873, "lat": 30.9000,
     "name": "Punjab farmland, India", "protected_area": False, "climatic_region": "Semi-Arid"},
    {"id": "agri_002", "ecosystem": "agricultural", "lon": -95.3698, "lat": 41.2565,
     "name": "Iowa farmland, USA", "protected_area": False, "climatic_region": "Humid Continental"},
    {"id": "agri_003", "ecosystem": "agricultural", "lon": -120.5000, "lat": 36.8000,
     "name": "Central Valley Farmland, California, USA", "protected_area": False, "climatic_region": "Mediterranean"},
    {"id": "agri_004", "ecosystem": "agricultural", "lon": 31.2000, "lat": 29.8000,
     "name": "Nile River Valley Farms, Egypt", "protected_area": False, "climatic_region": "Arid Irrigated"},
    {"id": "agri_005", "ecosystem": "agricultural", "lon": -98.5000, "lat": 38.5000,
     "name": "Great Plains Wheat Belt, Kansas, USA", "protected_area": False, "climatic_region": "Temperate Semi-Arid"},
    {"id": "agri_006", "ecosystem": "agricultural", "lon": -60.5000, "lat": -34.5000,
     "name": "Pampas Cropland, Argentina", "protected_area": False, "climatic_region": "Humid Subtropical"},
    {"id": "agri_007", "ecosystem": "agricultural", "lon": 104.5000, "lat": 30.5000,
     "name": "Sichuan Basin Farmland, China", "protected_area": False, "climatic_region": "Humid Subtropical"},
    {"id": "agri_008", "ecosystem": "agricultural", "lon": 146.0000, "lat": -34.3000,
     "name": "Murrumbidgee Cropland, Australia", "protected_area": False, "climatic_region": "Semi-Arid"},
    {"id": "agri_009", "ecosystem": "agricultural", "lon": -7.9000, "lat": 38.0000,
     "name": "Alentejo Fields, Portugal", "protected_area": False, "climatic_region": "Mediterranean"},
    {"id": "agri_010", "ecosystem": "agricultural", "lon": 31.5000, "lat": 49.0000,
     "name": "Ukraine Black Earth Belt, Ukraine", "protected_area": False, "climatic_region": "Humid Continental"},
    {"id": "agri_011", "ecosystem": "agricultural", "lon": 172.2000, "lat": -43.6000,
     "name": "Canterbury Plains Cropland, New Zealand", "protected_area": False, "climatic_region": "Marine West Coast"},
    {"id": "agri_012", "ecosystem": "agricultural", "lon": -106.0000, "lat": 52.0000,
     "name": "Saskatchewan Canola Fields, Canada", "protected_area": False, "climatic_region": "Subarctic"},
    {"id": "agri_013", "ecosystem": "agricultural", "lon": 105.8000, "lat": 10.2500,
     "name": "Mekong Delta Rice Fields, Vietnam", "protected_area": False, "climatic_region": "Tropical Monsoon"},
    {"id": "agri_014", "ecosystem": "agricultural", "lon": -47.8000, "lat": -15.8000,
     "name": "Cerrado Soybean Fields, Brazil", "protected_area": False, "climatic_region": "Tropical Savanna"},
    {"id": "agri_015", "ecosystem": "agricultural", "lon": 142.5000, "lat": 43.5000,
     "name": "Hokkaido Potato Farms, Japan", "protected_area": False, "climatic_region": "Humid Continental"},

    # Urban green spaces (15)
    {"id": "urban_green_001", "ecosystem": "urban_green", "lon": 77.5946, "lat": 12.9716,
     "name": "Cubbon Park, Bangalore, India", "protected_area": True, "climatic_region": "Tropical Savanna"},
    {"id": "urban_green_002", "ecosystem": "urban_green", "lon": -73.9654, "lat": 40.7829,
     "name": "Central Park, New York, USA", "protected_area": True, "climatic_region": "Humid Subtropical"},
    {"id": "urban_green_003", "ecosystem": "urban_green", "lon": -0.1657, "lat": 51.5073,
     "name": "Hyde Park, London, UK", "protected_area": True, "climatic_region": "Marine West Coast"},
    {"id": "urban_green_004", "ecosystem": "urban_green", "lon": 2.2500, "lat": 48.8600,
     "name": "Bois de Boulogne, Paris, France", "protected_area": True, "climatic_region": "Marine West Coast"},
    {"id": "urban_green_005", "ecosystem": "urban_green", "lon": -122.4862, "lat": 37.7690,
     "name": "Golden Gate Park, San Francisco, USA", "protected_area": True, "climatic_region": "Mediterranean"},
    {"id": "urban_green_006", "ecosystem": "urban_green", "lon": 13.3600, "lat": 52.5100,
     "name": "Tiergarten, Berlin, Germany", "protected_area": True, "climatic_region": "Marine West Coast"},
    {"id": "urban_green_007", "ecosystem": "urban_green", "lon": 139.7100, "lat": 35.6800,
     "name": "Shinjuku Gyoen, Tokyo, Japan", "protected_area": True, "climatic_region": "Humid Subtropical"},
    {"id": "urban_green_008", "ecosystem": "urban_green", "lon": -46.6570, "lat": -23.5850,
     "name": "Ibirapuera Park, Sao Paulo, Brazil", "protected_area": True, "climatic_region": "Humid Subtropical"},
    {"id": "urban_green_009", "ecosystem": "urban_green", "lon": 151.2300, "lat": -33.9000,
     "name": "Centennial Park, Sydney, Australia", "protected_area": True, "climatic_region": "Humid Subtropical"},
    {"id": "urban_green_010", "ecosystem": "urban_green", "lon": -99.1860, "lat": 19.4200,
     "name": "Chapultepec Park, Mexico City, Mexico", "protected_area": True, "climatic_region": "Subtropical Highland"},
    {"id": "urban_green_011", "ecosystem": "urban_green", "lon": 144.9800, "lat": -37.8300,
     "name": "Royal Botanic Gardens, Melbourne, Australia", "protected_area": True, "climatic_region": "Temperate Oceanic"},
    {"id": "urban_green_012", "ecosystem": "urban_green", "lon": -123.1400, "lat": 49.3000,
     "name": "Stanley Park, Vancouver, Canada", "protected_area": True, "climatic_region": "Marine West Coast"},
    {"id": "urban_green_013", "ecosystem": "urban_green", "lon": 100.5400, "lat": 13.7300,
     "name": "Lumpini Park, Bangkok, Thailand", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
    {"id": "urban_green_014", "ecosystem": "urban_green", "lon": 103.8150, "lat": 1.3100,
     "name": "Singapore Botanic Gardens, Singapore", "protected_area": True, "climatic_region": "Equatorial"},
    {"id": "urban_green_015", "ecosystem": "urban_green", "lon": 77.2200, "lat": 28.5900,
     "name": "Lodhi Gardens, New Delhi, India", "protected_area": True, "climatic_region": "Semi-Arid"},

    # ==================================================================
    # PHASE 1 EXPANSION (added 3 Sep) -- the five ecosystem categories the
    # faculty review named that did not exist: savanna, grassland, tundra,
    # boreal/taiga, dryland/shrubland. Takes 81 -> 131 configured locations
    # and 5 -> 10 categories (implementation.md Phase 1 / comment C5).
    #
    # EVERY coordinate below was verified against RESOLVE Ecoregions 2017
    # BEFORE being added: its actual BIOME_NAME was checked to match the
    # intended category, and the "realm" field is RESOLVE's own REALM at
    # that point, not a hand-typed guess. Two candidates were REJECTED and
    # relocated by that check rather than being discovered later from the
    # pixels:
    #   * Hortobagy Puszta, Hungary (21.00, 47.60) -> RESOLVE says
    #     "Pannonian mixed forests"; the puszta is a grassland enclave too
    #     small for the polygon. Replaced with Askania-Nova, Ukraine.
    #   * Kolyma lowland (155.00, 69.00) -> "Northeast Siberian taiga";
    #     the treeline is further north. Moved to (158.00, 70.50).
    # RESOLVE stays ADVISORY at this footprint (it called Jiuzhaigou
    # grassland where the imagery showed 58% forest), so 00's pixel-level
    # CONTENT check still governs -- see implementation.md 9, "trust pixels
    # over maps". This check is cheap insurance, not the final word.
    #
    # "realm" is included on the new entries only; Phase 3's cross-region
    # protocol should derive it for ALL locations from geo_lookups rather
    # than relying on this field being hand-maintained.
    # ==================================================================

    # ---- savanna (Tropical & Subtropical Grasslands, Savannas & Shrublands) ----
    {"id": "savanna_001", "ecosystem": "savanna", "lon": 34.8300, "lat": -2.3300,
     "name": "Serengeti Plains, Tanzania", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry", "realm": "Afrotropic"},
    {"id": "savanna_002", "ecosystem": "savanna", "lon": 35.1000, "lat": -1.5000,
     "name": "Maasai Mara, Kenya", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry", "realm": "Afrotropic"},
    {"id": "savanna_003", "ecosystem": "savanna", "lon": 31.5000, "lat": -24.0000,
     "name": "Kruger National Park, South Africa", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry", "realm": "Afrotropic"},
    {"id": "savanna_004", "ecosystem": "savanna", "lon": -47.5000, "lat": -15.5000,
     "name": "Cerrado, Goias, Brazil", "protected_area": False, "climatic_region": "Tropical Wet-and-Dry", "realm": "Neotropic"},
    # RELOCATED 3 Sep: (-68.00, 7.50) is the EXACT coordinate of wetland_010 "Llanos
    # Swamps, Venezuela". 00 flagged it DUPLICATE at 0.0 km -- one patch cannot be both
    # savanna and wetland, and a label pair that sits on identical pixels is unlearnable
    # by construction (cf. the two "wetland" sites inside the Sundarbans mangrove biome,
    # §3.1). Moved 511 km west into the Colombian Llanos, still RESOLVE "Llanos" ecoregion
    # but well-drained savanna rather than seasonally flooded ground.
    {"id": "savanna_005", "ecosystem": "savanna", "lon": -71.5000, "lat": 4.5000,
     "name": "Colombian Llanos, Meta", "protected_area": False, "climatic_region": "Tropical Wet-and-Dry", "realm": "Neotropic"},
    {"id": "savanna_006", "ecosystem": "savanna", "lon": 132.5000, "lat": -12.8000,
     "name": "Kakadu savanna, Australia", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry", "realm": "Australasia"},
    {"id": "savanna_007", "ecosystem": "savanna", "lon": 24.5000, "lat": -18.5000,
     "name": "Chobe woodland, Botswana", "protected_area": True, "climatic_region": "Semi-Arid", "realm": "Afrotropic"},
    {"id": "savanna_008", "ecosystem": "savanna", "lon": 27.5000, "lat": -13.5000,
     "name": "Miombo woodland, Zambia", "protected_area": False, "climatic_region": "Tropical Wet-and-Dry", "realm": "Afrotropic"},
    {"id": "savanna_009", "ecosystem": "savanna", "lon": -60.0000, "lat": -21.0000,
     "name": "Gran Chaco, Paraguay", "protected_area": False, "climatic_region": "Semi-Arid", "realm": "Neotropic"},
    {"id": "savanna_010", "ecosystem": "savanna", "lon": -1.5000, "lat": 12.0000,
     "name": "Sudanian savanna, Burkina Faso", "protected_area": False, "climatic_region": "Semi-Arid", "realm": "Afrotropic"},

    # ---- grassland (Temperate Grasslands, Savannas & Shrublands) ----
    {"id": "grassland_001", "ecosystem": "grassland", "lon": -96.6000, "lat": 39.1000,
     "name": "Konza Prairie, Kansas, USA", "protected_area": True, "climatic_region": "Humid Continental", "realm": "Nearctic"},
    {"id": "grassland_002", "ecosystem": "grassland", "lon": -60.0000, "lat": -35.0000,
     "name": "Pampas, Buenos Aires, Argentina", "protected_area": False, "climatic_region": "Humid Subtropical", "realm": "Neotropic"},
    {"id": "grassland_003", "ecosystem": "grassland", "lon": 105.0000, "lat": 47.0000,
     "name": "Mongolian Steppe", "protected_area": False, "climatic_region": "Cold Semi-Arid", "realm": "Palearctic"},
    {"id": "grassland_004", "ecosystem": "grassland", "lon": 68.0000, "lat": 50.5000,
     "name": "Kazakh Steppe", "protected_area": False, "climatic_region": "Cold Semi-Arid", "realm": "Palearctic"},
    # RELOCATED from Hortobagy Puszta, Hungary (21.00, 47.60) -- RESOLVE placed that point
    # in "Pannonian mixed forests", see the block comment above.
    {"id": "grassland_005", "ecosystem": "grassland", "lon": 33.8800, "lat": 46.4500,
     "name": "Askania-Nova Steppe, Ukraine", "protected_area": True, "climatic_region": "Cold Semi-Arid", "realm": "Palearctic"},
    {"id": "grassland_006", "ecosystem": "grassland", "lon": -105.0000, "lat": 51.0000,
     "name": "Saskatchewan Prairie, Canada", "protected_area": False, "climatic_region": "Humid Continental", "realm": "Nearctic"},
    {"id": "grassland_007", "ecosystem": "grassland", "lon": -69.0000, "lat": -45.0000,
     "name": "Patagonian Steppe, Argentina", "protected_area": False, "climatic_region": "Cold Semi-Arid", "realm": "Neotropic"},
    {"id": "grassland_008", "ecosystem": "grassland", "lon": 116.0000, "lat": 43.5000,
     "name": "Inner Mongolia Grassland, China", "protected_area": False, "climatic_region": "Cold Semi-Arid", "realm": "Palearctic"},
    {"id": "grassland_009", "ecosystem": "grassland", "lon": 115.0000, "lat": 50.5000,
     "name": "Daurian Steppe, Russia", "protected_area": True, "climatic_region": "Cold Semi-Arid", "realm": "Palearctic"},
    {"id": "grassland_010", "ecosystem": "grassland", "lon": -96.5000, "lat": 38.5000,
     "name": "Flint Hills, Kansas, USA", "protected_area": False, "climatic_region": "Humid Continental", "realm": "Nearctic"},

    # ---- tundra ----
    {"id": "tundra_001", "ecosystem": "tundra", "lon": -152.0000, "lat": 69.5000,
     "name": "North Slope tundra, Alaska, USA", "protected_area": False, "climatic_region": "Arctic", "realm": "Nearctic"},
    {"id": "tundra_002", "ecosystem": "tundra", "lon": 100.0000, "lat": 73.0000,
     "name": "Taimyr Peninsula, Russia", "protected_area": False, "climatic_region": "Arctic", "realm": "Palearctic"},
    {"id": "tundra_003", "ecosystem": "tundra", "lon": -72.0000, "lat": 69.0000,
     "name": "Baffin Island, Canada", "protected_area": False, "climatic_region": "Arctic", "realm": "Nearctic"},
    {"id": "tundra_004", "ecosystem": "tundra", "lon": 70.0000, "lat": 70.0000,
     "name": "Yamal Peninsula, Russia", "protected_area": False, "climatic_region": "Arctic", "realm": "Palearctic"},
    {"id": "tundra_005", "ecosystem": "tundra", "lon": -100.0000, "lat": 65.0000,
     "name": "Nunavut mainland tundra, Canada", "protected_area": False, "climatic_region": "Arctic", "realm": "Nearctic"},
    # RELOCATED 3 Sep after acquisition: (175.00, 67.00) came back 86% open water across
    # all 6 candidate scenes -- that point is on the Chukotka coast. Same structural
    # failure as the mangrove/salt-marsh losses (§3.2): coastal coordinates cannot fill a
    # 2.24 km patch with land. Moved inland to the Anadyr upland, still "Russian Bering
    # tundra" in RESOLVE, but elevated and away from the shoreline.
    {"id": "tundra_006", "ecosystem": "tundra", "lon": 173.0000, "lat": 65.5000,
     "name": "Anadyr upland tundra, Russia", "protected_area": False, "climatic_region": "Arctic", "realm": "Palearctic"},
    # RELOCATED 3 Sep: (-50.00, 67.00) West Greenland returned NDVI -0.122 with 79.3%
    # water -- icecap and fjord, not tundra. It passed the old gates because 79.3% sat
    # just under the 80% water limit and no content check applied to tundra at the time;
    # ECOSYSTEM_NDVI_BANDS now catches it. Moved to continental NW Russian tundra.
    {"id": "tundra_007", "ecosystem": "tundra", "lon": 57.0000, "lat": 68.0000,
     "name": "Bolshezemelskaya tundra, Russia", "protected_area": False, "climatic_region": "Arctic", "realm": "Palearctic"},
    # RELOCATED 3 Sep: (20.00, 69.50) Finnmark returned NDVI 0.047 with 26% water --
    # bare mountain plateau rather than vegetated tundra.
    {"id": "tundra_008", "ecosystem": "tundra", "lon": 170.0000, "lat": 67.0000,
     "name": "Chukchi Peninsula tundra, Russia", "protected_area": False, "climatic_region": "Arctic", "realm": "Palearctic"},
    {"id": "tundra_009", "ecosystem": "tundra", "lon": -164.0000, "lat": 65.0000,
     "name": "Seward Peninsula, Alaska, USA", "protected_area": False, "climatic_region": "Subarctic", "realm": "Nearctic"},
    # RELOCATED from (155.00, 69.00), which RESOLVE placed in "Northeast Siberian taiga".
    # RELOCATED TWICE. Originally Kolyma lowland (155.00, 69.00), which RESOLVE placed in
    # "Northeast Siberian taiga"; moved to the Kolyma delta (158.00, 70.50), which then
    # returned NDVI -0.366 with 68% water -- a river delta is water. Now on the Putorana
    # plateau: inland, elevated, and genuinely vegetated in season. Third attempt, and the
    # §9 rule applies -- if this one fails too, record the loss rather than keep guessing.
    {"id": "tundra_010", "ecosystem": "tundra", "lon": 94.0000, "lat": 69.5000,
     "name": "Putorana plateau tundra, Russia", "protected_area": True, "climatic_region": "Arctic", "realm": "Palearctic"},

    # ---- boreal / taiga ----
    {"id": "boreal_001", "ecosystem": "boreal", "lon": -113.0000, "lat": 56.0000,
     "name": "Boreal forest, Alberta, Canada", "protected_area": False, "climatic_region": "Subarctic", "realm": "Nearctic"},
    {"id": "boreal_002", "ecosystem": "boreal", "lon": 27.0000, "lat": 64.0000,
     "name": "Taiga, Kainuu, Finland", "protected_area": False, "climatic_region": "Subarctic", "realm": "Palearctic"},
    {"id": "boreal_003", "ecosystem": "boreal", "lon": 18.0000, "lat": 64.0000,
     "name": "Boreal forest, Vasterbotten, Sweden", "protected_area": False, "climatic_region": "Subarctic", "realm": "Palearctic"},
    {"id": "boreal_004", "ecosystem": "boreal", "lon": 130.0000, "lat": 62.0000,
     "name": "Yakutia larch taiga, Russia", "protected_area": False, "climatic_region": "Subarctic", "realm": "Palearctic"},
    {"id": "boreal_005", "ecosystem": "boreal", "lon": -85.0000, "lat": 50.0000,
     "name": "Boreal shield, Ontario, Canada", "protected_area": False, "climatic_region": "Subarctic", "realm": "Nearctic"},
    {"id": "boreal_006", "ecosystem": "boreal", "lon": -72.0000, "lat": 52.0000,
     "name": "Boreal forest, Quebec, Canada", "protected_area": False, "climatic_region": "Subarctic", "realm": "Nearctic"},
    {"id": "boreal_007", "ecosystem": "boreal", "lon": -148.0000, "lat": 64.5000,
     "name": "Interior Alaska taiga, USA", "protected_area": False, "climatic_region": "Subarctic", "realm": "Nearctic"},
    {"id": "boreal_008", "ecosystem": "boreal", "lon": 92.0000, "lat": 60.0000,
     "name": "Krasnoyarsk taiga, Russia", "protected_area": False, "climatic_region": "Subarctic", "realm": "Palearctic"},
    # RELOCATED 3 Sep: (58.50, 63.50) returned NDVI 0.214 with only 3.5% forest cover from
    # a MAY scene -- pre-greenup snow at 63.5N, not canopy. Adding "boreal" to
    # VEGETATED_ECOSYSTEMS makes 01 reject that combination. Moved ~2 degrees south, where
    # the growing season is long enough that a cloud-free summer scene is available.
    {"id": "boreal_009", "ecosystem": "boreal", "lon": 52.0000, "lat": 61.5000,
     "name": "South Komi taiga, Russia", "protected_area": False, "climatic_region": "Subarctic", "realm": "Palearctic"},
    {"id": "boreal_010", "ecosystem": "boreal", "lon": 42.0000, "lat": 63.0000,
     "name": "Arkhangelsk taiga, Russia", "protected_area": False, "climatic_region": "Subarctic", "realm": "Palearctic"},

    # ---- dryland / shrubland (Deserts & Xeric Shrublands; Mediterranean Woodlands & Scrub) ----
    {"id": "shrubland_001", "ecosystem": "shrubland", "lon": -111.5000, "lat": 32.2000,
     "name": "Sonoran Desert, Arizona, USA", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Nearctic"},
    {"id": "shrubland_002", "ecosystem": "shrubland", "lon": -116.5000, "lat": 35.0000,
     "name": "Mojave Desert, California, USA", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Nearctic"},
    {"id": "shrubland_003", "ecosystem": "shrubland", "lon": -104.0000, "lat": 28.0000,
     "name": "Chihuahuan Desert, Mexico", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Nearctic"},
    {"id": "shrubland_004", "ecosystem": "shrubland", "lon": 22.0000, "lat": -32.5000,
     "name": "Great Karoo, South Africa", "protected_area": False, "climatic_region": "Cold Semi-Arid", "realm": "Afrotropic"},
    {"id": "shrubland_005", "ecosystem": "shrubland", "lon": 122.0000, "lat": -25.0000,
     "name": "Mulga shrubland, Australia", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Australasia"},
    {"id": "shrubland_006", "ecosystem": "shrubland", "lon": -69.5000, "lat": -24.0000,
     "name": "Atacama margin, Chile", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Neotropic"},
    {"id": "shrubland_007", "ecosystem": "shrubland", "lon": 15.5000, "lat": -23.5000,
     "name": "Namib escarpment, Namibia", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Afrotropic"},
    {"id": "shrubland_008", "ecosystem": "shrubland", "lon": 71.5000, "lat": 27.0000,
     "name": "Thar Desert margin, India", "protected_area": False, "climatic_region": "Hot Desert", "realm": "Indomalayan"},
    {"id": "shrubland_009", "ecosystem": "shrubland", "lon": 105.0000, "lat": 43.0000,
     "name": "Gobi Desert, Mongolia", "protected_area": False, "climatic_region": "Cold Desert", "realm": "Palearctic"},
    {"id": "shrubland_010", "ecosystem": "shrubland", "lon": -4.0000, "lat": 38.0000,
     "name": "Iberian matorral, Spain", "protected_area": False, "climatic_region": "Mediterranean", "realm": "Palearctic"},
]

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

# PROJECT ROOT -- every path below is anchored here rather than to the process's working
# directory. Before the 15 Sep restructure these were bare relative strings ("patches",
# "results"), so a script only found its data when launched from the repo root; running one
# from anywhere else silently created a second, empty tree. Deriving the root from this
# file's own location removes that whole class of failure.
import os as _os
PROJECT_ROOT = _os.path.dirname(_os.path.abspath(__file__))

def _p(*parts):
    """Path under the project root, with forward slashes (harmless on Windows, and it
    keeps paths stored in catalog.json portable across machines)."""
    return _os.path.join(PROJECT_ROOT, *parts).replace("\\", "/")

# Inputs and intermediate artifacts live under data/, everything the pipeline produces
# under outputs/. Keeping the two apart is what makes "delete outputs and re-run" safe.
DATA_DIR = _p("data")
OUTPUTS_DIR = _p("outputs")
MODELS_DIR = _p("models")

PATCHES_DIR = _p("data", "patches")
PROCESSED_PATCHES_DIR = _p("data", "patches_processed")
# Records the exact coordinates, scene and quality of every patch on disk.
# Written by 01_acquire_patches.py and read back on the next run to decide
# whether an existing patch still matches config. catalog.json cannot serve
# this purpose because step 02 overwrites it with jittered sub-crop entries.
ACQUISITION_MANIFEST_PATH = f"{PATCHES_DIR}/acquisition_manifest.json"
METADATA_DIR = _p("data", "metadata")
EMBEDDINGS_DIR = _p("data", "embeddings", "prithvi")
METADATA_CATALOG_PATH = f"{METADATA_DIR}/catalog.json"

# ---------------------------------------------------------------
# Objective 2 — Retrieval Engine Configuration
# ---------------------------------------------------------------

RESULTS_DIR = _p("outputs", "results")
LOGS_DIR = _p("outputs", "logs")
DASHBOARDS_DIR = _p("outputs", "dashboards")
DEFAULT_TOP_K = 10           # Default number of similar ecosystems to retrieve
EVALUATION_K_VALUES = [1, 3, 5, 10]  # K values for Precision@K, Recall@K evaluation

# ---------------------------------------------------------------
# Multi-Model Comparison Configuration
# ---------------------------------------------------------------

SUPPORTED_MODELS = {
    "prithvi": {
        "label": "Prithvi-100M",
        "description": "NASA/IBM Geospatial Foundation Model (6-band, 768D)",
        "embedding_dim": 768,
        "embeddings_dir": _p("data", "embeddings", "prithvi"),
        "timm_name": None,                       # Not a timm model
    },
    "vit": {
        "label": "ViT-Base",
        "description": "Vision Transformer Base (ImageNet, RGB, 768D)",
        "embedding_dim": 768,
        "embeddings_dir": _p("data", "embeddings", "vit"),
        "timm_name": "vit_base_patch16_224",
    },
    "resnet": {
        "label": "ResNet-50",
        "description": "ResNet-50 (ImageNet, RGB, 2048D)",
        "embedding_dim": 2048,
        "embeddings_dir": _p("data", "embeddings", "resnet"),
        "timm_name": "resnet50",
    },
    "clay": {
        "label": "Clay-v1.5",
        "description": "Clay Foundation Model v1.5, large (real weights, 1024D)",
        "embedding_dim": 1024,
        "embeddings_dir": _p("data", "embeddings", "clay"),
        "timm_name": None,                       # Not a timm model -- see load_clay_model()
    },
    "satlas": {
        "label": "Satlas-ResNet50",
        "description": "AllenAI SatlasPretrain Sentinel2_Resnet50_SI_RGB (real weights, 2048D)",
        "embedding_dim": 2048,
        "embeddings_dir": _p("data", "embeddings", "satlas"),
        "timm_name": None,                       # Not a timm model -- see load_satlas_model()
    },
}

DEFAULT_MODEL = "prithvi"

# ImageNet normalization stats (for ViT-Base and ResNet-50)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------
# Real geospatial reference data (used by geo_lookups.py)
#
# These replace the hash-based fabricated "physical descriptors"
# that used to live in 09_explainability_engine.py. Every path
# below is a local file you download ONCE (see README.md ->
# "Reference data setup" for exact download links/instructions).
# geo_lookups.py will refuse to fabricate a value if a file is
# missing -- it returns None for that field and the caller must
# label it explicitly as unavailable, never guess.
# ---------------------------------------------------------------

GEO_DATA_DIR = _p("data", "reference")

# WDPA (World Database on Protected Areas), UNEP-WCMC/IUCN.
# Download the "shapefile" export from protectedplanet.net and point
# this at the combined polygons layer (merge the 0/1/2 parts first).
# Built by 26_build_wdpa_layer.py from the public WDPA release: terrestrial + coastal,
# designated/inscribed/established only, 299,473 polygons. GeoPackage rather than
# shapefile -- shapefiles cap at 2 GB and this layer is 4.94 GB -- and its R-tree is
# what makes is_protected() a bbox pushdown instead of a full-layer scan.
WDPA_POLYGONS_PATH = f"{GEO_DATA_DIR}/WDPA_terrestrial_designated.gpkg"
WDPA_ACCEPTED_STATUSES = ["Designated", "Inscribed", "Established"]  # excludes "Proposed"

# WorldClim v2.1 bioclim rasters, worldclim.org.
# BIO1 = annual mean temperature (deg C * 10), BIO12 = annual precipitation (mm)
#
# RESOLUTION: 2.5 arc-minute (~4.6 km), NOT the 30 arc-sec (~1 km) these paths
# used to name. WorldClim only ships bioclim as a single 19-variable bundle per
# resolution, and the 30s bundle is a 10.4 GB download to obtain two of those
# 19 rasters -- not a proportionate cost for a variable this project reads as
# regional context. The 2.5m bundle is ~628 MB.
#
# What this costs: 4.6 km cells are COARSER than the 2.24 km patches, so a
# single climate value now spans roughly four patches. Temperature and rainfall
# were already documented as describing "the region around a point, not
# something specific to a given patch" (see README, "Resolution mismatches in
# geo_lookups data") -- this widens that gap rather than introducing a new kind
# of error. Do not present these as patch-level measurements.
#
# To upgrade later: download wc2.1_30s_bio.zip, extract bio_1 and bio_12 to
# GEO_DATA_DIR, repoint the two paths below, and re-run 09 and 10. Nothing else
# needs to change -- geo_lookups.py samples by coordinate, not by grid size.
WORLDCLIM_TEMP_PATH = f"{GEO_DATA_DIR}/wc2.1_2.5m_bio_1.tif"
WORLDCLIM_PRECIP_PATH = f"{GEO_DATA_DIR}/wc2.1_2.5m_bio_12.tif"

# Elevation: SRTM 30m or Copernicus GLO-30 DEM, mosaicked/VRT covering
# your patch locations.
# Elevation: SRTM 30m or Copernicus GLO-30 DEM. Tiles are looked up
# per-point on demand (same pattern as Hansen tiles in
# 10_grid_tiling_labels.py) rather than mosaicked into one file --
# this project's locations are scattered worldwide, and mosaicking
# globally-scattered 1x1 degree tiles into a single dense array means
# allocating a raster covering the full combined bounding box, which
# for worldwide points is most of the planet (a multi-terabyte
# allocation in practice -- this is what a real run hit). Keep the
# downloaded tiles in DEM_TILES_DIR; geo_lookups.py finds the right
# one per query point by filename convention.
DEM_TILES_DIR = f"{GEO_DATA_DIR}/dem_tiles"

# RESOLVE Ecoregions 2017 (resolve.org/ecoregions), single global shapefile.
ECOREGIONS_PATH = f"{GEO_DATA_DIR}/Ecoregions2017.shp"

# Local cache of (lat, lon) -> descriptor lookups, to avoid re-sampling
# the same rasters repeatedly across 700+ sub-crops.
GEO_LOOKUP_CACHE_PATH = f"{GEO_DATA_DIR}/geo_lookup_cache.json"

# ---------------------------------------------------------------
# Evaluation methodology
# ---------------------------------------------------------------

# 02_preprocess_patches.py expands each base location into GROUP_SIZE
# sub-crops with heavy pixel overlap (see docstring there). Standard
# per-patch leave-one-out evaluation therefore lets a query retrieve
# near-duplicate crops of itself, inflating P@1/mAP. GROUP_AWARE_EVAL
# controls whether 07_evaluate_retrieval.py also computes a
# leave-one-LOCATION-out variant that excludes all sub-crops sharing
# a base_id from both the candidate pool and the relevant set. This
# does not replace the standard ("leaked") metrics -- both are
# reported side by side so the gap between them is visible.
GROUP_AWARE_EVAL = True

# ---------------------------------------------------------------
# Objective 4 -- Forest-Loss Risk Forecasting
# ---------------------------------------------------------------

# Hansen Global Forest Change (Hansen/UMD/Google/USGS/NASA), tiled
# GeoTIFFs, public GCS bucket, no auth required:
#   https://storage.googleapis.com/earthenginepartners-hansen/GFC-<version>/<layer>_<lat>_<lon>.tif
HANSEN_DATA_DIR = _p("data", "hansen")
HANSEN_VERSION = "GFC-2023-v1.11"
HANSEN_BASE_URL = f"https://storage.googleapis.com/earthenginepartners-hansen/{HANSEN_VERSION}"
# Minimum canopy cover (%) in Hansen's "treecover2000" layer for a
# pixel to count as forest at baseline -- standard literature default.
HANSEN_TREECOVER_THRESHOLD = 30

# Grid-tiling for risk modeling: forest-type base locations are tiled
# into GRID_CELL_SIZE_M x GRID_CELL_SIZE_M cells so the model trains
# on hundreds-to-thousands of independent cell-year samples instead
# of ~30-45 named locations (see README.md -> "Sample size").
GRID_CELL_SIZE_M = 1000          # 1km cells, matched to WorldClim resolution
GRID_REGION_BUFFER_KM = 15       # radius around each base location to tile
RISK_FOREST_ECOSYSTEMS = ["forest"]

# Prediction target: was there tree-cover loss within this many years
# after the observation year, in a Hansen-labeled cell.
RISK_HORIZON_YEARS = 2
RISK_MODEL_DIR = _p("outputs", "risk_model")
RISK_FEATURES_PATH = f"{RISK_MODEL_DIR}/cell_year_features.csv"
RISK_MODEL_PATH = f"{RISK_MODEL_DIR}/forest_loss_risk_model.joblib"

# ---------------------------------------------------------------
# Directory skeleton
# ---------------------------------------------------------------
# Created on import, because config is imported by every script in the pipeline and this is
# the only place that sees all of the paths at once. Four scripts (05, 07, 07b, 08) write
# results without calling makedirs themselves; on a fresh clone -- where data/ and outputs/
# are gitignored and therefore absent -- they would fail on their first write. Creating the
# tree here means a from-scratch run reproduces exactly this layout with no manual setup.
#
# exist_ok=True throughout: this is idempotent and never touches existing content.
_ENSURE_DIRS = [
    DATA_DIR, OUTPUTS_DIR, MODELS_DIR,
    PATCHES_DIR, PROCESSED_PATCHES_DIR, METADATA_DIR,
    GEO_DATA_DIR, HANSEN_DATA_DIR, DEM_TILES_DIR,
    RESULTS_DIR, LOGS_DIR, DASHBOARDS_DIR, RISK_MODEL_DIR,
] + [m["embeddings_dir"] for m in SUPPORTED_MODELS.values()]

for _d in _ENSURE_DIRS:
    try:
        _os.makedirs(_d, exist_ok=True)
    except OSError:
        # A read-only checkout should not stop a caller that only wants to READ config
        # constants (e.g. to print a path); the write itself will fail loudly later.
        pass
