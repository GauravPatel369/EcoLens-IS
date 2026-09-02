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

SEARCH_DATE_RANGE = "2024-01-01/2024-06-30"
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
    {"id": "forest_014", "ecosystem": "forest", "lon": 103.9184, "lat": 33.2613,
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
    {"id": "wetland_002", "ecosystem": "wetland", "lon": -81.3930, "lat": 25.8650,
     "name": "Everglades, Florida, USA", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
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
    {"id": "wetland_011", "ecosystem": "wetland", "lon": 48.0000, "lat": 46.0000,
     "name": "Volga Delta, Russia", "protected_area": True, "climatic_region": "Temperate Wetland"},
    # CORRECTED 89.6500, 22.0500 -> 85.3500, 19.7000: duplicated the Sundarbans delta already covered by mangrove_001; moved to Chilika, a distinct brackish lagoon wetland
    {"id": "wetland_012", "ecosystem": "wetland", "lon": 85.3500, "lat": 19.7000,
     "name": "Chilika Lake Wetland, India", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "wetland_013", "ecosystem": "wetland", "lon": -111.5000, "lat": 58.7500,
     "name": "Peace-Athabasca Delta, Canada", "protected_area": True, "climatic_region": "Subarctic"},
    {"id": "wetland_014", "ecosystem": "wetland", "lon": 47.0000, "lat": 31.0000,
     "name": "Mesopotamian Marshes, Iraq", "protected_area": False, "climatic_region": "Arid Marshland"},
    {"id": "wetland_015", "ecosystem": "wetland", "lon": 6.2000, "lat": 53.4500,
     "name": "Wadden Sea Salt Marshes, Netherlands", "protected_area": True, "climatic_region": "Temperate Coastal"},
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
    {"id": "mangrove_004", "ecosystem": "mangrove", "lon": -79.7000, "lat": -2.4300,
     "name": "Gulf of Guayaquil Mangroves, Ecuador", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_005", "ecosystem": "mangrove", "lon": 79.7820, "lat": 11.4310,
     "name": "Pichavaram Mangroves, India", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_006", "ecosystem": "mangrove", "lon": 86.8500, "lat": 20.6500,
     "name": "Bhitarkanika Mangroves, India", "protected_area": True, "climatic_region": "Tropical Monsoon"},
    # CORRECTED 106.7750, -6.1100 -> 108.8000, -7.7000: Muara Angke reserve is ~25 ha, far smaller than a 2.24 km patch, so 72% of it was water; swapped for Segara Anakan, a mangrove system larger than the patch footprint
    {"id": "mangrove_007", "ecosystem": "mangrove", "lon": 108.8000, "lat": -7.7000,
     "name": "Segara Anakan Mangroves, Indonesia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED 44.2500, -16.0000 -> 44.4500, -19.6500: patch measured 100% open water (Mozambique Channel); moved to the Tsiribihina delta
    {"id": "mangrove_008", "ecosystem": "mangrove", "lon": 44.4500, "lat": -19.6500,
     "name": "Madagascar Mangroves, Madagascar", "protected_area": True, "climatic_region": "Tropical Dry"},
    # CORRECTED 34.6000, 25.0000 -> 35.0200, 24.2800: was 539 m up in the Red Sea Hills; moved to the Hamata coastal mangroves
    {"id": "mangrove_009", "ecosystem": "mangrove", "lon": 35.0200, "lat": 24.2800,
     "name": "Red Sea Mangroves, Egypt", "protected_area": True, "climatic_region": "Arid Coastal"},
    {"id": "mangrove_010", "ecosystem": "mangrove", "lon": 145.2500, "lat": -15.4500,
     "name": "Great Barrier Reef Mangroves, Australia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_011", "ecosystem": "mangrove", "lon": -61.4500, "lat": 10.5800,
     "name": "Caroni Swamp Mangroves, Trinidad", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_012", "ecosystem": "mangrove", "lon": -87.5000, "lat": 20.1500,
     "name": "Yucatan Peninsula Mangroves, Mexico", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED -16.2000, 11.8000 -> -16.0500, 12.3000: patch measured 100% open water (offshore Atlantic); moved to the Rio Cacheu estuary
    {"id": "mangrove_013", "ecosystem": "mangrove", "lon": -16.0500, "lat": 12.3000,
     "name": "Guinea-Bissau Mangroves, Guinea-Bissau", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_014", "ecosystem": "mangrove", "lon": 100.6200, "lat": 4.8500,
     "name": "Matang Mangrove Forest, Malaysia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_015", "ecosystem": "mangrove", "lon": 113.6800, "lat": -25.8000,
     "name": "Shark Bay Mangroves, Australia", "protected_area": True, "climatic_region": "Semi-Arid Coastal"},
    {"id": "mangrove_016", "ecosystem": "mangrove", "lon": -81.0000, "lat": 24.5000,
     "name": "Key West Mangroves, USA", "protected_area": True, "climatic_region": "Tropical Coastal"},
    # CORRECTED 47.5000, -15.5000 -> 47.1500, -15.4500: was inland at 84 m; moved into Mahajamba Bay proper
    {"id": "mangrove_017", "ecosystem": "mangrove", "lon": 47.1500, "lat": -15.4500,
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
]

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------

PATCHES_DIR = "patches"
# Records the exact coordinates, scene and quality of every patch on disk.
# Written by 01_acquire_patches.py and read back on the next run to decide
# whether an existing patch still matches config. catalog.json cannot serve
# this purpose because step 02 overwrites it with jittered sub-crop entries.
ACQUISITION_MANIFEST_PATH = f"{PATCHES_DIR}/acquisition_manifest.json"
METADATA_DIR = "metadata"
EMBEDDINGS_DIR = "embeddings"
METADATA_CATALOG_PATH = f"{METADATA_DIR}/catalog.json"

# ---------------------------------------------------------------
# Objective 2 — Retrieval Engine Configuration
# ---------------------------------------------------------------

RESULTS_DIR = "results"
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
        "embeddings_dir": "embeddings",          # Original location (backward-compat)
        "timm_name": None,                       # Not a timm model
    },
    "vit": {
        "label": "ViT-Base",
        "description": "Vision Transformer Base (ImageNet, RGB, 768D)",
        "embedding_dim": 768,
        "embeddings_dir": "embeddings_vit",
        "timm_name": "vit_base_patch16_224",
    },
    "resnet": {
        "label": "ResNet-50",
        "description": "ResNet-50 (ImageNet, RGB, 2048D)",
        "embedding_dim": 2048,
        "embeddings_dir": "embeddings_resnet",
        "timm_name": "resnet50",
    },
    "clay": {
        "label": "Clay-v1.5",
        "description": "Clay Foundation Model v1.5, large (real weights, 1024D)",
        "embedding_dim": 1024,
        "embeddings_dir": "embeddings_clay",
        "timm_name": None,                       # Not a timm model -- see load_clay_model()
    },
    "satlas": {
        "label": "Satlas-ResNet50",
        "description": "AllenAI SatlasPretrain Sentinel2_Resnet50_SI_RGB (real weights, 2048D)",
        "embedding_dim": 2048,
        "embeddings_dir": "embeddings_satlas",
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

GEO_DATA_DIR = "geo_data"

# WDPA (World Database on Protected Areas), UNEP-WCMC/IUCN.
# Download the "shapefile" export from protectedplanet.net and point
# this at the combined polygons layer (merge the 0/1/2 parts first).
WDPA_POLYGONS_PATH = f"{GEO_DATA_DIR}/WDPA_polygons.shp"
WDPA_ACCEPTED_STATUSES = ["Designated", "Inscribed", "Established"]  # excludes "Proposed"

# WorldClim v2 bioclim rasters (30 arc-sec), worldclim.org.
# BIO1 = annual mean temperature (deg C * 10), BIO12 = annual precipitation (mm)
WORLDCLIM_TEMP_PATH = f"{GEO_DATA_DIR}/wc2.1_30s_bio_1.tif"
WORLDCLIM_PRECIP_PATH = f"{GEO_DATA_DIR}/wc2.1_30s_bio_12.tif"

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
HANSEN_DATA_DIR = "hansen_data"
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
RISK_MODEL_DIR = "risk_model"
RISK_FEATURES_PATH = f"{RISK_MODEL_DIR}/cell_year_features.csv"
RISK_MODEL_PATH = f"{RISK_MODEL_DIR}/forest_loss_risk_model.joblib"