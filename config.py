"""
EcoLens Phase 1 — Configuration

Defines the 10 proof-of-concept patch locations across different
ecosystem categories, plus shared constants used by every script
in the pipeline.

Pick real coordinates for ecosystems you can verify on Google Maps
satellite view first — this avoids wasting a STAC query on a patch
that turns out to be the wrong land cover type.
"""


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
    {"id": "forest_011", "ecosystem": "forest", "lon": 37.3000, "lat": -0.1500,
     "name": "Mount Kenya Forest, Kenya", "protected_area": True, "climatic_region": "Montane Forest"},
    {"id": "forest_012", "ecosystem": "forest", "lon": 130.5000, "lat": 30.3500,
     "name": "Yakushima Forest, Japan", "protected_area": True, "climatic_region": "Warm Temperate"},
    {"id": "forest_013", "ecosystem": "forest", "lon": -73.2500, "lat": -39.8000,
     "name": "Valdivian Rainforest, Chile", "protected_area": True, "climatic_region": "Temperate Rainforest"},
    {"id": "forest_014", "ecosystem": "forest", "lon": 103.9184, "lat": 33.2613,
     "name": "Jiuzhaigou Forest, China", "protected_area": True, "climatic_region": "Montane Deciduous"},
    {"id": "forest_015", "ecosystem": "forest", "lon": -1.0730, "lat": 53.2030,
     "name": "Sherwood Forest, UK", "protected_area": True, "climatic_region": "Temperate Deciduous"},

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
    {"id": "wetland_006", "ecosystem": "wetland", "lon": 89.5000, "lat": 22.3000,
     "name": "Sundarbans Freshwater Swamps, Bangladesh", "protected_area": True, "climatic_region": "Tropical Swamps"},
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
    {"id": "wetland_012", "ecosystem": "wetland", "lon": 89.6500, "lat": 22.0500,
     "name": "Sunderbans Delta Wetlands, Bangladesh", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "wetland_013", "ecosystem": "wetland", "lon": -111.5000, "lat": 58.7500,
     "name": "Peace-Athabasca Delta, Canada", "protected_area": True, "climatic_region": "Subarctic"},
    {"id": "wetland_014", "ecosystem": "wetland", "lon": 47.0000, "lat": 31.0000,
     "name": "Mesopotamian Marshes, Iraq", "protected_area": False, "climatic_region": "Arid Marshland"},
    {"id": "wetland_015", "ecosystem": "wetland", "lon": 6.2000, "lat": 53.4500,
     "name": "Wadden Sea Salt Marshes, Netherlands", "protected_area": True, "climatic_region": "Temperate Coastal"},

    # Mangroves (15)
    {"id": "mangrove_001", "ecosystem": "mangrove", "lon": 88.8500, "lat": 21.9500,
     "name": "Sundarbans, West Bengal, India", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
    {"id": "mangrove_002", "ecosystem": "mangrove", "lon": -80.1500, "lat": 25.3500,
     "name": "Florida Bay mangroves, USA", "protected_area": True, "climatic_region": "Tropical Wet-and-Dry"},
    {"id": "mangrove_003", "ecosystem": "mangrove", "lon": 6.0000, "lat": 4.5000,
     "name": "Niger Delta Mangroves, Nigeria", "protected_area": False, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_004", "ecosystem": "mangrove", "lon": -80.0000, "lat": -2.7500,
     "name": "Gulf of Guayaquil Mangroves, Ecuador", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_005", "ecosystem": "mangrove", "lon": 79.7820, "lat": 11.4310,
     "name": "Pichavaram Mangroves, India", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_006", "ecosystem": "mangrove", "lon": 86.8500, "lat": 20.6500,
     "name": "Bhitarkanika Mangroves, India", "protected_area": True, "climatic_region": "Tropical Monsoon"},
    {"id": "mangrove_007", "ecosystem": "mangrove", "lon": 106.7750, "lat": -6.1100,
     "name": "Muara Angke Mangroves, Indonesia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_008", "ecosystem": "mangrove", "lon": 44.2500, "lat": -16.0000,
     "name": "Madagascar Mangroves, Madagascar", "protected_area": True, "climatic_region": "Tropical Dry"},
    {"id": "mangrove_009", "ecosystem": "mangrove", "lon": 34.6000, "lat": 25.0000,
     "name": "Red Sea Mangroves, Egypt", "protected_area": True, "climatic_region": "Arid Coastal"},
    {"id": "mangrove_010", "ecosystem": "mangrove", "lon": 145.2500, "lat": -15.4500,
     "name": "Great Barrier Reef Mangroves, Australia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_011", "ecosystem": "mangrove", "lon": -61.4500, "lat": 10.5800,
     "name": "Caroni Swamp Mangroves, Trinidad", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_012", "ecosystem": "mangrove", "lon": -87.5000, "lat": 20.1500,
     "name": "Yucatan Peninsula Mangroves, Mexico", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_013", "ecosystem": "mangrove", "lon": -16.2000, "lat": 11.8000,
     "name": "Guinea-Bissau Mangroves, Guinea-Bissau", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_014", "ecosystem": "mangrove", "lon": 100.6200, "lat": 4.8500,
     "name": "Matang Mangrove Forest, Malaysia", "protected_area": True, "climatic_region": "Tropical Coastal"},
    {"id": "mangrove_015", "ecosystem": "mangrove", "lon": 113.6800, "lat": -25.8000,
     "name": "Shark Bay Mangroves, Australia", "protected_area": True, "climatic_region": "Semi-Arid Coastal"},

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
     "name": "Great Plains Wheat Belt, Kansas, USA", "protected_area": False, "climatic_region": "Temperated Semi-Arid"},
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
METADATA_DIR = "metadata"
EMBEDDINGS_DIR = "embeddings"
METADATA_CATALOG_PATH = f"{METADATA_DIR}/catalog.json"

# ---------------------------------------------------------------
# Objective 2 — Retrieval Engine Configuration
# ---------------------------------------------------------------

RESULTS_DIR = "results"
DEFAULT_TOP_K = 10           # Default number of similar ecosystems to retrieve
EVALUATION_K_VALUES = [1, 3, 5, 10]  # K values for Precision@K, Recall@K evaluation
