import gin
from pathlib import Path

BASE_PATH: Path = Path(__file__).parent

# Add gin config search paths
gin.add_config_file_search_path(BASE_PATH)
gin.add_config_file_search_path(BASE_PATH.joinpath('pipeline/configs'))