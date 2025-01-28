import os 
import urllib.request as request
import zipfile
from CNNproject import logger
from CNNproject.utils.common import get_size
from pathlib import Path
from CNNproject.entity.config_entity import DataIngestionConfig


class DataIngestion: 
    def __init__(self, config: DataIngestionConfig): 
        self.config = config 
    
    def download_file(self):
        if not os.path.exists(self.config.local_dir):
            file_name, headers = request.urlretrieve (
                url = self.config.source_url, 
                filename= self.config.local_dir
            )
            logger.info(f"{file_name} downloaded wiht the following info: \n {headers}")
        else: 
            logger.info(f"{file_name} is already exists of size: {get_size(Path(self.config.local_data_file))}")
        
    
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_dir, "r") as zip_ref:
            zip_ref.extractall(unzip_path)