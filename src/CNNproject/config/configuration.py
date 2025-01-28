from CNNproject.constants import * 
from CNNproject.utils.common import read_yaml, create_directories
from CNNproject.entity.config_entity import DataIngestionConfig

class ConfigurationManager: 
    def __init__ ( self,config_file_path = CONFIG_YAML, params_file_path = PARMAS_YAML):
        self.config = read_yaml(config_file_path)
        self.params = read_yaml (params_file_path)
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig: 
        config = self.config.data_ingestion 
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir= config.root_dir,
            source_url= config.source_url,
            local_dir= config.local_dir,
            unzip_dir= config.unzip_dir)

        return data_ingestion_config 
