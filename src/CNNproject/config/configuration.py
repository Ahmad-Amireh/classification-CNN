from CNNproject.constants import * 
from CNNproject.utils.common import read_yaml, create_directories
from CNNproject.entity.config_entity import DataIngestionConfig,PrepareBaseModelConfig,PrepareCallbacksConfig

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

    def get_prepare_base_model_config (self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(root_dir = Path(config.root_dir),
        base_model_path = Path(config.base_model_path),
        updated_base_model_path = Path(config.updated_base_model_path),
        params_learning_rate= self.params.LEARNING_RATE,
        params_include_top = self.params.INCLUDE_TOP,
        params_weight = self.params.WEIGHTS,
        params_classes = self.params.CLASSES,
        params_image_size = self.params.IMAGE_SIZE,)


        return prepare_base_model_config
    

    def get_prepare_call_back_config(self) -> PrepareCallbacksConfig : 
        config = self.config.prepare_callbacks
        model_ckpt_dir = os.path.dirname(config.checkpoint_model_filepath)
        create_directories([
            Path (model_ckpt_dir),
            Path(config.transorboard_root_log_dir)
        ])

        prepare_call_back_config = PrepareCallbacksConfig (
            root_dir = Path(config.root_dir),
            transorboard_root_log_dir = Path(config.transorboard_root_log_dir), 
            checkpoint_model_filepath= Path(config.checkpoint_model_filepath)
            )

        return prepare_call_back_config


    
        