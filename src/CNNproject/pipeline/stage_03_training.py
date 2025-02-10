from CNNproject.config.configuration import ConfigurationManager
from CNNproject.components.training import Training
from CNNproject import logger
from CNNproject.components.prepare_callbacks import PrepareCallback

STAGE_NAME="Training Stage"

class TrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepareCallbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepareCallbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config = training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.save_test_data()
        training.train(callback_list=callback_list)



if __name__ == "__main__":
    try:
        logger.info(f">>> stage {STAGE_NAME} started >>>")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} completed <<<<<< \n\n ====== ")
    except Exception as e :
        logger.exception(e)
        raise e 