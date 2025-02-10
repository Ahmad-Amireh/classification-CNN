from CNNproject.config.configuration import ConfigurationManager
from CNNproject.components.evaluation import Evaluation 
from CNNproject import logger
from CNNproject.components.prepare_callbacks import PrepareCallback

STAGE_NAME="Evaluation Stage"

class EvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config= ConfigurationManager()
        test_config = config.get_evaluation_config()
        evaluation = Evaluation(config=test_config)
        evaluation.evaluation()
        evaluation.save_score()



if __name__ == "__main__":
    try:
        logger.info(f">>> stage {STAGE_NAME} started >>>")
        obj = EvaluationPipeline()
        obj.main()
        logger.info(f">>> stage {STAGE_NAME} completed <<<<<< \n\n ====== ")
    except Exception as e :
        logger.exception(e)
        raise e 