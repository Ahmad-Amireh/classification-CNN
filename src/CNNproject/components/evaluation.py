from CNNproject.utils.common import load_bin
from CNNproject.entity.config_entity import EvaluationConfig 
import tensorflow as tf
from pathlib import Path
from CNNproject.utils.common import save_json

class Evaluation: 
    def __init__(self, config: EvaluationConfig): 
        self.config = config 
        tf.config.run_functions_eagerly(True)

    def _valid_generator(self):
        file_path = self.config.test_data_info
        test_data_info = load_bin(file_path)  # Load test data

        dataflow_kwargs = dict(                                                 
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Step 1: Create Test Generator
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=0.25
        )
        
        test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.training_data, 
            subset="validation",  # This will be the test set
            shuffle=False,  
            **dataflow_kwargs
        )

        # Step 2: Restore Attributes from Saved Test Data
        test_generator.filenames = test_data_info.get('filenames', test_generator.filenames)
        test_generator.classes = test_data_info.get('classes', test_generator.classes)
        test_generator.samples = test_data_info.get('samples', test_generator.samples)
        test_generator.batch_size = test_data_info.get('batch_size', test_generator.batch_size)

        return test_generator
    

    @staticmethod 
    def load_model (path:Path) -> tf.keras.Model : 
        return tf.keras.models.load_model(path)
    

    def evaluation(self) :
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator = self._valid_generator()
        self.score = self.model.evaluate(self._valid_generator)

    def save_score(self) :
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data= scores)
     