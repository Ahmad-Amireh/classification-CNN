import os 
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf 
import time
from CNNproject.entity.config_entity import TrainingConfig
from pathlib import Path
import joblib


class Training: 
    def __init__(self, config: TrainingConfig) : 
        self.config = config 
        tf.config.run_functions_eagerly(True)
        
    def get_base_model (self) :
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        # General data generator parameters
        datagenerator_kwargs = dict(rescale=1./255)

        dataflow_kwargs = dict(                                                
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Step 1: Create Test Generator (20% of the dataset)
        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=0.25
        )
        
        self.test_generator = test_datagenerator.flow_from_directory(
            directory=self.config.training_data, 
            subset="validation",  # This will be the test set
            shuffle=False,  
            **dataflow_kwargs
        )

   

        # Step 2: Create Training & Validation Generators (80% training, 20% validation from training)
        train_valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=0.2  # 20% of training data will be used for validation
        )

        self.valid_generator = train_valid_datagenerator.flow_from_directory(
            directory=self.config.training_data, 
            subset="validation",  # Validation set (from training data)
            shuffle=False,  
            **dataflow_kwargs
        )

        # Step 3: Augment the Training Set
        if self.config.params_is_augmentaion: 
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator( 
                rotation_range=40, 
                horizontal_flip=True, 
                width_shift_range=0.2, 
                height_shift_range=0.2, 
                shear_range=0.2, 
                zoom_range=0.2, 
                rescale=1./255, 
                validation_split=0.2  # Same split as validation
            )
        else:
            train_datagenerator = train_valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data, 
            subset="training",  # Training set
            shuffle=True,  
            **dataflow_kwargs
        )




    @staticmethod 
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)

    def train (self, callback_list: list):
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size 
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size
        print("Training samples:", self.train_generator.samples)
        print("Validation samples:", self.valid_generator.samples)
        print(f"Total test images: {self.test_generator.samples}")

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch = self.steps_per_epoch,
            validation_steps = self.validation_steps,
            validation_data = self.valid_generator,
            callbacks = callback_list

        )

        self.save_model(
            path = self.config.trained_model_path, 
            model =self.model 
        )

    def save_test_data(self, path="artifacts/test_data/test_data.pkl"):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save only the necessary data from the test generator
        test_data = {
            'filenames': self.test_generator.filenames,
            'classes': self.test_generator.classes,
            'samples': self.test_generator.samples,
            'batch_size': self.test_generator.batch_size
            }

        # Save the test data to a file
        with open(path, "wb") as f:
            joblib.dump(test_data, f)
