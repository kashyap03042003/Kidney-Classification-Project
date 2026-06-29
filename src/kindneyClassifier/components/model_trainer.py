import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from pathlib import Path
from kindneyClassifier.entity.config_entity import TrainingConfig


class Training:
    def __init__(self,config=TrainingConfig):
        self.config=config

    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )


    def train_valid_generator(self):

        image_size = self.config.params_image_size[:-1]
        batch_size = self.config.params_batch_size

        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="training",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            interpolation="bilinear",
            shuffle=True
        )

        valid_ds = tf.keras.utils.image_dataset_from_directory(
            directory=self.config.training_data,
            validation_split=0.20,
            subset="validation",
            seed=123,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="categorical",
            interpolation="bilinear",
            shuffle=True
        )

        rescale = tf.keras.layers.Rescaling(1./255)
        train_ds = train_ds.map(lambda x, y: (rescale(x), y))
        valid_ds = valid_ds.map(lambda x, y: (rescale(x), y))

        if self.config.params_is_augmentation:
            data_augmentation = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.2),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomTranslation(0.2, 0.2),
            ])
            train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        self.train_generator = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        self.valid_generator = valid_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
    

    def train(self):
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )


