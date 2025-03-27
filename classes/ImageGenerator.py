import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageGenerator:

    @staticmethod
    def get_train_valid_test_generators(target_size):
        train_data_generator = ImageDataGenerator(rescale=1. / 255)
        valid_data_generator = ImageDataGenerator(rescale=1. / 255)
        test_data_generator = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_data_generator.flow_from_directory(
            '../data/train',
            target_size=target_size,
            batch_size=32,
            class_mode='binary'
        )

        valid_generator = valid_data_generator.flow_from_directory(
            '../data/val',
            target_size=target_size,
            batch_size=32,
            class_mode='binary'
        )

        test_generator = test_data_generator.flow_from_directory(
            '../data/test',
            target_size=target_size,
            batch_size=32,
            class_mode='binary'
        )

        return train_generator, valid_generator, test_generator

    @staticmethod
    def clear():
        backend.clear_session()
        tf.config.set_visible_devices([], 'GPU')
