from tensorflow.keras import backend
from tensorflow.keras.applications import VGG19, ResNet50
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential

from classes.DataVisualizer import DataVisualizer
from classes.ImageGenerator import ImageGenerator
from classes.ModelProcessor import ModelProcessor


def main():
    target_size = (150, 150)
    epochs = 5
    epochs_retrain = 10

    train_generator, valid_generator, test_generator = ImageGenerator.get_train_valid_test_generators(target_size)

    model_1 = Sequential([
        Flatten(input_shape=(150, 150, 3)),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_2 = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_3 = VGG19(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )
    model_4 = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(150, 150, 3)
    )

    model_1_name = 'Fully Connected NN'
    model_2_name = 'Convolutional NN'
    model_3_name = 'VGG19'
    model_4_name = 'ResNet50'

    history_1 = ModelProcessor.compile_and_fit_model(model_1, train_generator, epochs, valid_generator)
    history_2 = ModelProcessor.compile_and_fit_model(model_2, train_generator, epochs, valid_generator)

    backend.clear_session()

    history_3 = ModelProcessor.compile_and_fit_model(model_3, train_generator, epochs, valid_generator, False)

    backend.clear_session()

    history_4 = ModelProcessor.compile_and_fit_model(model_4, train_generator, epochs, valid_generator, False)

    ModelProcessor.evaluate_model(model_1, test_generator, model_1_name)
    ModelProcessor.evaluate_model(model_2, test_generator, model_2_name)
    ModelProcessor.evaluate_model(model_3, test_generator, model_3_name)
    ModelProcessor.evaluate_model(model_4, test_generator, model_4_name)

    DataVisualizer.plot_train_and_loss_curves(history_1, model_1_name)
    DataVisualizer.plot_train_and_loss_curves(history_2, model_2_name)
    DataVisualizer.plot_train_and_loss_curves(history_3, model_3_name)
    DataVisualizer.plot_train_and_loss_curves(history_4, model_4_name)

    history_5 = ModelProcessor.fit_model(model_1, train_generator, epochs_retrain, valid_generator)
    history_6 = ModelProcessor.fit_model(model_2, train_generator, epochs_retrain, valid_generator)

    ModelProcessor.evaluate_model(model_1, test_generator, model_1_name)
    ModelProcessor.evaluate_model(model_2, test_generator, model_2_name)

    DataVisualizer.plot_train_curves(history_5, history_6, model_1_name, model_2_name)
    DataVisualizer.plot_loss_curves(history_5, history_6, model_1_name, model_2_name)


if __name__ == '__main__':
    main()
