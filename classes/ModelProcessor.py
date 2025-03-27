from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential


class ModelProcessor:

    @staticmethod
    def fit_model(model, train_generator, epochs, valid_generator):
        return model.fit(
            train_generator,
            epochs=epochs,
            validation_data=valid_generator
        )

    @staticmethod
    def compile_and_fit_model(model, train_generator, epochs, valid_generator, sequential: bool = True):
        if sequential:
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )

            history = ModelProcessor.fit_model(
                model,
                train_generator,
                epochs,
                valid_generator
            )
        else:
            for layer in model.layers:
                if 'conv' in layer.name:
                    layer.trainable = False

            modified_model = Sequential([
                model,
                Flatten(),
                Dense(512, activation='relu'),
                Dense(1, activation='sigmoid')
            ])

            modified_model.compile(
                loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )

            history = ModelProcessor.fit_model(
                modified_model,
                train_generator,
                epochs,
                valid_generator
            )

        return history

    @staticmethod
    def evaluate_model(model, test_generator, model_name):
        _, test_acc = model.evaluate(test_generator)

        print(f'Test accuracy of {model_name}: {test_acc}')
