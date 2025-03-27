import matplotlib.pyplot as plt


class DataVisualizer:

    @staticmethod
    def plot_train_and_loss_curves(history, model_name):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].plot(history.history['accuracy'], label='Training accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation accuracy')
        axes[0].set_title(f'Training and validation accuracy of {model_name}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()

        axes[1].plot(history.history['loss'], label='Training loss')
        axes[1].plot(history.history['val_loss'], label='Validation loss')
        axes[1].set_title(f'Training and validation loss of {model_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        plt.show()

    @staticmethod
    def plot_train_curves(history_1, history_2, model_1_name, model_2_name):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].plot(history_1.history['accuracy'], label='Training accuracy')
        axes[0].plot(history_1.history['val_accuracy'], label='Validation accuracy')
        axes[0].set_title(f'{model_1_name}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()

        axes[1].plot(history_2.history['accuracy'], label='Training accuracy')
        axes[1].plot(history_2.history['val_accuracy'], label='Validation accuracy')
        axes[1].set_title(f'{model_2_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()

        plt.show()

    @staticmethod
    def plot_loss_curves(history_1, history_2, model_1_name, model_2_name):
        _, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].plot(history_1.history['loss'], label='Training loss')
        axes[0].plot(history_1.history['val_loss'], label='Validation loss')
        axes[0].set_title(f'{model_1_name}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()

        axes[1].plot(history_2.history['loss'], label='Training loss')
        axes[1].plot(history_2.history['val_loss'], label='Validation loss')
        axes[1].set_title(f'{model_2_name}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()

        plt.show()
