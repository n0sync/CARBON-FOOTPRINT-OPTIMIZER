import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import os

class CarbonFootprintModel:
    def __init__(self):
        self.model = None
        self.history = None
        self.is_trained = False
        self.input_dim = None
        self.hidden_layers = [128, 64, 32]
        self.activation = 'relu'
        self.output_activation = 'linear'
        self.optimizer = 'adam'
        self.loss = 'mse'
        self.metrics = ['mae']
        
    def build_model(self, input_dim):
        self.input_dim = input_dim
        self.model = keras.Sequential()
        self.model.add(keras.layers.Dense(
            self.hidden_layers[0], 
            activation=self.activation, 
            input_dim=input_dim,
            name='hidden_layer_1'
        ))
        self.model.add(keras.layers.Dropout(0.2, name='dropout_1'))
        self.model.add(keras.layers.Dense(
            self.hidden_layers[1], 
            activation=self.activation,
            name='hidden_layer_2'
        ))
        self.model.add(keras.layers.Dropout(0.2, name='dropout_2'))
        self.model.add(keras.layers.Dense(
            self.hidden_layers[2], 
            activation=self.activation,
            name='hidden_layer_3'
        ))
        self.model.add(keras.layers.Dropout(0.1, name='dropout_3'))
        self.model.add(keras.layers.Dense(
            1, 
            activation=self.output_activation,
            name='output_layer'
        ))
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        print("Model Architecture:")
        self.model.summary()
        return self.model
    
    def train_model(self, X, y, test_size=0.2, validation_split=0.2, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        if self.model is None:
            self.build_model(X_train.shape[1])
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        print(f"Starting training for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        print("\nEvaluating model on test set...")
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        self.evaluation_results = {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'mse': mse,
            'r2_score': r2,
            'mae': mae
        }
        print(f"\nModel Performance:")
        print(f"Test Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        self.is_trained = True
        self.save_model()
        return self.model
    
    def predict(self, X):
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions!")
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def predict_single(self, X_single):
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions!")
        prediction = self.model.predict(X_single.reshape(1, -1))
        return prediction[0][0]
    
    def save_model(self, filepath='models/carbon_model.h5'):
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")
    
    def load_model(self, filepath='models/carbon_model.keras'):
        try:
            self.model = keras.models.load_model(filepath)
            self.is_trained = True
            print(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def plot_training_history(self):
        if self.history is None:
            print("No training history available!")
            return None
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['loss'], label='Training Loss', color='blue')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(self.history.history['mae'], label='Training MAE', color='blue')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', color='red')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        return fig
    
    def get_model_summary(self):
        if self.model is None:
            return "Model not built yet!"
        summary = {
            'architecture': 'Multi-Layer Perceptron (MLP)',
            'input_dimension': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'total_parameters': self.model.count_params(),
            'activation_function': self.activation,
            'optimizer': self.optimizer,
            'loss_function': self.loss,
            'is_trained': self.is_trained
        }
        if hasattr(self, 'evaluation_results'):
            summary.update(self.evaluation_results)
        return summary

class ModelExperiments:
    def __init__(self):
        self.experiments = []
    
    def run_experiment(self, X, y, hidden_layers, activation='relu', 
                      optimizer='adam', epochs=100):
        model = CarbonFootprintModel()
        model.hidden_layers = hidden_layers
        model.activation = activation
        model.optimizer = optimizer
        trained_model = model.train_model(X, y, epochs=epochs)
        experiment = {
            'hidden_layers': hidden_layers,
            'activation': activation,
            'optimizer': optimizer,
            'epochs': epochs,
            'results': model.evaluation_results,
            'model': model
        }
        self.experiments.append(experiment)
        return experiment
    
    def compare_experiments(self):
        if not self.experiments:
            print("No experiments to compare!")
            return
        print("\n" + "="*50)
        print("EXPERIMENT COMPARISON")
        print("="*50)
        for i, exp in enumerate(self.experiments):
            print(f"\nExperiment {i+1}:")
            print(f"Architecture: {exp['hidden_layers']}")
            print(f"Activation: {exp['activation']}")
            print(f"Optimizer: {exp['optimizer']}")
            print(f"R² Score: {exp['results']['r2_score']:.4f}")
            print(f"MAE: {exp['results']['mae']:.4f}")
            print(f"RMSE: {np.sqrt(exp['results']['mse']):.4f}")
            print("-" * 30)
        best_exp = max(self.experiments, key=lambda x: x['results']['r2_score'])
        print(f"\nBest Model: Experiment {self.experiments.index(best_exp) + 1}")
        print(f"Best R² Score: {best_exp['results']['r2_score']:.4f}")
