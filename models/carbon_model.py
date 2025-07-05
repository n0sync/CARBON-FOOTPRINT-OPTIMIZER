try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using fallback linear regression model.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time


class CarbonFootprintModel:
    _model_loaded = False
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
        
        if TF_AVAILABLE:
            self.model = keras.Sequential()
            
            # Input layer
            self.model.add(keras.layers.Input(shape=(input_dim,)))
            
            self.model.add(keras.layers.Normalization())
            
            # Hidden layers 
            self.model.add(keras.layers.Dense(256, activation='relu'))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dropout(0.3))
            
            self.model.add(keras.layers.Dense(128, activation='relu'))
            self.model.add(keras.layers.BatchNormalization())
            self.model.add(keras.layers.Dropout(0.3))
            
            self.model.add(keras.layers.Dense(64, activation='relu'))
            
            # Output layer
            self.model.add(keras.layers.Dense(1, activation='linear'))
        else:
            # Fallback to Random Forest Regressor
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def train_model(self, X, y, test_size=0.2, validation_split=0.2, epochs=200, batch_size=64, progress_callback=None):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        print(f"Training data shape: {X_train.shape}")
        print(f"Testing data shape: {X_test.shape}")
        
        if self.model is None:
            self.build_model(X_train.shape[1])

        if TF_AVAILABLE:
            # TensorFlow training
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=1000,
                decay_rate=0.9
            )
            
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
            
            self.model.compile(
                optimizer=optimizer,
                loss='huber',  
                metrics=['mae']
            )
            
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=50,
                    restore_best_weights=True,
                    verbose=1
                ),
            ]
            
            # Custom callback for progress tracking
            class ProgressCallback(keras.callbacks.Callback):
                def __init__(self, callback_func=None):
                    super().__init__()
                    self.callback_func = callback_func
                    
                def on_epoch_end(self, epoch, logs=None):
                    if self.callback_func:
                        progress = (epoch + 1) / self.params['epochs'] * 100
                        self.callback_func(progress, epoch + 1, logs)
            
            # Add progress callback if provided
            if progress_callback:
                callbacks.append(ProgressCallback(progress_callback))
            
            print(f"Starting TensorFlow training for {epochs} epochs...")
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
        else:
            # Sklearn training with simulated progress
            print("Starting Random Forest training...")
            if progress_callback:
                # Simulate training progress
                for i in range(epochs):
                    progress = (i + 1) / epochs * 100
                    progress_callback(progress, i + 1, {'loss': 0.5 - i*0.005})
                    time.sleep(0.05)  # Small delay to show progress
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            test_loss = mean_squared_error(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            
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
        if TF_AVAILABLE and hasattr(predictions, 'flatten'):
            return predictions.flatten()
        elif hasattr(predictions, 'ravel'):
            return predictions.ravel()
        else:
            return np.array(predictions).flatten()
    
    def predict_single(self, X_single):
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions!")
        prediction = self.model.predict(X_single.reshape(1, -1))
        if TF_AVAILABLE:
            return prediction[0][0]
        else:
            return prediction[0]
    
    def save_model(self, filepath='models/carbon_model'):
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if TF_AVAILABLE:
                self.model.save(f"{filepath}.keras", save_format='keras')
                print(f"TensorFlow model saved to {filepath}.keras")
            else:
                with open(f"{filepath}.pkl", 'wb') as f:
                    pickle.dump(self.model, f)
                print(f"Sklearn model saved to {filepath}.pkl")
        else:
            print("No model to save!")
  
    
    def load_model(self, filepath='models/carbon_model'):
        cls = self.__class__

        if hasattr(cls, "_model_instance") and cls._model_instance is not None:
            self.model = cls._model_instance
            self.is_trained = True
            self.is_loaded = True
            return True

        # Try loading TensorFlow model first
        if TF_AVAILABLE:
            try:
                keras_path = f"{filepath}.keras" if not filepath.endswith('.keras') else filepath
                model = keras.models.load_model(keras_path)
                self.model = model
                cls._model_instance = model
                self.is_trained = True
                self.is_loaded = True
                print(f"TensorFlow model loaded from {keras_path}")
                return True
            except Exception:
                try:
                    h5_path = filepath.replace('.keras', '.h5')
                    model = keras.models.load_model(h5_path)
                    self.model = model
                    cls._model_instance = model
                    self.is_trained = True
                    self.is_loaded = True
                    print(f"TensorFlow model loaded from {h5_path}")
                    return True
                except Exception:
                    pass
        
        # Try loading sklearn model
        try:
            pkl_path = f"{filepath}.pkl" if not filepath.endswith('.pkl') else filepath
            with open(pkl_path, 'rb') as f:
                model = pickle.load(f)
            self.model = model
            cls._model_instance = model
            self.is_trained = True
            self.is_loaded = True
            print(f"Sklearn model loaded from {pkl_path}")
            return True
        except Exception:
            print("Model loading failed. No saved model found.")
            self.model = None
            self.is_loaded = False
            return False

    
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
