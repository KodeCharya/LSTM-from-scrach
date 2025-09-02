import numpy as np
from stacked_lstm import StackedLSTM
from dense_layer import Dense
from loss_functions import LossFunctions
from adam_optimizer import AdamOptimizer

class LSTMModel:
    """
    Complete LSTM model with stacked LSTM layers and dense output layer.
    """
    
    def __init__(self, input_size, hidden_sizes, output_size, loss_type='mse'):
        """
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden sizes for LSTM layers
            output_size: Size of output
            loss_type: 'mse' for regression, 'cross_entropy' for classification
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.loss_type = loss_type
        
        # Create model components
        self.lstm = StackedLSTM(input_size, hidden_sizes)
        self.dense = Dense(hidden_sizes[-1], output_size)
        
        # Initialize optimizer
        self.optimizer = AdamOptimizer()
        
        # Gradient clipping threshold
        self.clip_threshold = 5.0
    
    def forward(self, x, return_sequences=False):
        """
        Forward pass through the entire model
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            return_sequences: If True, return outputs for all timesteps
            
        Returns:
            output: Model predictions
        """
        # Forward through LSTM layers
        lstm_outputs, lstm_cache = self.lstm.forward(x)
        
        if return_sequences:
            # Return outputs for all timesteps
            output = self.dense.forward(lstm_outputs)
        else:
            # Return output for last timestep only
            last_output = lstm_outputs[:, -1, :]  # (batch_size, hidden_size)
            output = self.dense.forward(last_output)
        
        return output, (lstm_outputs, lstm_cache)
    
    def compute_loss(self, y_true, y_pred):
        """Compute loss based on loss type"""
        if self.loss_type == 'mse':
            return LossFunctions.mse_loss(y_true, y_pred)
        elif self.loss_type == 'cross_entropy':
            return LossFunctions.cross_entropy_loss(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def clip_gradients(self, grads):
        """
        Clip gradients to prevent exploding gradients
        
        Args:
            grads: Dictionary of gradients
        """
        def clip_grad_dict(grad_dict, prefix=""):
            for key, value in grad_dict.items():
                if isinstance(value, dict):
                    clip_grad_dict(value, f"{prefix}{key}_")
                else:
                    # Compute gradient norm
                    grad_norm = np.linalg.norm(value)
                    if grad_norm > self.clip_threshold:
                        # Scale gradient
                        value *= self.clip_threshold / grad_norm
        
        clip_grad_dict(grads)
    
    def train_step(self, x, y_true):
        """
        Single training step
        
        Args:
            x: Input batch (batch_size, sequence_length, input_size)
            y_true: True labels (batch_size, output_size)
            
        Returns:
            loss: Training loss for this batch
        """
        # Forward pass
        y_pred, (lstm_outputs, lstm_cache) = self.forward(x)
        
        # Compute loss
        loss, dloss = self.compute_loss(y_true, y_pred)
        
        # Backward pass through dense layer
        if len(y_pred.shape) == 3:  # Sequence output
            dx_dense, dense_grads = self.dense.backward(dloss)
            # Backpropagate through LSTM
            lstm_grads = self.lstm.backward(dx_dense, lstm_cache)
        else:  # Single output
            dx_dense, dense_grads = self.dense.backward(dloss)
            # Create gradient tensor for LSTM (only last timestep)
            batch_size, seq_length, hidden_size = lstm_outputs.shape
            dlstm_outputs = np.zeros((batch_size, seq_length, hidden_size))
            dlstm_outputs[:, -1, :] = dx_dense
            # Backpropagate through LSTM
            lstm_grads = self.lstm.backward(dlstm_outputs, lstm_cache)
        
        # Combine gradients
        all_grads = {
            'lstm': lstm_grads,
            'dense': dense_grads
        }
        
        # Clip gradients
        self.clip_gradients(all_grads)
        
        # Update parameters
        all_params = {
            'lstm': self.lstm.get_params(),
            'dense': self.dense.get_params()
        }
        
        self.optimizer.update(all_params, all_grads)
        
        return loss
    
    def predict(self, x):
        """
        Make predictions without computing gradients
        
        Args:
            x: Input tensor
            
        Returns:
            predictions: Model predictions
        """
        predictions, _ = self.forward(x)
        
        if self.loss_type == 'cross_entropy':
            # Apply softmax for classification
            predictions = LossFunctions.softmax(predictions)
        
        return predictions
    
    def reset_states(self, batch_size=1):
        """Reset LSTM hidden states"""
        self.lstm.reset_states(batch_size)