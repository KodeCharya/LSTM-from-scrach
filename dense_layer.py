import numpy as np
import math

class Dense:
    """
    Fully connected (dense) layer for output.
    """
    
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        # Initialize weights using Xavier initialization
        limit = math.sqrt(6.0 / (input_size + output_size))
        self.W = np.random.uniform(-limit, limit, (input_size, output_size))
        self.b = np.zeros((1, output_size))
        
        # Cache for backpropagation
        self.cache = {}
    
    def forward(self, x):
        """
        Forward pass through dense layer
        
        Args:
            x: Input tensor (batch_size, input_size) or (batch_size, seq_length, input_size)
            
        Returns:
            output: Linear transformation of input
        """
        self.cache['x'] = x
        
        if len(x.shape) == 3:  # (batch, seq, features)
            batch_size, seq_length, _ = x.shape
            x_reshaped = x.reshape(-1, self.input_size)
            output = np.dot(x_reshaped, self.W) + self.b
            output = output.reshape(batch_size, seq_length, self.output_size)
        else:  # (batch, features)
            output = np.dot(x, self.W) + self.b
        
        return output
    
    def backward(self, doutput):
        """
        Backward pass through dense layer
        
        Args:
            doutput: Gradient w.r.t. output
            
        Returns:
            dx: Gradient w.r.t. input
            grads: Dictionary of parameter gradients
        """
        x = self.cache['x']
        
        if len(x.shape) == 3:  # (batch, seq, features)
            batch_size, seq_length, _ = x.shape
            x_reshaped = x.reshape(-1, self.input_size)
            doutput_reshaped = doutput.reshape(-1, self.output_size)
            
            # Gradients
            dW = np.dot(x_reshaped.T, doutput_reshaped)
            db = np.sum(doutput_reshaped, axis=0, keepdims=True)
            dx = np.dot(doutput_reshaped, self.W.T)
            dx = dx.reshape(batch_size, seq_length, self.input_size)
        else:  # (batch, features)
            # Gradients
            dW = np.dot(x.T, doutput)
            db = np.sum(doutput, axis=0, keepdims=True)
            dx = np.dot(doutput, self.W.T)
        
        grads = {'W': dW, 'b': db}
        
        return dx, grads
    
    def get_params(self):
        """Return parameters as dictionary"""
        return {'W': self.W, 'b': self.b}
    
    def set_params(self, params):
        """Set parameters from dictionary"""
        self.W = params['W']
        self.b = params['b']