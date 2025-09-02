import numpy as np

class AdamOptimizer:
    """
    Adam optimizer implementation for parameter updates.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Args:
            learning_rate: Learning rate for parameter updates
            beta1: Exponential decay rate for first moment estimates
            beta2: Exponential decay rate for second moment estimates
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Initialize moment estimates
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.t = 0   # Time step
    
    def update(self, params, grads):
        """
        Update parameters using Adam optimization
        
        Args:
            params: Dictionary of parameters to update
            grads: Dictionary of gradients
        """
        self.t += 1
        
        # Bias correction terms
        lr_t = self.learning_rate * math.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        
        def update_param(param_key, param_value, grad_value):
            # Initialize moments if first time
            if param_key not in self.m:
                self.m[param_key] = np.zeros_like(param_value)
                self.v[param_key] = np.zeros_like(param_value)
            
            # Update biased first moment estimate
            self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * grad_value
            
            # Update biased second raw moment estimate
            self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (grad_value ** 2)
            
            # Update parameters
            param_value -= lr_t * self.m[param_key] / (np.sqrt(self.v[param_key]) + self.epsilon)
        
        # Recursively update all parameters
        self._update_nested_params(params, grads, "", update_param)
    
    def _update_nested_params(self, params, grads, prefix, update_func):
        """Recursively update nested parameter dictionaries"""
        for key in params:
            param_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(params[key], dict):
                # Recursively handle nested dictionaries
                self._update_nested_params(
                    params[key], 
                    grads[key], 
                    f"{param_key}_", 
                    update_func
                )
            else:
                # Update parameter
                if key in grads:
                    update_func(param_key, params[key], grads[key])
    
    def reset(self):
        """Reset optimizer state"""
        self.m = {}
        self.v = {}
        self.t = 0

# Import math for bias correction calculation
import math