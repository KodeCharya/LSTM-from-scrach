import numpy as np
import math

class LSTMCell:
    """
    A single LSTM cell implementation with input, forget, and output gates.
    """
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights using Xavier initialization
        self._init_weights()
        
        # Cache for backpropagation
        self.cache = {}
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        # Input gate weights
        self.W_i = self._xavier_init(self.input_size, self.hidden_size)
        self.U_i = self._xavier_init(self.hidden_size, self.hidden_size)
        self.b_i = np.zeros((1, self.hidden_size))
        
        # Forget gate weights
        self.W_f = self._xavier_init(self.input_size, self.hidden_size)
        self.U_f = self._xavier_init(self.hidden_size, self.hidden_size)
        self.b_f = np.ones((1, self.hidden_size))  # Initialize to 1 for better gradient flow
        
        # Output gate weights
        self.W_o = self._xavier_init(self.input_size, self.hidden_size)
        self.U_o = self._xavier_init(self.hidden_size, self.hidden_size)
        self.b_o = np.zeros((1, self.hidden_size))
        
        # Cell state weights
        self.W_c = self._xavier_init(self.input_size, self.hidden_size)
        self.U_c = self._xavier_init(self.hidden_size, self.hidden_size)
        self.b_c = np.zeros((1, self.hidden_size))
        
    def _xavier_init(self, fan_in, fan_out):
        """Xavier weight initialization"""
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, (fan_in, fan_out))
    
    def sigmoid(self, x):
        """Sigmoid activation function with numerical stability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass through LSTM cell
        
        Args:
            x: Input at current timestep (batch_size, input_size)
            h_prev: Previous hidden state (batch_size, hidden_size)
            c_prev: Previous cell state (batch_size, hidden_size)
            
        Returns:
            h: New hidden state
            c: New cell state
        """
        # Input gate
        i = self.sigmoid(np.dot(x, self.W_i) + np.dot(h_prev, self.U_i) + self.b_i)
        
        # Forget gate
        f = self.sigmoid(np.dot(x, self.W_f) + np.dot(h_prev, self.U_f) + self.b_f)
        
        # Output gate
        o = self.sigmoid(np.dot(x, self.W_o) + np.dot(h_prev, self.U_o) + self.b_o)
        
        # Candidate cell state
        c_tilde = self.tanh(np.dot(x, self.W_c) + np.dot(h_prev, self.U_c) + self.b_c)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # New hidden state
        h = o * self.tanh(c)
        
        # Cache for backpropagation
        self.cache = {
            'x': x, 'h_prev': h_prev, 'c_prev': c_prev,
            'i': i, 'f': f, 'o': o, 'c_tilde': c_tilde,
            'c': c, 'h': h
        }
        
        return h, c
    
    def backward(self, dh, dc):
        """
        Backward pass through LSTM cell
        
        Args:
            dh: Gradient w.r.t. hidden state
            dc: Gradient w.r.t. cell state
            
        Returns:
            dx: Gradient w.r.t. input
            dh_prev: Gradient w.r.t. previous hidden state
            dc_prev: Gradient w.r.t. previous cell state
            grads: Dictionary of parameter gradients
        """
        # Retrieve cached values
        x = self.cache['x']
        h_prev = self.cache['h_prev']
        c_prev = self.cache['c_prev']
        i = self.cache['i']
        f = self.cache['f']
        o = self.cache['o']
        c_tilde = self.cache['c_tilde']
        c = self.cache['c']
        
        # Gradient w.r.t. output gate
        do = dh * self.tanh(c)
        do_raw = do * o * (1 - o)
        
        # Gradient w.r.t. cell state
        dc = dc + dh * o * (1 - self.tanh(c)**2)
        
        # Gradient w.r.t. forget gate
        df = dc * c_prev
        df_raw = df * f * (1 - f)
        
        # Gradient w.r.t. input gate
        di = dc * c_tilde
        di_raw = di * i * (1 - i)
        
        # Gradient w.r.t. candidate cell state
        dc_tilde = dc * i
        dc_tilde_raw = dc_tilde * (1 - c_tilde**2)
        
        # Gradients w.r.t. weights and biases
        grads = {}
        
        # Input gate gradients
        grads['W_i'] = np.dot(x.T, di_raw)
        grads['U_i'] = np.dot(h_prev.T, di_raw)
        grads['b_i'] = np.sum(di_raw, axis=0, keepdims=True)
        
        # Forget gate gradients
        grads['W_f'] = np.dot(x.T, df_raw)
        grads['U_f'] = np.dot(h_prev.T, df_raw)
        grads['b_f'] = np.sum(df_raw, axis=0, keepdims=True)
        
        # Output gate gradients
        grads['W_o'] = np.dot(x.T, do_raw)
        grads['U_o'] = np.dot(h_prev.T, do_raw)
        grads['b_o'] = np.sum(do_raw, axis=0, keepdims=True)
        
        # Cell state gradients
        grads['W_c'] = np.dot(x.T, dc_tilde_raw)
        grads['U_c'] = np.dot(h_prev.T, dc_tilde_raw)
        grads['b_c'] = np.sum(dc_tilde_raw, axis=0, keepdims=True)
        
        # Gradients w.r.t. inputs
        dx = (np.dot(di_raw, self.W_i.T) + 
              np.dot(df_raw, self.W_f.T) + 
              np.dot(do_raw, self.W_o.T) + 
              np.dot(dc_tilde_raw, self.W_c.T))
        
        dh_prev = (np.dot(di_raw, self.U_i.T) + 
                   np.dot(df_raw, self.U_f.T) + 
                   np.dot(do_raw, self.U_o.T) + 
                   np.dot(dc_tilde_raw, self.U_c.T))
        
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev, grads
    
    def get_params(self):
        """Return all parameters as a dictionary"""
        return {
            'W_i': self.W_i, 'U_i': self.U_i, 'b_i': self.b_i,
            'W_f': self.W_f, 'U_f': self.U_f, 'b_f': self.b_f,
            'W_o': self.W_o, 'U_o': self.U_o, 'b_o': self.b_o,
            'W_c': self.W_c, 'U_c': self.U_c, 'b_c': self.b_c
        }
    
    def set_params(self, params):
        """Set parameters from a dictionary"""
        for key, value in params.items():
            setattr(self, key, value)