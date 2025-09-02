import numpy as np
from lstm_cell import LSTMCell

class StackedLSTM:
    """
    Multi-layer LSTM implementation where output of one layer feeds into the next.
    """
    
    def __init__(self, input_size, hidden_sizes):
        """
        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden sizes for each LSTM layer
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)
        
        # Create LSTM layers
        self.layers = []
        layer_input_size = input_size
        
        for hidden_size in hidden_sizes:
            self.layers.append(LSTMCell(layer_input_size, hidden_size))
            layer_input_size = hidden_size
        
        # Initialize hidden and cell states
        self.reset_states()
    
    def reset_states(self, batch_size=1):
        """Reset hidden and cell states for all layers"""
        self.hidden_states = []
        self.cell_states = []
        
        for hidden_size in self.hidden_sizes:
            self.hidden_states.append(np.zeros((batch_size, hidden_size)))
            self.cell_states.append(np.zeros((batch_size, hidden_size)))
    
    def forward(self, x):
        """
        Forward pass through all LSTM layers
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            outputs: Output from last layer for each timestep
            all_hidden_states: Hidden states from all layers and timesteps
            all_cell_states: Cell states from all layers and timesteps
        """
        batch_size, seq_length, _ = x.shape
        
        # Ensure states have correct batch size
        if self.hidden_states[0].shape[0] != batch_size:
            self.reset_states(batch_size)
        
        # Store all states and caches for backpropagation
        all_hidden_states = [[] for _ in range(self.num_layers)]
        all_cell_states = [[] for _ in range(self.num_layers)]
        all_caches = [[] for _ in range(self.num_layers)]
        
        outputs = []
        
        for t in range(seq_length):
            layer_input = x[:, t, :]
            
            # Forward through each layer
            for layer_idx in range(self.num_layers):
                h, c = self.layers[layer_idx].forward(
                    layer_input, 
                    self.hidden_states[layer_idx], 
                    self.cell_states[layer_idx]
                )
                
                # Update states
                self.hidden_states[layer_idx] = h
                self.cell_states[layer_idx] = c
                
                # Store states
                all_hidden_states[layer_idx].append(h.copy())
                all_cell_states[layer_idx].append(c.copy())
                all_caches[layer_idx].append(self.layers[layer_idx].cache.copy())
                
                # Output of this layer becomes input to next layer
                layer_input = h
            
            # Store final output
            outputs.append(h)
        
        # Convert lists to arrays
        outputs = np.array(outputs).transpose(1, 0, 2)  # (batch, seq, hidden)
        forward_cache = (all_hidden_states, all_cell_states, all_caches)
        
        return outputs, forward_cache
    
    def backward(self, doutputs, forward_cache):
        """
        Backward pass through all LSTM layers using BPTT
        
        Args:
            doutputs: Gradients w.r.t. outputs (batch_size, seq_length, hidden_size)
            forward_cache: Tuple containing states and caches from forward pass
            
        Returns:
            grads: Dictionary of gradients for all parameters
        """
        all_hidden_states, all_cell_states, all_caches = forward_cache
        batch_size, seq_length, _ = doutputs.shape
        
        # Initialize gradients
        grads = {}
        for layer_idx in range(self.num_layers):
            layer_grads = {
                'W_i': np.zeros_like(self.layers[layer_idx].W_i),
                'U_i': np.zeros_like(self.layers[layer_idx].U_i),
                'b_i': np.zeros_like(self.layers[layer_idx].b_i),
                'W_f': np.zeros_like(self.layers[layer_idx].W_f),
                'U_f': np.zeros_like(self.layers[layer_idx].U_f),
                'b_f': np.zeros_like(self.layers[layer_idx].b_f),
                'W_o': np.zeros_like(self.layers[layer_idx].W_o),
                'U_o': np.zeros_like(self.layers[layer_idx].U_o),
                'b_o': np.zeros_like(self.layers[layer_idx].b_o),
                'W_c': np.zeros_like(self.layers[layer_idx].W_c),
                'U_c': np.zeros_like(self.layers[layer_idx].U_c),
                'b_c': np.zeros_like(self.layers[layer_idx].b_c)
            }
            grads[f'layer_{layer_idx}'] = layer_grads
        
        # Initialize gradient flows
        dh_next = [np.zeros((batch_size, hidden_size)) for hidden_size in self.hidden_sizes]
        dc_next = [np.zeros((batch_size, hidden_size)) for hidden_size in self.hidden_sizes]
        
        # Backpropagate through time
        for t in reversed(range(seq_length)):
            # Gradient from the layer above (at the same timestep)
            # For the top layer, this is the gradient from the dense layer.
            d_input_from_above = doutputs[:, t, :]
            
            # Backpropagate through layers (from top to bottom)
            for layer_idx in reversed(range(self.num_layers)):
                # Total gradient w.r.t. h_t of this layer is the sum of:
                # 1. Gradient from the next timestep (t+1) flowing back (dh_next)
                # 2. Gradient from the layer above at the current timestep (t) (d_input_from_above)
                dh = dh_next[layer_idx] + d_input_from_above
                
                # Set the cache for the current cell and timestep
                self.layers[layer_idx].cache = all_caches[layer_idx][t]
                
                # Backward pass through layer
                dx, dh_prev, dc_prev, layer_grads = self.layers[layer_idx].backward(
                    dh, dc_next[layer_idx]
                )
                
                # Accumulate gradients
                for key, grad in layer_grads.items():
                    grads[f'layer_{layer_idx}'][key] += grad
                
                # Update gradient flows for the next BPTT step (t-1)
                dh_next[layer_idx] = dh_prev
                dc_next[layer_idx] = dc_prev
                
                # The gradient w.r.t. the cell's input (dx) becomes the gradient
                # for the output of the layer below.
                d_input_from_above = dx
        
        return grads
    
    def get_params(self):
        """Get all parameters from all layers"""
        params = {}
        for layer_idx, layer in enumerate(self.layers):
            params[f'layer_{layer_idx}'] = layer.get_params()
        return params
    
    def set_params(self, params):
        """Set parameters for all layers"""
        for layer_idx, layer in enumerate(self.layers):
            if f'layer_{layer_idx}' in params:
                layer.set_params(params[f'layer_{layer_idx}'])