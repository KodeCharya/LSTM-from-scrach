# LSTM Neural Network from Scratch

A complete implementation of Long Short-Term Memory (LSTM) neural networks using only Python's standard libraries and NumPy. No external machine learning frameworks required!

## Features

- **Complete LSTM Cell**: Input, forget, and output gates with cell state updates
- **Stacked LSTM Layers**: Multiple LSTM layers with proper gradient flow
- **Dense Output Layer**: Fully connected layer for final predictions
- **Loss Functions**: MSE for regression, Cross-entropy for classification
- **Backpropagation Through Time (BPTT)**: Proper gradient computation for sequences
- **Adam Optimizer**: Advanced optimization with momentum and adaptive learning rates
- **Mini-batch Training**: Efficient batch processing
- **Gradient Clipping**: Prevents exploding gradients
- **Comprehensive Training**: Complete training pipeline with validation

## Architecture

```
Input → LSTM Layer 1 → LSTM Layer 2 → Dense Layer → Output
```

## Files

- `lstm_cell.py`: Core LSTM cell implementation
- `stacked_lstm.py`: Multi-layer LSTM with proper state management
- `dense_layer.py`: Fully connected output layer
- `loss_functions.py`: MSE and cross-entropy loss functions
- `adam_optimizer.py`: Adam optimizer implementation
- `lstm_model.py`: Complete model combining all components
- `trainer.py`: Training utilities and dataset generation
- `main.py`: Regression demo (predict next number in sequence)
- `demo_classification.py`: Classification demo (sequence pattern recognition)

## Usage

### Regression Task (Predict Next Number)
```bash
python main.py
```

### Classification Task (Sequence Patterns)
```bash
python demo_classification.py
```

## Example Output

The model learns to predict the next number in arithmetic sequences:
- Input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] → Predicted: ~10.0
- Input: [15, 16, 17, 18, 19, 20, 21, 22, 23, 24] → Predicted: ~25.0

## Implementation Details

- **Xavier Weight Initialization**: Proper weight initialization for stable training
- **Numerical Stability**: Clipped activations and stable softmax implementation
- **Memory Efficient**: Proper state management and gradient computation
- **Modular Design**: Clean separation of concerns for easy understanding and modification

## Mathematical Foundation

The LSTM cell implements the following equations:

```
f_t = σ(W_f·x_t + U_f·h_{t-1} + b_f)  # Forget gate
i_t = σ(W_i·x_t + U_i·h_{t-1} + b_i)  # Input gate
o_t = σ(W_o·x_t + U_o·h_{t-1} + b_o)  # Output gate
c̃_t = tanh(W_c·x_t + U_c·h_{t-1} + b_c)  # Candidate values
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t      # Cell state
h_t = o_t ⊙ tanh(c_t)                 # Hidden state
```

Where σ is the sigmoid function and ⊙ denotes element-wise multiplication.