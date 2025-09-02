#!/usr/bin/env python3
"""
LSTM Neural Network from Scratch
================================

This script demonstrates a complete LSTM implementation using only NumPy and Python's
standard libraries. The model is trained to predict the next number in a sequence.

Example: Given [0, 1, 2, 3, 4], predict 5
"""

import numpy as np
from lstm_model import LSTMModel
from trainer import Trainer

def main():
    """Main training and testing script"""
    print("="*60)
    print("LSTM NEURAL NETWORK FROM SCRATCH")
    print("="*60)
    print("Building LSTM model to predict next number in sequence...")
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Model configuration
    input_size = 1      # Single number input
    hidden_sizes = [64, 32]  # Two LSTM layers
    output_size = 1     # Single number output
    
    # Create model
    model = LSTMModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        loss_type='mse'
    )
    
    print(f"Model Architecture:")
    print(f"  Input size: {input_size}")
    print(f"  LSTM layers: {hidden_sizes}")
    print(f"  Output size: {output_size}")
    print(f"  Total parameters: {count_parameters(model)}")
    print()
    
    # Create trainer
    trainer = Trainer(model)
    
    # Generate training data
    print("Generating training data...")
    sequence_length = 10
    num_samples = 2000
    
    X, y = trainer.create_sequence_dataset(
        sequence_length=sequence_length,
        num_samples=num_samples,
        task_type='next_number'
    )
    
    print(f"Dataset: {num_samples} sequences of length {sequence_length}")
    print(f"Task: Predict the next number in sequence")
    print()
    
    # Show example
    print("Example training sample:")
    print(f"  Input sequence: {X[0].flatten()}")
    print(f"  Target: {y[0, 0]}")
    print()
    
    # Train model
    print("Starting training...")
    trainer.train(
        X=X,
        y=y,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=True
    )
    
    # Plot training progress
    trainer.plot_training_loss()
    
    # Test predictions
    trainer.test_predictions(num_tests=8, sequence_length=sequence_length)
    
    # Additional test: longer sequences
    print("\n" + "="*50)
    print("TESTING ON LONGER SEQUENCES")
    print("="*50)
    
    longer_tests = [
        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],  # Should predict 20
        [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],       # Should predict 15
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]  # Should predict 110
    ]
    
    for i, test_seq in enumerate(longer_tests):
        X_test = np.array(test_seq).reshape(1, len(test_seq), 1)
        model.reset_states(1)
        prediction = model.predict(X_test)
        expected = test_seq[-1] + 1
        
        print(f"Test {i+1}:")
        print(f"  Sequence: {test_seq}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {prediction[0, 0]:.2f}")
        print(f"  Error: {abs(prediction[0, 0] - expected):.2f}")
        print()

def count_parameters(model):
    """Count total number of trainable parameters"""
    total = 0
    
    # LSTM parameters
    lstm_params = model.lstm.get_params()
    for layer_key, layer_params in lstm_params.items():
        for param_name, param_value in layer_params.items():
            total += param_value.size
    
    # Dense parameters
    dense_params = model.dense.get_params()
    for param_name, param_value in dense_params.items():
        total += param_value.size
    
    return total

if __name__ == "__main__":
    main()