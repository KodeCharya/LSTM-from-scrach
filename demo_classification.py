#!/usr/bin/env python3
"""
LSTM Classification Demo
========================

This script demonstrates LSTM for sequence classification tasks.
The model learns to classify sequences as increasing, decreasing, or mixed patterns.
"""

import numpy as np
from lstm_model import LSTMModel
from trainer import Trainer

def main():
    """Classification demo"""
    print("="*60)
    print("LSTM SEQUENCE CLASSIFICATION DEMO")
    print("="*60)
    print("Training LSTM to classify sequence patterns...")
    print()
    
    # Set random seed
    np.random.seed(42)
    
    # Model configuration for classification
    input_size = 1
    hidden_sizes = [32, 16]
    output_size = 3  # 3 classes: increasing, decreasing, mixed
    
    # Create classification model
    model = LSTMModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        loss_type='cross_entropy'
    )
    
    print(f"Classification Model:")
    print(f"  Classes: 0=Increasing, 1=Decreasing, 2=Mixed")
    print(f"  LSTM layers: {hidden_sizes}")
    print()
    
    # Create trainer
    trainer = Trainer(model)
    
    # Generate classification data
    print("Generating classification dataset...")
    X, y = trainer.create_classification_dataset(
        sequence_length=5,
        num_samples=1500
    )
    
    # Show examples
    print("Example training samples:")
    for i in range(3):
        sequence = X[i].flatten()
        label = y[i]
        class_names = ['Increasing', 'Decreasing', 'Mixed']
        print(f"  Sequence: {sequence} -> Class: {label} ({class_names[label]})")
    print()
    
    # Train model
    print("Training classification model...")
    trainer.train(
        X=X,
        y=y,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        verbose=True
    )
    
    # Test classification
    print("\n" + "="*50)
    print("TESTING CLASSIFICATION")
    print("="*50)
    
    test_sequences = [
        [1, 2, 3, 4, 5],      # Increasing
        [5, 4, 3, 2, 1],      # Decreasing
        [1, 3, 2, 4, 3],      # Mixed
        [2, 2, 2, 2, 2],      # Constant (should be increasing)
        [7, 8, 9, 10, 11],    # Increasing
    ]
    
    class_names = ['Increasing', 'Decreasing', 'Mixed']
    
    for i, test_seq in enumerate(test_sequences):
        X_test = np.array(test_seq).reshape(1, len(test_seq), 1)
        model.reset_states(1)
        
        # Get predictions (probabilities)
        predictions = model.predict(X_test)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0, predicted_class]
        
        # Determine true class
        diffs = np.diff(test_seq)
        if np.all(diffs >= 0):
            true_class = 0
        elif np.all(diffs <= 0):
            true_class = 1
        else:
            true_class = 2
        
        print(f"Test {i+1}:")
        print(f"  Sequence: {test_seq}")
        print(f"  True class: {true_class} ({class_names[true_class]})")
        print(f"  Predicted: {predicted_class} ({class_names[predicted_class]})")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Correct: {'✓' if predicted_class == true_class else '✗'}")
        print()

if __name__ == "__main__":
    main()