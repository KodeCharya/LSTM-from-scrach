import numpy as np
from lstm_model import LSTMModel

class Trainer:
    """
    Trainer class for LSTM model with various training utilities.
    """

    def __init__(self, model):
        """
        Args:
            model: LSTMModel instance to train
        """
        self.model = model
        self.train_losses = []
    
    def create_sequence_dataset(self, sequence_length=10, num_samples=1000, task_type='next_number'):
        """
        Create a toy dataset for training
        
        Args:
            sequence_length: Length of input sequences
            num_samples: Number of training samples
            task_type: 'next_number' or 'sum_sequence'
            
        Returns:
            X: Input sequences (num_samples, sequence_length, 1)
            y: Target values (num_samples, 1)
        """
        X = []
        y = []
        
        for _ in range(num_samples):
            if task_type == 'next_number':
                # Predict next number in sequence
                start = np.random.randint(0, 100)
                sequence = np.arange(start, start + sequence_length)
                target = start + sequence_length
                
                X.append(sequence.reshape(-1, 1))
                y.append([target])
                
            elif task_type == 'sum_sequence':
                # Predict sum of sequence
                sequence = np.random.randint(0, 10, sequence_length)
                target = np.sum(sequence)
                
                X.append(sequence.reshape(-1, 1))
                y.append([target])
        
        return np.array(X), np.array(y)
    
    def create_classification_dataset(self, sequence_length=5, num_samples=1000):
        """
        Create a toy classification dataset
        
        Returns:
            X: Input sequences
            y: Class labels (0: increasing, 1: decreasing, 2: mixed)
        """
        X = []
        y = []
        
        for _ in range(num_samples):
            # Generate random sequence
            sequence = np.random.randint(0, 10, sequence_length)
            
            # Determine pattern
            diffs = np.diff(sequence)
            if np.all(diffs >= 0):
                label = 0  # Increasing
            elif np.all(diffs <= 0):
                label = 1  # Decreasing
            else:
                label = 2  # Mixed
            
            X.append(sequence.reshape(-1, 1))
            y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2, verbose=True):
        """
        Train the LSTM model
        
        Args:
            X: Input data (num_samples, sequence_length, input_size)
            y: Target data (num_samples, output_size)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            validation_split: Fraction of data to use for validation
            verbose: Whether to print training progress
        """
        # Split data into train/validation
        num_samples = X.shape[0]
        num_val = int(num_samples * validation_split)
        indices = np.random.permutation(num_samples)
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        num_train = X_train.shape[0]
        num_batches = (num_train + batch_size - 1) // batch_size
        
        if verbose:
            print(f"Training on {num_train} samples, validating on {num_val} samples")
            print(f"Epochs: {epochs}, Batch size: {batch_size}")
            print("-" * 50)
        
        for epoch in range(epochs):
            # Shuffle training data
            shuffle_indices = np.random.permutation(num_train)
            X_train_shuffled = X_train[shuffle_indices]
            y_train_shuffled = y_train[shuffle_indices]
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_train)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Reset states for each batch
                self.model.reset_states(X_batch.shape[0])
                
                # Training step
                batch_loss = self.model.train_step(X_batch, y_batch)
                epoch_loss += batch_loss
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            self.train_losses.append(avg_loss)
            
            # Validation
            if num_val > 0:
                val_loss = self.evaluate(X_val, y_val, batch_size)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} - "
                          f"Train Loss: {avg_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1:3d}/{epochs} - "
                          f"Train Loss: {avg_loss:.6f}")
    
    def evaluate(self, X, y, batch_size=32):
        """
        Evaluate model on given data
        
        Args:
            X: Input data
            y: Target data
            batch_size: Batch size for evaluation
            
        Returns:
            Average loss
        """
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size
        total_loss = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Reset states
            self.model.reset_states(X_batch.shape[0])
            
            # Forward pass only
            y_pred, _ = self.model.forward(X_batch)
            loss, _ = self.model.compute_loss(y_batch, y_pred)
            
            total_loss += loss
        
        return total_loss / num_batches
    
    def test_predictions(self, num_tests=5, sequence_length=10):
        """
        Test model predictions on new sequences
        
        Args:
            num_tests: Number of test sequences to generate
            sequence_length: Length of test sequences
        """
        print("\n" + "="*50)
        print("TESTING MODEL PREDICTIONS")
        print("="*50)
        
        for i in range(num_tests):
            # Generate test sequence
            start = np.random.randint(0, 50)
            test_sequence = np.arange(start, start + sequence_length)
            true_next = start + sequence_length
            
            # Prepare input
            X_test = test_sequence.reshape(1, sequence_length, 1)
            
            # Reset states and predict
            self.model.reset_states(1)
            prediction = self.model.predict(X_test)
            
            print(f"Test {i+1}:")
            print(f"  Sequence: {test_sequence}")
            print(f"  True next: {true_next}")
            print(f"  Predicted: {prediction[0, 0]:.2f}")
            print(f"  Error: {abs(prediction[0, 0] - true_next):.2f}")
            print()
    
    def plot_training_loss(self):
        """Print training loss progression"""
        print("\n" + "="*50)
        print("TRAINING LOSS PROGRESSION")
        print("="*50)
        
        if not self.train_losses:
            print("No training history available.")
            return
        
        # Print loss every 10 epochs
        step = max(1, len(self.train_losses) // 10)
        for i in range(0, len(self.train_losses), step):
            epoch = i + 1
            loss = self.train_losses[i]
            print(f"Epoch {epoch:3d}: Loss = {loss:.6f}")
        
        # Print final loss
        if len(self.train_losses) > 1:
            final_epoch = len(self.train_losses)
            final_loss = self.train_losses[-1]
            print(f"Epoch {final_epoch:3d}: Loss = {final_loss:.6f} (Final)")