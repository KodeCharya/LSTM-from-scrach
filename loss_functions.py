import numpy as np

class LossFunctions:
    """
    Collection of loss functions for training.
    """
    
    @staticmethod
    def mse_loss(y_true, y_pred):
        """
        Mean Squared Error loss for regression tasks
        
        Args:
            y_true: True values (batch_size, output_size)
            y_pred: Predicted values (batch_size, output_size)
            
        Returns:
            loss: Scalar loss value
            grad: Gradient w.r.t. predictions
        """
        diff = y_pred - y_true
        loss = np.mean(diff ** 2)
        grad = 2 * diff / y_true.size
        
        return loss, grad
    
    @staticmethod
    def cross_entropy_loss(y_true, y_pred):
        """
        Cross entropy loss for classification tasks
        
        Args:
            y_true: True labels (batch_size,) or one-hot (batch_size, num_classes)
            y_pred: Predicted logits (batch_size, num_classes)
            
        Returns:
            loss: Scalar loss value
            grad: Gradient w.r.t. predictions
        """
        batch_size = y_pred.shape[0]
        
        # Apply softmax to get probabilities
        y_pred_softmax = LossFunctions.softmax(y_pred)
        
        # Convert labels to one-hot if necessary
        if len(y_true.shape) == 1:
            num_classes = y_pred.shape[1]
            y_true_onehot = np.zeros((batch_size, num_classes))
            y_true_onehot[np.arange(batch_size), y_true.astype(int)] = 1
        else:
            y_true_onehot = y_true
        
        # Compute loss (with numerical stability)
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred_softmax, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true_onehot * np.log(y_pred_clipped), axis=1))
        
        # Gradient w.r.t. logits (before softmax)
        grad = (y_pred_softmax - y_true_onehot) / batch_size
        
        return loss, grad
    
    @staticmethod
    def softmax(x):
        """
        Numerically stable softmax function
        
        Args:
            x: Input logits (batch_size, num_classes)
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)