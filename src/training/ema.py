"""Exponential Moving Average implementation for model weights."""


class EMA:
    """Exponential Moving Average for model weights.
    
    This maintains a shadow copy of model weights that gets updated using an 
    exponential moving average. This typically leads to better model performance
    and more robust convergence.
    """
    
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: The model whose weights will be tracked
            decay: The decay rate for the EMA (closer to 1 means slower updates)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update the EMA weights after each optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply the EMA weights to the model for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore the original weights to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
