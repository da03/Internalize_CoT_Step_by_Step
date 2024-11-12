import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict
import os

class ActivationCache:
    def __init__(self, cache_dir: str, checkpoint_every: int = 100):
        self.cache_dir = cache_dir
        self.activations: Dict[str, list] = {}  # layer_name -> list of activations
        self.input_length = None
        self.first_pred_captured = False
        self.checkpoint_every = checkpoint_every
        self.sample_count = 0
        os.makedirs(cache_dir, exist_ok=True)
        
    def set_input_length(self, length: int):
        """Set the input length to know when prediction starts"""
        self.input_length = length
        self.first_pred_captured = False
        
    def cache_activation(self, activation: Union[torch.Tensor, Tuple], layer_name: str):
        """Cache activation for the first prediction token only"""
        if isinstance(activation, tuple):
            activation = activation[0]
        
        # Only cache if this is the first prediction token
        if activation.size(1) == 1 and not self.first_pred_captured:
            # Take the activation for the last position (first prediction token)
            pred_activation = activation[:, -1, :]
            act_np = pred_activation.detach().cpu().numpy()
            
            if layer_name not in self.activations:
                self.activations[layer_name] = []
            
            self.activations[layer_name].append(act_np)
            
            # Only count sample once per input (not per layer)
            if layer_name == sorted(self.activations.keys())[0]:  # Count only for first layer
                self.sample_count += 1
                
                # Check if we should save a checkpoint
                if self.sample_count % self.checkpoint_every == 0:
                    self.save_checkpoint()
            
            self.first_pred_captured = True
    
    def save_checkpoint(self):
        """Save current activations as a checkpoint"""
        checkpoint_dir = os.path.join(self.cache_dir, f'checkpoint_{self.sample_count}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for layer_name, acts in self.activations.items():
            if acts:
                combined = np.concatenate(acts, axis=0)
                save_path = os.path.join(checkpoint_dir, f"{layer_name}_first_pred.npy")
                np.save(save_path, combined)
                print(f"Checkpoint {self.sample_count}: Saved {layer_name} activations of shape {combined.shape} to {save_path}")
    
    def save_to_disk(self, final: bool = True):
        """Save final activations to disk"""
        if final:
            final_dir = os.path.join(self.cache_dir, 'final')
            os.makedirs(final_dir, exist_ok=True)
            
            for layer_name, acts in self.activations.items():
                if acts:
                    combined = np.concatenate(acts, axis=0)
                    save_path = os.path.join(final_dir, f"{layer_name}_first_pred.npy")
                    np.save(save_path, combined)
                    print(f"Final: Saved {layer_name} activations of shape {combined.shape} to {save_path}")
            
    def clear(self):
        """Clear the activation cache"""
        self.activations.clear()
        self.first_pred_captured = False

def get_activation_hook(cache: ActivationCache, layer_name: str):
    """Creates a hook function for capturing activations"""
    def hook(module, input, output):
        cache.cache_activation(output, layer_name)
    return hook

def get_layer_by_name(model, layer_name: str):
    """Get a specific layer from the model based on the layer name"""
    if layer_name.startswith('transformer_layer_'):
        layer_idx = int(layer_name.split('_')[-1])
        return model.base_model.transformer.h[layer_idx]
    elif layer_name == 'embedding':
        return model.base_model.transformer.wte
    else:
        raise ValueError(f"Unknown layer name: {layer_name}")

def attach_hooks_to_layers(model, layer_names: list, cache: ActivationCache):
    """Attach hooks to multiple layers in the model"""
    hooks = []
    for layer_name in layer_names:
        layer = get_layer_by_name(model, layer_name)
        hook = layer.register_forward_hook(get_activation_hook(cache, layer_name))
        hooks.append(hook)
    return hooks