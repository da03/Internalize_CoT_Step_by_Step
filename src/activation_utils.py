# import torch
# import numpy as np
# from typing import Optional, Union, Tuple, Dict
# import os

# class ActivationCache:
#     def __init__(self, cache_dir: str, checkpoint_every: int = 100, pred_token_idx: int = -1):
#         self.cache_dir = cache_dir
#         self.activations: Dict[str, list] = {}  # layer_name -> list of activations
#         self.input_length = None
#         self.first_pred_captured = {} # layer_name -> bool 
#         self.checkpoint_every = checkpoint_every
#         self.sample_count = 0
#         self.pred_token_idx = pred_token_idx  # New parameter to specify which prediction token to cache
#         os.makedirs(cache_dir, exist_ok=True)
        
#     def set_input_length(self, length: int):
#         """Set the input length to know when prediction starts"""
#         self.input_length = length
#         self.first_pred_captured = { layer_name: False for layer_name in self.activations.keys() }
        
#     def cache_activation(self, activation: Union[torch.Tensor, Tuple], layer_name: str):
#         """Cache activation for the specified prediction token"""
#         if isinstance(activation, tuple):
#             activation = activation[0]
        
#         if activation.size(1) == 1:
#             # For streaming predictions, use last token
#             pred_activation = activation[:, -1, :]
#             act_np = pred_activation.detach().cpu().numpy()
            
#             if layer_name not in self.activations:
#                 self.activations[layer_name] = []
#                 self.first_pred_captured[layer_name] = False 
            
#             if self.first_pred_captured[layer_name]:
#                 return 
            
#             # Store the activation
#             self.activations[layer_name].append(act_np)
            
#             # Only count sample once per input (not per layer)
#             if layer_name == sorted(self.activations.keys())[0]:  # Count only for first layer
#                 self.sample_count += 1
                
#                 # Check if we should save a checkpoint
#                 if self.sample_count % self.checkpoint_every == 0:
#                     self.save_checkpoint()
            
#             self.first_pred_captured[layer_name] = True
#         else:
#             # For batch predictions, use specified token index
#             if self.pred_token_idx == -1:
#                 # Use last token by default
#                 pred_activation = activation[:, -1, :]
#             else:
#                 # Use specified token index
#                 pred_activation = activation[:, self.pred_token_idx, :]
            
#             act_np = pred_activation.detach().cpu().numpy()
            
#             if layer_name not in self.activations:
#                 self.activations[layer_name] = []
#                 self.first_pred_captured[layer_name] = False
            
#             if self.first_pred_captured[layer_name]:
#                 return
            
#             self.activations[layer_name].append(act_np)
            
#             if layer_name == sorted(self.activations.keys())[0]:
#                 self.sample_count += 1
#                 if self.sample_count % self.checkpoint_every == 0:
#                     self.save_checkpoint()
            
#             self.first_pred_captured[layer_name] = True
    
#     def save_checkpoint(self):
#         """Save current activations as a checkpoint"""
#         checkpoint_dir = os.path.join(self.cache_dir, f'checkpoint_{self.sample_count}')
#         os.makedirs(checkpoint_dir, exist_ok=True)
        
#         token_suffix = 'last_pred' if self.pred_token_idx == -1 else f'pred_token_{self.pred_token_idx}'
        
#         for layer_name, acts in self.activations.items():
#             if acts:
#                 combined = np.concatenate(acts, axis=0)
#                 save_path = os.path.join(checkpoint_dir, f"{layer_name}_{token_suffix}.npy")
#                 np.save(save_path, combined)
#                 print(f"Checkpoint {self.sample_count}: Saved {layer_name} activations of shape {combined.shape} to {save_path}")
    
#     def save_to_disk(self, final: bool = True):
#         """Save final activations to disk"""
#         if final:
#             final_dir = os.path.join(self.cache_dir, 'final')
#             os.makedirs(final_dir, exist_ok=True)
            
#             token_suffix = 'last_pred' if self.pred_token_idx == -1 else f'pred_token_{self.pred_token_idx}'
            
#             for layer_name, acts in self.activations.items():
#                 if acts:
#                     combined = np.concatenate(acts, axis=0)
#                     save_path = os.path.join(final_dir, f"{layer_name}_{token_suffix}.npy")
#                     np.save(save_path, combined)
    
#     def clear(self):
#         """Clear the activation cache"""
#         self.activations.clear()
#         self.first_pred_captured = { layer_name: False for layer_name in self.activations.keys() }

# def get_activation_hook(cache: ActivationCache, layer_name: str):
#     """Creates a hook function for capturing activations"""
#     # def hook(module, input, output):
#     return lambda module, input, output: cache.cache_activation(output, layer_name)

# def get_layer_by_name(model, layer_name: str):
#     """Get a specific layer from the model based on the layer name"""
#     if layer_name.startswith('transformer_layer_'):
#         layer_idx = int(layer_name.split('_')[-1])
#         return model.base_model.transformer.h[layer_idx]
#     elif layer_name == 'embedding':
#         return model.base_model.transformer.wte
#     else:
#         raise ValueError(f"Unknown layer name: {layer_name}")

# def attach_hooks_to_layers(model, layer_names: list, cache: ActivationCache):
#     """Attach hooks to multiple layers in the model"""
#     hooks = []
#     for layer_name in layer_names:
#         print(f"Attaching hook to {layer_name}")
#         layer = get_layer_by_name(model, layer_name)
#         hook = layer.register_forward_hook(get_activation_hook(cache, layer_name), always_call=True)
#         hooks.append(hook)
#     print(f"Attached {len(hooks)} hooks, {cache.activations}")
#     return hooks

import torch
import numpy as np
from typing import Optional, Union, Tuple, Dict
import os

class ActivationCache:
    def __init__(self, cache_dir: str, pred_token_idx: int = -1):
        self.cache_dir = cache_dir
        self.activations: Dict[str, list] = {}  # layer_name -> list of activations
        self.pred_token_idx = pred_token_idx  # Specifies which prediction token to cache
        os.makedirs(cache_dir, exist_ok=True)
        
    def cache_activation(self, activation: Union[torch.Tensor, Tuple], layer_name: str):
        """Cache activation for the specified prediction token"""
        if isinstance(activation, tuple):
            activation = activation[0]
        
        if layer_name not in self.activations:
            self.activations[layer_name] = []
        
        seq_len = activation.size(1)
        
        # Handle negative and positive indices
        index = self.pred_token_idx if self.pred_token_idx >= 0 else seq_len + self.pred_token_idx
        
        if 0 <= index < seq_len:
            pred_activation = activation[:, index, :]
            act_np = pred_activation.detach().cpu().numpy()
            self.activations[layer_name].append(act_np)
        else:
            # Index out of bounds
            pass  # You may log this event if necessary
        
    def save_checkpoint(self, checkpoint_number: int):
        """Save current activations as a checkpoint"""
        checkpoint_dir = os.path.join(self.cache_dir, f'checkpoint_{checkpoint_number}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        token_suffix = f'pred_token_{self.pred_token_idx}'
        
        for layer_name, acts in self.activations.items():
            if acts:
                combined = np.concatenate(acts, axis=0)
                save_path = os.path.join(checkpoint_dir, f"{layer_name}_{token_suffix}.npy")
                np.save(save_path, combined)
                print(f"Checkpoint {checkpoint_number}: Saved {layer_name} activations of shape {combined.shape} to {save_path}")
        # Clear activations after checkpointing
        self.activations.clear()
    
    def save_to_disk(self, final: bool = True):
        """Save final activations to disk"""
        if final and self.activations:
            final_dir = os.path.join(self.cache_dir, 'final')
            os.makedirs(final_dir, exist_ok=True)
            
            token_suffix = f'pred_token_{self.pred_token_idx}'
            
            for layer_name, acts in self.activations.items():
                if acts:
                    combined = np.concatenate(acts, axis=0)
                    save_path = os.path.join(final_dir, f"{layer_name}_{token_suffix}.npy")
                    np.save(save_path, combined)
                    print(f"Final Save: Saved {layer_name} activations of shape {combined.shape} to {save_path}")
            # Clear activations after saving
            self.activations.clear()
def get_activation_hook(cache: ActivationCache, layer_name: str):
    """Creates a hook function for capturing activations"""
    return lambda module, input, output: cache.cache_activation(output, layer_name)

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
        print(f"Attaching hook to {layer_name}")
        layer = get_layer_by_name(model, layer_name)
        hook = layer.register_forward_hook(get_activation_hook(cache, layer_name))
        hooks.append(hook)
    print(f"Attached {len(hooks)} hooks.")
    return hooks

