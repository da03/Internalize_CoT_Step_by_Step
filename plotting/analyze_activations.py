# # # analyze_activations.py
#%% 

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import seaborn as sns
import argparse
from typing import Dict, List, Tuple
import plotly.graph_objects as go
import sys

#%% 

def extract_labels_from_expressions(expressions_file: str) -> Tuple[List[str], np.ndarray]:
    """Extract sequences and convert them to numbers"""
    full_labels = []
    numeric_values = []
    
    with open(expressions_file, 'r') as f:
        for line in f:
            parts = line.strip().split('#')
            parts2 = line.strip().split('||')
            if len(parts) > 1 and len(parts2) > 1:
                input = parts2[0].strip()[::-1]
                output = parts[-1].strip()[::-1]
                full_labels.append(f"{input} || {output}")
                # Convert space-separated sequence to single number
                num = int(''.join(output.split()))
                numeric_values.append(num)
    return full_labels, np.array(numeric_values)

full_labels, numeric_values = extract_labels_from_expressions("data/4_by_4_mult/test_bigbench.txt")
print(full_labels)
#%% 
def load_activations_with_labels(activation_dir: str, expressions_file: str):
    """Load activations from final folder and their corresponding labels"""
    full_labels, numeric_values = extract_labels_from_expressions(expressions_file)
    print(f"Loaded {len(full_labels)} labels")
    
    final_dir = os.path.join(activation_dir, "final")
    if not os.path.exists(final_dir):
        raise ValueError(f"Final directory not found at {final_dir}")
        
    activations = {}
    for filename in os.listdir(final_dir):
        if not filename.endswith('_first_pred.npy'):
            continue
            
        layer_name = filename.replace('_first_pred.npy', '')
        filepath = os.path.join(final_dir, filename)
        
        print(f"Loading activations from {filepath}")
        data = np.load(filepath)
        print(f"Loaded activation shape: {data.shape}")
        
        if len(data) > len(full_labels):
            data = data[:len(full_labels)]
            print(f"Trimmed to {len(data)} samples to match labels")
        
        activations[layer_name] = data
    
    return activations, full_labels, numeric_values

def analyze_pca_with_math_labels(
    activations: Dict[str, np.ndarray], 
    full_labels: List[str],
    numeric_values: np.ndarray,
    save_dir: str = 'pca_results_labels'
):
    """Perform PCA analysis with scatter plots colored by numeric value"""
    os.makedirs(save_dir, exist_ok=True)
    
    for layer_name, acts in activations.items():
        print(f"\nAnalyzing {layer_name}...")
        
        # Perform PCA
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(acts)
        
        # Create interactive scatter plot with plotly
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=transformed[:, 0],
            y=transformed[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=numeric_values,
                colorscale='Viridis',
                colorbar=dict(title='Numeric Value'),
                opacity=0.6
            ),
            text=full_labels,  # Add hover text showing full labels
            hovertemplate='<b>Value:</b> %{marker.color}<br>' +
                         '<b>Label:</b> %{text}<br>' +
                         '<b>PC1:</b> %{x:.2f}<br>' +
                         '<b>PC2:</b> %{y:.2f}<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{layer_name} - PCA Projection',
            xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
            yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
            width=1000,
            height=800,
            template='plotly_white'  # Clean white background with grid
        )
        
        # Show plot interactively
        fig.show()
        fig.write_image(os.path.join(save_dir, f'{layer_name}_pca.png'), scale=2)

# def extract_labels_from_expressions(expressions_file: str) -> List[str]:
#     """Extract full sequence labels from math expressions (all numbers after #)"""
#     labels = []
#     with open(expressions_file, 'r') as f:
#         for line in f:
#             # Find the part after the # symbols
#             parts = line.strip().split('#')
#             if len(parts) > 1:
#                 # Get all numbers after the last # as a single string
#                 label = parts[-1].strip()
#                 labels.append(label)
#             else:
#                 print(f"Warning: No label found in line: {line}")
#                 labels.append(None)
    
#     return labels

# def load_activations_with_labels(activation_dir: str, expressions_file: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
#     """Load activations from final folder and their corresponding labels"""
#     # Load labels first
#     labels = extract_labels_from_expressions(expressions_file)
#     print(f"Loaded {len(labels)} labels")
    
#     # Load activations from final folder only
#     final_dir = os.path.join(activation_dir, "final")
#     if not os.path.exists(final_dir):
#         raise ValueError(f"Final directory not found at {final_dir}")
        
#     activations = {}
#     for filename in os.listdir(final_dir):
#         if not filename.endswith('_first_pred.npy'):
#             continue
            
#         layer_name = filename.replace('_first_pred.npy', '')
#         filepath = os.path.join(final_dir, filename)
        
#         print(f"Loading activations from {filepath}")
#         data = np.load(filepath)
#         print(f"Loaded activation shape: {data.shape}")
        
#         # Trim to match number of labels if necessary
#         if len(data) > len(labels):
#             data = data[:len(labels)]
#             print(f"Trimmed to {len(data)} samples to match labels")
        
#         activations[layer_name] = data
    
#     return activations, labels



# def analyze_pca_with_math_labels(
#     activations: Dict[str, np.ndarray], 
#     labels: np.ndarray,
#     save_dir: str = 'pca_results',
#     visualization_type: str = 'both'  # 'scatter', 'density', or 'both'
# ):
#     """Perform PCA analysis with math expression labels"""
#     os.makedirs(save_dir, exist_ok=True)
    
#     for layer_name, acts in activations.items():
#         print(f"\nAnalyzing {layer_name}...")
        
#         # Perform PCA
#         n_components = min(acts.shape[0], acts.shape[1])
#         pca = PCA(n_components=n_components)
#         transformed = pca.fit_transform(acts)
        
#         if visualization_type in ['scatter', 'both']:
#             # Create labeled scatter plot
#             plt.figure(figsize=(12, 8))
#             scatter = plt.scatter(
#                 transformed[:, 0],
#                 transformed[:, 1],
#                 c=labels,
#                 cmap='viridis',
#                 alpha=0.6,
#                 s=50
#             )
#             plt.colorbar(scatter, label='First Predicted Number')
#             plt.title(f'{layer_name} - PCA Projection by Predicted First Digit')
#             plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
#             plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, f'{layer_name}_scatter_labeled.png'))
#             plt.close()
        
#         if visualization_type in ['density', 'both']:
#             # Create density plot for each unique label
#             plt.figure(figsize=(15, 10))
#             unique_labels = np.unique(labels)
#             for label in unique_labels:
#                 mask = labels == label
#                 sns.kdeplot(
#                     x=transformed[mask, 0],
#                     y=transformed[mask, 1],
#                     label=f'Pred: {label}',
#                     alpha=0.5,
#                     levels=5
#                 )
#             plt.title(f'{layer_name} - Density by Predicted First Digit')
#             plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
#             plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
#             plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#             plt.grid(True)
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, f'{layer_name}_density_labeled.png'))
#             plt.close()
            
#             # Additional analysis: cluster separation
#             print(f"\nCluster Analysis for {layer_name}:")
#             for label in unique_labels:
#                 mask = labels == label
#                 center = np.mean(transformed[mask, :2], axis=0)
#                 std = np.std(transformed[mask, :2], axis=0)
#                 print(f"Prediction {label}:")
#                 print(f"  Count: {np.sum(mask)}")
#                 print(f"  Center: ({center[0]:.3f}, {center[1]:.3f})")
#                 print(f"  Std Dev: ({std[0]:.3f}, {std[1]:.3f})")

# def analyze_full_pca(activations: Dict[str, np.ndarray], save_dir: str = 'pca_results'):
#     """Perform full PCA analysis on activations and create 2D scatter plots."""
#     os.makedirs(save_dir, exist_ok=True)
#     results = {}
    
#     for layer_name, acts in activations.items():
#         print(f"\nAnalyzing {layer_name}...")
#         print(f"Activation shape: {acts.shape}")
        
#         # Check if there are enough samples for PCA
#         if acts.shape[0] < 2:
#             print(f"Not enough samples in {layer_name} for PCA. Skipping.")
#             continue
        
#         # Perform PCA with all components
#         n_components = min(acts.shape[0], acts.shape[1])
#         pca = PCA(n_components=n_components)
#         transformed = pca.fit_transform(acts)
        
#         # Save results
#         results[layer_name] = {
#             'explained_variance_ratio': pca.explained_variance_ratio_,
#             'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
#             'singular_values': pca.singular_values_,
#             'components': pca.components_,
#             'transformed': transformed
#         }
        
#         # 1. Plot full spectrum of explained variance
#         plt.figure(figsize=(15, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(pca.explained_variance_ratio_, 'b-', alpha=0.7)
#         plt.title(f'{layer_name} - Full Explained Variance Spectrum')
#         plt.xlabel('Principal Component')
#         plt.ylabel('Explained Variance Ratio')
#         plt.yscale('log')  # Log scale to see small values

#         plt.subplot(1, 2, 2)
#         plt.plot(np.cumsum(pca.explained_variance_ratio_), 'r-')
#         plt.title(f'{layer_name} - Cumulative Explained Variance')
#         plt.xlabel('Number of Components')
#         plt.ylabel('Cumulative Explained Variance')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'{layer_name}_full_spectrum.png'))
#         plt.close()
        
#         # 2. Analyze dimensionality
#         variance_thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
#         cumsum = np.cumsum(pca.explained_variance_ratio_)
#         dims_needed = [np.argmax(cumsum >= threshold) + 1 for threshold in variance_thresholds]
        
#         print("\nDimensionality analysis:")
#         for threshold, dims in zip(variance_thresholds, dims_needed):
#             print(f"Dimensions needed for {threshold*100}% variance: {dims}")
            
#         # 3. Save detailed results
#         np.save(os.path.join(save_dir, f'{layer_name}_full_pca.npy'), {
#             'explained_variance_ratio': pca.explained_variance_ratio_,
#             'cumulative_variance': cumsum,
#             'singular_values': pca.singular_values_,
#             'dims_for_thresholds': dict(zip(variance_thresholds, dims_needed))
#         })
        
#         # 4. Plot dimensionality histogram
#         plt.figure(figsize=(12, 6))
#         bins = np.logspace(0, np.log10(len(pca.explained_variance_ratio_)), 50)
#         plt.hist(np.arange(1, len(pca.explained_variance_ratio_) + 1), 
#                  bins=bins, 
#                  weights=pca.explained_variance_ratio_,
#                  alpha=0.7)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.title(f'{layer_name} - Distribution of Variance Across Components')
#         plt.xlabel('Component (log scale)')
#         plt.ylabel('Explained Variance Ratio (log scale)')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'{layer_name}_variance_dist.png'))
#         plt.close()
        
#         # 5. Scree plot with elbow analysis
#         plt.figure(figsize=(10, 6))
#         variance = pca.explained_variance_
#         plt.plot(range(1, len(variance) + 1), variance, 'bo-')
#         plt.yscale('log')
#         plt.title(f'{layer_name} - Scree Plot')
#         plt.xlabel('Principal Component')
#         plt.ylabel('Eigenvalue (log scale)')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'{layer_name}_scree.png'))
#         plt.close()
        
#         # 6. Create a 2D scatter plot of the data projected onto the first two principal components
#         plt.figure(figsize=(10, 8))
#         plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.5)
#         plt.title(f'{layer_name} - 2D Projection onto First Two Principal Components')
#         plt.xlabel('Principal Component 1')
#         plt.ylabel('Principal Component 2')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, f'{layer_name}_2d_projection.png'))
#         plt.close()
        
#     return results

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--activation_dir', type=str, required=True, help='Directory containing activation files')
#     parser.add_argument('--save_dir', type=str, default='pca_results', help='Directory to save PCA results')
#     args = parser.parse_args()
    
#     print(f"Loading activations from {args.activation_dir}")
#     activations = load_activations(args.activation_dir)
    
#     print(f"\nPerforming full PCA analysis")
#     results = analyze_full_pca(activations, args.save_dir)
    
#     print(f"\nResults saved to {args.save_dir}")

# if __name__ == "__main__":
#     main()

def main():
    # Check if we're in a notebook environment
    in_notebook = 'ipykernel' in sys.modules
    
    if not in_notebook:
        parser = argparse.ArgumentParser()
        parser.add_argument('--activation_dir', type=str, required=True,
                          help='Directory containing activation files')
        parser.add_argument('--expressions_file', default='data/4_by_4_mult/test_bigbench.txt', type=str,
                          help='File containing math expressions with labels')
        parser.add_argument('--save_dir', type=str, default='pca_results_labels_2',
                          help='Directory to save PCA results')
        parser.add_argument('--visualization', type=str, default='both',
                          choices=['scatter', 'density', 'both'],
                          help='Type of visualization to generate')
        args = parser.parse_args()
    else:
        # Default values for notebook environment
        class Args:
            def __init__(self):
                self.activation_dir = "cached_activations"
                self.expressions_file = "data/4_by_4_mult/test_bigbench.txt"
                self.save_dir = "pca_results_labels_2"
                self.visualization = "both"
        args = Args()
    # print(f"Loading activations and labels...")
    # activations, labels = load_activations_with_labels(args.activation_dir, args.expressions_file)
    
    # print(f"\nPerforming PCA analysis with math expression labels")
    # analyze_pca_with_math_labels(
    #     activations,
    #     labels,
    #     args.save_dir,
    #     args.visualization
    # )
    
    # print(f"\nResults saved to {args.save_dir}")
    activations, full_labels, numeric_values = load_activations_with_labels(
        args.activation_dir, 
        args.expressions_file
    )
    
    analyze_pca_with_math_labels(activations, full_labels, numeric_values, args.save_dir)
    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()


# %%
