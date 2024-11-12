# analyze_activations.py

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import seaborn as sns
from typing import Dict, List
import argparse

def load_activations(activation_dir: str) -> Dict[str, np.ndarray]:
    """Load all activation files from the directory structure"""
    activations = {}
    
    # First load and combine all checkpoints
    for dirname in os.listdir(activation_dir):
        dirpath = os.path.join(activation_dir, dirname)
        if not os.path.isdir(dirpath):
            continue
        print(f"Processing directory: {dirpath}")
        
        for filename in os.listdir(dirpath):
            if not filename.endswith('_first_pred.npy'):
                continue
                
            layer_name = filename.replace('_first_pred.npy', '')
            filepath = os.path.join(dirpath, filename)
            print(f"Loading activations from {filepath}")
            
            if layer_name not in activations:
                activations[layer_name] = []
            
            data = np.load(filepath)
            activations[layer_name].append(data)
    
    # Combine all checkpoints for each layer
    return {layer: np.concatenate(acts, axis=0) for layer, acts in activations.items()}

def analyze_full_pca(activations: Dict[str, np.ndarray], save_dir: str = 'pca_results'):
    """Perform full PCA analysis on activations and create 2D scatter plots."""
    os.makedirs(save_dir, exist_ok=True)
    results = {}
    
    for layer_name, acts in activations.items():
        print(f"\nAnalyzing {layer_name}...")
        print(f"Activation shape: {acts.shape}")
        
        # Check if there are enough samples for PCA
        if acts.shape[0] < 2:
            print(f"Not enough samples in {layer_name} for PCA. Skipping.")
            continue
        
        # Perform PCA with all components
        n_components = min(acts.shape[0], acts.shape[1])
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(acts)
        
        # Save results
        results[layer_name] = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'singular_values': pca.singular_values_,
            'components': pca.components_,
            'transformed': transformed
        }
        
        # 1. Plot full spectrum of explained variance
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(pca.explained_variance_ratio_, 'b-', alpha=0.7)
        plt.title(f'{layer_name} - Full Explained Variance Spectrum')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.yscale('log')  # Log scale to see small values

        plt.subplot(1, 2, 2)
        plt.plot(np.cumsum(pca.explained_variance_ratio_), 'r-')
        plt.title(f'{layer_name} - Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name}_full_spectrum.png'))
        plt.close()
        
        # 2. Analyze dimensionality
        variance_thresholds = [0.5, 0.75, 0.9, 0.95, 0.99]
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        dims_needed = [np.argmax(cumsum >= threshold) + 1 for threshold in variance_thresholds]
        
        print("\nDimensionality analysis:")
        for threshold, dims in zip(variance_thresholds, dims_needed):
            print(f"Dimensions needed for {threshold*100}% variance: {dims}")
            
        # 3. Save detailed results
        np.save(os.path.join(save_dir, f'{layer_name}_full_pca.npy'), {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumsum,
            'singular_values': pca.singular_values_,
            'dims_for_thresholds': dict(zip(variance_thresholds, dims_needed))
        })
        
        # 4. Plot dimensionality histogram
        plt.figure(figsize=(12, 6))
        bins = np.logspace(0, np.log10(len(pca.explained_variance_ratio_)), 50)
        plt.hist(np.arange(1, len(pca.explained_variance_ratio_) + 1), 
                 bins=bins, 
                 weights=pca.explained_variance_ratio_,
                 alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f'{layer_name} - Distribution of Variance Across Components')
        plt.xlabel('Component (log scale)')
        plt.ylabel('Explained Variance Ratio (log scale)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name}_variance_dist.png'))
        plt.close()
        
        # 5. Scree plot with elbow analysis
        plt.figure(figsize=(10, 6))
        variance = pca.explained_variance_
        plt.plot(range(1, len(variance) + 1), variance, 'bo-')
        plt.yscale('log')
        plt.title(f'{layer_name} - Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue (log scale)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name}_scree.png'))
        plt.close()
        
        # 6. Create a 2D scatter plot of the data projected onto the first two principal components
        plt.figure(figsize=(10, 8))
        plt.scatter(transformed[:, 0], transformed[:, 1], alpha=0.5)
        plt.title(f'{layer_name} - 2D Projection onto First Two Principal Components')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{layer_name}_2d_projection.png'))
        plt.close()
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--activation_dir', type=str, required=True, help='Directory containing activation files')
    parser.add_argument('--save_dir', type=str, default='pca_results', help='Directory to save PCA results')
    args = parser.parse_args()
    
    print(f"Loading activations from {args.activation_dir}")
    activations = load_activations(args.activation_dir)
    
    print(f"\nPerforming full PCA analysis")
    results = analyze_full_pca(activations, args.save_dir)
    
    print(f"\nResults saved to {args.save_dir}")

if __name__ == "__main__":
    main()
