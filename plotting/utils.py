import os
import matplotlib.pyplot as plt
import glob
from PIL import Image
import argparse

def concat_plots_from_folder(folder_path="pca_results_labels_2"):
    """
    Concatenates all images in the given folder into a single plot
    
    Args:
        folder_path (str): Path to folder containing images to concatenate
    """
    # Get all image files in folder
    image_files = [
        os.path.join(folder_path, f"transformer_layer_{i}_pca.png") for i in range(12)
    ]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
        
    # Open all images
    images = [Image.open(f) for f in image_files if not f.endswith("combined_plot.png")]
    
    # Get dimensions
    widths, heights = zip(*(i.size for i in images))
    
    # Calculate layout
    n = len(images)
    nrows = (n + 3) // 4  # Ceiling division to get number of rows needed for 4 per row
    ncols = 4
    
    # Create figure
    fig = plt.figure(figsize=(20, 5*nrows))
    
    # Add each image as a subplot
    for idx, img in enumerate(images):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(image_files[idx]).replace('.png', ''))
    
    plt.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.5)  # Adjust spacing
    plt.savefig(os.path.join(folder_path, "combined_plot.png"), bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="pca_results_labels_2")
    args = parser.parse_args()  
    concat_plots_from_folder(args.folder_path)
