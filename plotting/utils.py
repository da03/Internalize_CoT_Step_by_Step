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
    image_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
        
    # Open all images
    images = [Image.open(f) for f in image_files]
    
    # Get dimensions
    widths, heights = zip(*(i.size for i in images))
    
    # Calculate layout
    n = len(images)
    nrows = int(n**0.5)  # Square root for roughly square layout
    ncols = (n + nrows - 1) // nrows  # Ceiling division
    
    # Create figure
    fig = plt.figure(figsize=(20, 20))
    
    # Add each image as a subplot
    for idx, img in enumerate(images):
        ax = fig.add_subplot(nrows, ncols, idx+1)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(os.path.basename(image_files[idx]))
    
    plt.tight_layout(pad=0.5) # Reduced padding between subplots
    plt.savefig(os.path.join(folder_path, "combined_plot.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="pca_results_labels_2")
    args = parser.parse_args()  
    concat_plots_from_folder(args.folder_path)
