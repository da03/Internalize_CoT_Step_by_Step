for i in $(seq 25 5 50); do
    echo "Running checkpoint $i"
    python plotting/analyze_activations.py \
        --activation_dir cached_activations/model_checkpoint_$i \
        --save_dir pca_results_labels/checkpoint_$i
    python plotting/utils.py --folder_path pca_results_labels/checkpoint_$i
done


for i in $(seq 1 5 26); do
    echo "Running checkpoint $i"
    python plotting/analyze_activations.py \
        --activation_dir cached_activations/model_checkpoint_$i \
        --save_dir pca_results_labels/checkpoint_$i
    python plotting/utils.py --folder_path pca_results_labels/checkpoint_$i
done

for i in $(seq 101 5 126); do 
    echo "Running checkpoint $i"
    python plotting/analyze_activations.py \
        --activation_dir cached_activations/model_checkpoint_$i \
        --save_dir pca_results_labels/checkpoint_$i
    python plotting/utils.py --folder_path pca_results_labels/checkpoint_$i
done