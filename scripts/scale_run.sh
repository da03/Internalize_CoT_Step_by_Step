for i in $(seq 101 5 126); do
    echo "Running checkpoint $i"
    python src/generate.py \
        --from_pretrained trained_models/launch2/checkpoint_$i \
        --test_path data/4_by_4_mult/test_bigbench.txt \
        --batch_size 1 \
        --layer_names embedding transformer_layer_0 transformer_layer_1 transformer_layer_2 transformer_layer_3 transformer_layer_4 \
            transformer_layer_5 transformer_layer_6 transformer_layer_7 transformer_layer_8 transformer_layer_9 \
            transformer_layer_10 transformer_layer_11 \
        --cache_dir ./cached_activations/model_checkpoint_$i
done

