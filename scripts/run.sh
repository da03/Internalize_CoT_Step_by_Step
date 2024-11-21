python src/generate.py \
    --from_pretrained final_checkpoint \
    --test_path data/4_by_4_mult/test_bigbench.txt \
    --batch_size 1 \
    --layer_names embedding transformer_layer_0 transformer_layer_1 transformer_layer_2 transformer_layer_3 transformer_layer_4 \
        transformer_layer_5 transformer_layer_6 transformer_layer_7 transformer_layer_8 transformer_layer_9 \
        transformer_layer_10 transformer_layer_11 \
    --cache_dir ./cached_activations