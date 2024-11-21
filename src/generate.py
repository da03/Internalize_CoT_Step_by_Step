import math
import time
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
from model import ImplicitModel
from data import CoTDataset, CoTDataCollator, extract_answer

import traceback

from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)


@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, max_new_tokens,  layer_names=None, cache_dir=None):
    model.eval()
    total_instances = 0
    total_correct = 0
    total_time = 0

    # Set up activation caching if layer_names specified
    activation_cache = None
    hooks = None
    if layer_names:
        from activation_utils import ActivationCache, attach_hooks_to_layers
        activation_cache = ActivationCache(cache_dir)
        hooks = attach_hooks_to_layers(model, layer_names, activation_cache)
        print(f'hooks: {hooks}')
    try:

        for batch in tqdm.tqdm(dataloader):
            input_ids_all = batch['input_ids_all'].to(device)
            labels = batch['labels_all']
            sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
            input_ids = input_ids_all[:, :sep_positions.max()+1]
            batch_size = input_ids.shape[0]
            total_instances += batch_size

            # Set input length for activation caching
            if activation_cache is not None:
                activation_cache.set_input_length(input_ids.size(1))
            # Generate
            start_time = time.time()
            beam_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                stop_on_two_eos=True,
            )
            end_time = time.time()
            total_time += end_time - start_time


            # Evaluate
            for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
                sep_position = sep_positions[i].item()
                tgt = input_ids_all_i[sep_position+1:]
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text)
                pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
                pred_ans = extract_answer(pred_text)
                if ans == pred_ans:
                    total_correct += 1
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
            
            if layer_names and activation_cache is not None:
                activation_cache.save_to_disk(final=True)

            # break
        
    finally:
        if hooks:
            for hook in hooks:
                hook.remove()

    accuracy = total_correct / total_instances
    throughput = total_instances / total_time
    return accuracy, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--layer_names', type=str, nargs='+', help='Name of layer to cache activations from (e.g., transformer_layer_0, embedding)')
    parser.add_argument('--cache_dir', type=str, default='activation_cache', help='Directory to store cached activations')
    parser.set_defaults(bf16=False)
    args = parser.parse_args()

    print(args)

    if args.bf16:
        dtype = 'bfloat16'
    else:
        dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    ctx = torch.amp.autocast(device_type='cpu', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Load model
    print (f'Loading from {args.from_pretrained}')
    model = ImplicitModel.from_pretrained(args.from_pretrained).to(device).to(ptdtype)
    model = model.to(device).to(ptdtype)
    model.eval()
    tokenizer = model.tokenizer

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    
    accuracy, throughput = evaluate(
        test_dataloader, 
        tokenizer, 
        device, 
        ctx, 
        model, 
        args.max_new_tokens,
        layer_names=args.layer_names,
        cache_dir=args.cache_dir
    )

    print (f"Test Accuracy: {accuracy}. Throughput: {throughput}")

if __name__ == "__main__":
    main()