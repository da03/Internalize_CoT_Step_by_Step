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
from configuration_model import ImplicitModelConfig
from data import CoTDataset, CoTDataCollator, extract_answer

from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def batch_ids(input_ids_list, pad_token_id, device, dtype):
    max_seq_len = max([len(item) for item in input_ids_list])
    batch_size = len(input_ids_list)
    input_ids = torch.Tensor(batch_size, max_seq_len).to(dtype).to(device)
    input_ids.fill_(pad_token_id)
    for batch_id in range(batch_size):
        input_ids[batch_id, :len(input_ids_list[batch_id])] = input_ids_list[batch_id]
    return input_ids


def save_model(model, tokenizer, model_dir):
    print ('saving', model_dir)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, max_new_tokens, to_remove_min, removal_side, remove_eos, keep_position=False, keep_allposition=False, start_from=0):
    model.eval()
    total_instances = 0
    total_correct = 0
    total_time = 0
    #stop_on_two_eos = True
    position_ids_all = None
    position_ids = None
    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]
        first_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        #assert all(first_sep_positions == first_sep_positions[0])
        second_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=1)
        #assert all(second_sep_positions == second_sep_positions[0])
        #import pdb; pdb.set_trace()
        if to_remove_min > 0:
            if keep_position or keep_allposition:
                position_ids_all = torch.arange(0, input_ids_all.shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
            input_ids_all_tmp = []
            labels_tmp = []
            to_remove = to_remove_min
            if removal_side == 'left':
                start_positions = first_sep_positions + 1 # remove from, including
                end_positions = first_sep_positions + 1 + to_remove # remove to, not including
            else:
                assert removal_side == 'right'
                end_positions = second_sep_positions
                start_positions = second_sep_positions - to_remove
            for batch_id in range(input_ids_all.shape[0]):
                start_position = start_positions[batch_id]
                end_position = end_positions[batch_id]
                start_position = max(start_position, first_sep_positions[batch_id]+1)
                if remove_eos:
                    end_position = min(end_position, second_sep_positions[batch_id] + 1)
                else:
                    end_position = min(end_position, second_sep_positions[batch_id])
                if keep_position:
                    position_ids_all[batch_id, start_position-1:] += end_position-start_position - start_from
                elif keep_allposition:
                    position_ids_all[batch_id, :] += end_position-start_position - start_from
                input_ids_all_tmp.append(torch.cat((input_ids_all[batch_id, :start_position], input_ids_all[batch_id, end_position:]), dim=-1))
                labels_tmp.append(torch.cat((labels[batch_id, :start_position], labels[batch_id, end_position:]), dim=-1))
            input_ids_all = batch_ids(input_ids_all_tmp, tokenizer.eos_token_id, input_ids_all.device, input_ids_all.dtype)
            labels = batch_ids(labels_tmp, tokenizer.eos_token_id, input_ids.device, input_ids.dtype)
        total_instances += batch_size

        # Generate
        if remove_eos:
            stop_on_two_eos = False
        else:
            stop_on_two_eos = True
        if keep_position or keep_allposition:
            position_ids = position_ids_all[:, :input_ids.shape[-1]]
        start_time = time.time()
        beam_output = model.generate(
            input_ids=input_ids,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            stop_on_two_eos=stop_on_two_eos,
        )
        # Evaluate
        #import pdb; pdb.set_trace()
        for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
            sep_position = sep_positions[i].item()
            tgt = input_ids_all_i[sep_position+1:]
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            pred_text = tokenizer.decode(beam_output_i[0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)
            if ans == pred_ans:
                total_correct += 1
            if i == 0:
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target: {tgt_text}')
                print (f'Predicted: {pred_text}')
                print ('')
        end_time = time.time()
        total_time += end_time - start_time
    accuracy = total_correct / total_instances
    throughput = total_instances / total_time
    return accuracy, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--start_from', type=str, default='0')
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--removal_side', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--remove_eos', action='store_true')
    parser.set_defaults(remove_eos=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--keep_allposition', action='store_true')
    parser.set_defaults(keep_allposition=False)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    args = parser.parse_args()

    print (args)

    if args.bf16:
        dtype = 'bfloat16'
    else:
        dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Load model
    print (f'Loading from {args.from_pretrained}')
    model = ImplicitModel.from_pretrained(args.from_pretrained).to(device).to(ptdtype)
    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    ## Create Optimizer
    #trainable_params = list(model.parameters())
    #use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #extra_args = dict(fused=True) if use_fused else dict()
    #optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    model.eval()

    start_from = args.start_from
    if start_from == 'inf':
        start_from = float('inf')
    else:
        start_from = int(start_from)
    accuracy, throughput = evaluate(test_dataloader, tokenizer, device, ctx, model, args.max_new_tokens, start_from, args.removal_side, args.remove_eos, keep_position=args.keep_position, keep_allposition=args.keep_allposition)
    print (f"Test Accuracy: {accuracy}. Throughput: {throughput}")

if __name__ == "__main__":
    main()
