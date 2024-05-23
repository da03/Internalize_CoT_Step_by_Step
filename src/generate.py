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
from models.teacher import Teacher
from models.configuration_teacher import TeacherConfig
from lowdata import CoTDataset, CoTDataCollator, extract_answer

from utils import get_sep_position

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING) # disable WARNING, INFO and DEBUG logging everywhere

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

def compute_distribution(lamb, truncate_length=100):
    if lamb == float('inf'):
        probs = torch.zeros(truncate_length)
        probs[0] = 1
    else:
        positions = torch.arange(truncate_length)
        probs = (1 - math.exp(-lamb)) * positions.mul(-lamb).exp()
        cum_prob = probs.sum()
        assert cum_prob <= 1
        probs[-1] = probs[-1] + (1-cum_prob)
    return probs

    
@torch.no_grad()
def evaluate(dataloader, tokenizer, ctx, teacher, max_new_tokens, to_delete_min, delete_side, delete_eos, lamb, probs, keep_position=False, keep_allposition=False, start_from=0, disable_offset=False):
    teacher.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
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
        if to_delete_min > 0 or lamb != float('inf'):
            if keep_position or keep_allposition:
                position_ids_all = torch.arange(0, input_ids_all.shape[-1], dtype=torch.long, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
            input_ids_all_tmp = []
            labels_tmp = []
            offset = torch.multinomial(probs, batch_size, replacement=True)
            if disable_offset:
                offset = offset * 0
            to_delete = to_delete_min + offset.to(input_ids.device)
            if delete_side == 'left':
                start_positions = first_sep_positions + 1 # remove from, including
                end_positions = first_sep_positions + 1 + to_delete # remove to, not including
            else:
                assert delete_side == 'right'
                end_positions = second_sep_positions
                start_positions = second_sep_positions - to_delete
            for batch_id in range(input_ids_all.shape[0]):
                start_position = start_positions[batch_id]
                end_position = end_positions[batch_id]
                start_position = max(start_position, first_sep_positions[batch_id]+1)
                if delete_eos:
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
        #with ctx:
        #    if keep_position or keep_allposition:
        #        position_ids_all = position_ids_all[:, :input_ids_all.shape[-1]]
        #    outputs = teacher.compute_loss(input_ids=input_ids_all, labels=labels, position_ids=position_ids_all)
        #total_loss += outputs.total_loss.item()
        #total_correct_tokens += outputs.total_correct.item()
        #total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        if delete_eos:
            stop_on_two_eos = False
        else:
            stop_on_two_eos = True
        if keep_position or keep_allposition:
            position_ids = position_ids_all[:, :input_ids.shape[-1]]
        start_time = time.time()
        beam_output = teacher.generate(
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
    #token_accuracy = total_correct_tokens / total_tokens
    #loss = total_loss / total_tokens
    #ppl = math.exp(loss)
    return accuracy, throughput #token_accuracy, ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--base_model', type=str, default='gpt2')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--truncation', type=int, default=1024)
    parser.add_argument('--delete_per_epoch', type=float, default=0)
    parser.add_argument('--delete_beyond', type=int, default=2048)
    parser.add_argument('--delete_side', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--delete_type', type=str, choices=['epoch', 'step', 'threshold'], default='epoch')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--pretrain_epoch', type=int, default=0)
    parser.add_argument('--minus_start_from', type=int, default=None)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--lamb', type=float, default=float('inf'))
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--delete_eos', action='store_true')
    parser.set_defaults(delete_eos=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--keep_allposition', action='store_true')
    parser.set_defaults(keep_allposition=False)
    parser.add_argument('--disable_dropout', action='store_true')
    parser.set_defaults(disable_dropout=False)
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--nopretrain', action='store_true')
    parser.set_defaults(nopretrain=False)
    parser.add_argument('--oraccuracy', action='store_true')
    parser.set_defaults(oraccuracy=False)
    args = parser.parse_args()

    print (args)
    if args.seed > 0:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    #import pdb; pdb.set_trace()
    probs = compute_distribution(args.lamb)
    print (probs.tolist()[:10])

    dtype = 'float32'
    if args.base_model == 'meta-llama/Llama-2-7b-hf':
        dtype = 'bfloat16'
    if args.base_model == 'mistralai/Mistral-7B-v0.1':
        dtype = 'bfloat16'
    if args.base_model == 'microsoft/Phi-3-mini-4k-instruct':
        dtype = 'bfloat16'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create Teacher 
    if args.from_pretrained is None:
        config = TeacherConfig(base_model=args.base_model)
        teacher = Teacher(config).to(device).to(ptdtype)
    else:
        print (f'Loading from {args.from_pretrained}')
        teacher = Teacher.from_pretrained(args.from_pretrained).to(device).to(ptdtype)
    if args.nopretrain:
        print ('reinitializing weights')
        teacher.base_model.apply(teacher.base_model._init_weights)
    if args.keep_position:
        assert 'gpt2' in args.base_model # only implemented for gpt2 generate
    if 'gpt2' in args.base_model:
        old_length = teacher.base_model.transformer.wpe.weight.shape[0]
        if args.truncation > old_length:
            #import pdb; pdb.set_trace()
            print ('EXPANDING POSITIONs')
            new_wpe = torch.nn.Embedding(args.truncation, teacher.hidden_size)
            new_wpe.weight.data[:old_length] = teacher.base_model.transformer.wpe.weight
            new_wpe.weight.data[old_length:] = teacher.base_model.transformer.wpe.weight[-1].view(1, -1).expand(args.truncation-old_length, -1)
            teacher.base_model.transformer.wpe = new_wpe

            for block in teacher.base_model.transformer.h:
                block.attn.register_buffer(
                    "bias",
                    torch.tril(torch.ones((args.truncation, args.truncation), dtype=torch.bool)).view(
                        1, 1, args.truncation, args.truncation
                ),
                persistent=False,
            )
    teacher = teacher.to(device).to(ptdtype)
    if args.tokenizer is not None:
        #import pdb; pdb.set_trace()
        from transformers import AutoTokenizer
        teacher.config.tokenizer_name = args.tokenizer
        teacher.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load data
    tokenizer = teacher.tokenizer

    #if args.base_model == 'mistralai/Mistral-7B-v0.1':
    #    tokenizer.padding_side  = 'right'
    #    print ('PADDING SIDE CHANGED TO RIGHT')
    #    print (tokenizer)
    #import pdb; pdb.set_trace()
    #if tokenizer.eos_token == '#':
    #    import pdb; pdb.set_trace()
    #    print ('WARNING: changing tokenizer\'s eos token to bos token!')
    #    eos_token_id = tokenizer.encode(tokenizer.eos_token)[0]
    #    del tokenizer.added_tokens_decoder[eos_token_id]
    #    tokenizer.eos_token = tokenizer.bos_token
    #    tokenizer.eos_token_id = tokenizer.bos_token_id
    #    pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    #    del tokenizer.added_tokens_decoder[pad_token_id]
    #    tokenizer.pad_token = tokenizer.bos_token
    #    tokenizer.pad_token_id = tokenizer.bos_token_id
    collate_fn = CoTDataCollator(tokenizer)
    args.batch_size = 1
    #train_dataset = CoTDataset(tokenizer, args.train_path, args.truncation)
    #train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    #val_dataset = CoTDataset(tokenizer, args.val_path, args.truncation)
    #val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    ## Create Optimizer
    #trainable_params = list(teacher.parameters())
    #use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    #extra_args = dict(fused=True) if use_fused else dict()
    #optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    if not args.disable_dropout:
        teacher.train()
    else:
        teacher.eval()
    teacher.eval()

    #import numpy as np
    #for dataloader in [train_dataloader, val_dataloader]:
    #    deltas = []
    #    for batch in tqdm.tqdm(dataloader):
    #        input_ids = batch['input_ids_all'].to(device)
    #        labels = batch['labels_all'].to(device)
    #        first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
    #        #assert all(first_sep_positions == first_sep_positions[0])
    #        second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
    #        third_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)
    #        #deltas.extend((second_sep_positions-first_sep_positions).cpu().tolist())
    #        deltas.extend(third_sep_positions.cpu().tolist())
    #        #if (second_sep_positions-first_sep_positions).max() > 80:
    #        #    import pdb; pdb.set_trace()
    #    deltas = np.array(deltas)
    #    for percentile in [0, 5, 10, 50, 60, 70, 80, 85, 90, 95, 97, 98, 99 100]:
    #        print (f'percentile: {percentile}, {np.percentile(deltas, percentile)}')
    #        #assert all(second_sep_positions == second_sep_positions[0])
    #        #to_delete_min = epoch
    #sys.exit(1)
    # Train
    #step = 0
    #steps_per_epoch = len(train_dataloader)
    to_delete_min = 0
    if args.start_from > 0:
        print (f'starting from {args.start_from}')
        to_delete_min = args.start_from
    #position_ids = None
    #if args.minus_start_from is None:
    #    args.minus_start_from = 0 #args.start_from

    #steps_per_delete = int(round(steps_per_epoch / args.delete_per_epoch))
    #delete_step_timer = 0
    #best_accuracy = 0

    #prev_flag_all_deleted = False
    #for epoch in range(args.epochs):
    #    if args.delete_type == 'epoch':
    #        assert False
    #        steps_per_epoch = len(train_dataloader)
    #        to_delete_min = epoch * args.delete_per_epoch
    #    elif args.delete_type == 'step':
    #        steps_per_epoch = len(train_dataloader)
    #        #to_delete_min = step / steps_per_epoch * args.delete_per_epoch + args.start_from
    #    elif args.delete_type == 'threshold':
    #        assert False
    #    else:
    #        assert False
    #    if to_delete_min < float('inf'):
    #        to_delete_min = int(round(to_delete_min))
    #    if args.delete_beyond > 0 and to_delete_min >= args.delete_beyond:
    #        to_delete_min = float('inf') # delete all
    #    #if epoch < args.pretrain_epoch:
    #    #    to_delete_min = 0
    #    print(f"Epoch {epoch}. Deleting: {to_delete_min}")
    #    #teacher.train()
    #    if not args.disable_dropout:
    #        teacher.train()
    #    else:
    #        teacher.eval()
    #    #import pdb; pdb.set_trace()
    #    deleted = 0
    #    #prev_flag_all_deleted = False
    #    for batch in tqdm.tqdm(train_dataloader):
    #        if args.delete_type == 'step':
    #            prev_to_delete_min = to_delete_min
    #            #to_delete_min = step / steps_per_epoch * args.delete_per_epoch + args.start_from
    #            #to_delete_min = int(round(to_delete_min))
    #            if delete_step_timer == steps_per_delete or steps_per_delete == 0:
    #                deleted += 1
    #                if deleted == args.delete_per_epoch + 1:
    #                    break
    #                to_delete_min += 1
    #                delete_step_timer = 0
    #            if epoch >= args.pretrain_epoch:
    #                delete_step_timer += 1
    #            if to_delete_min > prev_to_delete_min:
    #                print(f" -epoch {epoch}. step {step}. deleting: {to_delete_min}")
    #                if args.reset_optimizer and (not prev_flag_all_deleted):
    #                    print ('RESETTING OPTIMIZER')
    #                    optimizer.zero_grad(set_to_none=True)
    #                    del optimizer
    #                    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    #                sys.stdout.flush()
    #            if args.delete_beyond > 0 and to_delete_min >= args.delete_beyond:
    #                to_delete_min = float('inf') # delete all
    #        input_ids = batch['input_ids_all'].to(device)
    #        #import pdb; pdb.set_trace()
    #        labels = batch['labels_all'].to(device)
    #        first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
    #        #assert all(first_sep_positions == first_sep_positions[0])
    #        second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
    #        #assert all(second_sep_positions == second_sep_positions[0])
    #        #to_delete_min = epoch
    #        #to_delete_min = 1000
    #        #max_to_delete_min = second_sep_positions[0] - first_sep_positions[0]
    #        #to_delete_min = (epoch / args.epochs) * max_to_delete_min * 2
    #        #to_delete_min = int(round(to_delete_min.item()))

    #        #import pdb; pdb.set_trace()
    #        batch_size = input_ids.shape[0]
    #        flag_all_deleted = False
    #        if to_delete_min > 0 or args.lamb != float('inf'):
    #            input_ids_tmp = []
    #            labels_tmp = []
    #            offset = torch.multinomial(probs, batch_size, replacement=True)
    #            to_delete = to_delete_min + offset.to(input_ids.device)
    #            if epoch < args.pretrain_epoch:
    #                to_delete.fill_(args.start_from)
    #            if args.keep_position or args.keep_allposition:
    #                position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
    #            if args.delete_side == 'left':
    #                start_positions = first_sep_positions + 1 # remove from, including
    #                end_positions = first_sep_positions + 1 + to_delete # remove to, not including
    #            else:
    #                assert args.delete_side == 'right'
    #                end_positions = second_sep_positions
    #                start_positions = second_sep_positions - to_delete
    #            flag_all_deleted = True
    #            for batch_id in range(input_ids.shape[0]):
    #                start_position = start_positions[batch_id]
    #                end_position = end_positions[batch_id]
    #                start_position = max(start_position, first_sep_positions[batch_id]+1)
    #                if args.delete_eos:
    #                    if end_position < second_sep_positions[batch_id] + 1:
    #                        flag_all_deleted = False
    #                    end_position = min(end_position, second_sep_positions[batch_id] + 1)
    #                else:
    #                    if end_position < second_sep_positions[batch_id]:
    #                        flag_all_deleted = False
    #                    end_position = min(end_position, second_sep_positions[batch_id])
    #                if args.keep_position:
    #                    assert not args.keep_allposition
    #                    position_ids[batch_id, start_position-1:] += end_position-start_position - args.minus_start_from
    #                elif args.keep_allposition:
    #                    position_ids[batch_id, :] += end_position-start_position - args.minus_start_from
    #                input_ids_tmp.append(torch.cat((input_ids[batch_id, :start_position], input_ids[batch_id, end_position:]), dim=-1))
    #                labels_tmp.append(torch.cat((labels[batch_id, :start_position], labels[batch_id, end_position:]), dim=-1))
    #            input_ids = batch_ids(input_ids_tmp, tokenizer.eos_token_id, input_ids.device, input_ids.dtype)
    #            labels = batch_ids(labels_tmp, tokenizer.eos_token_id, input_ids.device, input_ids.dtype)
    #        #import pdb; pdb.set_trace()
    #        print (input_ids.shape)
    #        prev_flag_all_deleted = flag_all_deleted
    #        if args.max_len_train > 0 and input_ids.shape[-1] > args.max_len_train:
    #            print ('skipped')
    #            if args.delete_type == 'step':
    #                steps_per_epoch -= 1
    #            #sys.stdout.flush()
    #            continue
    #        #sys.stdout.flush()
    #       
    #        with ctx:
    #            if args.keep_position or args.keep_allposition:
    #                position_ids = position_ids[:, :input_ids.shape[-1]]
    #            outputs = teacher.compute_loss(input_ids=input_ids, labels=labels, position_ids=position_ids)
    #        loss = outputs.loss
    #        token_accuracy = outputs.token_accuracy.item()

    #        loss.div(args.accumulate).backward()
    #        if step % args.accumulate == 0:
    #            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
    #            optimizer.step()
    #            optimizer.zero_grad(set_to_none=True)
    #        #torch.cuda.empty_cache() 

    #        ppl = loss.exp().item()
    #        if args.oraccuracy and epoch >= args.pretrain_epoch and token_accuracy == 1:
    #            deleted += 1
    #            if deleted == args.delete_per_epoch + 1:
    #                break
    #            prev_to_delete_min = to_delete_min
    #            to_delete_min += 1
    #            if to_delete_min > prev_to_delete_min:
    #                print(f" -epoch {epoch}. step {step}. deleting: {to_delete_min} ppl: {ppl} token accuracy: {token_accuracy}")
    #                if args.reset_optimizer and (not flag_all_deleted):
    #                    print ('RESETTING OPTIMIZER')
    #                    optimizer.zero_grad(set_to_none=True)
    #                    del optimizer
    #                    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    #                sys.stdout.flush()
    #            delete_step_timer = 0

    #        if args.delete_type == 'threshold':
    #            assert False
    #            flag_acc = False
    #            if args.oraccuracy:
    #                if token_accuracy == 1:
    #                    flag_acc = True
    #            if ppl < args.threshold or flag_acc:
    #                prev_to_delete_min = to_delete_min
    #                to_delete_min += 1
    #                if args.delete_beyond > 0 and to_delete_min >= args.delete_beyond:
    #                    to_delete_min = float('inf') # delete all
    #                if to_delete_min > prev_to_delete_min:
    #                    print(f" -epoch {epoch}. step {step}. deleting: {to_delete_min} ppl: {ppl} token accuracy: {token_accuracy}")
    #                    if args.reset_optimizer and (not flag_all_deleted):
    #                        print ('RESETTING OPTIMIZER')
    #                        optimizer.zero_grad(set_to_none=True)
    #                        del optimizer
    #                        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
    #                    sys.stdout.flush()
    #        if step % 100 == 0:
    #            print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
    #            sys.stdout.flush()
    #        step += 1
    #        #break
    #    print (f'deleted: {to_delete_min}')
    #    #accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, args.max_new_tokens, to_delete_min, args.delete_side, args.delete_eos, args.lamb, probs, keep_position=args.keep_position, keep_allposition=args.keep_allposition, start_from=args.minus_start_from)
    #    #print (f'Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
    #    #if accuracy > best_accuracy:
    #    #    print ('***best***')
    #    #    best_accuracy = accuracy
    #    if args.delete_beyond > 0 and to_delete_min < 10 and args.pretrain_epoch == 0:
    #        pass
    #    else:
    #        accuracy, token_accuracy, ppl = evaluate(val_dataloader, tokenizer, ctx, teacher, args.max_new_tokens, to_delete_min, args.delete_side, args.delete_eos, args.lamb, probs, keep_position=args.keep_position, keep_allposition=args.keep_allposition, start_from=args.minus_start_from, disable_offset=True)
    #        print (f'Disable Offset Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
    #        if accuracy > best_accuracy:
    #            print ('***best***')
    #            best_accuracy = accuracy
    if True:
        if True:
            accuracy, throughput = evaluate(test_dataloader, tokenizer, ctx, teacher, args.max_new_tokens, to_delete_min, args.delete_side, args.delete_eos, args.lamb, probs, keep_position=args.keep_position, keep_allposition=args.keep_allposition, start_from=args.minus_start_from)
            #print (f'Test. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
            print (f"Test Accuracy: {accuracy}. Throughput: {throughput}")
            sys.stdout.flush()
        #teacher.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
