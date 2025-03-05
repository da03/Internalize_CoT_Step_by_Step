import math
import argparse
import os
import sys
import tqdm
import inspect
import logging
import random
import torch
import re

from torch.utils.data import DataLoader
from transformers import AdamW

from model import ImplicitModel
from configuration_model import ImplicitModelConfig
from data import CoTDataset, CoTDataCollator, extract_answer
from utils import get_sep_position, batch_ids, save_model


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)

def detect_nan_hook(grad):
    if torch.isnan(grad).any():
        import pdb; pdb.set_trace()
        print("NaN gradient detected!")
    return grad.nan_to_num(nan=0.0)

def compute_lambda_distribution(removal_smoothing_lambda, truncate_length=100):
    if removal_smoothing_lambda == float('inf'):
        lambda_distribution = torch.zeros(truncate_length)
        lambda_distribution[0] = 1
    else:
        positions = torch.arange(truncate_length)
        lambda_distribution = (1 - math.exp(-removal_smoothing_lambda)) * positions.mul(-removal_smoothing_lambda).exp()
        cum_prob = lambda_distribution.sum()
        assert cum_prob <= 1
        lambda_distribution[-1] = lambda_distribution[-1] + (1-cum_prob)
    return lambda_distribution

def is_correct(ans, pred_ans):
    ans = ans.replace('#', '').strip().replace(' ', '')
    pred_ans = pred_ans.replace('#', '').strip().replace(' ', '')
    try:
        # Remove leading zeros in numbers before evaluating
        def remove_leading_zeros(expr):
            # Replace any number with leading zeros (but not just '0')
            return re.sub(r'\b0+([1-9][0-9]*)\b', r'\1', expr)
        
        ans = remove_leading_zeros(ans)
        pred_ans = remove_leading_zeros(pred_ans)
        #import pdb; pdb.set_trace()
        # evaluate ans and pred_ans
        ans_result = eval(ans)
        pred_ans_result = eval(pred_ans)
        return ans_result == pred_ans_result, ans_result, pred_ans_result
    except:
        return ans == pred_ans, ans, pred_ans

@torch.no_grad()
def evaluate(dataloader, tokenizer, device, ctx, model, max_new_tokens, scheduled_to_remove, removal_side, removal_smoothing_lambda, lambda_distribution, keep_position=False, disable_random_removal_offset=False, thought_length=10, with_z=False, num_zs=10):
    model.eval()
    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0
    position_ids_all = None
    position_ids = None
    total_any_correct = 0
    total_avg_correct = 0
    for batch in tqdm.tqdm(dataloader):
        input_ids_all = batch['input_ids_all'].to(device)
        labels = batch['labels_all'].to(device)
        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        input_ids = input_ids_all[:, :sep_positions.max()+1]
        batch_size = input_ids.shape[0]
        first_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        second_sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=1)
        eos_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id, skip=2)

        if scheduled_to_remove > 0 or removal_smoothing_lambda != float('inf'):
            if keep_position:
                position_ids_all = torch.arange(0, input_ids_all.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
            input_ids_all_tmp = []
            labels_tmp = []
            random_removal_offset = torch.multinomial(lambda_distribution, batch_size, replacement=True).to(device)
            if disable_random_removal_offset:
                random_removal_offset.fill_(0)
            to_remove = scheduled_to_remove + random_removal_offset
            if removal_side == 'left':
                removal_from_positions = first_sep_positions + 1 # remove from, including
                removal_to_positions = first_sep_positions + 1 + to_remove # remove to, not including
            else: # removal_side == 'right'
                removal_to_positions = second_sep_positions
                removal_from_positions = second_sep_positions - to_remove

            for batch_id in range(input_ids_all.shape[0]):
                eos_position = eos_positions[batch_id]
                removal_from_position = removal_from_positions[batch_id]
                removal_to_position = removal_to_positions[batch_id]
                removal_from_position = max(removal_from_position, first_sep_positions[batch_id]+1)
                removal_to_position = min(removal_to_position, second_sep_positions[batch_id])
                if keep_position:
                    position_ids_all[batch_id, removal_from_position-1:] += removal_to_position-removal_from_position
                input_ids_all_tmp.append(torch.cat((input_ids_all[batch_id, :removal_from_position], input_ids_all[batch_id, removal_to_position:eos_position+1]), dim=-1))
                labels_tmp.append(torch.cat((labels[batch_id, :removal_from_position], labels[batch_id, removal_to_position:eos_position+1]), dim=-1))
            input_ids_all = batch_ids(input_ids_all_tmp, tokenizer.eos_token_id, device, input_ids_all.dtype)
            labels = batch_ids(labels_tmp, -100, device, input_ids.dtype)

        with ctx:
            if keep_position:
                position_ids_all = position_ids_all[:, :input_ids_all.shape[-1]]
            outputs = model.compute_loss(input_ids=input_ids_all, labels=labels, position_ids=position_ids_all, thought_length=thought_length)

        total_loss += outputs.total_loss.item()
        total_correct_tokens += outputs.total_correct.item()
        total_tokens += outputs.total_tokens
        total_instances += batch_size

        # Generate
        stop_on_two_eos = True
        if keep_position:
            position_ids = position_ids_all[:, :input_ids.shape[-1]]

        if not with_z:
            num_zs = 1
        correct_tensor = torch.zeros((batch_size, num_zs), dtype=torch.bool)
        beam_output_all = model.generate(
                input_ids=input_ids,
                position_ids=position_ids,
                max_new_tokens=max_new_tokens,
                stop_on_two_eos=stop_on_two_eos,
                thought_length=thought_length,
                with_z=with_z,
                num_zs=num_zs,
        )
        for z_idx in range(num_zs):
            beam_output = beam_output_all[:, z_idx]

            # Evaluate
            for i, (input_ids_all_i, beam_output_i) in enumerate(zip(input_ids_all, beam_output)):
                sep_position = sep_positions[i].item()
                tgt = input_ids_all_i[sep_position+1:]
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text)
                pred_text = tokenizer.decode(beam_output_i, skip_special_tokens=True)
                pred_ans = extract_answer(pred_text)
                correct, ans_result, pred_ans_result = is_correct(ans, pred_ans)
                if correct:
                    total_correct += 1
                    correct_tensor[i, z_idx] = True
                print (f'Input: {tokenizer.decode(input_ids_all_i[:sep_position], skip_special_tokens=True)}')
                print (f'Target {i}: {tgt_text} ({ans_result})')
                print (f'Pred   {i}: {pred_text} ({pred_ans_result})')
                print ('')
        total_any_correct += correct_tensor.any(dim=1).sum().item()
        total_avg_correct += correct_tensor.float().mean(dim=1).sum().item()
    accuracy = total_correct / total_instances / num_zs
    accuracy_avg_correct = total_avg_correct / total_instances
    accuracy_any_correct = total_any_correct / total_instances
    token_accuracy = total_correct_tokens / total_tokens
    loss = total_loss / total_tokens
    ppl = math.exp(loss)
    return accuracy, token_accuracy, ppl, accuracy_avg_correct, accuracy_any_correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--remove_per_epoch', type=float, default=8)
    parser.add_argument('--remove_all_when_remove_beyond', type=str, default='inf')
    parser.add_argument('--removal_smoothing_lambda', type=float, default=float('inf'))
    parser.add_argument('--removal_side', type=str, choices=['left', 'right'], default='left')
    parser.add_argument('--pretrain_epochs', type=int, default=0)
    parser.add_argument('--truncation', type=int, default=-1)
    parser.add_argument('--max_len_train', type=int, default=-1)
    parser.add_argument('--max_new_tokens', type=int, default=800)
    parser.add_argument('--max_size', type=int, default=-1)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--from_pretrained', type=str, default=None)
    parser.add_argument('--remove_start_from', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--thought_length', type=int, default=10, help='Length of thought vector')
    parser.add_argument('--bf16', action='store_true')
    parser.set_defaults(bf16=False)
    parser.add_argument('--reset_optimizer', action='store_true')
    parser.set_defaults(reset_optimizer=False)
    parser.add_argument('--keep_position', action='store_true')
    parser.set_defaults(keep_position=False)
    parser.add_argument('--reinitialize_weights', action='store_true')
    parser.set_defaults(reinitialize_weights=False)
    parser.add_argument('--with_z', action='store_true')
    parser.set_defaults(with_z=False)
    parser.add_argument('--num_zs', type=int, default=10)
    parser.add_argument('--entropy_loss_lambda', type=float, default=1.0)
    parser.add_argument('--grpo_epsilon', type=float, default=0.2)
    parser.add_argument('--grpo_beta', type=float, default=0.0) # 0.04 from https://huggingface.co/docs/trl/main/en/grpo_trainer, but we are not implementing it first to simplify implementation
    parser.add_argument('--grpo_mu', type=int, default=1) # we use 1 to simplify implementation, such that clipping is not needed
    parser.add_argument('--grpo_learn_sigma', action='store_true')
    parser.set_defaults(grpo_learn_sigma=False)
    parser.add_argument('--sample_during_training', action='store_true')
    parser.set_defaults(sample_during_training=False)
    args = parser.parse_args()

    if args.remove_all_when_remove_beyond == 'inf':
        args.remove_all_when_remove_beyond = float('inf')
    else:
        args.remove_all_when_remove_beyond = int(args.remove_all_when_remove_beyond)
    print (args)
    assert args.grpo_beta == 0, 'GRPO beta is not implemented yet'
    assert args.grpo_mu == 1, 'GRPO mu is not implemented yet'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    lambda_distribution = compute_lambda_distribution(args.removal_smoothing_lambda)
    print (lambda_distribution.tolist()[:10])

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype)
    print (ptdtype, dtype, device)

    # Create model
    if args.from_pretrained is None:
        config = ImplicitModelConfig(base_model=args.model)
        model = ImplicitModel(config).to(device).to(ptdtype)
        assert False, 'not implemented for learn sigma'
    else:
        print (f'Loading from {args.from_pretrained}')
        model = ImplicitModel.from_pretrained(args.from_pretrained, args.grpo_learn_sigma).to(device).to(ptdtype)
    if 'gpt2' in args.model:
        old_length = model.base_model.transformer.wpe.weight.shape[0]
        if args.truncation > old_length and args.from_pretrained is None:
            #import pdb; pdb.set_trace()
            print ('EXPANDING POSITIONs')
            new_wpe = torch.nn.Embedding(args.truncation, model.base_model.transformer.wpe.weight.shape[-1])
            new_wpe.weight.data[:old_length] = model.base_model.transformer.wpe.weight
            new_wpe.weight.data[old_length:] = model.base_model.transformer.wpe.weight[-1].view(1, -1).expand(args.truncation-old_length, -1)
            model.base_model.transformer.wpe = new_wpe

            for block in model.base_model.transformer.h:
                block.attn.register_buffer(
                    "bias",
                    torch.tril(torch.ones((args.truncation, args.truncation), dtype=torch.bool)).view(
                        1, 1, args.truncation, args.truncation
                ),
                persistent=False,
            )
    model = model.to(device).to(ptdtype)
    tokenizer = model.tokenizer

    if args.reinitialize_weights:
        print ('reinitializing weights')
        model.model.apply(model.model._init_weights)

    if args.keep_position:
        assert 'gpt2' in args.model # only implemented for gpt2 generate TODO: the code for this is not checked in yet

    # Load data
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, args.truncation, max_size=args.max_size)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, args.truncation)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
    if args.test_path:
        test_dataset = CoTDataset(tokenizer, args.test_path, args.truncation)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Create Optimizer
    trainable_params = list(model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    # Train
    step = 0
    scheduled_to_remove = 0
    if args.remove_start_from > 0:
        print (f'the number of removed CoT tokens starts from {args.remove_start_from}')
        scheduled_to_remove = args.remove_start_from

    position_ids = None

    steps_per_epoch = len(train_dataloader)
    steps_per_removed_token = int(round(steps_per_epoch / args.remove_per_epoch))
    remove_step_counter = 0
    best_val_accuracy = float('-inf')

    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(detect_nan_hook)

    all_cot_removed_in_prev_batch = False
    for epoch in range(args.epochs):
        if scheduled_to_remove < float('inf'):
            scheduled_to_remove = int(round(scheduled_to_remove))
        if scheduled_to_remove >= args.remove_all_when_remove_beyond:
            scheduled_to_remove = float('inf') # remove all
        print(f"Epoch {epoch}. Scheduled to remove: {scheduled_to_remove}")
        model.train() # we need to disable dropout
        model.eval()
        for batch in tqdm.tqdm(train_dataloader):
            prev_scheduled_to_remove = scheduled_to_remove
            if remove_step_counter == steps_per_removed_token or steps_per_removed_token == 0:
                scheduled_to_remove += 1
                remove_step_counter = 0
            if epoch >= args.pretrain_epochs:
                remove_step_counter += 1
            if scheduled_to_remove > prev_scheduled_to_remove:
                print(f" -epoch {epoch}. step {step}. removing: {scheduled_to_remove}")
                if args.reset_optimizer and (not all_cot_removed_in_prev_batch):
                    print ('RESETTING OPTIMIZER')
                    optimizer.zero_grad(set_to_none=True)
                    del optimizer
                    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)
            if scheduled_to_remove >= args.remove_all_when_remove_beyond:
                scheduled_to_remove = float('inf') # remove all
            input_ids = batch['input_ids_all'].to(device)
            labels = batch['labels_all'].to(device)
            batch_size = input_ids.shape[0]

            first_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id)
            second_sep_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=1)
            eos_positions = get_sep_position(input_ids, tokenizer.eos_token_id, skip=2)

            all_cot_removed_in_batch = False
            if scheduled_to_remove > 0 or args.removal_smoothing_lambda != float('inf'):
                input_ids_tmp = []
                labels_tmp = []
                random_removal_offset = torch.multinomial(lambda_distribution, batch_size, replacement=True).to(device)
                to_remove = scheduled_to_remove + random_removal_offset
                if epoch < args.pretrain_epochs:
                    to_remove.fill_(args.remove_start_from)
                if args.keep_position:
                    position_ids = torch.arange(0, input_ids.shape[-1], dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
                if args.removal_side == 'left':
                    removal_from_positions = first_sep_positions + 1 # remove from, including
                    removal_to_positions = first_sep_positions + 1 + to_remove # remove to, not including
                else: # removal_side == 'right'
                    removal_to_positions = second_sep_positions
                    removal_from_positions = second_sep_positions - to_remove

                all_cot_removed_in_batch = True
                for batch_id in range(input_ids.shape[0]):
                    eos_position = eos_positions[batch_id]
                    removal_from_position = removal_from_positions[batch_id]
                    removal_to_position = removal_to_positions[batch_id]
                    removal_from_position = max(removal_from_position, first_sep_positions[batch_id]+1)
                    if removal_to_position < second_sep_positions[batch_id]:
                        all_cot_removed_in_batch = False
                    removal_to_position = min(removal_to_position, second_sep_positions[batch_id])
                    if args.keep_position:
                        position_ids[batch_id, removal_from_position-1:] += removal_to_position-removal_from_position
                    input_ids_tmp.append(torch.cat((input_ids[batch_id, :removal_from_position], input_ids[batch_id, removal_to_position:eos_position+1]), dim=-1))
                    labels_tmp.append(torch.cat((labels[batch_id, :removal_from_position], labels[batch_id, removal_to_position:eos_position+1]), dim=-1))
                input_ids = batch_ids(input_ids_tmp, tokenizer.eos_token_id, device, input_ids.dtype)
                labels = batch_ids(labels_tmp, -100, device, input_ids.dtype)
                if not all_cot_removed_in_batch:
                    best_val_accuracy = float('-inf')
            #print (input_ids.shape)
            all_cot_removed_in_prev_batch = all_cot_removed_in_batch
            if args.max_len_train > 0 and input_ids.shape[-1] > args.max_len_train:
                print ('skipped')
                continue
           
            with ctx:
                if args.keep_position:
                    position_ids = position_ids[:, :input_ids.shape[-1]]
                # TODO: 1. train; 2. visualize average sampling acc; 3. visualize top-k acc
                outputs = model.compute_loss_grpo(input_ids=input_ids, labels=labels, position_ids=position_ids, thought_length=args.thought_length, with_z=args.with_z, num_zs=args.num_zs, epsilon=args.grpo_epsilon, beta=args.grpo_beta, mu=args.grpo_mu, max_new_tokens=args.max_new_tokens, sample_during_training=args.sample_during_training)
            loss = outputs.loss
            total_loss = loss
            #if args.with_z:
            #    entropy_loss = outputs.entropy_loss
            #    total_loss += entropy_loss * args.entropy_loss_lambda
            total_loss.div(args.accumulate).backward()
            if step % args.accumulate == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                if args.with_z and args.grpo_learn_sigma:
                    ppl_entropy = math.exp(outputs.entropy_loss.item())
                    any_correct_accuracy = outputs.any_correct_accuracy.item()
                    avg_correct_accuracy = outputs.avg_correct_accuracy.item()
                    sigma = outputs.sigma
                    print (f"Step: {step}. Total Loss: {total_loss.item()}. Any Correct Acc: {any_correct_accuracy}. Avg Correct Acc: {avg_correct_accuracy}. PPL Entropy: {ppl_entropy}. Sigma: {sigma}")
                
                elif args.with_z:
                    ppl_entropy = math.exp(outputs.entropy_loss.item())
                    any_correct_accuracy = outputs.any_correct_accuracy.item()
                    avg_correct_accuracy = outputs.avg_correct_accuracy.item()
                    print (f"Step: {step}. Total Loss: {total_loss.item()}. Any Correct Acc: {any_correct_accuracy}. Avg Correct Acc: {avg_correct_accuracy}. PPL Entropy: {ppl_entropy}")
                else:
                    ppl = loss.exp().item()
                    token_accuracy = outputs.token_accuracy.item()
                    print (f"Step: {step}. PPL: {ppl}. Token Accuracy: {token_accuracy}")
                sys.stdout.flush()
            step += 1
        print (f'Scheduled to remove: {scheduled_to_remove}')
        accuracy, token_accuracy, ppl, accuracy_avg_correct, accuracy_any_correct = evaluate(val_dataloader, tokenizer, device, ctx, model, args.max_new_tokens, scheduled_to_remove, args.removal_side, args.removal_smoothing_lambda, lambda_distribution, keep_position=args.keep_position, disable_random_removal_offset=True, thought_length=args.thought_length, with_z=args.with_z, num_zs=args.num_zs)
        print (f'Disable Offset Val. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}. Avg Correct Acc: {accuracy_avg_correct}. Any Correct Acc: {accuracy_any_correct}')
        if accuracy > best_val_accuracy and False:
            print ('***best so far or removed more CoT tokens***')
            best_val_accuracy = accuracy
            if args.test_path:
                accuracy, token_accuracy, ppl = evaluate(test_dataloader, tokenizer, device, ctx, model, args.max_new_tokens, scheduled_to_remove, args.removal_side, args.removal_smoothing_lambda, lambda_distribution, keep_position=args.keep_position, disable_random_removal_offset=True, thought_length=args.thought_length)
                print (f'Test. PPL: {ppl}; Accuracy: {accuracy}; Token Accuracy: {token_accuracy}.')
        model.save_pretrained(os.path.join(args.save_model, f'checkpoint_{epoch}'))

if __name__ == "__main__":
    main()
