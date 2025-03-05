import os
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
import sys

from configuration_model import ImplicitModelConfig
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor
import math
import re
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
        # evaluate ans and pred_ans
        ans_result = eval(ans)
        pred_ans_result = eval(pred_ans)
        #print (f'ans_result: {ans_result}, pred_ans_result: {pred_ans_result}')
        return ans_result == pred_ans_result
    except:
        return ans == pred_ans

class ImplicitModel(nn.Module):
    def __init__(self, config, reinitialize_weights=False):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model, trust_remote_code=True)
        if reinitialize_weights:
            print ('Reinitializing model weights!')
            self.base_model.apply(self.base_model._init_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.grpo_learn_sigma = config.grpo_learn_sigma if hasattr(config, 'grpo_learn_sigma') else False
        if self.grpo_learn_sigma:
            print ('Learning sigma!')
            self.log_sigma_layer = nn.Sequential(
                nn.Linear(self.base_model.config.hidden_size, 4*self.base_model.config.hidden_size),
                nn.ReLU(),
                nn.Linear(4*self.base_model.config.hidden_size, 1)
            )
            # Use proper initialization for weights (weights will use default PyTorch initialization)
            # Only initialize the final layer's bias to a small value to start with reasonably small variance
            self.log_sigma_layer[-1].bias.data.fill_(0.0)  # Initialize to exp(0.0) ≈ 1 for the standard deviation
        

    def forward(self, input_ids, position_ids=None, output_attentions=False, thought_length=10, with_z=False, num_zs=10):
        assert position_ids is None
        self.thought_length = thought_length
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        assert sep_positions.eq(sep_positions[0]).all()
        sep_position = sep_positions[0].item()
        src = input_ids[:, :sep_position+1]
        tgt = input_ids[:, sep_position+1:]

        # Get initial hidden states from source
        src_outputs = self.base_model.forward(input_ids=src, output_attentions=output_attentions, output_hidden_states=True, use_cache=True)
        hidden_states = src_outputs.hidden_states[-1] # [batch_size, seq_len, hidden_size]
        last_hidden_state = hidden_states[:, -1].unsqueeze(1) # [batch_size, 1, hidden_size]
        past_key_values = src_outputs.past_key_values

        # Thinking loop - feed hidden states back as inputs
        #import pdb; pdb.set_trace()
        batch_size = input_ids.shape[0]
        #with_z = True
        #print ('with_z', with_z)
        if with_z:
            # Create random noise vectors
            zs = torch.randn(batch_size, num_zs, self.base_model.config.hidden_size).to(last_hidden_state.device)
            
            # Expand last_hidden_state to incorporate multiple samples per batch example
            # Shape: [batch_size, 1, hidden_size] -> [batch_size * num_zs, 1, hidden_size]
            last_hidden_state = last_hidden_state.unsqueeze(1).repeat(1, num_zs, 1, 1)
            last_hidden_state = last_hidden_state.view(batch_size * num_zs, 1, self.base_model.config.hidden_size)
            
            # Add the noise to the expanded hidden states
            zs_reshaped = zs.view(batch_size * num_zs, 1, self.base_model.config.hidden_size)
            last_hidden_state = last_hidden_state + zs_reshaped
            
            # Expand past_key_values for each batch item
            #import pdb; pdb.set_trace()
            expanded_past_key_values = []
            for layer_past in past_key_values:
                # Each layer_past is a tuple of (key, value) tensors
                expanded_layer_past = []
                for tensor in layer_past:
                    # tensor shape: [batch_size, num_heads, seq_len, head_dim]
                    expanded_tensor = tensor.unsqueeze(1).repeat(1, num_zs, 1, 1, 1)
                    expanded_tensor = expanded_tensor.view(batch_size * num_zs, *tensor.shape[1:])
                    expanded_layer_past.append(expanded_tensor)
                expanded_past_key_values.append(tuple(expanded_layer_past))
            past_key_values = tuple(expanded_past_key_values)

        for i in range(thought_length):
            #import pdb; pdb.set_trace()
            outputs = self.base_model.forward(
                inputs_embeds=last_hidden_state,
                output_attentions=output_attentions,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            last_hidden_state = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
            #last_hidden_state = hidden_states[:, -1].unsqueeze(1) # [32, 1, 768]
        # Final forward pass using processed hidden states and target
        if with_z:
            # Repeat target tokens for each noise vector
            tgt = tgt.unsqueeze(1).repeat(1, num_zs, 1).view(batch_size * num_zs, -1)
        outputs = self.base_model.forward(
            input_ids=tgt,
            output_attentions=output_attentions,
            use_cache=True,
            past_key_values=past_key_values,
        )
        if with_z:
            outputs.zs = zs
        return outputs

    def compute_loss(self, input_ids, labels, position_ids=None, output_attentions=False, thought_length=10, with_z=False, num_zs=10):
        #with_z = True
        outputs = self.forward(input_ids=input_ids, position_ids=position_ids, output_attentions=output_attentions, thought_length=thought_length, with_z=with_z, num_zs=num_zs)
        
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        assert sep_positions.eq(sep_positions[0]).all()
        sep_position = sep_positions[0].item()
        labels = labels[:, sep_position+1:]
        #import pdb; pdb.set_trace()
        
        mask = labels[...,1:].ge(0)
        total_tokens = mask.sum()

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        batch_size = input_ids.shape[0]

        if with_z:
            #import pdb; pdb.set_trace()
            zs = outputs.zs # batch_size, num_zs, hidden_size
            shift_logits = shift_logits.log_softmax(-1)
            shift_logits = shift_logits.view(batch_size, num_zs, -1, shift_logits.shape[-1]) # batch_size, num_zs, seq_len, vocab_size
            probs = shift_logits.exp()
            # compute entropy to encourage the model to be deterministic under each z
            entropy_loss = - (probs * shift_logits).sum(dim=-1)
            entropy_loss = entropy_loss * mask.unsqueeze(1).repeat(1, num_zs, 1)
            entropy_loss = entropy_loss.sum() / total_tokens / num_zs
            # compute log E_z[p(y | x, z)]
            # compute any correct and average correct
            shift_labels_pred = shift_logits.argmax(-1)  # batch_size, num_zs, seq_len
            #import pdb; pdb.set_trace()
            correct_any = ((shift_labels_pred == shift_labels.unsqueeze(1)) | ~mask.unsqueeze(1)).all(-1).any(-1).sum()
            correct_avg = ((shift_labels_pred == shift_labels.unsqueeze(1)) | ~mask.unsqueeze(1)).all(-1).float().mean(-1).sum()
            #import pdb; pdb.set_trace()
            shift_logits = torch.logsumexp(shift_logits, dim=1) - math.log(num_zs) # batch_size, seq_len, vocab_size

        labels_pred = shift_logits.argmax(-1)
        correct_tokens = ((labels_pred == shift_labels) * mask).sum()
        token_accuracy = correct_tokens / total_tokens

        if with_z:
            outputs.entropy_loss = entropy_loss

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        if with_z:
            outputs.total_correct_any = correct_any
            outputs.total_correct_avg = correct_avg
            outputs.any_correct_accuracy = correct_any / batch_size
            outputs.avg_correct_accuracy = correct_avg / batch_size
        return outputs

    def compute_loss_grpo(self, input_ids, labels, position_ids=None, output_attentions=False, thought_length=10, with_z=False, num_zs=10, epsilon=0.2, beta=0.0, mu=1, max_new_tokens=512, sample_during_training=False):
        self.thought_length = thought_length
        assert with_z, 'GRPO requires with_z'
        if beta != 0.0 or mu != 1:
            raise NotImplementedError('GRPO beta and mu are not implemented yet')
        #with_z = True
        #outputs = self.forward(input_ids=input_ids, position_ids=position_ids, output_attentions=output_attentions, thought_length=thought_length, with_z=with_z, num_zs=num_zs)

        # First run generation
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Process source and get initial hidden states
        sep_position = sep_positions[0].item()
        src = input_ids[:, :sep_position+1]
        
        # Initial forward pass on source
        src_outputs = self.base_model.forward(input_ids=src, output_attentions=False, use_cache=True, output_hidden_states=True)
        hidden_states = src_outputs.hidden_states[-1]  # Get last layer's hidden states
        past_key_values = src_outputs.past_key_values

        # Thinking loop - same as in forward()
        last_hidden_state = hidden_states[:, -1].unsqueeze(1)  # [batch_size, 1, hidden_size]
        #self.base_model.train()
        #import pdb; pdb.set_trace()
        if with_z:
            #print ('with_z')
            zs = torch.randn(batch_size, num_zs, self.base_model.config.hidden_size).to(last_hidden_state.device)
            if self.grpo_learn_sigma:
                log_sigma_squared = self.log_sigma_layer(last_hidden_state.squeeze(1)).unsqueeze(-1) # batch_size, 1, 1
                sigma_squared = log_sigma_squared.exp()
                #print (f'sigma_squared: {sigma_squared.view(-1)}')
                zs = zs * sigma_squared.sqrt()
            last_hidden_state = last_hidden_state.unsqueeze(1).repeat(1, num_zs, 1, 1)
            last_hidden_state = last_hidden_state.view(batch_size * num_zs, 1, self.base_model.config.hidden_size)
            zs_reshaped = zs.view(batch_size * num_zs, 1, self.base_model.config.hidden_size)
            last_hidden_state_orig = last_hidden_state
            last_hidden_state = last_hidden_state + zs_reshaped
            if self.grpo_learn_sigma:
                total_logprobs = -0.5 * ((last_hidden_state.detach() - last_hidden_state_orig).norm(dim=-1) ** 2).reshape(batch_size, num_zs) / sigma_squared - 0.5 * self.base_model.config.hidden_size * log_sigma_squared
            else:
                total_logprobs = -0.5 * ((last_hidden_state.detach() - last_hidden_state_orig).norm(dim=-1) ** 2).reshape(batch_size, num_zs)
        # Expand past_key_values for each batch item
            #import pdb; pdb.set_trace()
            expanded_past_key_values = []
            for layer_past in past_key_values:
                # Each layer_past is a tuple of (key, value) tensors
                expanded_layer_past = []
                for tensor in layer_past:
                    # tensor shape: [batch_size, num_heads, seq_len, head_dim]
                    expanded_tensor = tensor.unsqueeze(1).repeat(1, num_zs, 1, 1, 1)
                    expanded_tensor = expanded_tensor.view(batch_size * num_zs, *tensor.shape[1:])
                    expanded_layer_past.append(expanded_tensor)
                expanded_past_key_values.append(tuple(expanded_layer_past))
                past_key_values = tuple(expanded_past_key_values)

        for i in range(self.thought_length):
            outputs = self.base_model.forward(
                inputs_embeds=last_hidden_state,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            last_hidden_state = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
        #import pdb; pdb.set_trace()
        # Initialize generation
        generated = []
        cur_token_id = src[:, -1:]  # Start with last token of source
        cur_token_id = cur_token_id.unsqueeze(1).repeat(1, num_zs, 1).view(batch_size * num_zs, -1)

        # Greedy decoding loop
        #cur_token_id = None
        #import pdb; pdb.set_trace()
        #total_logprobs = -0.5 * zs.norm(dim=-1) ** 2
        entropy_loss = 0
        for t in range(max_new_tokens):
            #print (f't={t}')
            outputs = self.base_model.forward(
                input_ids=cur_token_id,
                use_cache=True,
                past_key_values=past_key_values
            )
            next_token_logits = outputs.logits[:, -1, :]
            if t > 0 and cur_token_id.eq(self.tokenizer.eos_token_id).any():
                print ('Stopping at eos, setting logits to -1e4')
                next_token_logits[cur_token_id.eq(self.tokenizer.eos_token_id).view(-1)] = - 1e4 # -float('inf')
                next_token_logits[cur_token_id.eq(self.tokenizer.eos_token_id).view(-1)][..., self.tokenizer.eos_token_id] = 0
            #entropy_loss += next_token_logits.log_softmax(-1).entropy().mean()

            next_token_logprobs = next_token_logits.log_softmax(-1) # batch_size * num_zs, vocab_size
            next_token_probs = next_token_logprobs.exp()
            entropy_loss += - (next_token_probs * next_token_logprobs).sum(dim=-1).mean()

            
                
            #import pdb; pdb.set_trace()
            if not sample_during_training:
                next_token_id = next_token_logits.argmax(dim=-1, keepdim=True) # batch_size * num_zs, 1
            else:
                next_token_id = torch.multinomial(next_token_probs, num_samples=1) # batch_size * num_zs, 1
            next_token_logprobs_selected = next_token_logprobs.gather(dim=-1, index=next_token_id) # batch_size * num_zs, 1
            total_logprobs += next_token_logprobs_selected.view(batch_size, num_zs)
            #import pdb; pdb.set_trace()
            #next_token_id = next_token_id.view(batch_size, num_zs, -1)
            #if t > 0 and cur_token_id.eq(self.tokenizer.eos_token_id).any():
            #    next_token_id[cur_token_id.eq(self.tokenizer.eos_token_id)] = self.tokenizer.eos_token_id
            past_key_values = outputs.past_key_values
            
            generated.append(next_token_id)
            cur_token_id = next_token_id
            
            if cur_token_id.eq(self.tokenizer.eos_token_id).all():
                #assert t == 34
                break
            if t == 34:
                break
        #print (f'Example generated 1: {self.tokenizer.decode([generated[i][0].item() for i in range(len(generated))], skip_special_tokens=True)}')
        #print (f'Example generated 2: {self.tokenizer.decode([generated[i][1].item() for i in range(len(generated))], skip_special_tokens=True)}')
        # Concatenate all generated tokens
        generated_tokens = torch.cat(generated, dim=-1)
        generated_tokens = generated_tokens.view(batch_size, num_zs, -1)

        # Compute rewards
        rewards = torch.zeros(batch_size, num_zs).to(device)
        for i in range(batch_size):
            labels_i = labels[i]
            labels_i = labels_i[labels_i >= 0]
            ground_truth_str = self.tokenizer.decode(labels_i, skip_special_tokens=True)
            for j in range(num_zs):
                sequence = generated_tokens[i, j]
                sequence_str = self.tokenizer.decode(sequence, skip_special_tokens=True)
                correct = is_correct(ground_truth_str, sequence_str)
                rewards[i, j] = correct
        # compute correct_any and correct_avg
        correct_any = rewards.any(dim=1).sum()
        correct_avg = rewards.mean(dim=1).sum()
        #print (f'correct_any: {correct_any}, correct_avg: {correct_avg}')
        outputs.total_correct_any = correct_any
        outputs.total_correct_avg = correct_avg 
        outputs.any_correct_accuracy = correct_any / batch_size
        outputs.avg_correct_accuracy = correct_avg / batch_size
        if self.grpo_learn_sigma:
            outputs.sigma = sigma_squared.detach().sqrt().view(-1).cpu().tolist()
        outputs.entropy_loss = entropy_loss
        #print (f'outputs.total_correct_any: {outputs.total_correct_any}, outputs.total_correct_avg: {outputs.total_correct_avg}, outputs.any_correct_accuracy: {outputs.any_correct_accuracy}, outputs.avg_correct_accuracy: {outputs.avg_correct_accuracy}')

        #import pdb; pdb.set_trace()
        rewards_mean = rewards.mean(dim=1, keepdim=True)
        rewards_std = rewards.std(dim=1, keepdim=True).clamp(min=1e-6)
        rewards_normalized = (rewards - rewards_mean) / rewards_std
        obj = rewards_normalized * (total_logprobs - total_logprobs.detach()).exp()
        outputs.loss = -obj.mean()
        
        return outputs

    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True, position_ids=None, thought_length=10, with_z=False, num_zs=10):
        sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Process source and get initial hidden states
        sep_position = sep_positions[0].item()
        src = input_ids[:, :sep_position+1]
        
        # Initial forward pass on source
        src_outputs = self.base_model.forward(input_ids=src, output_attentions=False, use_cache=True, output_hidden_states=True)
        hidden_states = src_outputs.hidden_states[-1]  # Get last layer's hidden states
        past_key_values = src_outputs.past_key_values

        # Thinking loop - same as in forward()
        last_hidden_state = hidden_states[:, -1].unsqueeze(1)  # [batch_size, 1, hidden_size]
        #self.base_model.train()
        #import pdb; pdb.set_trace()
        if with_z:
            #print ('with_z')
            if self.grpo_learn_sigma:   
                sigma_squared = self.log_sigma_layer(last_hidden_state.squeeze(1)).exp().unsqueeze(-1) # batch_size, 1, 1
                print (f'sigma_squared: {sigma_squared.view(-1).cpu().tolist()}')
            zs = torch.randn(batch_size, num_zs, self.base_model.config.hidden_size).to(last_hidden_state.device)
            if self.grpo_learn_sigma:
                zs = zs * sigma_squared.sqrt()
            last_hidden_state = last_hidden_state.unsqueeze(1).repeat(1, num_zs, 1, 1)
            last_hidden_state = last_hidden_state.view(batch_size * num_zs, 1, self.base_model.config.hidden_size)
            zs_reshaped = zs.view(batch_size * num_zs, 1, self.base_model.config.hidden_size)
            last_hidden_state = last_hidden_state + zs_reshaped
        # Expand past_key_values for each batch item
            #import pdb; pdb.set_trace()
            expanded_past_key_values = []
            for layer_past in past_key_values:
                # Each layer_past is a tuple of (key, value) tensors
                expanded_layer_past = []
                for tensor in layer_past:
                    # tensor shape: [batch_size, num_heads, seq_len, head_dim]
                    expanded_tensor = tensor.unsqueeze(1).repeat(1, num_zs, 1, 1, 1)
                    expanded_tensor = expanded_tensor.view(batch_size * num_zs, *tensor.shape[1:])
                    expanded_layer_past.append(expanded_tensor)
                expanded_past_key_values.append(tuple(expanded_layer_past))
                past_key_values = tuple(expanded_past_key_values)

        for i in range(self.thought_length):
            outputs = self.base_model.forward(
                inputs_embeds=last_hidden_state,
                output_hidden_states=True,
                use_cache=True,
                past_key_values=past_key_values
            )
            last_hidden_state = outputs.hidden_states[-1]
            past_key_values = outputs.past_key_values
        
        # Initialize generation
        generated = []
        cur_token_id = src[:, -1:]  # Start with last token of source
        cur_token_id = cur_token_id.unsqueeze(1).repeat(1, num_zs, 1).view(batch_size * num_zs, -1)

        # Greedy decoding loop
        #cur_token_id = None
        #import pdb; pdb.set_trace()
        for t in range(max_new_tokens):
            outputs = self.base_model.forward(
                input_ids=cur_token_id,
                use_cache=True,
                past_key_values=past_key_values
            )
            next_token_logits = outputs.logits[:, -1, :]
            
                
            #import pdb; pdb.set_trace()
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True) # batch_size * num_zs, 1
            #import pdb; pdb.set_trace()
            #next_token_id = next_token_id.view(batch_size, num_zs, -1)
            if t > 0 and cur_token_id.eq(self.tokenizer.eos_token_id).any():
                next_token_id[cur_token_id.eq(self.tokenizer.eos_token_id)] = self.tokenizer.eos_token_id
            past_key_values = outputs.past_key_values
            
            generated.append(next_token_id)
            cur_token_id = next_token_id
            if cur_token_id.eq(self.tokenizer.eos_token_id).all():
                break

        # Concatenate all generated tokens
        generated_tokens = torch.cat(generated, dim=-1)
        generated_tokens = generated_tokens.view(batch_size, num_zs, -1)
        #import pdb; pdb.set_trace()
        return generated_tokens  # Return list to match original interface

    @classmethod
    def from_pretrained(self, pretrained_path, learn_sigma=False):
        config = ImplicitModelConfig.from_pretrained(pretrained_path)
        config.grpo_learn_sigma = learn_sigma
        print (f'config.grpo_learn_sigma: {config.grpo_learn_sigma}')
        model = ImplicitModel(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        load_result = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {load_result.missing_keys}")
        print(f"Unexpected keys: {load_result.unexpected_keys}")
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
