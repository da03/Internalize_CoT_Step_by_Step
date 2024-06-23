import os
import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
import sys

from configuration_model import ImplicitModelConfig
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor


class ImplicitModel(nn.Module):
    def __init__(self, config, reinitialize_weights=False):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model, trust_remote_code=True)
        if reinitialize_weights:
            print ('Reinitializing model weights!')
            self.base_model.apply(self.base_model._init_weights)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def forward(self, input_ids, position_ids=None, output_attentions=False):
        if position_ids is not None:
            outputs = self.base_model.forward(input_ids=input_ids, output_attentions=output_attentions, position_ids=position_ids)
        else:
            outputs = self.base_model.forward(input_ids=input_ids, output_attentions=output_attentions)
        return outputs

    def compute_loss(self, input_ids, labels, position_ids=None, output_attentions=False):
        outputs = self.forward(input_ids=input_ids, position_ids=position_ids, output_attentions=output_attentions)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs

    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True, position_ids=None):
        eos_token_id = self.tokenizer(' =')['input_ids'][0]
        sep_positions = get_sep_position(input_ids, eos_token_id)
        batch_size = input_ids.shape[0]

        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if hasattr(generation_config, 'pad_token_id'):
            #generation_config.pad_token_id = -1 #TODO: this might not be necessary
            generation_config.pad_token_id = None #TODO: this might not be necessary
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None

        if sep_positions.eq(sep_positions[0]).all():
            input_ids = input_ids[:, :sep_positions[0]+1]
            if position_ids is not None:
                position_ids = position_ids[:, :sep_positions[0]+1]
            if position_ids is not None:
                beam_output = self.base_model.generate(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
            else:
                beam_output = self.base_model.generate(
                    input_ids=input_ids,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                    num_return_sequences=1,
                    logits_processor=logits_processor,
                    stopping_criteria=stopping_criteria,
                )
            beam_output = beam_output.unsqueeze(1)
        else:
            beam_output = []
            for i in range(batch_size):
                input_ids_i = input_ids[i:i+1]
                sep_positions_i = sep_positions[i:i+1]
                input_ids_i = input_ids_i[:, :sep_positions_i+1]
                if position_ids is not None:
                    position_ids_i = position_ids[i:i+1, :sep_positions_i+1]
                else:
                    position_ids_i = None
                if position_ids_i is not None:
                    beam_output_i = self.base_model.generate(
                        input_ids=input_ids_i,
                        position_ids=position_ids_i,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        early_stopping=True,
                        num_return_sequences=1,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                    )
                else:
                    beam_output_i = self.base_model.generate(
                        input_ids=input_ids_i,
                        generation_config=generation_config,
                        max_new_tokens=max_new_tokens,
                        num_beams=num_beams,
                        early_stopping=True,
                        num_return_sequences=1,
                        logits_processor=logits_processor,
                        stopping_criteria=stopping_criteria,
                    )
                beam_output.append(beam_output_i)
        return beam_output

    @classmethod
    def from_pretrained(self, pretrained_path, truncation):
        config = ImplicitModelConfig.from_pretrained(pretrained_path)
        model = ImplicitModel(config)
        old_length = model.base_model.transformer.wpe.weight.shape[0]
        if truncation > old_length:
            print ('EXPANDING POSITIONs')
            new_wpe = torch.nn.Embedding(truncation, model.base_model.transformer.wpe.weight.shape[-1])
            new_wpe.weight.data[:old_length] = model.base_model.transformer.wpe.weight
            new_wpe.weight.data[old_length:] = model.base_model.transformer.wpe.weight[-1].view(1, -1).expand(truncation-old_length, -1)
            model.base_model.transformer.wpe = new_wpe

            for block in model.base_model.transformer.h:
                block.attn.register_buffer(
                    "bias",
                    torch.tril(torch.ones((truncation, truncation), dtype=torch.bool)).view(
                        1, 1, truncation, truncation
                ),
                persistent=False,
            )
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'), map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=True)
        return model

    def save_pretrained(self, save_directory):
        print (f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))
