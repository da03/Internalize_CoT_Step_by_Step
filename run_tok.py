from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
import pdb; pdb.set_trace()
eos_token_id = tokenizer(' =')['input_ids'][0]
