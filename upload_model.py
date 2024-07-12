from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('src')
from model import ImplicitModel

model_path = '/n/holyscratch01/rush_lab/Users/yuntian/gsm_jul9/train_models/gsm8k_nosharp_deleteeos/checkpoint_7'
#model_path = '/n/holyscratch01/rush_lab/Users/yuntian/gsm_jul9/train_models/gsm8k_nosharp_deleteeos_nospace/checkpoint_8'
# Load your locally fine-tuned model and tokenizer
model = ImplicitModel.from_pretrained(model_path)
base_model = model.base_model
tokenizer = model.tokenizer

# Your Hugging Face repository name
repo_name = "yuntian-deng/implicit-cot-math-mistral7b"

# Push the model to the Hugging Face Hub
base_model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)
