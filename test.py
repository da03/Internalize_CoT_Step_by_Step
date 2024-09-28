from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

checkpoint = "mistralai/Mathstral-7b-v0.1"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)

prompt = [{"role": "user", "content": "What are the roots of unity?"}]
prompt = [{"role": "user", "content": "What is the value of $-a-b^3+ab$ if $a=-3$ and $b=2$?"}]
prompt = [{"role": "user", "content": "Find the smallest positive integer $b$ for which $x^2 + bx + 2008$ factors into a product of two polynomials, each having integer coefficients."}]
prompt = [{"role": "user", "content": "Six balls, numbered 2, 3, 4, 5, 6, 7, are placed in a hat.  Each ball is equally likely to be chosen.  If one ball is chosen, what is the probability that the number on the selected ball is a prime number?"}]
prompt = [{"role": "user", "content": "A box contains six cards. Three of the cards are black on both sides, one card is black on one side and red on the other, and two of the cards are red on both sides.  You pick a card uniformly at random from the box and look at a random side.  Given that the side you see is red, what is the probability that the other side is red?  Express your answer as a common fraction."}]
tokenized_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
import pdb; pdb.set_trace()
out = model.generate(**tokenized_prompt, max_new_tokens=512)
print (tokenizer.decode(out[0]))
#>>> '<s>[INST] What are the roots of unity?[/INST] The roots of unity are the complex numbers that satisfy the equation $z^n = 1$, where $n$ is a positive integer. These roots are evenly spaced around the unit circle in the complex plane, and they have a variety of interesting properties and applications in mathematics and physics.</s>'

