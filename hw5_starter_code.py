from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random
import csv
import sys
import os

datadir = '/kaggle/input/storycloze-2018'
sys.path.append(datadir)

class LanguageModel:
	
	def __init__(self, model_name='gpt2', device='cpu', mode='greedy', p=0.8):
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.mode = mode
		self.p = p
	
	def start(self, s):
		# return a dictionary with keys 'input_ids' and 'attention_masks'. 
		# 'input_ids' should be a tensor of shape [1,n], where n is the number of tokens in the sequence s
		# tokens can be obtained from self.tokenizer(s)['input_ids']
		# 'attention_masks' should be a tensor of the same shape [1,n] filled with ones.
		raise NotImplementedError
	
	def step(self, state):
		# state should be a dictionary with keys 'input_ids' and 'attention_masks' as returned by self.start()
		# outputs from the LM are obtained by self.model(**state, labels=state['input_ids'])
		# Logit scores can be obtained from outputs.logits. These last column pertains to the newest prediction. 
		# The probability distribution coming from the last column should be used for the decoding algorithms.
		# step should return the updated state (new prediction added to state['input_ids'], another 1 added to state['attention_masks'])
		raise NotImplementedError
	
	def ids_to_string(self, ids):
		# converts the token ids to strings to print out and view
		return self.tokenizer.decode(ids)

# Store both datasets - short_context_data and long_context data. Remember to strip newlines.
# For each dataset, try each decoder and save the generated text to a file. Include this file in your submission and use it to answer the questions.
# For the top-p decoder, remember to try different values of p.
# Use gpt2 with cpu for development, but use gpt2-xl with gpu to answer the questions for the assignment.