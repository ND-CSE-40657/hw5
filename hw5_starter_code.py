import sys
import torch

# If you're running on your own computer, you may need to run
#     pip install transformers
datadir = 'storycloze-2018'

# On Kaggle:
#!pip install -q transformers
#!git clone https://github.com/ND-CSE-40657/hw5
#sys.path.append('hw5')
#datadir = 'hw5/storycloze-2018'

import gpt

# To load a language model, do the following.
# Use 'gpt2' for debugging, but change to 'gpt2-xl' when actually generating outputs.
lm = gpt.LanguageModel('gpt2', 'cuda' if torch.cuda.is_available() else 'cpu')

# The LanguageModel object has methods tokenize() and detokenize() for
# converting strings to/from lists of numbers:
s = 'My hovercraft is full of eels.'
nums = lm.tokenize(s)
assert s == lm.detokenize(nums)

# Otherwise, the interface is similar to HW1. Here, we show how to
# compute the log-probability of nums:
q = lm.start()
prev = lm.bos()
total = 0.
for num in nums + [lm.eos()]:
    q, p = lm.step(q, prev)
    total += p[num].item()
    prev = num
print(total)

