import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LanguageModel:
    """A wrapper around HuggingFace's GPT-2 implementation to give it
    a simple interface in the style of HW1.

    Parameters:
    - model_name (str): 'gpt2' for debugging, 'gpt2-xl' for better quality
    - device (str): 'cuda' to use a GPU or 'cpu' otherwise
    """
    
    def __init__(self, model_name='gpt2', device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    def start(self):
        """Return the initial state of the language model."""
        return (0, None)

    def bos(self):
        return self.tokenizer.bos_token_id
    def eos(self):
        return self.tokenizer.eos_token_id
    
    def step(self, state, num):
        """Given state and a numberized token, update the state with
        the token and return the log-probability distribution over the
        next token.
        
        Parameters:
        - state: The language model's state.
        - num: The next numberized token. The first token should be self.bos().

        Return: (state, out), where
        - state: The updated state.
        - out: The tensor of log-probabilities.

        """
        t, past_key_values = state
        out = self.model(past_key_values=past_key_values,
                         input_ids=torch.tensor([[num]]),
                         attention_mask=torch.ones(1,t+1))
        state = (t+1, out.past_key_values)
        logits = out.logits.squeeze((0,1))
        return (state, torch.log_softmax(logits, dim=-1))

    def tokenize(self, s):
        """Tokenize and numberize a string.
        Parameter:
        - s (str): The string to tokenize and numberize.

        Returns:
        - A list of n numberized tokens (ints).
        """
        
        return self.tokenizer(s)['input_ids']
    
    def detokenize(self, nums):
        """Denumberize and detokenize a string.
        Parameter:
        - nums (list of int): The numberized tokens.

        Returns:
        - The denumberized and detokenized string (str).
        """
        
        return self.tokenizer.decode(nums)
