import numpy as np
import torch

from .load_model import load_model
from .pretrained_utils.restoration import AbRestore
from .pretrained_utils.encodings import AbEncoding
from .pretrained_utils.alignment import AbAlignment
from .pretrained_utils.scores import AbScores

valid_modes = [
    'rescoding', 'seqcoding', 'restore', 'likelihood', 'probability',
    'pseudo_log_likelihood', 'confidence'
]


class pretrained(AbEncoding, AbRestore, AbAlignment, AbScores):
    """
    Initializes AbLang for heavy or light chains.    
    """
    
    def __init__(self, model_to_use = "ablang2-paired", random_init = False, ncpu = 1, device = 'cpu'):
        super().__init__()
        
        self.used_device = torch.device(device)
        
        self.AbLang, self.tokenizer, self.hparams = load_model(model_to_use)
        self.AbLang.to(self.used_device)
        self.AbLang.eval() # Default 
        self.AbRep = self.AbLang.AbRep
        
        self.ncpu = ncpu
        self.spread = 11 # Based on get_spread_sequences function
        
    def freeze(self):
        self.AbLang.eval()
        
    def unfreeze(self):
        self.AbLang.train()
        
    def __call__(self, seqs, mode = 'seqcoding', align = False, fragmented = False, batch_size = 50):
        """
        Use different modes for different usecases
        """
        if not mode in valid_modes: raise SyntaxError(f"Given mode doesn't exist. Please select one of the following: {valid_modes}.")
        
        seqs, chain = format_seq_input(seqs, fragmented = fragmented) 

        if align:
            numbered_seqs, seqs, number_alignment = self.number_sequences(
                seqs, chain = chain, fragmented = fragmented
            )
        else:
            numbered_seqs = None
            number_alignment = None
        
        subset_list = []
        for subset in [seqs[x:x+batch_size] for x in range(0, len(seqs), batch_size)]:
            subset_list.append(getattr(self, mode)(subset, align = align))

        return self.reformat_subsets(
            subset_list, 
            mode = mode, 
            align = align,
            numbered_seqs = numbered_seqs, 
            seqs = seqs,
            number_alignment = number_alignment,
        )
        
        
def format_seq_input(seqs, fragmented = False):
    """
    Formats input sequences into the correct format for the tokenizer.
    """
    if isinstance(seqs[0], str):
        seqs = [seqs]
    
    seqs = [add_extra_tokens(seq) for seq in seqs]

    return seqs, determine_chain(seqs[0])
    
    
def add_extra_tokens(seq, fragmented = False):
    
    heavy, light = seq
    
    if fragmented:
        return f"{heavy}|{light}"
    else:
        return f"<{heavy}>|<{light}>".replace("<>","")
    
        
def determine_chain(seq): 
    h, l = seq.split('|')
      
    chain = ''
    if len(h)>2: chain+='H'
    if len(l)>2: chain+='L'
    
    return chain 
