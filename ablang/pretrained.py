import numpy as np
import torch
from dataclasses import dataclass


from .load_model import load_model
from .pretrained_utils.restoration import AbRestore
from .pretrained_utils.encodings import AbEncoding
from .pretrained_utils.alignment import AbAlignment


class pretrained(AbEncoding, AbRestore, AbAlignment):
    """
    Initializes AbLang for heavy or light chains.    
    """
    
    def __init__(self, model_to_use="download", random_init=False, ncpu=1, device='cpu'):
        super().__init__()
        
        self.used_device = torch.device(device)
        
        self.AbLang, self.tokenizer, self.hparams = load_model(model_to_use)
        self.AbLang.eval() # Default 
        
        self.ncpu = ncpu
        self.spread = 11 # Based on get_spread_sequences function
        
    def freeze(self):
        self.AbLang.eval()
        
    def unfreeze(self):
        self.AbLang.train()
        
    def __call__(self, seqs, mode='seqcoding', align=False, fragmented = False, chunk_size=50):
        """
        Mode: sequence, residue, restore or likelihood.
        """
        if not mode in ['rescoding', 'seqcoding', 'restore', 'likelihood']:
            raise SyntaxError("Given mode doesn't exist.")
        
        seqs, chain = prepare_sequences(seqs, fragmented = False) 
        if align:
            numbered_seqs, seqs, number_alignment = self.number_sequences(seqs, chain = chain)
        
        subset_list = []
        for subset in [seqs[x:x+chunk_size] for x in range(0, len(seqs), chunk_size)]:
            subset_list.append(getattr(self, mode)(subset, align, chain = chain))
            
        return self.reformat_subsets(
            subset_list, 
            mode = mode, 
            align = align,
            numbered_seqs = numbered_seqs, 
            seqs = seqs,
            number_alignment = number_alignment,
        )
        
        
def prepare_sequences(seqs, fragmented = False):
        
    if isinstance(seqs, list):
        if isinstance(seqs[0], dict):
            seqs = convert_many_dicts(seqs, fragmented = fragmented)
        else:
            seqs = [seqs] 
        
    if isinstance(seqs, dict): 
        seqs = convert_many_dicts(seqs, fragmented = fragmented)
        
    return seqs, determine_chain(seqs[0])
    
        
        
def convert_many_dicts(dict_list, fragmented = False):
    if isinstance(dict_list, dict): dict_list = [dict_list]

    return [convert_dicts(adict, fragmented = fragmented) for adict in dict_list]
     
def convert_dicts(a_dict, fragmented = False):
    heavy = a_dict.get('H') or ''
    light = a_dict.get('L') or ''
    return f"{heavy}|{light}" if fragmented else f"<{heavy}>|<{light}>".replace("<>","")
    
        
def determine_chain(seq):
    
    chain = ''
    h, l = seq.split('|')
    if len(h)>2: chain+='H'
    if len(l)>2: chain+='L'
    
    return chain 
