import numpy as np
import torch

from .load_model import load_model
from .pretrained_utils.restoration import AbRestore
from .pretrained_utils.encodings import AbEncoding


class pretrained(AbEncoding):
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
            
        self.restore_antibody = AbRestore(
            self.AbLang, 
            self.tokenizer, 
            self.spread, 
            self.used_device, 
            ncpu
        )
        
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
        
        subset_list = []
        for subset in [seqs[x:x+chunk_size] for x in range(0, len(seqs), chunk_size)]:
            subset_list.append(getattr(self, mode)(subset, align, chain = chain))
        
        return self.group_subsets(subset_list, mode, align)
            
    
    
    def group_subsets(self, subset_list, mode = 'seqcoding', align = False):
        
        if mode=='seqcoding':
            return np.concatenate(subset_list)
        elif not align:
            return sum(subset_list, [])
        else:
            return subset_list # this needs to be changed
    
    
    
    def restore(self, seqs, align=False):
        """
        Restore sequences
        """

        return self.restore_antibody.restore(seqs, align=align)
    
    def likelihood(self, tokens, align=False):
        """
        Possible Mutations
        """
        
        with torch.no_grad():
            predictions = self.AbLang(tokens)

        return predictions
        
        
def prepare_sequences(seqs, fragmented = False):
        
    if isinstance(seqs, list):
        if isinstance(seqs[0], dict):
            seqs = convert_dicts_to_seqs(seqs, fragmented = fragmented)
        else:
            seqs = [seqs] 
        
    if isinstance(seqs, dict): 
        seqs = convert_dicts_to_seqs(seqs, fragmented = fragmented)
        
    return seqs, determine_chain(seqs[0])
    
        
        
def convert_dicts_to_seqs(dict_list, fragmented = False):
    if isinstance(dict_list, dict): dict_list = [dict_list]

    if fragmented:
        return [f"{adict['H']}|{adict['L']}" for adict in dict_list]
    else:
        return [f"<{adict['H']}>|<{adict['L']}>" for adict in dict_list] 
     
        
def determine_chain(seq):
    
    chain = ''
    h, l = seq.split('|')
    if len(h)>2: chain+='H'
    if len(l)>2: chain+='L'
    
    return chain
        
        
#if mode == 'likelihood':
            
            #tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
            #aList = []
            #for sequence_part in [tokens[x:x+chunk_size] for x in range(0, len(tokens), chunk_size)]:
            #    aList.append(getattr(self, mode)(sequence_part, align, chain))
                
            #return torch.cat(aList)