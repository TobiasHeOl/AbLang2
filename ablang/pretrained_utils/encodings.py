from dataclasses import dataclass
import numpy as np
import torch

from .extra_utils import paired_msa_numbering, unpaired_msa_numbering, create_alignment, res_to_list, res_to_seq


class AbEncoding:

    def __init__(self, device = 'cpu', ncpu = 1):
        
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abencoding(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer
        
    def encode_sequences(self, seqs):
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
        with torch.no_grad():
            return self.AbLang.AbRep(tokens).last_hidden_states.numpy()
        
    def seqcoding(self, seqs, align=False, chain = 'H'):
        """
        Sequence specific representations
        """
        
        residue_states = self.encode_sequences(seqs)
        
        lens = np.vectorize(len)(seqs)
        lens = np.tile(lens.reshape(-1,1,1), (residue_states.shape[2], 1))
        seqcodings = np.apply_along_axis(res_to_seq, 2, np.c_[np.swapaxes(residue_states,1,2), lens])
        
        del lens
        del residue_states
        
        return seqcodings
        
    def rescoding(self, seqs, align=False, chain = 'H'):
        """
        Residue specific representations.
        """
           
        if not align:
            residue_states = self.encode_sequences(seqs)
            residue_output = [res_to_list(state, seq) for state, seq in zip(residue_states, seqs)]
            return residue_output
              
        else:
            if chain == 'HL':
                numbered_seqs, seqs, number_alignment = paired_msa_numbering(seqs, self.ncpu)
            else:
                numbered_seqs, seqs, number_alignment = unpaired_msa_numbering(seqs, chain = chain, ncpus = self.ncpu)
                
            residue_states = self.encode_sequences(seqs)

            residue_output = np.concatenate(
                [[
                    create_alignment(
                        res_embed, numbered_seq, seq, number_alignment
                    ) for res_embed, numbered_seq, seq in zip(residue_states, numbered_seqs, seqs)
                ]], axis=0
            )
            
            del residue_states
            
            return aligned_results(
                aligned_embeds=residue_output, 
                number_alignment=number_alignment.apply(lambda x: '{}{}'.format(*x[0]), axis=1).values
            )
        
        

@dataclass
class aligned_results():
    """
    Dataclass used to store output.
    """

    aligned_embeds: None
    number_alignment: None