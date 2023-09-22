from dataclasses import dataclass
import numpy as np
import torch

from .extra_utils import paired_msa_numbering, unpaired_msa_numbering, create_alignment, res_to_list, res_to_seq


class AbAlignment:

    def __init__(self, device = 'cpu', ncpu = 1):
        
        self.device = device
        self.ncpu = ncpu
        
    def number_sequences(self, seqs, chain = 'H', fragmented = False):
        if chain == 'HL':
            numbered_seqs, seqs, number_alignment = paired_msa_numbering(seqs, fragmented = fragmented, n_jobs = self.ncpu)
        else:
            numbered_seqs, seqs, number_alignment = unpaired_msa_numbering(
                seqs, chain = chain, ncpus = self.ncpu, fragmented = fragmented
            )
        
        return numbered_seqs, seqs, number_alignment
    
    def align_encodings(self, encodings, numbered_seqs, seqs, number_alignment):
        
        aligned_encodings = np.concatenate(
            [[
                create_alignment(
                    res_embed, numbered_seq, seq, number_alignment
                ) for res_embed, numbered_seq, seq in zip(encodings, numbered_seqs, seqs)
            ]], axis=0
        )
        return aligned_encodings
        
        
    def reformat_subsets(
        self, 
        subset_list, 
        mode = 'seqcoding', 
        align = False,
        numbered_seqs = None, 
        seqs = None,
        number_alignment = None,
    ):
        
        if mode in ['seqcoding', 'restore']:
            return np.concatenate(subset_list)
        elif align:
            subset_list = [
                self.align_encodings(
                    subset, 
                    numbered_seqs[num*len(subset):(num+1)*len(subset)],
                    seqs[num*len(subset):(num+1)*len(subset)], 
                    number_alignment
                ) for num, subset in enumerate(subset_list)
            ]            
            return aligned_results(
                aligned_embeds=np.concatenate(subset_list),
                number_alignment=number_alignment.apply(lambda x: '{}{}'.format(*x[0]), axis=1).values
            ) 
    
        elif not align:
            return sum(subset_list, [])
        else:
            return np.concatenate(subset_list) # this needs to be changed
        

@dataclass
class aligned_results():
    """
    Dataclass used to store output.
    """

    aligned_embeds: None
    number_alignment: None