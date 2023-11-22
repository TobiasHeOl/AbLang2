import numpy as np
import torch

from .extra_utils import res_to_seq, get_sequences_from_anarci


class AbRestore:
    def __init__(self, spread = 11, device = 'cpu', ncpu = 1):
        self.spread = spread
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abrestore(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer

    def restore(self, seqs, align = False, chain = 'H'):
        """
        Restore sequences
        """

        if align:
            seqs = self._sequence_aligning(seqs)
            
            nr_seqs = len(seqs)//self.spread
            
            tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)          
            predictions = self.AbLang(tokens)[:,:,1:21]

            # Reshape
            tokens = tokens.reshape(nr_seqs, self.spread, -1)
            predictions = predictions.reshape(nr_seqs, self.spread, -1, 20)
            seqs = seqs.reshape(nr_seqs, -1)

            # Find index of best predictions
            best_seq_idx = torch.argmax(torch.max(predictions, -1).values[:,:,1:2].mean(2), -1)

            # Select best predictions           
            tokens = tokens.gather(1, best_seq_idx.view(-1, 1).unsqueeze(1).repeat(1, 1, tokens.shape[-1])).squeeze(1)
            predictions = predictions[range(predictions.shape[0]), best_seq_idx]
            seqs = np.take_along_axis(seqs, best_seq_idx.view(-1, 1).cpu().numpy(), axis=1)

        else:
            tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
            predictions = self.AbLang(tokens)[:,:,1:21]

        predicted_tokens = torch.max(predictions, -1).indices + 1
        restored_tokens = torch.where(tokens==23, predicted_tokens, tokens)

        restored_seqs = self.tokenizer(restored_tokens, mode="decode")

        return np.array([res_to_seq(seq, 'restore') for seq in np.c_[restored_seqs, np.vectorize(len)(seqs)]])
    
    def _sequence_aligning(self, seqs, chain):
        
        import pandas as pd
        import anarci

        anarci_out = anarci.run_anarci(
            pd.DataFrame([seq.replace('*', 'X') for seq in seqs]).reset_index().values.tolist(), 
            ncpu=self.ncpu, 
            scheme='imgt',
            allowed_species=['human', 'mouse'],
        )
        
        anarci_data = pd.DataFrame(
            [str(anarci[0][0]) if anarci else 'ANARCI_error' for anarci in anarci_out[1]], 
            columns=['anarci']
        ).astype('<U90')
        
        max_position = 128 if chain == 'H' else 127
        
        seqs = anarci_data.apply(
            lambda x: get_sequences_from_anarci(
                x.anarci, 
                max_position, 
                self.spread
            ), axis=1, result_type='expand'
        ).to_numpy().reshape(-1)
         
        return seqs