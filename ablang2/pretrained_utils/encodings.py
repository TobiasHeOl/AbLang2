import numpy as np
import torch

from .extra_utils import res_to_list, res_to_seq


class AbEncoding:

    def __init__(self, device = 'cpu', ncpu = 1):
        
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abencoding(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer
        
    def _encode_sequences(self, seqs):
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
        with torch.no_grad():
            return self.AbLang.AbRep(tokens).last_hidden_states
        
    def _predict_logits(self, seqs):
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
        with torch.no_grad():
            return self.AbLang(tokens)

    def seqcoding(self, seqs, **kwargs):
        """
        Sequence specific representations
        """
        
        encodings = self._encode_sequences(seqs).cpu().numpy()
        
        lens = np.vectorize(len)(seqs)
        lens = np.tile(lens.reshape(-1,1,1), (encodings.shape[2], 1))
        return np.apply_along_axis(res_to_seq, 2, np.c_[np.swapaxes(encodings,1,2), lens])
        
    def rescoding(self, seqs, align=False):
        """
        Residue specific representations.
        """
        encodings = self._encode_sequences(seqs).cpu().numpy()
           
        if align: return encodings
            
        else: return [res_to_list(state, seq) for state, seq in zip(encodings, seqs)]
        
    def likelihood(self, seqs, align=False):
        """
        Likelihood of mutations
        """
        logits = self._predict_logits(seqs).cpu().numpy()
        
        if align: return logits
            
        else: return [res_to_list(state, seq) for state, seq in zip(logits, seqs)]
        
    def probability(self, seqs, align=False):
        """
        Probability of mutations
        """
        logits = self._predict_logits(seqs)
        probs = logits.softmax(1).cpu().numpy()
        
        if align: return probs
            
        else: return [res_to_list(state, seq) for state, seq in zip(probs, seqs)]
            