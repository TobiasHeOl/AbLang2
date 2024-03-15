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
        
    def _predict_logits_with_step_masking(self, seqs):
        
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
        
        logits = []
        for single_seq_tokens in tokens:
            
            tkn_len = len(single_seq_tokens)
            masked_tokens = single_seq_tokens.repeat(tkn_len, 1)
            for num in range(tkn_len):
                masked_tokens[num, num] = self.tokenizer.mask_token
            
            with torch.no_grad():
                logits_tmp = self.AbLang(masked_tokens)
                       
            logits_tmp = torch.stack([logits_tmp[num, num] for num in range(tkn_len)])

            logits.append(logits_tmp)
    
        return torch.stack(logits, dim=0)        

    def seqcoding(self, seqs, **kwargs):
        """
        Sequence specific representations
        """
        
        encodings = self._encode_sequences(seqs).cpu().numpy()
        
        lens = np.vectorize(len)(seqs)
        lens = np.tile(lens.reshape(-1,1,1), (encodings.shape[2], 1))

        return np.apply_along_axis(res_to_seq, 2, np.c_[np.swapaxes(encodings,1,2), lens])
        
    def rescoding(self, seqs, align=False, **kwargs):
        """
        Residue specific representations.
        """
        encodings = self._encode_sequences(seqs).cpu().numpy()
           
        if align: return encodings
            
        else: return [res_to_list(state, seq) for state, seq in zip(encodings, seqs)]
        
    def likelihood(self, seqs, align=False, stepwise_masking=False, **kwargs):
        """
        Likelihood of mutations
        """
        if stepwise_masking:
            logits = self._predict_logits_with_step_masking(seqs).cpu().numpy()
        else:
            logits = self._predict_logits(seqs).cpu().numpy()
        
        if align: return logits
            
        else: return [res_to_list(state, seq) for state, seq in zip(logits, seqs)]
        
    def probability(self, seqs, align=False, stepwise_masking=False, **kwargs):
        """
        Probability of mutations
        """
        if stepwise_masking:
            logits = self._predict_logits_with_step_masking(seqs)
        else:
            logits = self._predict_logits(seqs)
        probs = logits.softmax(-1).cpu().numpy()
        
        if align: return probs
            
        else: return [res_to_list(state, seq) for state, seq in zip(probs, seqs)]
        