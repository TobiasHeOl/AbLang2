import numpy as np
import torch

from .extra_utils import res_to_list, res_to_seq


class AbScores:

    def __init__(self, device = 'cpu', ncpu = 1):
        
        self.device = device
        self.ncpu = ncpu
        
    def _initiate_abencoding(self, model, tokenizer):
        self.AbLang = model
        self.tokenizer = tokenizer
        
    def _encode_sequences(self, seqs):
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
        with torch.no_grad():
            return self.AbLang.AbRep(tokens).last_hidden_states.numpy()
        
    def _predict_logits(self, seqs):
        tokens = self.tokenizer(seqs, pad=True, w_extra_tkns=False, device=self.used_device)
        with torch.no_grad():
            return self.AbLang(tokens), tokens
        
    def pseudo_log_likelihood(self, seqs, **kwargs):
        """
        Pseudo log likelihood of sequences.
        """
        
        plls = []
        for seq in seqs:
            
            labels = self.tokenizer(
                seq, pad=True, w_extra_tkns=False, device=self.used_device
            )
            
            idxs = (
                ~torch.isin(labels, torch.Tensor(self.tokenizer.all_special_tokens).to(self.used_device))
            ).nonzero()
            
            masked_tokens = labels.repeat(len(idxs), 1)
            for num, idx in enumerate(idxs):
                masked_tokens[num, idx[1]] = self.tokenizer.mask_token
            
            with torch.no_grad():
                logits = self.AbLang(masked_tokens)
            
            logits[:, :, self.tokenizer.all_special_tokens] = -float("inf")            
            logits = torch.stack([logits[num, idx[1]] for num, idx in enumerate(idxs)])
  
            labels = labels[:,idxs[:,1:]].squeeze(2)[0]

            nll = torch.nn.functional.cross_entropy(
                    logits,
                    labels,
                    reduction="mean",
                )
            
            pll = -nll

            plls.append(pll)
    
        plls = torch.stack(plls, dim=0).cpu().numpy()

        return plls  
    
    def confidence(self, seqs, **kwargs):
        """
        Log likelihood of sequences without masking.
        """
        
        labels = self.tokenizer(
                seqs, pad=True, w_extra_tkns=False, device=self.used_device
            )
        with torch.no_grad():
            logits = self.AbLang(labels)
            logits[:, :, self.tokenizer.all_special_tokens] = -float("inf")  
            
        plls = []
        for label, logit in zip(labels, logits):
            
            idxs = (
                ~torch.isin(label, torch.Tensor(self.tokenizer.all_special_tokens).to(self.used_device))
            ).nonzero().squeeze(1)

            nll = torch.nn.functional.cross_entropy(
                        logit[idxs],
                        label[idxs],
                        reduction="mean",
                    )

            pll = -nll
            plls.append(pll)

        return torch.stack(plls, dim=0).cpu().numpy()