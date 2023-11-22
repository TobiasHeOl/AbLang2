from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .encoderblock import TransformerEncoder, get_activation_fn


class AbLang(torch.nn.Module):
    """
    AbLang inspired by ESM-2's architecture.
    """
    
    def __init__(
        self,
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
                
        self.AbRep = AbRep(
            vocab_size,
            hidden_embed_size,
            n_attn_heads,
            n_encoder_blocks,
            padding_tkn,
            mask_tkn,
            layer_norm_eps,
            a_fn,
            dropout,
        )       
        self.AbHead = AbHead(
            vocab_size,
            hidden_embed_size,
            self.AbRep.aa_embed_layer.weight,
            layer_norm_eps,
            a_fn,
        )
        
    def forward(self, tokens, return_attn_weights=False, return_rep_layers=[]):
        
        representations = self.AbRep(tokens, return_attn_weights, return_rep_layers)
        
        if return_attn_weights:
            return representations.attention_weights
        
        elif return_rep_layers != []:
            return representations.many_hidden_states
        else:
            likelihoods = self.AbHead(representations.last_hidden_states)
            return likelihoods
    
    def get_aa_embeddings(self):
        "Extracts the trained aa_embeddings."
        return self.AbRep.aa_embed_layer

    
class AbRep(torch.nn.Module):
    """
    AbRep (antibody representations), takes the tokenized sequence and create hidden_embed (representations).
    """
    
    def __init__(
        self, 
        vocab_size,
        hidden_embed_size,
        n_attn_heads,
        n_encoder_blocks,
        padding_tkn,
        mask_tkn,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
        dropout: float = 0.1, 
    ):
        super().__init__()
        self.padding_tkn = padding_tkn
        self.mask_tkn = mask_tkn
        
        self.aa_embed_layer = nn.Embedding(
            vocab_size, 
            hidden_embed_size, 
            padding_idx=padding_tkn,
        )   
        self.encoder_blocks = nn.ModuleList(
            [TransformerEncoder(
                hidden_embed_size,
                n_attn_heads,
                attn_dropout = dropout,
                layer_norm_eps = layer_norm_eps,
                a_fn = a_fn,
            ) for _ in range(n_encoder_blocks)]
        )
        self.layer_norm_after_encoder_blocks = nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps)

    def forward(self, 
                tokens, 
                return_attn_weights=False, 
                return_rep_layers=[],
               ):
        
        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_tkn)

        hidden_embed = self.aa_embed_layer(tokens)       
        
        return_rep_layers = set(return_rep_layers)
        rep_layers = {}
        if 0 in return_rep_layers: rep_layers[0] = hidden_embed
            
        all_attn_weights = []
        
        for n_layer, encoder_block in enumerate(self.encoder_blocks):
            hidden_embed, attn_weights = encoder_block(hidden_embed, padding_mask, return_attn_weights)
            
            if (n_layer + 1) in return_rep_layers: 
                rep_layers[n_layer + 1] = hidden_embed
            
            if return_attn_weights: 
                all_attn_weights.append(attn_weights)
           
        hidden_embed = self.layer_norm_after_encoder_blocks(hidden_embed)

        return DataAbRep(
            last_hidden_states=hidden_embed, 
            many_hidden_states=rep_layers, 
            attention_weights=all_attn_weights
        )
    

class AbHead(torch.nn.Module):
    """
    AbHead (antibody head model), creates amino acid probabilities for each position based on the hidden_embed (representations).
    """
    
    def __init__(
        self, 
        vocab_size,
        hidden_embed_size,
        weights,
        layer_norm_eps: float = 1e-12,
        a_fn: str = "gelu",
    ):
        super().__init__()
        
        activation_fn, scale = get_activation_fn(a_fn)
        
        self.ff = torch.nn.Sequential(
            nn.Linear(hidden_embed_size, hidden_embed_size * scale),
            activation_fn(),
            nn.LayerNorm(hidden_embed_size, eps=layer_norm_eps),
        )

        self.weights = weights
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_embed):
        
        hidden_embed = self.ff(hidden_embed)
        logits = F.linear(hidden_embed, self.weights) + self.bias
        
        return logits

    
@dataclass
class DataAbRep():
    """
    Dataclass used to store AbRep output.
    """

    last_hidden_states: torch.FloatTensor
    many_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attention_weights: Optional[Tuple[torch.FloatTensor]] = None