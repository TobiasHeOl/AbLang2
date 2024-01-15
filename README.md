
---

<div align="center">    
 
# AbLang-2 
## Addressing the antibody germline bias and its effect on language models for improved antibody design 

</div> 

**Motivation:** The versatile pathogen-binding properties of antibodies have made them an extremely important class of biotherapeutics. However, therapeutic antibody development is a complex, expensive and time-consuming task, with the final antibody needing to not only have strong and specific binding, but also be minimally impacted by any developability issues. The success of transformer-based language models for language tasks and the availability of vast amounts of antibody sequences, has led to the development of many antibody-specific language models to help guide antibody discovery and design. Antibody diversity primarily arises from V(D)J recombination, mutations within the CDRs, or from a small number of mutations away from the germline outside the CDRs. Consequently, a significant portion of the variable domain of all natural antibody sequences remains germline. This affects the pre-training of antibody-specific language models, where this facet of the sequence data introduces a prevailing bias towards germline residues. This poses a challenge, as mutations away from the germline are often vital for generating specific and potent binding to a target, meaning that language models need be able to suggest key mutations away from germline.

**Results:** In this study, we explored the implications of the germline bias, examining its impact on both general-protein and antibody-specific language models. We developed and trained a series of new antibody-specific language models optimised for predicting non-germline residues. We then compared our final model, AbLang-2, with current models and show how it suggests a diverse set of valid mutations with high cumulative probability. AbLang-2 is trained on both unpaired and paired data, and is the first freely available paired VH-VL language model (https://github.com/oxpig/AbLang2.git).

**Availability and implementation:** AbLang2 is a python package available at https://github.com/oxpig/AbLang2.git.



-----------

# Install AbLang2

AbLang is freely available and can be installed with pip.

~~~.sh
    pip install ablang2
~~~

or directly from github.

~~~.sh
    pip install -U git+https://github.com/oxpig/AbLang2.git
~~~

**NB:** If you want to have your returned output aligned (i.e. use the argument "align=True"), you need to manually install **Pandas** and a version of **[ANARCI](https://github.com/oxpig/ANARCI)** in the same environment. ANARCI can also be installed using bioconda; however, this version is maintained by a third party.

~~~.sh
    conda install -c bioconda anarci
~~~


----------

# AbLang2 usecases

   
AbLang2 can be used in different ways and for a variety of usecases. The central building blocks are the tokenizer, AbRep, and AbLang.
    
- Tokenizer: Converts sequences and amino acids to tokens, and vice versa
- AbRep: Generates residue embeddings from tokens
- AbLang: Generates amino acid likelihoods from tokens
    
```{r, engine='python', count_lines}
import ablang2

# Download and initialise the model
ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device='cpu')

seq = [
'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS',
'DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK'
]

# Tokenize input sequences
seqs = [f"{seq[0]}|{seq[1]}"] # Input needs to be a list
tokenized_seq = ablang.tokenizer(seqs, pad=True, w_extra_tkns=False, device="cpu")
        
# Generate rescodings
with torch.no_grad():
    rescoding = ablang.AbRep(tokenized_seq).last_hidden_states

# Generate logits/likelihoods
with torch.no_grad():
    likelihoods = ablang.AbLang(tokenized_seq)
```
    
**We have build a wrapper for specific usecases which can be explored via a the following [Jupyter notebook](https://github.com/TobiasHeOl/AbLang2/blob/main/notebooks/pretrained_module.ipynb).**



### Citation   
```
@article{Olsen2024,
  title={},
  author={Tobias H. Olsen, Iain H. Moal and Charlotte M. Deane},
  journal={in-preparation},
  doi={},
  year={2024}
}