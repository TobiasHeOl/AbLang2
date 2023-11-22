
---

<div align="center">    
 
# AbLang2: XX  

    

-----------

# Install AbLang2

AbLang is freely available and can be installed with pip.

~~~.sh
    pip install ablang2
~~~

or directly from github.

~~~.sh
    pip install -U git+https://github.com/oxpig/AbLang.git
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

ablang = ablang2.pretrained()

seq = [
'EVQLLESGGEVKKPGASVKVSCRASGYTFRNYGLTWVRQAPGQGLEWMGWISAYNGNTNYAQKFQGRVTLTTDTSTSTAYMELRSLRSDDTAVYFCARDVPGHGAAFMDVWGTGTTVTVSS',
'DIQLTQSPLSLPVTLGQPASISCRSSQSLEASDTNIYLSWFQQRPGQSPRRLIYKISNRDSGVPDRFSGSGSGTHFTLRISRVEADDVAVYYCMQGTHWPPAFGQGTKVDIK'
]

tokenized_seq = ablang.tokenizer([seq])

rescoding = ablang.AbRep(tokenized_seq)

likelihoods = ablang.AbLang(tokenized_seq)
```
    
**We have build a wrapper for specific usecases which can be explored via a the following [Jupyter notebook](https://github.com/TobiasHeOl/AbLang2/blob/main/notebooks/pretrained_module.ipynb).**



### Citation   
```
@article{Olsen2023,
  title={},
  author={Tobias H. Olsen, Iain H. Moal and Charlotte M. Deane},
  journal={in-preparation},
  doi={},
  year={2023}
}