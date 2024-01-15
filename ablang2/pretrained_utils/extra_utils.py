import string, re
import numpy as np


def res_to_list(state, seq):
    return state[1:1+len(seq)]

def res_to_seq(a, mode='mean'):
    """
    Function for how we go from n_values for each amino acid to n_values for each sequence.
    
    We leave out the start, end and padding tokens.
    """
    if mode=='sum':
        return a[1:(1+int(a[-1]))].sum()
    
    elif mode=='mean':
        return a[1:(1+int(a[-1]))].mean()
    
    elif mode=='restore':
        
        return a[0][0:(int(a[-1]))]

def get_number_alignment(numbered_seqs):
    """
    Creates a number alignment from the anarci results.
    """
    import pandas as pd

    alist = [pd.DataFrame(aligned_seq, columns = [0,1,'resi']) for aligned_seq in numbered_seqs]
    unsorted_alignment = pd.concat(alist).drop_duplicates(subset=0)
    max_alignment = get_max_alignment()    
    
    return max_alignment.merge(unsorted_alignment.query("resi!='-'"), left_on=0, right_on=0)[[0,1]]

def get_max_alignment():
    """
    Create maximum possible alignment for sorting
    """
    import pandas as pd

    sortlist = [[("<", "")]]
    for num in range(1, 128+1):
        if num in [33,61,112]:
            for char in string.ascii_uppercase[::-1]:
                sortlist.append([(num, char)])

            sortlist.append([(num,' ')])
        else:
            sortlist.append([(num,' ')])
            for char in string.ascii_uppercase:
                sortlist.append([(num, char)])
                
    return pd.DataFrame(sortlist + [[(">", "")]])


def paired_msa_numbering(ab_seqs, fragmented = False, n_jobs = 10):
    
    import pandas as pd
    
    tmp_seqs = [pairs.replace(">", "").replace("<", "").split("|") for pairs in ab_seqs]

    numbered_seqs_heavy, seqs_heavy, number_alignment_heavy = unpaired_msa_numbering(
        [i[0] for i in tmp_seqs], 'H', fragmented = fragmented, n_jobs = n_jobs
    )
    numbered_seqs_light, seqs_light, number_alignment_light = unpaired_msa_numbering(
        [i[1] for i in tmp_seqs], 'L', fragmented = fragmented, n_jobs = n_jobs
    )
    
    number_alignment = pd.concat([
        number_alignment_heavy, 
        pd.DataFrame([[("|",""), "|"]]), 
        number_alignment_light]
    ).reset_index(drop=True)
    
    seqs = [f"{heavy}|{light}" for heavy, light in zip(seqs_heavy, seqs_light)]
    numbered_seqs = [
        heavy + [(("|",""), "|", "|")] + light for heavy, light in zip(numbered_seqs_heavy, numbered_seqs_light)
    ]
    
    return numbered_seqs, seqs, number_alignment


def unpaired_msa_numbering(seqs, chain = 'H', fragmented = False, n_jobs = 10):
    
    numbered_seqs = number_with_anarci(seqs, chain = chain, fragmented = fragmented, n_jobs = n_jobs)
    number_alignment = get_number_alignment(numbered_seqs)
    number_alignment[1] = chain   
    
    seqs = [''.join([i[2] for i in numbered_seq]).replace('-','') for numbered_seq in numbered_seqs]    
    return numbered_seqs, seqs, number_alignment


def number_with_anarci(seqs, chain = 'H', fragmented = False, n_jobs = 1):
    
    import anarci
    import pandas as pd

    anarci_out = anarci.run_anarci(
        pd.DataFrame(seqs).reset_index().values.tolist(), 
        ncpu=n_jobs, 
        scheme='imgt',
        allowed_species=['human', 'mouse'],
    )
    
    numbered_seqs = []
    for onarci in anarci_out[1]:            
        numbered_seq = []
        for i in onarci[0][0]:
            if i[1] != '-':
                numbered_seq.append((i[0], chain, i[1]))
            
        if fragmented:
            numbered_seqs.append(numbered_seq) 
        else:
            numbered_seqs.append([(("<",""), chain, "<")] + numbered_seq + [((">",""), chain, ">")])
    
    return numbered_seqs


def create_alignment(res_embeds, numbered_seqs, seq, number_alignment):
    
    import pandas as pd
    
    datadf = pd.DataFrame(numbered_seqs)
    sequence_alignment = number_alignment.merge(datadf, how='left', on=[0, 1]).fillna('-')[2]

    idxs = np.where(sequence_alignment.values == '-')[0]
    idxs = [idx-num for num, idx in enumerate(idxs)]
    
    aligned_embeds = pd.DataFrame(np.insert(res_embeds[:len(seq)], idxs , 0, axis=0))
    
    return pd.concat([aligned_embeds, sequence_alignment], axis=1).values


def get_spread_sequences(seq, spread, start_position):
    """
    Test sequences which are 8 positions shorter (position 10 + max CDR1 gap of 7) up to 2 positions longer (possible insertions).
    """
    spread_sequences = []

    for diff in range(start_position-8, start_position+2+1):
        spread_sequences.append('*'*diff+seq)
    
    return np.array(spread_sequences)

def get_sequences_from_anarci(out_anarci, max_position, spread):
    """
    Ensures correct masking on each side of sequence
    """
    
    if out_anarci == 'ANARCI_error':
        return np.array(['ANARCI-ERR']*spread)
    
    end_position = int(re.search(r'\d+', out_anarci[::-1]).group()[::-1])
    # Fixes ANARCI error of poor numbering of the CDR1 region
    start_position = int(re.search(r'\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+\'\),\s\(\(\d+,\s\'.\'\),\s\'[^-]+',
                                   out_anarci).group().split(',')[0]) - 1
    
    sequence = "".join(re.findall(r"(?i)[A-Z*]", "".join(re.findall(r'\),\s\'[A-Z*]', out_anarci))))

    sequence_j = ''.join(sequence).replace('-','').replace('X','*') + '*'*(max_position-int(end_position))

    return get_spread_sequences(sequence_j, spread, start_position)

