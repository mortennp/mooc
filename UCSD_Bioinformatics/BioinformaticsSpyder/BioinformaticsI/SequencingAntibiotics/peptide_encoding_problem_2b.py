# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:45:04 2015

@author: mortennp
"""

import copy
import fileinput

codon_length = 3
chunk_size = 4

def build_reverse_complement_translation_table(chunk_size):
    map = {}
    map['A'] = 'T'
    map['T'] = 'A'
    map['G'] = 'C'
    map['C'] = 'G'

    tt = copy.deepcopy(map)

    for i in range(1, chunk_size):
        buffer = copy.deepcopy(tt)
        for strand, strandComplement in buffer.items():
            for nucleotide, nucleotideComplement in map.items():
                newStrand = strand + nucleotide
                newStrandComplement = strandComplement + nucleotideComplement
                tt[newStrand] = newStrandComplement

    return tt


def generate_reverse_complement(dna, tt):
    chunk_complements = []
    for i in range(0, len(dna), chunk_size):
        chunk = dna[i:i+chunk_size]
        chunk_complements.append(tt[chunk])
    chunk = dna[i+chunk_size:]
    if len(chunk) > 0:
        chunk_complements.append(tt[dna[i+chunk_size:]])
    complement = ''.join(chunk_complements)
    return complement[::-1]
    
    
def build_peptide_encoding_table():
    et = {}
    for mapping in codons.split(","):
        three_mer = mapping[0:codon_length]
        amino_acid = mapping[codon_length:].rstrip().lstrip()
        et[three_mer] = amino_acid
    return et


def yield_codons(rna):
    idx = 0
    while idx < len(rna):
        yield rna[idx:idx+codon_length]
        idx += codon_length


def encode(rna, tt):
    peptide = ''
    for codon in yield_codons(rna):
        if not codon:
            return peptide
        peptide += tt[codon]
    return peptide
    
    
def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]     
    
    
def main():
    with fileinput.input() as fi:
        dna = fi.readline().rstrip()
        peptide = fi.readline().rstrip()

    rctt = build_reverse_complement_translation_table(chunk_size)    
    rnatt = str.maketrans('T', 'U')
    pet = build_peptide_encoding_table()
    pattern_len = len(peptide) * codon_length
    matches = []
    for kmer in yield_kmers(dna, pattern_len):
        reverse_complement = generate_reverse_complement(kmer, rctt)
        encoded = encode(kmer.translate(rnatt), pet)
        reverse_encoded = encode(reverse_complement.translate(rnatt), pet)
        if encoded == peptide or reverse_encoded == peptide:
            matches.append(kmer)
    print('\n'.join(sorted(matches)))
        

codons = "\
AAA K,\
AAC N,\
AAG K,\
AAU N,\
ACA T,\
ACC T,\
ACG T,\
ACU T,\
AGA R,\
AGC S,\
AGG R,\
AGU S,\
AUA I,\
AUC I,\
AUG M,\
AUU I,\
CAA Q,\
CAC H,\
CAG Q,\
CAU H,\
CCA P,\
CCC P,\
CCG P,\
CCU P,\
CGA R,\
CGC R,\
CGG R,\
CGU R,\
CUA L,\
CUC L,\
CUG L,\
CUU L,\
GAA E,\
GAC D,\
GAG E,\
GAU D,\
GCA A,\
GCC A,\
GCG A,\
GCU A,\
GGA G,\
GGC G,\
GGG G,\
GGU G,\
GUA V,\
GUC V,\
GUG V,\
GUU V,\
UAA ,\
UAC Y,\
UAG ,\
UAU Y,\
UCA S,\
UCC S,\
UCG S,\
UCU S,\
UGA ,\
UGC C,\
UGG W,\
UGU C,\
UUA L,\
UUC F,\
UUG L,\
UUU F"


if __name__ == "__main__":
    main()
    