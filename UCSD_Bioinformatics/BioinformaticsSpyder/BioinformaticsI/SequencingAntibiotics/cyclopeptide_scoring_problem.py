# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:07:04 2015

@author: mortennp
"""

import fileinput
from itertools import product
from collections import defaultdict


def build_weights_table():
    peptide_len = 1
    wt = {}
    for mapping in daltons.split(","):
        peptide = mapping[0:peptide_len]
        weight = int(mapping[peptide_len:].rstrip().lstrip())
        wt[peptide] = weight
    return wt


def yield_circular_sub_peptides(peptide):
    buffer = peptide + peptide
    for (i,j) in product(range(len(peptide)), range(len(peptide)-1)):
        yield buffer[i:i+j+1]
    yield ""
    yield peptide
    
    
def yield_linear_sub_peptides(peptide):
    buffer = peptide + peptide
    for (i,j) in product(range(len(peptide)), range(len(peptide)-1)):
        if i+j < len(peptide):
            yield buffer[i:i+j+1]
    yield ""
    yield peptide   
            
        
def generate_cyclospectrum(peptide, wt):
    weights = []    
    for sub_peptide in yield_circular_sub_peptides(peptide):
        weight = sum(wt[p] for p in sub_peptide)
        weights.append(weight)
    return sorted(weights)
    
    
def uniquify(spectrum):
    counters = defaultdict(int)
    for w in spectrum:
        yield (w, counters[w])
        counters[w] += 1
    
        
def calc_score(spectrum1, spectrum2, wt):
    return len(set(uniquify(spectrum1)).intersection(set(uniquify(spectrum2))))


def main():
    with fileinput.input() as fi:
        peptide = fi.readline().rstrip()
        empirical_spectrum_line = fi.readline().rstrip()
    wt = build_weights_table()
    empirical_spectrum = [int(s) for s in empirical_spectrum_line.split(" ")]
    theoretical_spectrum = generate_cyclospectrum(peptide, wt)
    score = calc_score(empirical_spectrum, theoretical_spectrum, wt)
    print(score)
    

daltons = "\
G 57,\
A 71,\
S 87,\
P 97,\
V 99,\
T 101,\
C 103,\
I 113,\
L 113,\
N 114,\
D 115,\
K 128,\
Q 128,\
E 129,\
M 131,\
H 137,\
F 147,\
R 156,\
Y 163,\
W 186"


if __name__ == "__main__":
    main()
    