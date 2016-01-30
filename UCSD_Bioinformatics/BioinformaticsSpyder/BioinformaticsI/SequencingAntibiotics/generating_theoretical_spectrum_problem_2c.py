# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:44:25 2015

@author: mortennp
"""

import fileinput
from itertools import product


def build_weights_table():
    peptide_len = 1
    wt = {}
    for mapping in daltons.split(","):
        peptide = mapping[0:peptide_len]
        weight = int(mapping[peptide_len:].rstrip().lstrip())
        wt[peptide] = weight
    return wt


def yield_sub_peptides(peptide):
    buffer = peptide + peptide
    for (i,j) in product(range(len(peptide)), range(len(peptide)-1)):
        yield buffer[i:i+j+1]
    yield ""
    yield peptide
    
        
        
def generate_weights(peptide, wt):
    weights = []    
    for sub_peptide in yield_sub_peptides(peptide):
        weight = sum(wt[p] for p in sub_peptide)
        weights.append(weight)
    return weights


def main():
    with fileinput.input() as fi:
        peptide = fi.readline().rstrip()
    wt = build_weights_table()
    weights = sorted(generate_weights(peptide, wt))
    print(' '.join(str(weight) for weight in weights))


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
    