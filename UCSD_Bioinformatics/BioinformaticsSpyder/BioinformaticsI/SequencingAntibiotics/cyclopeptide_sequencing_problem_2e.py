# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 21:15:28 2015

@author: mortennp
"""

import fileinput
from itertools import product


def build_weights_list():
    peptide_len = 1
    wt = {}
    for mapping in daltons.split(","):
        peptide = mapping[0:peptide_len]
        weight = int(mapping[peptide_len:].rstrip().lstrip())
        wt[peptide] = weight
    return list(set(wt.values()))
           
        
def generate_cyclespectrum(peptide):
    weights = []
    buffer = peptide + peptide
    for (i,j) in product(range(len(peptide)), range(len(peptide)-1)):
        weights.append(calc_mass(buffer[i:i+j+1]))
    weights.append(0)
    weights.append(calc_mass(peptide))
    spectrum = sorted(weights)
    return spectrum
    

def generate_linear_spectrum(peptide):
    weights = []
    for (i,j) in product(range(len(peptide)), range(len(peptide)-1)):
        if i+j < len(peptide):
            weights.append(calc_mass(peptide[i:i+j+1]))
    weights.append(0)
    weights.append(calc_mass(peptide))
    spectrum = sorted(weights)
    return spectrum
       

def expand(peptides, weights):
    expanded = []
    for peptide in peptides:
        for weight in weights:
            expanded.append(peptide + [weight])
    return expanded
    
    
def calc_mass(peptide):
    return sum(peptide)
    

def calc_parent_mass(spectrum):
    return max(spectrum)
    
    
def is_consistent(peptide, spectrum):
    peptide_spectrum = generate_linear_spectrum(peptide)
    diff = [w for w in peptide_spectrum if w not in spectrum]
    return not diff

    
def cmp(peptide, spectrum):
    peptide_spectrum = generate_cyclespectrum(peptide)
    if len(peptide_spectrum) != len(spectrum):
        return False
    diffs = [(w1,w2) for (w1,w2) in zip(peptide_spectrum, spectrum) if w1 != w2]
    return not diffs
    

def sequence(spectrum, weights):
    output = []
    peptides = [[]]
    parent_mass = calc_parent_mass(spectrum)
    while peptides:
        peptides = expand(peptides, weights)
        to_remove = []
        for peptide in peptides:
            if calc_mass(peptide) == parent_mass:
                if cmp(peptide, spectrum):
                    output.append(peptide)
                to_remove.append(peptide)
            else:
                if not is_consistent(peptide, spectrum):
                    to_remove.append(peptide)
        peptides = [peptide for peptide in peptides if peptide not in to_remove]
    return output


def main():
    with fileinput.input() as fi:
        spectrum_line = fi.readline().rstrip()
    spectrum = [int(s) for s in spectrum_line.split(" ")]
    weights = build_weights_list()
    output = sequence(spectrum, weights)
    formatted = ["-".join(map(str, peptide)) for peptide in output]
    print(" ".join(formatted))
    

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
    