# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 21:28:45 2015

@author: mortennp
"""

import fileinput
from itertools import product
from collections import defaultdict
from operator import itemgetter


uniquified_spectrum_set = []


def build_weights_list():
    peptide_len = 1
    wt = {}
    for mapping in daltons.split(","):
        peptide = mapping[0:peptide_len]
        weight = int(mapping[peptide_len:].rstrip().lstrip())
        wt[peptide] = weight
    return list(set(wt.values()))
           
        
def generate_linear_spectrum(peptide):
    weights = []
    for (i,j) in product(range(len(peptide)), range(len(peptide)-1)):
        if i+j < len(peptide):
            weights.append(calc_mass(peptide[i:i+j+1]))
    weights.append(0)
    weights.append(calc_mass(peptide))
    return weights
       

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
    
    
def uniquify(spectrum):
    counters = defaultdict(int)
    for w in spectrum:
        yield (w, counters[w])
        counters[w] += 1
    
        
def calc_score(spectrum):
    return len(set(uniquify(spectrum)).intersection(uniquified_spectrum_set))
    
    
def trim(leaderboard, spectrum, N):    
    global counter
    counter = 0
    scores = [(i, calc_score(generate_linear_spectrum(peptide))) for (i,peptide) in enumerate(leaderboard)]
    ordered = sorted(scores, key=itemgetter(1), reverse=True)
    if len(leaderboard) > N:        
        cut_rank = ordered[N-1][1]
        trimmed = ordered[:N] + [item for item in ordered[N:] if item[1] == cut_rank];
        return [leaderboard[i] for (i, score) in trimmed]
    else:
        return leaderboard
    

def sequence(spectrum, N, weights):
    global uniquified_spectrum_set 
    uniquified_spectrum_set = set(uniquify(spectrum))
    leader_peptide = []
    leaderboard = [[]]
    parent_mass = calc_parent_mass(spectrum)
    while leaderboard:
        leaderboard = expand(leaderboard, weights)
        to_remove = [False] * len(leaderboard)
        for (i,peptide) in enumerate(leaderboard):
            mass = calc_mass(peptide)
            if mass == parent_mass:
                if calc_score(generate_linear_spectrum(peptide)) > calc_score(generate_linear_spectrum(leader_peptide)):
                    leader_peptide = peptide
            else:
                if mass > parent_mass:
                    to_remove[i] = True
        leaderboard = [peptide for (i,peptide) in enumerate(leaderboard) if not to_remove[i]]
        leaderboard = trim(leaderboard, spectrum, N)
        print(len(leaderboard))
    return [leader_peptide]


def main():
    with fileinput.input() as fi:
        N = int(fi.readline().rstrip())
        spectrum_line = fi.readline().rstrip()
    spectrum = [int(s) for s in spectrum_line.split(" ")]
    weights = build_weights_list()
    output = sequence(spectrum, N, weights)
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
    