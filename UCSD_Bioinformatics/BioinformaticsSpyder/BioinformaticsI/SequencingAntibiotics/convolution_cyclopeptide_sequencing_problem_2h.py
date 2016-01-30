# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 08:51:23 2015

@author: mortennp
"""

import fileinput
from itertools import product
from collections import defaultdict
from operator import itemgetter


uniquified_spectrum_set = []


def convolute(spectrum):
    diffs = []
    for i in range(len(spectrum)):
        for j in range(i):
            diff = spectrum[i] - spectrum[j]
            if diff > 0:
                diffs.append(diff)
    return sorted(diffs)
    
    
def trim_weights(diffs, M):
    filtered = [diff for diff in diffs if 57 <= diff and diff <= 200]
    weights = list(set(filtered))
    scores_dict = defaultdict(int)
    for diff in filtered:
        scores_dict[diff] += 1
    scores = [(i, scores_dict[weight]) for (i,weight) in enumerate(weights)]
    ordered = sorted(scores, key=itemgetter(1), reverse=True)
    if len(weights) > M:
        cut_rank = ordered[M-1][1]
        trimmed = ordered[:M] + [item for item in ordered[M:] if item[1] == cut_rank];
        return [weights[i] for (i, score) in trimmed]
    else:
        return weights    


def build_weights_list(M, spectrum):
    diffs = convolute(spectrum)
    weights = trim_weights(diffs, M)
    return weights
           
        
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
        M = int(fi.readline().rstrip())
        N = int(fi.readline().rstrip())
        spectrum_line = fi.readline().rstrip()
    spectrum = [int(s) for s in spectrum_line.split(" ")]
    weights = build_weights_list(M, spectrum)
    output = sequence(spectrum, N, weights)
    formatted = ["-".join(map(str, peptide)) for peptide in output]
    print(" ".join(formatted))
    

if __name__ == "__main__":
    main()
    