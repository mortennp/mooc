# -*- coding: utf-8 -*-
"""
Created on Wed May  6 20:33:49 2015

@author: Morten
"""

import fileinput
import operator
import random
import copy

def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]   
        
        
def get_nucleotide_index():
    idx = {}
    idx["A"] = 0
    idx["C"] = 1
    idx["G"] = 2
    idx["T"] = 3
    return idx


def calculate_counts(k, kmers, idx):
    # set init value to 1 for pseudo counts 
    matrix = [[1 for j in range(k)] for i in range(len(idx))]
    for kmer in kmers:
        for pos in range(len(kmer)):
            nucleotide = kmer[pos]
            matrix[idx[nucleotide]][pos] = matrix[idx[nucleotide]][pos] + 1
    return matrix

       
def calculate_profile(k, t, kmers, idx):    
    matrix = calculate_counts(k, kmers, idx)
    for i in range(len(idx)):
        for j in range(k):
            matrix[i][j] = matrix[i][j] / t
    return matrix


def calc_probability(kmer, matrix, idx):
    prob = 1
    for i in range(len(kmer)):
        nucleotide = kmer[i]        
        prob *= matrix[idx[nucleotide]][i]
    return prob


def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"
   

def choose_randomly_by_profile(text, k, matrix, idx):
    kmer_probabilities = [(kmer, calc_probability(kmer, matrix, idx)) for kmer in yield_kmers(text, k)]
    kmer = weighted_choice(kmer_probabilities)    
    return kmer
    
   
def calc_score(k, t, kmers, idx):
    matrix = calculate_counts(k, kmers, idx)
    transpose = list(zip(*matrix))
    score = 0
    for j in range(len(transpose)):
        column = transpose[j]
        i, val = max(enumerate(column), key=operator.itemgetter(1))
        score += (t - val)
    return score


def gibbs_sampler(k, t, N, dnas):
    idx = get_nucleotide_index()
    start_positions = [random.randint(0, len(dnas[i]) - k + 1) for i in range(t)]
    motifs = [dnas[i][start_positions[i]:start_positions[i] + k] for i in range(t)]
    best_motifs = copy.deepcopy(motifs)
    best_score = calc_score(k, t, best_motifs, idx)
    for j in range(N):
        x = random.randint(0, t - 1)
        profile = calculate_profile(k, t - 1, [motifs[i] for i in range(t) if i != x], idx)
        motifs[x] = choose_randomly_by_profile(dnas[x], k, profile, idx) 
        score = calc_score(k, t, motifs, idx)
        if score < best_score:
            best_score = score
            best_motifs = copy.deepcopy(motifs)
    print('.')
    return best_score, best_motifs


def main():
    with fileinput.input() as fi:
        ktN = fi.readline().rstrip().split(' ')
        k = int(ktN[0])
        t = int(ktN[1])
        N = int(ktN[2])
        dnas = []
        for line in fi:
            dnas.append(line.rstrip())
    results = [gibbs_sampler(k, t, N, dnas) for i in range(20)]    
    i, val = min(enumerate(results), key=operator.itemgetter(1))
    print(str(val[0]) + '\n\n' + '\n'.join(val[1]))
    #print('\n'.join(val[1]))


if __name__ == "__main__":
    main()