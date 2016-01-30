# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:33:18 2015

@author: mortennp
"""
import fileinput
import operator


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
    matrix = [[0 for j in range(k)] for i in range(len(idx))]
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


def find_most_probable(text, k, matrix, idx):
    best_prob = 0
    best_kmer = text[0:k]
    for kmer in yield_kmers(text, k):
        prob = calc_probability(kmer, matrix, idx)
        if prob > best_prob:
            best_prob = prob
            best_kmer = kmer
    return best_kmer
    
    
def calc_score(k, t, kmers, idx):
    matrix = calculate_counts(k, kmers, idx)
    transpose = list(zip(*matrix))
    score = 0
    for j in range(len(transpose)):
        column = transpose[j]
        i, val = max(enumerate(column), key=operator.itemgetter(1))
        score += (t - val)
    return score


def greedy_motif_search(k, t, dnas):
    idx = get_nucleotide_index()
    best_motifs = [dna[0:k] for dna in dnas]
    best_score = calc_score(k, t, best_motifs, idx)
    for motif1 in yield_kmers(dnas[0], k):
        motifs = [motif1]
        for i in range(1, t):
            profile = calculate_profile(k, i, motifs, idx)
            motif_i = find_most_probable(dnas[i], k, profile, idx)
            motifs.append(motif_i)
            
        score = calc_score(k, t, motifs, idx)
        if score < best_score:
            best_score = score
            best_motifs = motifs
                    
    return best_motifs


def main():
    with fileinput.input() as fi:
        kt = fi.readline().rstrip().split(' ')
        k = int(kt[0])
        t = int(kt[1])
        dnas = []
        for line in fi:
            dnas.append(line.rstrip())
    motifs = greedy_motif_search(k, t, dnas)
    print('\n'.join(motifs))

if __name__ == "__main__":
    main()