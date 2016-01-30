# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:08:02 2015

@author: mortennp
"""

import fileinput
           
            
def get_nucleotide_index():
    idx = {}
    idx["A"] = 0
    idx["C"] = 1
    idx["G"] = 2
    idx["T"] = 3
    return idx
    
    
def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]   


def calc_probability(kmer, matrix, idx):
    prob = 1
    for i in range(len(kmer)):
        nucleotide = kmer[i]        
        prob *= matrix[idx[nucleotide]][i]
    return prob


def find_most_probable(text, k, matrix):
    idx = get_nucleotide_index()
    best_prob = 0
    best_kmer = ''
    for kmer in yield_kmers(text, k):
        prob = calc_probability(kmer, matrix, idx)
        if prob > best_prob:
            best_prob = prob
            best_kmer = kmer
    return best_kmer

def main():
    with fileinput.input() as fi:
        text = fi.readline().rstrip()
        k = int(fi.readline().rstrip())
        rows = []
        for line in fi:
            rows.append(list(map(float, line.rstrip().split(' '))))
    matrix = [rows[i] for i in range(len(rows))]
    kmer = find_most_probable(text, k, matrix)
    print(kmer)
                

if __name__ == "__main__":
    main()