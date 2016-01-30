# -*- coding: utf-8 -*-

import sys
import fileinput

           
def yield_kmers_by_position(text, k):
    for i in range(len(text) - k + 1):
        yield i, text[i:i+k]        
        

def calc_hamming_distance(text1, text2):
    assert(len(text1) == len(text2))
    result = 0
    for i in range(len(text1)):
        if text1[i:i+1] != text2[i:i+1]:
            result += 1
    return result
    

def calculate_neighbors(pattern, d):
    nucleotides = ['A', 'C', 'G', 'T']

    if 0 == d:
        return [pattern]

    if 1 == len(pattern):
        return nucleotides

    neighbors = []
    first_symbol, suffix = pattern[0:1], pattern[1:]
    suffix_neighbors = calculate_neighbors(suffix, d)
    for suffix_neighbor in suffix_neighbors:
        if calc_hamming_distance(suffix_neighbor, suffix) < d:
                for nucleotide in nucleotides:
                    neighbors.append(nucleotide + suffix_neighbor)
        else:
            neighbors.append(first_symbol + suffix_neighbor)
    return neighbors
    
    
def find_approximate_matches(pattern, text, d):
    match_positions = []
    for position, kmer in yield_kmers_by_position(text, len(pattern)):
        if calc_hamming_distance(pattern, kmer) <= d:
            match_positions.append(position)
    return match_positions
    
    
def enumerate_motifs(k, d, dnas):
    patterns = {}
    discards = {}
    for dna in dnas:
        for _, pattern in yield_kmers_by_position(dna, k):
            for prime in calculate_neighbors(pattern, d):
                if prime in discards: continue
                if prime in patterns: continue                           
                matches = list(map(lambda dna: (len(find_approximate_matches(prime, dna, d)) > 0), dnas))
                ok = all(matches)
                if ok:
                    patterns[prime] = True
                else:
                    discards[prime] = False
    return patterns.keys()
    

def main():
    with fileinput.input() as fi:
        kd = fi.readline().rstrip().split(' ')
        k = int(kd[0])
        d = int(kd[1])
        dnas = []
        for line in fi:
            dnas.append(line.rstrip())
    motifs = enumerate_motifs(k, d, dnas)
    print(' '.join(sorted(motifs)))
                

if __name__ == "__main__":
    main()
