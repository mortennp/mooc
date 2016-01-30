# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:42:29 2015

@author: mortennp
"""
import fileinput
import sys

def yield_enumerate_kmers(k):
    if 0 == k:
        yield ""
        return
    nucleotides = ['A', 'C', 'G', 'T']
    for nucleotide in nucleotides:
        for kmer in yield_enumerate_kmers(k-1):
            yield nucleotide + kmer
            
            
def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]   
        

def calc_hamming_distance(text1, text2):
    assert(len(text1) == len(text2))
    result = 0
    for i in range(len(text1)):
        if text1[i:i+1] != text2[i:i+1]:
            result += 1
    return result
    
def calc_distance(pattern, k, dnas):
    d = 0
    for dna in dnas:
        d += min(calc_hamming_distance(pattern, kmer) for kmer in yield_kmers(dna, k))
    return d

def find_median_string(k, dnas):
    median = ''
    distance = sys.maxsize
    for pattern in yield_enumerate_kmers(k):
        d = calc_distance(pattern, k, dnas)
        if d < distance:
            median = pattern 
            distance = d
        else:
            if d == distance:
                median += ' ' + pattern
    return median

def main():
    with fileinput.input() as fi:
        k = int(fi.readline().rstrip())
        dnas = []
        for line in fi:
            dnas.append(line.rstrip())
    median = find_median_string(k, dnas)
    print(median)
                

if __name__ == "__main__":
    main()