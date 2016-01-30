# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:13:58 2015

@author: Morten
"""
import fileinput
import frequency_array_lib


def compute_frequency_array(text, k):
    fa = [0] * pow(4, k)
    for kmer in frequency_array_lib.yield_kmers(text, k):
        number = frequency_array_lib.pattern_to_number(kmer)
        fa[number] = fa[number] + 1
    return fa


def main():
    with fileinput.input() as fi:
        genome = fi.readline().rstrip()
        k = int(fi.readline().rstrip())
        fa = compute_frequency_array(genome, k)
        print(' '.join(map(str, fa)))


if __name__ == "__main__":
    main()
