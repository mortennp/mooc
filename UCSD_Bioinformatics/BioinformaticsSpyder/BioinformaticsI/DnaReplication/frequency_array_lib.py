# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 22:15:17 2015

@author: Morten
"""


def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]


def get_nucleotide_index():
    idx = {}
    idx["A"] = 0
    idx["C"] = 1
    idx["G"] = 2
    idx["T"] = 3
    return idx, len(idx)


def pattern_to_number(text):
    idx, alphabet_size = get_nucleotide_index()
    number = 0
    for char in text:
        number *= alphabet_size
        number += idx[char]
    return number


def get_reverse_nucleotide_index():
    idx = {}
    idx[0] = "A"
    idx[1] = "C"
    idx[2] = "G"
    idx[3] = "T"
    return idx, len(idx)


def number_to_pattern(number, k):
    idx, alphabet_size = get_reverse_nucleotide_index()
    pattern = ""
    for i in reversed(range(0, k)):
        divisor = pow(alphabet_size, i)
        key, number = divmod(number, divisor)
        pattern += idx[key]
    return pattern
