# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:22:32 2015

@author: Morten
"""
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


def find_approximate_matches(pattern, text, d):
    match_positions = []
    for position, kmer in yield_kmers_by_position(text, len(pattern)):
        if calc_hamming_distance(pattern, kmer) <= d:
            match_positions.append(position)
    return match_positions


def count_approximate_matches(pattern, text, d):
    return len(find_approximate_matches(pattern, text, d))


def main():
    with fileinput.input() as fi:
        pattern = fi.readline().rstrip()
        genome = fi.readline().rstrip()
        d = int(fi.readline().rstrip())
#        cnt = count_approximate_matches(pattern, genome, d)
#        print(str(cnt))
        matches = find_approximate_matches(pattern, genome, d)
        print(' '.join(map(str, matches)))


#if __name__ == "__main__":
#    main()
