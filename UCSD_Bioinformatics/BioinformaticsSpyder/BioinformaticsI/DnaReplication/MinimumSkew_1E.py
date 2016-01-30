# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:51:44 2015

@author: Morten
"""
import sys
import fileinput


def yield_skew_by_positions(text):
    skew = 0
    position = 0
    yield position, skew
    for i in range(len(text)):
        position += 1
        nucleotide = text[i:i+1]
        if "G" == nucleotide:
            skew += 1
        elif "C" == nucleotide:
            skew -= 1
        yield position, skew


def get_skews(text):
    skews = []
    for position, skew in yield_skew_by_positions(text):
        skews.append(str(position) + ' ' + str(skew))
    return skews


def find_minimum_skew_positions(text):
    min_skew = sys.maxsize
    min_positions = []
    for position, skew in yield_skew_by_positions(text):
        if skew < min_skew:
            min_skew = skew
            min_positions.clear()
            min_positions.append(position)
        elif skew == min_skew:
            min_positions.append(position)
    return min_positions


def main():
    with fileinput.input() as fi:
        genome = fi.readline().rstrip()
        min_positions = find_minimum_skew_positions(genome)
        print(' '.join(map(str, min_positions)))
#        skews = get_skews(genome)
#        print(' '.join(map(str, skews)))


#if __name__ == "__main__":
#    main()
