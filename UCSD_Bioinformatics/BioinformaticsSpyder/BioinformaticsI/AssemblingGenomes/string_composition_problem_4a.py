# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:32:16 2015

@author: mortennp
"""


import fileinput


def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]


def main():
    with fileinput.input() as fi:
        k = int(fi.readline().rstrip())
        text = fi.readline().rstrip()
    print('\n'.join(sorted(yield_kmers(text, k)))


if __name__ == "__main__":
    main()        