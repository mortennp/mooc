# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 23:15:36 2015

@author: mortennp
"""

import fileinput
from collections import defaultdict

        
def convolute(spectrum):
    diffs = []
    for i in range(len(spectrum)):
        for j in range(i):
            diff = spectrum[i] - spectrum[j]
            if diff > 0:
                diffs.append(diff)
    return sorted(diffs)


def main():
    with fileinput.input() as fi:
        spectrum_line = fi.readline().rstrip()
    spectrum = [int(s) for s in spectrum_line.split(" ")]
    output = convolute(sorted(spectrum))
    print(" ".join(map(str, output)))
    
if __name__ == "__main__":
    main()
    