# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:56:43 2015

@author: mortennp
"""

import fileinput


def reconstruct_string(k, d, kdmers):
    first_patterns = [kdmer[0:k] for kdmer in kdmers]
    second_patterns = [kdmer[k+1:2*k+1] for kdmer in kdmers]
    prefix_string = ''.join(pattern[0] for pattern in first_patterns)
    prefix_string += first_patterns[-1][1:]
    suffix_string = ''.join(pattern[0] for pattern in second_patterns)
    suffix_string += second_patterns[-1][1:]
    for i in range(k+d+1,len(prefix_string)):
        if prefix_string[i] != suffix_string[i-k-d]:
            return None
    return prefix_string + suffix_string[-k-d:]


def main():
    with fileinput.input() as fi:
        kd = fi.readline().rstrip().split(' ') 
        k = int(kd[0])
        d = int(kd[1])
        kdmers = []
        for line in fi:
            kdmers.append(line.rstrip())
    s = reconstruct_string(k, d, kdmers)
    print(s)
    

if __name__ == "__main__":
    main()        