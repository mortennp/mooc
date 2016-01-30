# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:58:06 2015

@author: mortennp
"""


import fileinput

def main():
    string = []
    last_kmer = None
    with fileinput.input() as fi:
        for line in fi:
               string.append(line[0]) 
               last_kmer = line
    string.append(last_kmer[1:])               
    print(''.join(string))


if __name__ == "__main__":
    main()        