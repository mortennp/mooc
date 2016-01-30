# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:16:08 2015

@author: mortennp
"""
import fileinput


def prep_translation_table():
    tt = {}
    for mapping in codons.split(","):
        three_mer = mapping[0:3]
        amino_acid = mapping[3:].rstrip().lstrip()
        tt[three_mer] = amino_acid
    return tt    


def yield_3mers(pattern):
    idx = 0
    while idx < len(pattern):
        yield pattern[idx:idx+3]
        idx += 3


def encode(pattern, tt):
    peptide = ''
    for three_mer in yield_3mers(pattern):
        if not three_mer:
            return peptide
        peptide += tt[three_mer]
    return peptide


def main():
    tt = prep_translation_table()
    with fileinput.input() as fi:
        pattern = fi.readline().rstrip()
    peptide = encode(pattern, tt)
    print(peptide)


codons = "\
AAA K,\
AAC N,\
AAG K,\
AAU N,\
ACA T,\
ACC T,\
ACG T,\
ACU T,\
AGA R,\
AGC S,\
AGG R,\
AGU S,\
AUA I,\
AUC I,\
AUG M,\
AUU I,\
CAA Q,\
CAC H,\
CAG Q,\
CAU H,\
CCA P,\
CCC P,\
CCG P,\
CCU P,\
CGA R,\
CGC R,\
CGG R,\
CGU R,\
CUA L,\
CUC L,\
CUG L,\
CUU L,\
GAA E,\
GAC D,\
GAG E,\
GAU D,\
GCA A,\
GCC A,\
GCG A,\
GCU A,\
GGA G,\
GGC G,\
GGG G,\
GGU G,\
GUA V,\
GUC V,\
GUG V,\
GUU V,\
UAA ,\
UAC Y,\
UAG ,\
UAU Y,\
UCA S,\
UCC S,\
UCG S,\
UCU S,\
UGA ,\
UGC C,\
UGG W,\
UGU C,\
UUA L,\
UUC F,\
UUG L,\
UUU F"


if __name__ == "__main__":
    main()
