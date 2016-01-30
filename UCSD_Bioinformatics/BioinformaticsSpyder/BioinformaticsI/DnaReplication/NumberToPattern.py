# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:59:07 2015

@author: Morten
"""

import fileinput
import frequency_array_lib


def main():
    with fileinput.input() as fi:
        number = int(fi.readline().rstrip())
        k = int(fi.readline().rstrip())
        genome = frequency_array_lib.number_to_pattern(number, k)
        print(genome)


if __name__ == "__main__":
    main()
