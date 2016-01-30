# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 21:53:42 2015

@author: Morten
"""
import fileinput
import frequency_array_lib


def main():
    with fileinput.input() as fi:
        genome = fi.readline().rstrip()
        number = frequency_array_lib.pattern_to_number(genome)
        print(number)


if __name__ == "__main__":
    main()
