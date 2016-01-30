# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:26:08 2015

@author: Morten
"""
import fileinput
import copy
import frequency_array_lib
import ApproximatePatternMatching_1F

def build_translation_table(n):
    map = {}
    map['A'] = 'T'
    map['T'] = 'A'
    map['G'] = 'C'
    map['C'] = 'G'

    tt = copy.deepcopy(map)

    for i in range(1, n):
        buffer = copy.deepcopy(tt)
        for strand, strandComplement in buffer.items():
            for nucleotide, nucleotideComplement in map.items():
                newStrand = strand + nucleotide
                newStrandComplement = strandComplement + nucleotideComplement
                tt[newStrand] = newStrandComplement

    return tt


def generate_strand_complement(text, tt, chunksize):
    chunk_complements = []
    for i in range(0, len(text), chunksize):
        chunk = text[i:i+chunksize]
        chunk_complements.append(tt[chunk])
    chunk = text[i+chunksize:]
    if len(chunk) > 0:
        chunk_complements.append(tt[text[i+chunksize:]])
    complement = ''.join(chunk_complements)
    return complement[::-1]


def calculate_neighbors(pattern, d):
    nucleotides = ['A', 'C', 'G', 'T']

    if 0 == d:
        return [pattern]

    if 1 == len(pattern):
        return nucleotides

    neighbors = []
    first_symbol, suffix = pattern[0:1], pattern[1:]
    suffix_neighbors = calculate_neighbors(suffix, d)
    for suffix_neighbor in suffix_neighbors:
        if ApproximatePatternMatching_1F.calc_hamming_distance(suffix_neighbor, suffix) < d:
                for nucleotide in nucleotides:
                    neighbors.append(nucleotide + suffix_neighbor)
        else:
            neighbors.append(first_symbol + suffix_neighbor)
    return neighbors


def get_approximate_frequent_words(text, k, d):
    # Flag indicies of candidate kmers from neighborhoods
    is_neighbor_flags = [0] * pow(4, k)
    for position, kmer in ApproximatePatternMatching_1F.yield_kmers_by_position(text, k):
        neighbors = calculate_neighbors(kmer, d)
        for neighbor in neighbors:
            number = frequency_array_lib.pattern_to_number(neighbor)
            is_neighbor_flags[number] = 1

    # Count approximate occurences of candidates
    chunk_size = 5
    translation_table = build_translation_table(chunk_size)
    frequency_array = [0] * pow(4, k)
    for i in range(len(is_neighbor_flags)):
        if 1 == is_neighbor_flags[i]:
            kmer = frequency_array_lib.number_to_pattern(i, k)
            reverse_complement = generate_strand_complement(kmer, translation_table, chunk_size)
            frequency = ApproximatePatternMatching_1F.count_approximate_matches(kmer, text, d) + ApproximatePatternMatching_1F.count_approximate_matches(reverse_complement, text, d)
            frequency_array[i] = frequency

    # Find max count
    max_count = -1
    for count in frequency_array:
        if count > max_count:
            max_count = count

    # Find most frequent kmers
    frequent = []
    for i in range(len(frequency_array)):
        if frequency_array[i] == max_count:
            kmer = frequency_array_lib.number_to_pattern(i, k)
            frequent.append(kmer)
    return frequent


def script_main(fi):
    genome = fi.readline().rstrip()
    kd = fi.readline().rstrip().split(' ')
    k = int(kd[0])
    d = int(kd[1])
    kmers = get_approximate_frequent_words(genome, k, d)
    print(' '.join(kmers))
#    pattern = fi.readline().rstrip()
#    d = int(fi.readline().rstrip())
#    print('\n'.join(calculate_neighbors(pattern, d)))


def profile_main():
    with fileinput.input("Input.txt") as fi:
        script_main(fi)


def main():
    with fileinput.input() as fi:
        script_main(fi)


if __name__ == "__main__":
    main()