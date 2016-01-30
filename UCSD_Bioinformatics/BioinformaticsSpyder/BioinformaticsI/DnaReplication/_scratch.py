# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 12:26:08 2015

@author: Morten
"""
import fileinput

def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]


def get_nucleotide_index():
    idx = {}
    idx["A"] = 0
    idx["C"] = 1
    idx["G"] = 2
    idx["T"] = 3
    return idx, len(idx)


def pattern_to_number(text):
    idx, alphabet_size = get_nucleotide_index()
    number = 0
    for char in text:
        number *= alphabet_size
        number += idx[char]
    return number


def get_reverse_nucleotide_index():
    idx = {}
    idx[0] = "A"
    idx[1] = "C"
    idx[2] = "G"
    idx[3] = "T"
    return idx, len(idx)


def number_to_pattern(number, k):
    idx, alphabet_size = get_reverse_nucleotide_index()
    pattern = ""
    for i in reversed(range(0, k)):
        divisor = pow(alphabet_size, i)
        key, number = divmod(number, divisor)
        pattern += idx[key]
    return pattern

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
        if calc_hamming_distance(suffix_neighbor, suffix) < d:
                for nucleotide in nucleotides:
                    neighbors.append(nucleotide + suffix_neighbor)
        else:
            neighbors.append(first_symbol + suffix_neighbor)
    return neighbors


#def yield_enumerate_kmers(k):
#    if 0 == k:
#        yield ""
#        return
#    nucleotides = ['A', 'C', 'G', 'T']
#    for nucleotide in nucleotides:
#        for kmer in yield_enumerate_kmers(k-1):
#            yield nucleotide + kmer
#
#
def get_approximate_frequent_words(text, k, d):
    # Flag indicies of candidate kmers from neighborhoods
    is_neighbor_flags = [0] * pow(4, k)
    for position, kmer in yield_kmers_by_position(text, k):
        neighbors = calculate_neighbors(kmer, d)
        for neighbor in neighbors:
            number = pattern_to_number(neighbor)
            is_neighbor_flags[number] = 1

    # Count approximate occurences of candidates
    frequency_array = [0] * pow(4, k)
    for i in range(len(is_neighbor_flags)):
        if 1 == is_neighbor_flags[i]:
            kmer = number_to_pattern(i, k)
            frequency_array[i] = count_approximate_matches(kmer, text, d)

    # Find max count
    max_count = -1
    for count in frequency_array:
        if count > max_count:
            max_count = count

    # Find most frequent kmers
    frequent = []
    for i in range(len(frequency_array)):
        if frequency_array[i] == max_count:
            kmer = number_to_pattern(i, k)
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
