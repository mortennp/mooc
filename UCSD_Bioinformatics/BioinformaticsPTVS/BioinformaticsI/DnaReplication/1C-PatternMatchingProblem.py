import fileinput

def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield i, text[i:i+k]

def get_matches(pattern, genome):
    positions = []
    for position, kmer in yield_kmers(genome, len(pattern)):
        if kmer == pattern:
            positions.append(str(position))
    return ' '.join(positions)

with fileinput.input() as fi:
    pattern = fi.readline().rstrip()
    genome = fi.readline().rstrip()
    print(get_matches(pattern, genome))