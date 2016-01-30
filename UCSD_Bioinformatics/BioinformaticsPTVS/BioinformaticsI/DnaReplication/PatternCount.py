import fileinput

def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]

def get_pattern_count(text, pattern):
    # Count occurances 
    count = 0
    for kmer in yield_kmers(text, len(pattern)):
        if kmer == pattern:
            count += 1
    return count

with fileinput.input() as fi:
    text = fi.readline().rstrip()
    pattern = fi.readline().rstrip()
count = get_pattern_count(text, pattern)
print(count)

