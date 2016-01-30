import fileinput

def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]

def get_frequent_words(text, k):
    # Count kmers
    dict = {}
    for kmer in yield_kmers(text, k):
        count = dict.get(kmer, 0);
        dict[kmer] = count + 1

    # Find max count
    max = -1
    for kmer, count in dict.items():
        if count > max:
            max = count

    # Find most frequent kmers
    frequent = []
    for kmer, count in dict.items():
        if count == max:
            frequent.append(kmer)

    # Sort and return
    frequent.sort()
    return frequent

with fileinput.input() as fi:
    text = fi.readline().rstrip()
    k = int(fi.readline().rstrip())
frequent = get_frequent_words(text, k)
print(' '.join(frequent))

