import fileinput

def get_lines():
    with fileinput.input() as fi:
        for line in fi:
            yield line.rstrip()

def get_kmers(k):
    buffer = ''
    for line in get_lines():        
        buffer = buffer + line
        for i in range(len(buffer) - k):
            yield buffer[i:i+k]
        buffer = buffer[i+1:]

def get_frequent_words(k):
    # Count kmers
    dict = {}
    for kmer in get_kmers(k):
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

    return frequent

print(' '.join(get_frequent_words(5)))
print(' '.join(get_frequent_words(3)))