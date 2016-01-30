import fileinput

def yield_kmers(text, k):
    for i in range(len(text) - k + 1):
        yield i, text[i:i+k]

def build_kmer_index(text, k):
    kmerindex = {}
    for position, kmer in yield_kmers(genome, k):
        positions = kmerindex.get(kmer, [])
        positions.append(position)
        kmerindex[kmer] = positions
    return kmerindex

def find_clumping_kmers(genome, k, L, t):
    kmerindex = build_kmer_index(genome,k)

    kmers = []
    for kmer, positions in kmerindex.items():
        if len(positions) >= t:
            for i in range(0, len(positions) - t + 1):
                clumpsize = positions[i+t-1] - positions[i] + k
                if clumpsize <= L:
                    kmers.append(kmer)
                    break

    #E-coli
    #return str(len(kmers))
    return ' '.join(kmers)

with fileinput.input() as fi:
    genome = fi.readline().rstrip()
    kLt = fi.readline().rstrip().split(' ')
    #E-coli:
    #k = 9
    #L = 500
    #t = 3
    k = int(kLt[0])
    L = int(kLt[1])
    t = int(kLt[2])
    kmers = find_clumping_kmers(genome, k, L, t)
    print(kmers)