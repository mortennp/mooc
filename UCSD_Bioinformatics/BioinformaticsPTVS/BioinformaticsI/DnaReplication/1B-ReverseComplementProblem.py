import fileinput
import copy


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
    return ''.join(chunk_complements)


chunksize = 3
with fileinput.input() as fi:
    text = fi.readline().rstrip()
tt = build_translation_table(chunksize)
complement = generate_strand_complement(text, tt, chunksize)
reverse = complement[::-1]
print(reverse)
