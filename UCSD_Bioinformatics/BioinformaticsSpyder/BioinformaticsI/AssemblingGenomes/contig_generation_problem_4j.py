# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:12:53 2015

@author: mortennp
"""

import fileinput
from itertools import groupby
from operator import itemgetter
from collections import defaultdict


def yield_edges():
    with fileinput.input() as fi:
        for line in fi:
            yield line.rstrip()
            

def calc_composition_graph():
    labels = []
    neighbors = []
    node_idx = 0
    for edge in yield_edges():
        k = len(edge)
        next_idx = node_idx + 1;
        labels.append(edge[0:k-1])
        labels.append(edge[1:k])
        neighbors.append([next_idx])
        neighbors.append([])
        node_idx += 2
    return (labels, neighbors)
    
        
def calc_de_bruijn_graph():
    (labels, neighbors) = calc_composition_graph()
    
    sorted_nodes = sorted(enumerate(labels), key=itemgetter(1))
    groups = groupby(sorted_nodes, key=itemgetter(1))    
    mapping = {idx: idx for idx, _ in enumerate(labels)}
    for _, group in groups:
        (survivor_node, *culled_nodes) = group
        survivor_idx = survivor_node[0]
        for culled_node in culled_nodes:
            culled_idx = culled_node[0]
            neighbors[survivor_idx].extend(neighbors[culled_idx])
            mapping[culled_idx] = survivor_idx            
            
    de_bruijn_neighbors = {}
    for (idx, label) in sorted_nodes:
        if mapping[idx] == idx:
            mapped_neighbors = sorted(labels[mapping[neighbor_idx]] for neighbor_idx in neighbors[idx])
            if mapped_neighbors:
                de_bruijn_neighbors[label] = mapped_neighbors
    return de_bruijn_neighbors
    
    
def calc_degrees(neighbors):
    in_degrees = defaultdict(int)
    out_degrees = defaultdict(int)
    nodes = set()
    for node in neighbors:
        adjacents = neighbors[node]
        nodes.add(node)
        out_degrees[node] = len(adjacents)
        for adjacent in adjacents:
            nodes.add(adjacent)
            in_degrees[adjacent] += 1
    return (in_degrees, out_degrees)


def walk_contig(node, neighbors, in_degrees, out_degrees):
    path = [node]
    while True:
        adjacents = neighbors[node]
        if not adjacents:
            break;
        next_node = adjacents.pop(0)
        path.append(next_node)
        if 1 != in_degrees[next_node] or 1 != out_degrees[next_node]:
            break;        
        node = next_node
    return path
    

def find_contigs(neighbors, in_degrees, out_degrees):
    contigs = []
    while True:
        node = next((key for key, val in neighbors.items() if val and not (1 == in_degrees[key] and 1 == out_degrees[key])), None)
        if not node:
            break
        contig = walk_contig(node, neighbors, in_degrees, out_degrees)
        contigs.append(contig)   
    return contigs
    
    
def reconstruct_string(contig):
    s = ''
    for node in contig:
        s += node[0]
    s += contig[-1][1:]
    return s
    

def main():
    neighbors = calc_de_bruijn_graph()
    (in_degrees, out_degrees) = calc_degrees(neighbors)
    contigs = find_contigs(neighbors, in_degrees, out_degrees)    
    print('\n'.join(sorted(reconstruct_string(contig) for contig in contigs)))


if __name__ == "__main__":
    main()        