# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 20:59:53 2015

@author: mortennp
"""

import fileinput
from itertools import groupby
from operator import itemgetter


def yield_edges(text, k):
    for i in range(len(text) - k + 1):
        yield text[i:i+k]


def calc_path_graph(k, text):
    size = len(text) - k + 2
    labels = [None] * size
    neighbors = [None] * size
    node_idx = 0
    labels[node_idx] = text[0:k-1]
    for edge in yield_edges(text, k):
        next_idx = node_idx + 1;
        neighbors[node_idx] = [next_idx]
        node_idx = next_idx
        labels[node_idx] = edge[1:k]
    neighbors[node_idx] = []
    return (labels, neighbors)
    
        
def calc_de_bruijn_graph(k, text):
    (labels, neighbors) = calc_path_graph(k, text)
    
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
            
    for (idx, label) in sorted_nodes:
        if mapping[idx] == idx:
            mapped_neighbors = sorted(labels[mapping[neighbor_idx]] for neighbor_idx in neighbors[idx])
            if mapped_neighbors:
                print('{0} -> {1}'.format(labels[idx],','.join(mapped_neighbors)))
    

def main():
    with fileinput.input() as fi:
        k = int(fi.readline().rstrip())
        text = fi.readline().rstrip()
        calc_de_bruijn_graph(k, text)


if __name__ == "__main__":
    main()        