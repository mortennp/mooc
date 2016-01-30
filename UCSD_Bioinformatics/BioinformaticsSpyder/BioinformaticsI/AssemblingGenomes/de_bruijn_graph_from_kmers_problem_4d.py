# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 21:34:13 2015

@author: mortennp
"""

import fileinput
from itertools import groupby
from operator import itemgetter


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
            
    for (idx, label) in sorted_nodes:
        if mapping[idx] == idx:
            mapped_neighbors = sorted(labels[mapping[neighbor_idx]] for neighbor_idx in neighbors[idx])
            if mapped_neighbors:
                print('{0} -> {1}'.format(labels[idx],','.join(mapped_neighbors)))
    

def main():
    calc_de_bruijn_graph()       


if __name__ == "__main__":
    main()        