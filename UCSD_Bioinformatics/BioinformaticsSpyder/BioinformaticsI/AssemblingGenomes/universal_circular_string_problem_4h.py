# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 20:16:28 2015

@author: mortennp
"""


import sys
import fileinput
from collections import defaultdict
from itertools import groupby
from itertools import product
from operator import itemgetter
from io import StringIO

# De Bruijn graph construction

def yield_edges(k):
    for lst in product(['0', '1'], repeat=k):
        yield ''.join(lst)
            

def calc_composition_graph(k):
    labels = []
    neighbors = []
    node_idx = 0
    for edge in yield_edges(k):
        k = len(edge)
        next_idx = node_idx + 1;
        labels.append(edge[0:k-1])
        labels.append(edge[1:k])
        neighbors.append([next_idx])
        neighbors.append([])
        node_idx += 2
    return (labels, neighbors)
    
        
def calc_de_bruijn_graph(k):
    (labels, neighbors) = calc_composition_graph(k)
    
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
                
# Eulerian cycle

def read_graph(fi):
    neighbors = {}
    for line in fi:
        (node, adjacents) = line.rstrip().split('->')
        node = node.rstrip().lstrip()
        adjacents = adjacents.rstrip().lstrip()
        neighbors[node] = adjacents.split(',')
    return neighbors
    
    
def random_walk(node, neighbors):
    path = []
    while True:
        path.append(node)
        adjacents = neighbors[node]
        if not adjacents:
            break;
        node = adjacents.pop(0)        
    return path


def is_cycle(path):
    return path[0] == path[-1]


def find_eulerian_cycle(neighbors):
    node = next(iter(neighbors.keys()))
    cycle = random_walk(node, neighbors)
    if not is_cycle(cycle):
        return None
    while True:
        (idx, node) = next(((idx, node) for (idx, node) in enumerate(cycle) if neighbors[node]), (None, None))
        if not node:
            break;
        cycle.pop(-1)
        l = len(cycle)
        cycle_prime = [cycle[(idx + i) % l] for i in range(l)] 
        cycle_extension = random_walk(node, neighbors)
        if not is_cycle(cycle_extension):
            return None
        cycle_prime.extend(cycle_extension)
        cycle = cycle_prime
    return cycle


# String reconstruction

def reconstruct_string(path):
    string = ''
    for node in path:
        string += node[-1]
    return string


def main():
    with fileinput.input() as fi:
        k = int(fi.readline().rstrip())

    stdout = sys.stdout
    sys.stdout = file = StringIO()
    calc_de_bruijn_graph(k)
    sys.stdout = stdout

    file.seek(0)
    neighbors = read_graph(file)
    cycle = find_eulerian_cycle(neighbors)
    cycle.pop(-1)
    s = reconstruct_string(cycle)
    print(s)


if __name__ == "__main__":
    main()        