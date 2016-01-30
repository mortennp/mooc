# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:13:17 2015

@author: mortennp
"""

import sys
import fileinput
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from io import StringIO

# De Bruijn graph construction

def yield_edges(fi):
    for line in fi:
        yield line.rstrip()
            

def calc_composition_graph(fi):
    labels = []
    neighbors = []
    node_idx = 0
    for edge in yield_edges(fi):
        k = len(edge)
        next_idx = node_idx + 1;
        labels.append(edge[0:k-1])
        labels.append(edge[1:k])
        neighbors.append([next_idx])
        neighbors.append([])
        node_idx += 2
    return (labels, neighbors)
    
        
def calc_de_bruijn_graph(fi):
    (labels, neighbors) = calc_composition_graph(fi)
    
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
                
# Eulerian path

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
    
    
def find_balancing_edge(neighbors):
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
    unbalanced = [node for node in nodes if in_degrees[node] != out_degrees[node]]
    if len(unbalanced) != 2:
        return None
    (node1, node2) = unbalanced
    if in_degrees[node1] < out_degrees[node1]:
        (node1, node2) = (node2, node1)
    return (node1, node2)


def find_eulerian_path(neighbors):
    (node1, node2) = find_balancing_edge(neighbors)
    if node1 in neighbors:
        neighbors[node1].append(node2)
    else:
        neighbors[node1] = [node2]
    cycle = find_eulerian_cycle(neighbors)
    cycle.pop(-1)
    idx = next((idx for (idx, node) in enumerate(cycle) if cycle[idx] == node1 and cycle[idx+1] == node2), None)
    l = len(cycle)
    path = [cycle[(idx + 1 + i) % l] for i in range(l)]
    return path


# String reconstruction

def reconstruct_string(path):
    string = path[0]
    for node in path[1:]:
        string += node[-1]
    return string


def main():
    stdout = sys.stdout
    sys.stdout = file = StringIO()
    with fileinput.input() as fi:
        k = int(fi.readline().rstrip())
        calc_de_bruijn_graph(fi)
    sys.stdout = stdout

    file.seek(0)
    neighbors = read_graph(file)
    path = find_eulerian_path(neighbors)
    s = reconstruct_string(path)
    print(s)


if __name__ == "__main__":
    main()