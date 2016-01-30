# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:04:45 2015

@author: mortennp
"""

import fileinput
from collections import defaultdict

def read_graph():
    neighbors = {}
    with fileinput.input() as fi:
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
    cycle = random_walk('0', neighbors)
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


def main():
    neighbors = read_graph()
    ep = find_eulerian_path(neighbors)
    print('->'.join(ep))


if __name__ == "__main__":
    main()        