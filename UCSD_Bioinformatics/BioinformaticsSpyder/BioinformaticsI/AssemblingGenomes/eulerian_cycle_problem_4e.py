# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 22:03:30 2015

@author: mortennp
"""

import fileinput


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
    

def main():
    neighbors = read_graph()
    ec = find_eulerian_cycle(neighbors)
    print('->'.join(ec))


if __name__ == "__main__":
    main()        