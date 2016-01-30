# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:17:34 2015

@author: mortennp
"""


import fileinput


def calc_overlap_graph(nodes):
    prefix_dict = {node[0:-1]: node for node in nodes}
    adjacency_list = [(node, prefix_dict.get(node[1:], '')) for node in nodes]
    return adjacency_list


def main():
    nodes = []
    with fileinput.input() as fi:
        for line in fi:
               nodes.append(line.rstrip())
    adjacency_list = calc_overlap_graph(sorted(nodes))
    for (node, adjacent) in adjacency_list:
        if adjacent:
            print('{0} -> {1}'.format(node, adjacent))


if __name__ == "__main__":
    main()        