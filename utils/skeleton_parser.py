import networkx as nx
from typing import Tuple, List, Dict
from collections import deque
import random


def get_graph(
    data: List[Tuple[int, int, float, float, float, float, int]]
) -> nx.DiGraph:
    """
    Generate networkx from list of tuples of form node_id, node_type, x, y, z, radius, parent_id
    """
    G = nx.DiGraph()
    for row in data:
        G.add_node(row[0], type=row[1], x=row[2], y=row[3], z=row[4])
        if row[6] != -1:
            G.add_edge(row[6], row[0])
    return G


def split_tree(
    G: nx.DiGraph, strahler: int = None, split_node: int = None
) -> Tuple[nx.DiGraph, nx.DiGraph, int, int]:
    """
    split tree: splits tree on edge between random node and its parent.
    Returns both halves of the 
    Node A can be specified by its id, or you can specify a strahler index to randomly
    select a node of a specific size. If neither option is specified, select a node at random
    """
    TempG = G.copy()
    if len(list(G.nodes)) < 2:
        raise ValueError("cannot split tree with less than 2 nodes")
    tries = 0
    while True:
        if split_node is None:
            potential_splits = [
                node
                for node in TempG.nodes
                if strahler is None or TempG.nodes[node]["strahler"] == strahler
            ]
            split_node = random.choice(potential_splits)
        try:
            parent = list(TempG.predecessors(split_node))[0]
            break
        except IndexError as e:
            print(e)
            tries += 1
            if tries > 3:
                raise ValueError("failed to split tree after 3 attempts")
            pass

    TempG.remove_edge(parent, split_node)

    full_node_set = set(G.nodes)
    split_set = set(dfs(TempG, split_node))
    rest_set = set([x for x in full_node_set if x not in split_set])

    S = nx.DiGraph()
    for parent in split_set:
        S.add_node(parent)
        for child in TempG.successors(parent):
            if child in split_set:
                S.add_edge(parent, child)
        S.nodes[parent].update(TempG.nodes[parent])

    R = nx.DiGraph()
    for parent in rest_set:
        R.add_node(parent)
        for child in TempG.successors(parent):
            if child in rest_set:
                R.add_edge(parent, child)
        R.nodes[parent].update(TempG.nodes[parent])

    S.nodes

    return S, R, split_node, parent


def re_index(G: nx.DiGraph) -> Tuple[nx.DiGraph, Dict]:
    node_map = {}
    F = nx.DiGraph()
    current = 0
    for node in dfs(G):
        node_map[node] = current
        current += 1
        F.add_node(node_map[node])
        F.nodes[node_map[node]].update(G.nodes[node])

    for u, v in G.edges:
        F.add_edge(node_map[u], node_map[v])
    return F, node_map


def merge_trees(
    A: nx.DiGraph,
    B: nx.DiGraph,
    node_indicies: Tuple[int, int] = None,
    split_b_tree: bool = True,
) -> Tuple[nx.DiGraph, int]:
    if node_indicies is not None:
        a, b = node_indicies
    else:
        a, b = random.choice(list(A.nodes.keys())), random.choice(list(B.nodes.keys()))

    B = new_root(B, b)

    offset = (
        A.nodes[a]["x"] - B.nodes[b]["x"],
        A.nodes[a]["y"] - B.nodes[b]["y"],
        A.nodes[a]["z"] - B.nodes[b]["z"],
    )
    translate_tree(B, offset)

    max_a_id = max(list(A.nodes.keys()))
    C = nx.DiGraph()
    for u, v in A.edges:
        C.add_edge(u, v)
        C.nodes[u].update(A.nodes[u])
        C.nodes[v].update(A.nodes[v])

    for b_node in B.nodes.keys():
        if b_node != b:
            C.add_node(b_node + max_a_id + 1)
            C.nodes[b_node + max_a_id + 1].update(B.nodes[b_node])
    for u, v in B.edges.keys():
        if u != b:
            C.add_edge(u + max_a_id + 1, v + max_a_id + 1)
        else:
            C.add_edge(a, v + max_a_id + 1)

    return C, a


def translate_tree(G: nx.DiGraph, direction: Tuple[float, float, float]) -> nx.DiGraph:
    """
    translate tree by some vector
    """
    for node in G.nodes:
        G.nodes[node]["x"] += direction[0]
        G.nodes[node]["y"] += direction[1]
        G.nodes[node]["z"] += direction[2]
    return G


def dfs(G: nx.DiGraph, start: int = None) -> List[int]:
    """
    depth first search of the tree...
    might also be breadth first, i didnt check which side append and pop go to
    """
    if not G.graph.get("root_node", False):
        calculate_root(G)

    dfs_list = []
    if start is None:
        queue = deque([G.graph["root_node"]])
    else:
        queue = deque([start])

    while len(queue) > 0:
        current = queue.pop()
        dfs_list.append(current)
        for node in G.successors(current):
            queue.append(node)

    return dfs_list


def calculate_root(G: nx.DiGraph, start: int = 0) -> None:
    """
    starting from a somewhat arbitrary node, move upstream until reaching a node
    """
    current = list(G.nodes)[start]
    predecessors = list(G.predecessors(current))
    while len(predecessors) == 1:
        current = predecessors[0]
        predecessors = list(G.predecessors(current))
    G.graph["root_node"] = current


def new_root(G: nx.DiGraph, new_root: int = 0) -> nx.DiGraph:
    """
    Just create a new tree starting at the new root
    iterate over nodes extracting neighbors
    keep track of seen nodes in a set
    i.e. do the "wave" over nodes in G
    """
    NG = nx.DiGraph()

    seen = set()
    current = [new_root]
    next_layer = []
    while len(current) > 0:
        for c in current:
            NG.add_node(c)
            NG.nodes[c].update(G.nodes[c])
            seen.add(c)
            for n in list(G.predecessors(c)) + list(G.successors(c)):
                if n not in seen:
                    next_layer.append(n)
                    NG.add_edge(c, n)
        current = next_layer
        next_layer = []
    NG.graph["root_node"] = new_root
    return NG


def calculate_strahlers(G: nx.DiGraph) -> None:
    """
    calculate a strahler for all nodes and store it
    """
    dfs_list = dfs(G)
    global_max_strahler = 1
    for node in iter(dfs_list[::-1]):
        successor_strahlers = [G.nodes[s]["strahler"] for s in G.successors(node)]
        max_strahler = 1
        count = 0
        for strahler in successor_strahlers:
            if strahler > max_strahler:
                max_strahler = strahler
                count = 1
            elif strahler == max_strahler:
                count += 1
        if count >= 2:
            max_strahler += 1
        global_max_strahler = (
            max_strahler if max_strahler > global_max_strahler else global_max_strahler
        )
        G.nodes[node]["strahler"] = max_strahler
    G.graph["max_strahler"] = global_max_strahler

