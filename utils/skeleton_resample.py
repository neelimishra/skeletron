import networkx
import numpy as np

def resample(tree, step_width):

    assert networkx.is_tree(tree)
    root = tree.graph['root_node']

    resampled_tree = networkx.DiGraph(directed=True, **tree.graph)
    resample_segment(tree, (root, root), step_width, resampled_tree)

    return resampled_tree

def resample_segment(
        tree,
        edge,
        step_width,
        resampled_tree,
        append_node=-1,
        next_id=0):

    # print("Resampling segment, starting with edge", edge)
    # print("Current append_node =", append_node)

    node = edge[0]
    next_node = edge[1]
    dist_to_next = None
    distance = 0.0

    first_after_branch = append_node >= 0

    while True:

        pos_node = np.array([tree.node[node][d] for d in ['x', 'y', 'z']])
        pos_next_node = np.array([tree.node[next_node][d] for d in ['x', 'y', 'z']])
        offset = pos_next_node - pos_node
        dist_to_next = np.linalg.norm(pos_node - pos_next_node)

        # O-----O---------O--O-----------O
        # *--*--*--*--*--*--*--*--*--*--**

        # print("dist_tp_next:", dist_to_next)

        while distance <= dist_to_next:

            alpha = distance/dist_to_next if dist_to_next != 0 else 0
            # print("current distance: {}".format(distance))
            # print("alpha:", alpha)
            position = pos_node + alpha*offset

            closest_node = node if alpha < 0.5 else next_node
            kwargs_closest_node = {k:tree.node[closest_node][k] for k in tree.node[closest_node] if k not in ['x', 'y', 'z', 'strahler']}


            if not first_after_branch:
                resampled_tree.add_node(
                    next_id,
                    x=position[0], y=position[1], z=position[2],
                    strahler=tree.node[next_node]['strahler'],
                    **kwargs_closest_node
                )
                if append_node >= 0:
                    resampled_tree.add_edge(append_node, next_id)
                append_node = next_id
                next_id += 1
            else:
                first_after_branch = False
            distance += step_width

        degree = tree.out_degree(next_node)
        # print("Out-degree of next node", next_node, "is", degree)

        if degree == 1:

            distance -= dist_to_next
            node = next_node
            next_node = list(tree.neighbors(node))[0]
            continue

        elif degree == 0:

            resampled_tree.add_node(
                next_id,
                **tree.node[next_node])
            resampled_tree.add_edge(
                append_node, next_id)

            return next_id + 1

        # branch point
        else:
            resampled_tree.add_node(
                next_id,
                **tree.node[next_node])
            resampled_tree.add_edge(
                append_node, next_id)
            append_node = next_id
            next_id += 1
            for child_edge in tree.out_edges(next_node):
                # print("Recursing into", child_edge)
                next_id = resample_segment(
                    tree,
                    child_edge,
                    step_width,
                    resampled_tree,
                    append_node,
                    next_id)
            break

    return next_id
