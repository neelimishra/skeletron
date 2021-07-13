import collections
import json
import pdb
import os
import numpy as np
import collections
import random
import torch
from torch_geometric.data import Data
from torch.utils import data

from .skeleton_parser import get_graph, split_tree, calculate_strahlers, calculate_root, merge_trees, re_index
from .skeleton_resample import resample

class Skeletron(data.Dataset):
    def __init__(self, root, split='train'):

        self.split = split 
        self.root = root
        self.files = collections.defaultdict(list)

        for split in ["train", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        #print(self.files[self.split][index], flush=True)
        raw_data = np.loadtxt(self.root + "/" + self.split + "/" + self.files[self.split][index])
        G = get_graph(raw_data)
        calculate_root(G)
        calculate_strahlers(G)

        try:
            S, R, a, b = split_tree(G)
        except:
            return self[0]

        G = resample(G, 2)

        if random.choice([0,1]) == 0:
            # split
            R, _ = re_index(R)
            processed_data = np.array([(node,
                                        R.nodes[node]["type"],
                                        R.nodes[node]["x"],
                                        R.nodes[node]["y"],
                                        R.nodes[node]["z"],
                                        R.nodes[node]["strahler"],
                                        list(R.predecessors(node))[0] if len(list(R.predecessors(node))) > 0 else - 1) for node in R.nodes])
            
            indices = processed_data[:,0].astype(np.int)
            merges = np.zeros(processed_data.shape[0])
            #print(np.sum((indices == b).astype(np.float)))
            #wreck()

        else:
            # merge
            #S, _, _, _ = split_tree(G)
            R, merge = merge_trees(G, S)
            R, index_map = re_index(R)


            processed_data = np.array([(node,
                                        R.nodes[node]["type"],
                                        R.nodes[node]["x"],
                                        R.nodes[node]["y"],
                                        R.nodes[node]["z"],
                                        R.nodes[node]["strahler"],
                                        list(R.predecessors(node))[0] if len(list(R.predecessors(node))) > 0 else - 1) for node in R.nodes])
        
            indices = processed_data[:,0].astype(np.int)
            node_ids = processed_data[:, 0]
            merges = (node_ids == index_map[merge])*1.0


        # Creating relative coordinates
        raw_coords = processed_data[:,2:5]
        node_ids = processed_data[:, 0]

        edges = list(R.edges.keys())
        #edges = np.array(edges).astype(np.int)
        rel_coords = []

        for edge in R.edges: 
            u, v = edge
            assert u in node_ids, "can't find node {}".format(u)
            assert v in node_ids, "can't find node {}".format(v)

            relative = np.array([R.nodes[u]["x"] - R.nodes[v]["x"],
                                     R.nodes[u]["y"] - R.nodes[v]["y"],
                                     R.nodes[u]["z"] - R.nodes[v]["z"]]) # <- one relative coord per edge. This is where we are missing a node
            rel_coords.append(relative)

        #rel_coords.append(rel_coords[-1])

        rel_coords = np.array(rel_coords)
        #print(rel_coords.shape)
        types = processed_data[:,1] - 1 #[0, 1, 2, 3]
        #indices = processed_data[:,0].astype(np.int)
        #parents = processed_data[1:,-1].astype(np.int) - 1
        #strahlers = processed_data[1:, -2] # integer
        split = (indices == b).astype(np.float)#.astype(int) # binary
        #print(np.unique(split), np.unique(merges))
        split[split == 1] += 1
        split += merges
        #wreck()
        
        target = np.vstack((split, types))

        edge_index = torch.tensor(edges, dtype=torch.long)
        #edge_index[1, 0] = 0
        #pdb.set_trace()
        data = Data(x=torch.tensor(raw_coords), pos=torch.tensor(rel_coords), edge_index=torch.transpose(edge_index, 1, 0), y=torch.tensor(target))

        return data

if __name__ == "__main__":
    dataset = Skeletron(root='/share/data/vision-greg/ivas/proj/skeletron/data')
    data = dataset[0]
    print(data)
