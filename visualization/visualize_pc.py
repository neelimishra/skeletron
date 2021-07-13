import pptk
import numpy as np
import sys
import os




label_colors = np.asarray([
					[255, 0, 0],
					[0, 255, 0],
					[0, 0, 255]])

def value_to_rgb(value, minimum, maximum):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def visualize_type(tree, filename=None):
	nb_nodes = len(tree.nodes)
	point_coords = np.empty((nb_nodes, 3))
	types = np.empty((nb_nodes,3))
	for i, node in enumerate(tree.nodes):
		point_coords[i] = np.array([tree.nodes[node]['x'],
							 tree.nodes[node]['y'],
							 tree.nodes[node]['z']])
		types[i] = label_colors[int(tree.nodes[node]['type']) - 1]
	v = pptk.viewer(point_coords)
	v.attributes(types)
	v.set(point_size=1.0)
	filename = 'screenshot.png' if filename is None else filename
	v.capture(os.path.join('./figures', filename))


def visualize_strahler(tree, filename=None):
	nb_nodes = len(tree.nodes)
	point_coords = np.empty((nb_nodes, 3))
	strahlers = np.zeros((nb_nodes, 3))


	for i, node in enumerate(tree.nodes):
		point_coords[i] = np.array([tree.nodes[node]['x'],
									tree.nodes[node]['y'],
									tree.nodes[node]['z']])
		strahlers[i, :] = np.array(value_to_rgb(tree.nodes[node]['strahler'], 0., tree.graph['max_strahler']))

	v = pptk.viewer(point_coords)
	v.attributes(strahlers)
	v.set(point_size=1.0)
	filename = 'screenshot.png' if filename is None else filename
	v.capture(os.path.join('./figures', filename))



if __name__ == '__main__':
	RAW_PATH = sys.argv[1]

	points = np.loadtxt(RAW_PATH)
	pc = points[:,2:5]
	labels = points[:,1].astype(np.int) - 1
	types = np.zeros((pc.shape[0], 3))


	for pt in range(pc.shape[0]):
		types[pt, :] = label_colors[labels[pt]]

	v = pptk.viewer(pc)
	v.attributes(types)
	v.set(point_size=1.0)
	v.capture('./figures/screenshot.png')


