# Skeletron: Automated Error Detection in Neuron Skeletons

We aim to teach a Graph Convolutional Network (GCN) to locate (and fix?) mistakes which are naturally ocurring in human-annotated nueron skeletons.
These skeletons are traced out by human annotators attempting to capture the large-scale structure of individual neurons in EM and fluorescence microscopy data.
Common mistakes in such skeletons include *missing branches* (false splits) and accidentilly tracing sections of *adjacent* neurons (false merges).
Given a large collection of fully curated skeletons from [VFB] we train a GCN to identify these common mistakes through artificially introducing damage (splits and merges) similar in character to the common mistakes found in uncurated, human-annotated neuron skeletons.

This project was developed at the Janelia Workshop on Machine Learning and Computer Vision (MLCV) in April 2019. This work was done by Alberto Bailoni, Coleman Broaddus, Melanie Dohmen, Neeli Mishra, William Patton and Igor Vasiljevic. We received very helpful mentorship from Jan Funke and Stephan Saalfeld.

# Graph Convolutional Networks

Graph Convolutional Networks have attributes assigned to nodes, and use the structure of the graph when performing convolutions.


# Relevant Literature

Semi-Supervised Classification with Graph Convolutional Networks
Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering
Gated Graph Sequence Neural Networks
Convolutional Networks on Graphs for Learning Molecular Fingerprints
Deep Convolutional Networks on Graph-Structured Data
Spectral Networks and Locally Connected Networks on Graphs

Recent dialog:

Initial paper: Semi-Supervised Classification with Graph Convolutional Networks
Commentary: https://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2/
Response by Authors: https://tkipf.github.io/graph-convolutional-networks/

---

NBLAST: Rapid, Sensitive Comparison of Neuronal Structure and Construction of Neuron Family Databases
Different task, but they capture relevant (hand-designed) features.

# Questions

What do GCN people mean by *spectral domain*?

How will re-sampling graphs deal with branching points? Leaves?
How will it deal with short sections between branches of length 1,2,3 (e.g. synapses)?

What shape is our input tensor? What is our encoding?

[VFB]: https://www.virtualflybrain.org/site/vfb_site/Chiang2010.htm
