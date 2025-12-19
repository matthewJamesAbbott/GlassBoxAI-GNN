# GlassBoxAI-GNN

**Author:** Matthew Abbott (2025)

A transparent, extensible CUDA implementation of Graph Neural Networks (GNNs) for research and educational use. This repository contains:

- **gnn.cu:** A direct, research-grade CUDA GNN with message passing layers, output/readout, and full GPU training.
- **facaded_gnn.cu:** A modern C++17/CUDA GNN facade for user-friendly scripting, rapid prototyping, and detailed graph/network diagnostics.

Both modules put a premium on direct access, step-by-step visibility, and hackability for advanced users and educators.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [gnn.cu](#gnncu)
  - [Design](#design)
  - [Usage](#usage)
  - [Arguments](#arguments)
  - [Public Methods](#public-methods)
- [facaded_gnn.cu (GNN Facade)](#facaded_gnncu-gnn-facade)
  - [Design](#design-1)
  - [Usage](#usage-1)
  - [Arguments](#arguments-1)
  - [Public Methods](#public-methods-1)
- [Data Structures](#data-structures)
- [Overview & Notes](#overview--notes)

---

## Features

- **Message Passing Graph Neural Network** supporting node-level, graph-level, and edge-level tasks.
- Multiple activation & loss functions (ReLU, LeakyReLU, Tanh, Sigmoid; MSE, BCE).
- Full forward and backward CUDA training; all major layers and graph operations GPU-accelerated.
- Configurable message-passing layers, hidden and output sizes.
- Research-focused utility and graph diagnostics: deduplication, undirected conversion, self-loops.
- Detailed gradient clipping and numerical stability equations.
- Model and graph serialization for checkpointing and experiments.
- Facade GNN exposes node/edge features, neighbor/topology functions, embeddings, and architecture summary.

---

## Requirements

- NVIDIA GPU with CUDA support (compute 6.0+ recommended, but 5.0+ works)
- CUDA toolkit (nvcc, tested up through CUDA 12)
- C++14 or later (C++17 for facade)
- No external deep learning framework required

---

## gnn.cu

### Design

Implements a flexible message passing GNN with explicit CPU-GPU data transfer and direct control. All weights, gradients, graph features, and model states are available for diagnosis or research hacking. CUDA kernels handle forward, backward, and aggregation passes.

### Usage

```bash
nvcc -O3 -std=c++14 gnn.cu -o gnn_cuda
```
Example: Use as a linked library/module in your custom driver (see code for instantiation and API).

### Arguments

#### Model Constructor

- `featureSize`: Per-node input feature dimension (`int`)
- `hiddenSize`: Hidden feature dimension (`int`)
- `outputSize`: Output vector size per graph (`int`)
- `numMPLayers`: Number of message-passing layers

#### Other Model Parameters (Setters or direct member access)

- `learningRate`: (double, default 0.01)
- `maxIterations`: (int, default 100)
- `activation`: Activation function: `atReLU`, `atLeakyReLU`, `atTanh`, `atSigmoid`
- `lossFunction`: Loss: `ltMSE`, `ltBinaryCrossEntropy`
- `Config` (on TGraph):: Controls undirected, self-loop, deduplication booleans

#### Graph Configuration

- `TGraph` struct inputs:
  - `NumNodes`
  - `NodeFeatures` (per node float vector)
  - `Edges` (array of `{Source, Target}`) with methods for deduplication, reversing, and self-loops
  - `AdjacencyList` (built automatically from Edges)

### Public Methods

Class: `TGraphNeuralNetwork`

- Model:
  - `Predict(TGraph&)` — Returns output for a graph.
  - `double Train(TGraph&, const TDoubleArray& target)` — Runs one training step.
  - `void TrainMultiple(TGraph&, const TDoubleArray& target, int iters)` — Trains for N iters.
  - `void SaveModel(std::string path)` and `LoadModel(std::string path)`
  - Getters/setters for architecture, learning rate, metrics, etc.
- Graph utilities:
  - `DeduplicateEdges`, `AddReverseEdges`, `AddSelfLoops`, `ValidateGraph`
  - Adjacency list logic baked in.
- Layers:
  - All layers (message, update, readout, output) accessible for custom inspection/hacking.
- GPU utilities:
  - `InitializeGPU`, `FreeGPU`, `SyncToGPU`, `SyncFromGPU`
- Loss:
  - `ComputeLoss`, `ComputeLossGradient` (MSE or BCE)

---

## facaded_gnn.cu (GNN Facade)

### Design

A modernized GNN abstraction with advanced graph and network diagnostics and user-friendly methods for data science, research, and education. Written in C++17 with CUDA, it supports Python-like exploration of graphs and the underlying network.

### Usage

```bash
nvcc -O3 -std=c++17 facaded_gnn.cu -o gnn_facade_cuda
```
Embed in your C++17 project or call as an executable for custom scripting/classroom use.

### Arguments

#### Facade Constructor

Class: `CUDAGNNFacade`

- `featureSize`: Number of features per node
- `hiddenSize`: Internal dimension
- `outputSize`: Model output dimension
- `numMPLayers`: Number of message passing layers

#### Graph Methods

- `createEmptyGraph(numNodes, featureSize)`
- `setNodeFeature(nodeIdx, featureIdx, value)` / `setNodeFeatures(nodeIdx, FloatArray)`
- `addEdge(source, target [, features])`
- `removeEdge(edgeIdx)`
- `rebuildAdjacencyList()`

#### Core Training/Prediction

- `predict()`: Returns output for the loaded/created graph
- `train(targetFloatArray)`: Returns loss on batch
- `trainMultiple(targetFloatArray, iters)`

#### Saving/Loading

- `saveModel(filename)`
- `loadModel(filename)`

### Public Methods

Class: `CUDAGNNFacade`

- **Node & Edge:**
  - `getNodeFeature(node, idx)`, `setNodeFeature(node, idx, value)`
  - `getNodeFeatures(node)`, `setNodeFeatures(node, FloatArray)`
  - `addEdge(source, target [, features])`, `removeEdge(edgeIdx)`
  - `getEdgeEndpoints(edgeIdx)`, `getEdgeFeatures(edgeIdx)`, `setEdgeFeatures(edgeIdx, FloatArray)`
  - `hasEdge(source, target)`, `findEdgeIndex(source, target)`
  - `getNeighbors(node)`, `getInDegree(node)`, `getOutDegree(node)`

- **Graph:**
  - `rebuildAdjacencyList()`, `createEmptyGraph(numNodes, featureSize)`

- **Prediction/Training:**
  - `predict()`, `train(target)`, `trainMultiple(target, iters)`

- **Persistence:**
  - `saveModel(filename)`, `loadModel(filename)`

- **Network Introspection:**
  - `getArchitectureSummary()`: Prints all network shapes, param counts, and hyperparameters.
  - `getGraphEmbedding()`: Returns the latest graph embedding vector.

- **(Advanced / Internal in main classes)**
  - Forward/backward through each layer, input/output gradient access, full weight serialization.

---

## Data Structures

- `TGraph` / `Graph`: Nodes, features, adjacency, edge lists.
- `TLayer`, `TGPULayer` / `GPULayer`: All layer weights/biases/gradients accessible.
- Facade edge/node masks, edge features.
- Embedding history and gradient flow info for detailed debugging (facade).
- All layer and model weights serializable via disk save/load.

---

## Overview & Notes

- **No icons/branding**—100% code and docs.
- Both "bare" and "facade" APIs are fully hackable; you can extract, modify, and visualize every tensor, graph, or parameter.
- The `facaded_gnn.cu` facade allows for rapid graph experimentation or as a learning/teaching reference model.
- For input/output, use the code's detailed type definitions and comments.
- No external deep learning libraries are ever required.

---

## License

MIT License, Copyright © 2025 Matthew Abbott

---
