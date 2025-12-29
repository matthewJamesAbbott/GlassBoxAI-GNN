# GlassBoxAI-GNN

**Author:** Matthew Abbott (2025)

A transparent, research-first Graph Neural Network (GNN) toolkit featuring both CUDA and OpenCL codepaths. Designed for maximum visibility, extensibility, and learning: inspect every activation, weight, embedding, and training step!

This repository provides all major GNN features in **two user modes**:
- **Direct CUDA/OpenCL Model:** Programmable/raw GNN for command-line or scripting (`gnn.cu`, `gnn_opencl.cpp`)
- **Facade/Introspectable GNN:** Research/teaching CLI and C++17 class, with node/edge/gradient inspection and graph utilities (`facaded_gnn.cu`, `facaded_gnn_opencl.cpp`)

No deep learning framework required. All logic, math, and graph ops are implemented from scratch.

---

## Table of Contents

- [Features](#features)
- [Module Types](#module-types)
- [Requirements](#requirements)
- [Quickstart: Compiling & Running](#quickstart-compiling--running)
- [CLI Usage and Help](#cli-usage-and-help)
  - [1. CUDA GNN (gnn.cu)](#1-cuda-gnn-gnncu)
  - [2. OpenCL GNN (gnn_opencl.cpp)](#2-opencl-gnn-gnn_openclcpp)
  - [3. CUDA Facade (facaded_gnn.cu)](#3-cuda-facade-facaded_gnncu)
  - [4. OpenCL Facade (facaded_gnn_opencl.cpp)](#4-opencl-facade-facaded_gnn_openclcpp)
- [Architecture Notes](#architecture-notes)
- [Data Structures & Internals](#data-structures--internals)
- [License](#license)

---

## Features

- CUDA *and* OpenCL support (select at compile-time)
- **Transparent message-passing GNN** — all weights, messages, and gradients are visible
- **Direct and facade paradigms:** Raw programmable GNN and high-level, CLI driven introspection
- **Multiple activation/loss functions:** ReLU, LeakyReLU, Tanh, Sigmoid | MSE, Binary Cross-Entropy
- **Fully GPU-accelerated forward and backward**
- Model save/load, checkpointing, and graph state serialization
- CLI and class exposes: node/edge inspection, degree, PageRank, neighbors, embedding, gradient flow, and more
- Built-in graph preprocessing: undirected conversion, self-loops, edge deduplication
- No PyTorch/TensorFlow required — pure C++/CUDA/OpenCL

---

## Module Types

There are **2 × 2 = 4 alternatives**:

| Type      | Direct/CMD line      | Facade/Introspectable          |
|-----------|----------------------|--------------------------------|
| CUDA      | `gnn.cu`             | `facaded_gnn.cu`               |
| OpenCL    | `gnn_opencl.cpp`     | `facaded_gnn_opencl.cpp`       |

**Direct** = programmable GNN, compile/run program, manual scripting  
**Facade** = research/CLI, Python-style command chains, model/graph internals visible

---

## Requirements

- **CUDA:** NVIDIA GPU (Compute 6.0+ recommended), CUDA toolkit, C++14+ (`gnn.cu`, `facaded_gnn.cu`)
- **OpenCL:** Any OpenCL 1.2+ device, C++11+ (`gnn_opencl.cpp`, `facaded_gnn_opencl.cpp`)
- **Standard build tools:** nvcc/g++, no deep learning framework needed

---

## Quickstart: Compiling & Running

**CUDA:**
```bash
# Direct GNN (classic, programmable)
nvcc -O3 -std=c++14 gnn.cu -o gnn_cuda

# Facade GNN (introspectable, C++17)
nvcc -O3 -std=c++17 facaded_gnn.cu -o facaded_gnn_cuda
```

**OpenCL:**
```bash
# Direct OpenCL GNN
g++ -O2 -std=c++11 -o gnn_opencl gnn_opencl.cpp -lOpenCL

# Facade OpenCL GNN
g++ -O2 -std=c++11 -o facaded_gnn_opencl facaded_gnn_opencl.cpp -lOpenCL
```

---

## CLI Usage and Help

Below are templates for running each GNN mode with command-line arguments, including help output and typical commands.

---

### 1. CUDA GNN (`gnn.cu`)

No built-in CLI — **use as a class/module** or write your own driver (see the file for API and function call usage). Key arguments/types for constructor:

- `featureSize`, `hiddenSize`, `outputSize`, `numMPLayers` (message-passing layers)  
- `SetLearningRate(double)`, `SetActivation(ActivationType)`, `SetLossFunction(LossType)`, ...  
- `Train(TGraph&, target)`, `Predict(TGraph&)`, `SaveModel`, `LoadModel`  
- Graph: Use `TGraph` struct (see code)

Example (in C++):
```cpp
#include "gnn.cu"
TGraphNeuralNetwork gnn(3, 16, 2, 2);
gnn.SetLearningRate(0.01);
gnn.SetLossFunction(ltMSE);
double loss = gnn.Train(your_graph, your_target_array);
auto output = gnn.Predict(your_graph);
```

---

### 2. OpenCL GNN (`gnn_opencl.cpp`)

Run with no args to print help, or use commands below.

```bash
./gnn_opencl help
```

##### **Help Output (abridged):**
```
GNN-OpenCL - Command-line Graph Neural Network (GPU-Accelerated)

Commands:
  create   Create a new GNN model
  train    Train an existing model with graph data
  predict  Make predictions with a trained model
  info     Display model information
  help     Show this help message

Create Options:
  --feature=N          Input feature size (required)
  --hidden=N           Hidden layer size (required)
  --output=N           Output size (required)
  --mp-layers=N        Message passing layers (required)
  --save=FILE          Save model to file (required)
  --lr=VALUE           Learning rate (default: 0.01)
  --activation=TYPE    relu|leakyrelu|tanh|sigmoid
  --loss=TYPE          mse|bce

Train Options:
  --model=FILE         Model file to load (required)
  --graph=FILE         Graph file (JSON format) (required)
  --save=FILE          Save trained model to file (required)
  --epochs=N           Number of training epochs (default: 100)
  --lr=VALUE           Override learning rate
  --verbose            Show training progress

Predict Options:
  --model=FILE         Model file to load (required)
  --graph=FILE         Graph file (required)

Info Options:
  --model=FILE         Model file to load (required)
```

**Examples:**
```bash
# Create a model
./gnn_opencl create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=my_gnn.bin

# Train for 500 epochs
./gnn_opencl train --model=my_gnn.bin --graph=your_graph.json --epochs=500 --save=gnn_trained.bin

# Predict output for a graph
./gnn_opencl predict --model=gnn_trained.bin --graph=your_graph.json

# Get info about a trained model
./gnn_opencl info --model=gnn_trained.bin
```

---

### 3. CUDA Facade (`facaded_gnn.cu`)

No command-line utility. Use as a C++17 class for fully **pythonic graph manipulation/introspection**.

**Key class:** `CUDAGNNFacade`

**Initialization and usage:**
```cpp
#include "facaded_gnn.cu"
CUDAGNNFacade gnn(3, 16, 2, 2);
gnn.createEmptyGraph(numNodes, 3);
gnn.setNodeFeature(0, 0, 1.0f);            // Set a node's feature
gnn.addEdge(0, 1);
auto output = gnn.predict();
float loss = gnn.train({0.1, 0.2});
gnn.saveModel("out.bin");
```

- **Node/edge/graph introspection**:  
  - `getNeighbors(node)`, `getInDegree(node)`, `getOutDegree(node)`, ...
  - `getNodeFeature(node,i)`, `setNodeFeature(node,i,v)` and batch setters
  - `getEdgeEndpoints(edgeIdx)`, `getEdgeFeatures(edgeIdx)`, `setEdgeFeatures(edgeIdx,v)`
  - Architecture/hyperparam info: `getArchitectureSummary()`
  - Graph embedding: `getGraphEmbedding()`
- **Train/save/load**:  
  - `train(target)`, `trainMultiple(target, iters)`, `saveModel`, `loadModel`

---

### 4. OpenCL Facade (`facaded_gnn_opencl.cpp`)

Run for **fully-featured introspectable CLI**:

```bash
./facaded_gnn_opencl      # Prints help/usage summary
./facaded_gnn_opencl help # Detailed help output
```

##### **Help Output (abridged):**
```
GNN-Facade - Graph Neural Network with Facade Pattern (GPU-Accelerated)
Usage:
  facade-gnn <command> [options]

COMMANDS:
  create         Create a new GNN model
  add-node       Add node to the graph
  add-edge       Add an edge
  remove-edge    Remove an edge
  predict        Predict for a graph
  train          Train the model
  degree         Node degree
  in-degree      Node in-degree
  out-degree     Node out-degree
  neighbors      Get node neighbors
  pagerank       Compute PageRank
  save           Save model to file
  load           Load model from file
  info           Model information
  gradient-flow  Gradient flow analysis
  help           Show this help message

NETWORK FUNCTIONS:
  create         --feature=N --hidden=N --output=N --mp-layers=N --model=FILE
  predict        --model=FILE --graph=FILE
  train          --model=FILE --graph=FILE --target=FILE --epochs=N --save=FILE

EXAMPLES:
  facade-gnn create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=my.bin
  facade-gnn degree --model=my.bin --node=0
  facade-gnn pagerank --model=my.bin --damping=0.85 --iterations=20
  facade-gnn train --model=my.bin --graph=my.csv --target=target.csv --epochs=100 --save=trained.bin
  facade-gnn predict --model=trained.bin --graph=my.csv
```

---


## License

MIT License  
© 2025 Matthew Abbott

---
