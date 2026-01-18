# GlassBoxAI-GNN  <img width="1459" height="821" alt="cisa-logo-01" src="https://github.com/user-attachments/assets/a36bfc70-6b57-48f4-89e4-cd8aa44348af" />


## **Graph Neural Network Suite**

### *GPU-Accelerated GNN Implementations with Formal Verification*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-12.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCL](https://img.shields.io/badge/OpenCL-3.0-blue.svg)](https://www.khronos.org/opencl/)
[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![Kani](https://img.shields.io/badge/Kani-Verified-brightgreen.svg)](https://model-checking.github.io/kani/)
[![CISA Compliant](https://img.shields.io/badge/CISA-Secure%20by%20Design-blue.svg)](https://www.cisa.gov/securebydesign)

---

## **Overview**

GlassBoxAI-GNN is a comprehensive, production-ready Graph Neural Network implementation suite featuring:

- **Multiple GPU backends**: CUDA and OpenCL acceleration
- **Multiple language implementations**: C++ and Rust
- **Facade pattern architecture**: Clean API separation for maintainability
- **Formal verification**: Kani-verified Rust implementation for memory safety guarantees
- **CISA/NSA Secure by Design compliance**: Built following government cybersecurity standards

This project demonstrates enterprise-grade software engineering practices including comprehensive testing, formal verification, cross-platform compatibility, and security-first development.

---

## **Table of Contents**

1. [Features](#features)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Prerequisites](#prerequisites)
5. [Installation & Compilation](#installation--compilation)
6. [CLI Reference](#cli-reference)
   - [Standard GNN Commands](#standard-gnn-commands)
   - [Facade GNN Commands](#facade-gnn-commands)
7. [Testing](#testing)
8. [Formal Verification with Kani](#formal-verification-with-kani)
9. [CISA/NSA Compliance](#cisansa-compliance)
10. [License](#license)
11. [Author](#author)

---

## **Features**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Message Passing** | Configurable multi-layer message passing neural network |
| **Graph Operations** | PageRank, degree analysis, neighbor queries |
| **Training** | Backpropagation with gradient clipping |
| **Activation Functions** | ReLU, LeakyReLU, Tanh, Sigmoid |
| **Loss Functions** | MSE, Binary Cross-Entropy |
| **Model Persistence** | Binary serialization for model save/load |

### GPU Acceleration

| Backend | Implementation | Performance |
|---------|---------------|-------------|
| **CUDA** | Native CUDA kernels | Optimal for NVIDIA GPUs |
| **OpenCL** | Cross-platform GPU | AMD, Intel, NVIDIA support |

### Safety & Security

| Feature | Technology |
|---------|------------|
| **Memory Safety** | Rust ownership model |
| **Formal Verification** | Kani proof harnesses |
| **Bounds Checking** | Verified array access |
| **Input Validation** | CLI argument validation |

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        GlassBoxAI-GNN                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   C++ CUDA  │  │ C++ OpenCL  │  │       Rust CUDA         │  │
│  ├─────────────┤  ├─────────────┤  ├─────────────────────────┤  │
│  │ • gnn.cu    │  │ • gnn_      │  │ • rust_cuda/            │  │
│  │ • facaded_  │  │   opencl.cpp│  │ • rust_cuda_facade/     │  │
│  │   gnn.cu    │  │ • facaded_  │  │   └─ kani_proofs/       │  │
│  │             │  │   gnn_      │  │      (Formal Verify)    │  │
│  │             │  │   opencl.cpp│  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    Shared Features                          ││
│  │  • Consistent CLI interface across all implementations      ││
│  │  • Binary-compatible model format                           ││
│  │  • Comprehensive test suites                                ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## **File Structure**

```
GlassBoxAI-GNN/
│
├── gnn.cu                      # C++ CUDA GNN implementation
├── gnn_opencl.cpp              # C++ OpenCL GNN implementation
├── facaded_gnn.cu              # C++ CUDA GNN with Facade pattern
├── facaded_gnn_opencl.cpp      # C++ OpenCL GNN with Facade pattern
│
├── rust_cuda/                  # Rust CUDA GNN implementation
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
│
├── rust_cuda_facade/           # Rust CUDA GNN with Facade pattern
│   ├── Cargo.toml
│   ├── src/
│   │   └── main.rs
│   ├── gui/                    # Qt-based GUI (optional)
|       └── kani_proofs         # Formal verification proofs
│       |   ├── Cargo.toml
│       |   ├── src/
│       |   │   └── lib.rs
│       ├── README.md
│       └── VERIFICATION_REPORT.md
│   │   └── src/
│   │       └── main.rs
│   └── kani_proofs/            # Formal verification proofs
│       ├── Cargo.toml
│       ├── src/
│       │   └── lib.rs
│       ├── README.md
│       └── VERIFICATION_REPORT.md
│
├── test_gnn_cuda.sh            # CUDA test suite
├── test_gnn_opencl.sh          # OpenCL test suite
├── test_gnn_rust_cuda.sh       # Rust CUDA test suite
│
├── index.html                  # Project documentation
├── license.md                  # MIT License
└── README.md                   # This file
```

---

## **Prerequisites**

### Required

| Dependency | Version | Purpose |
|------------|---------|---------|
| **GCC/G++** | 11+ | C++ compilation |
| **CUDA Toolkit** | 12.0+ | CUDA compilation |
| **Rust** | 1.75+ | Rust compilation |

### Optional

| Dependency | Version | Purpose |
|------------|---------|---------|
| **OpenCL SDK** | 3.0 | OpenCL compilation |
| **Kani** | 0.67+ | Formal verification |
| **Qt 6** | 6.x | GUI version |

---

## **Installation & Compilation**

### **C++ CUDA Implementation**

```bash
# Standard GNN
nvcc -std=c++17 -o gnn_cuda gnn.cu

# Facade GNN
nvcc -std=c++17 -o facaded_gnn_cuda facaded_gnn.cu
```

### **C++ OpenCL Implementation**

```bash
# Standard GNN
g++ -std=c++17 -o gnn_opencl gnn_opencl.cpp -lOpenCL

# Facade GNN
g++ -std=c++17 -o facaded_gnn_opencl facaded_gnn_opencl.cpp -lOpenCL
```

### **Rust CUDA Implementation**

```bash
# Standard GNN
cd rust_cuda
cargo build --release

# Facade GNN
cd rust_cuda_facade
cargo build --release
```

### **Build All**

```bash
# Build everything
nvcc -std=c++17 -o gnn_cuda gnn.cu
nvcc -std=c++17 -o facaded_gnn_cuda facaded_gnn.cu
g++ -std=c++17 -o gnn_opencl gnn_opencl.cpp -lOpenCL
g++ -std=c++17 -o facaded_gnn_opencl facaded_gnn_opencl.cpp -lOpenCL
(cd rust_cuda && cargo build --release)
(cd rust_cuda_facade && cargo build --release)
```

---

## **CLI Reference**

### **Standard GNN Commands**

The standard GNN implementations provide core neural network functionality.

#### Usage

```
gnn_cuda <command> [options]
gnn_opencl <command> [options]
gnn_cuda (Rust) <command> [options]
```

#### Commands

| Command | Description |
|---------|-------------|
| `create` | Create a new GNN model |
| `train` | Train the model with graph data |
| `predict` | Make predictions on a graph |
| `info` | Display model information |
| `save` | Save model to file |
| `load` | Load model from file |
| `degree` | Get node degree |
| `in-degree` | Get node in-degree |
| `out-degree` | Get node out-degree |
| `neighbors` | Get node neighbors |
| `pagerank` | Compute PageRank scores |
| `gradient-flow` | Show gradient flow analysis |
| `add-node` | Add a node to the graph |
| `add-edge` | Add an edge to the graph |
| `remove-edge` | Remove an edge from the graph |
| `help` | Show help message |

#### Network Functions

**create**
```
--feature=N          Input feature dimension (required)
--hidden=N           Hidden layer dimension (required)
--output=N           Output dimension (required)
--mp-layers=N        Message passing layers (required)
--save=FILE          Save initial model to file (required)
--lr=VALUE           Learning rate (default: 0.01)
--activation=TYPE    relu|leakyrelu|tanh|sigmoid (default: relu)
--loss=TYPE          mse|bce (default: mse)
```

**train**
```
--model=FILE         Model file (required)
--graph=FILE         Graph file in JSON format (required)
--save=FILE          Save trained model to file
--epochs=N           Training epochs (default: 100)
--lr=VALUE           Override learning rate
```

**predict**
```
--model=FILE         Model file (required)
--graph=FILE         Graph file in JSON format (required)
```

#### Graph Functions

**degree / in-degree / out-degree**
```
--model=FILE         Model file (required)
--node=N             Node index (required)
```

**neighbors**
```
--model=FILE         Model file (required)
--node=N             Node index (required)
```

**pagerank**
```
--model=FILE         Model file (required)
--damping=D          Damping factor (default: 0.85)
--iterations=N       Iterations (default: 20)
```

**gradient-flow**
```
--model=FILE         Model file (required)
--layer=N            Layer index (optional)
```

#### Examples

```bash
# Create a new model
gnn_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=model.bin

# Get node degree
gnn_cuda degree --model=model.bin --node=0

# Compute PageRank
gnn_cuda pagerank --model=model.bin --damping=0.85 --iterations=20

# Train the model
gnn_cuda train --model=model.bin --graph=graph.json --save=trained.bin --epochs=100

# Make predictions
gnn_cuda predict --model=trained.bin --graph=graph.json
```

---

### **Facade GNN Commands**

The Facade implementations provide an extended API with additional graph manipulation, masking, and analysis capabilities.

#### Usage

```
facaded_gnn_cuda <command> [options]
facaded_gnn_opencl <command> [options]
gnn_facade_cuda (Rust) <command> [options]
```

#### All Standard Commands Plus:

##### Graph Structure Commands

| Command | Description |
|---------|-------------|
| `create-graph <nodes> <features>` | Create empty graph with N nodes and feature dim |
| `load-graph <nodes.csv> <edges.csv>` | Load graph from CSV files |
| `save-graph <nodes.csv> <edges.csv>` | Save graph to CSV files |
| `export-json` | Export graph as JSON |

##### Node Operations

| Command | Description |
|---------|-------------|
| `get-node-feature <node_idx> <feature_idx>` | Get single node feature value |
| `set-node-feature <node_idx> <feature_idx> <value>` | Set single node feature value |
| `get-node-features <node_idx>` | Get all features for a node |
| `set-node-features <node_idx> <v1,v2,...>` | Set all features for a node |
| `get-neighbors <node_idx>` | Get neighbor node indices |
| `get-in-degree <node_idx>` | Get node in-degree |
| `get-out-degree <node_idx>` | Get node out-degree |
| `get-num-nodes` | Get total number of nodes |

##### Edge Operations

| Command | Description |
|---------|-------------|
| `add-edge <src> <tgt> [features]` | Add edge with optional features |
| `remove-edge <edge_idx>` | Remove edge by index |
| `get-edge-endpoints <edge_idx>` | Get source and target of edge |
| `has-edge <source> <target>` | Check if edge exists |
| `get-num-edges` | Get total number of edges |

##### Masking/Dropout

| Command | Description |
|---------|-------------|
| `set-node-mask <node_idx> <true\|false>` | Set node mask (true=active) |
| `set-edge-mask <edge_idx> <true\|false>` | Set edge mask (true=active) |
| `apply-node-dropout <rate>` | Apply random node dropout (0.0-1.0) |
| `apply-edge-dropout <rate>` | Apply random edge dropout (0.0-1.0) |

##### Model Analysis

| Command | Description |
|---------|-------------|
| `get-node-embedding <layer_idx> <node_idx>` | Get node embedding at layer |
| `get-activation-histogram <layer_idx> [num_bins]` | Get activation distribution |
| `get-parameter-count` | Get total trainable parameters |
| `get-gradient-flow <layer_idx>` | Get gradient flow info for layer |
| `compute-loss <pred1,pred2,...> <target1,target2,...>` | Compute loss between arrays |
| `compute-pagerank [damping] [iterations]` | Compute PageRank (default: 0.85, 100) |
| `export-embeddings <layer_idx> <output.csv>` | Export embeddings to CSV |
| `get-architecture` | Show model architecture summary |
| `get-graph-embedding` | Get graph-level embedding |

##### Configuration

| Command | Description |
|---------|-------------|
| `set-activation <relu\|leaky_relu\|tanh\|sigmoid>` | Set activation function |
| `set-loss <mse\|bce>` | Set loss function |
| `set-learning-rate <val>` | Set learning rate |
| `get-learning-rate` | Get current learning rate |

##### Training

| Command | Description |
|---------|-------------|
| `predict` | Run forward pass and get output |
| `train <t1,t2,...>` | Train one step with targets |
| `train-multiple <iters> <t1,t2,...>` | Train multiple iterations |

#### Facade Examples

```bash
# Create a new model
facaded_gnn_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=model.bin

# Create and manipulate a graph
facaded_gnn_cuda create-graph 5 3
facaded_gnn_cuda add-edge 0 1
facaded_gnn_cuda set-node-features 0 1.0,2.0,3.0

# Apply dropout and predict
facaded_gnn_cuda apply-node-dropout 0.2
facaded_gnn_cuda predict

# Get node degree
facaded_gnn_cuda get-in-degree 0
facaded_gnn_cuda get-out-degree 0

# Compute PageRank
facaded_gnn_cuda compute-pagerank 0.85 100

# Train the model
facaded_gnn_cuda train 0.5,0.3
facaded_gnn_cuda train-multiple 100 0.5,0.3
```

---

## **Testing**

### Running All Tests

```bash
# Run CUDA tests
./test_gnn_cuda.sh

# Run OpenCL tests
./test_gnn_opencl.sh

# Run Rust CUDA tests
./test_gnn_rust_cuda.sh
```

### Test Categories

Each test suite covers:

| Category | Tests |
|----------|-------|
| **Help & Usage** | Command-line interface verification |
| **Model Creation** | Various architecture configurations |
| **Hyperparameters** | Learning rate, activation, loss functions |
| **Model Info** | Metadata retrieval |
| **Save & Load** | Model persistence |
| **Graph Operations** | Degree, neighbors, PageRank |
| **Edge Operations** | Add/remove nodes and edges |
| **Gradient Flow** | Backpropagation analysis |
| **Error Handling** | Invalid input handling |
| **Cross-Implementation** | API compatibility |
| **Train & Predict** | End-to-end workflows |
| **Stress Tests** | Large configurations |

### Test Output Example

```
=========================================
GNN CUDA Comprehensive Test Suite
=========================================

Group: Help & Usage
Test 1: GNN help command... PASS
Test 2: GNN --help flag... PASS
Test 3: GNN -h flag... PASS
...

=========================================
Test Summary
=========================================
Total tests: 75
Passed: 75
Failed: 0

All tests passed!
```

---

## **Formal Verification with Kani**

### Overview

The Rust Facade implementation includes **Kani formal verification proofs** that mathematically prove the absence of certain classes of bugs. This goes beyond traditional testing to provide **mathematical guarantees** about code correctness.

### Verification Report

| Metric | Value |
|--------|-------|
| **Unit Tests** | 76 |
| **Kani Proof Harnesses** | 19 |
| **Total Verifications** | **95** |
| **Failures** | 0 |

### Kani Proof Harnesses

#### Node Feature Access (3 proofs)
- `proof_get_node_feature_never_panics` ✓
- `proof_set_node_feature_never_panics` ✓
- `proof_get_node_features_never_panics` ✓

#### Edge Operations (5 proofs)
- `proof_get_edge_bounds_safe` ✓
- `proof_add_edge_bounds_checked` ✓
- `proof_has_edge_never_panics` ✓
- `proof_find_edge_index_never_panics` ✓
- `proof_remove_edge_never_panics` ✓

#### Adjacency List (3 proofs)
- `proof_get_neighbors_never_panics` ✓
- `proof_get_in_degree_never_panics` ✓
- `proof_get_out_degree_never_panics` ✓

#### Node Mask Operations (2 proofs)
- `proof_node_mask_get_set_never_panic` ✓
- `proof_node_mask_toggle_never_panics` ✓

#### Edge Mask Operations (2 proofs)
- `proof_edge_mask_get_set_never_panic` ✓
- `proof_edge_mask_remove_never_panics` ✓

#### Buffer Index Validation (4 proofs)
- `proof_buffer_validator_node_correctness` ✓
- `proof_buffer_validator_edge_correctness` ✓
- `proof_node_feature_offset_bounds` ✓
- `proof_node_embedding_offset_bounds` ✓

### Running Kani Verification

```bash
cd rust_cuda_facade/kani_proofs

# Run all proofs
cargo kani

# Run specific proof
cargo kani --harness proof_get_node_feature_never_panics

# Run unit tests
cargo test
```

### Why Formal Verification Matters

Traditional testing can only verify specific test cases. Formal verification with Kani:

- **Exhaustively checks all possible inputs** within defined bounds
- **Mathematically proves** absence of panics, buffer overflows, and undefined behavior
- **Catches edge cases** that random testing might miss
- **Provides cryptographic-level assurance** for safety-critical code

---

## **CISA/NSA Compliance**

### Secure by Design

This project follows **CISA (Cybersecurity and Infrastructure Security Agency)** and **NSA (National Security Agency)** Secure by Design principles:

| Principle | Implementation |
|-----------|---------------|
| **Memory Safety** | Rust ownership model eliminates buffer overflows, use-after-free, and data races |
| **Formal Verification** | Kani proofs mathematically verify absence of critical bugs |
| **Input Validation** | All CLI inputs validated before processing |
| **Defense in Depth** | Multiple layers of safety (language, compiler, runtime checks) |
| **Secure Defaults** | Safe default configurations throughout |
| **Transparency** | Open source with full code visibility |

### Compliance Checklist

- [x] **Memory-safe language** (Rust implementation)
- [x] **Static analysis** (Rust compiler + Clippy)
- [x] **Formal verification** (Kani proof harnesses)
- [x] **Comprehensive testing** (Unit tests + integration tests)
- [x] **Bounds checking** (Verified array access)
- [x] **Input validation** (CLI argument parsing)
- [x] **No unsafe code in critical paths** (Where possible)
- [x] **Documentation** (Inline docs + README)
- [x] **Version control** (Git)
- [x] **License clarity** (MIT License)

### Attestation

This codebase has been developed following secure software development lifecycle (SSDLC) practices and demonstrates:

- **95 formal verifications passed** (76 unit tests + 19 Kani proofs)
- **Zero warnings** compilation across all implementations
- **Consistent API** across all language/backend combinations
- **Production-ready** code quality

---

## **License**

MIT License

Copyright (c) 2025 Matthew Abbott

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## **Author**

**Matthew Abbott**  
Email: mattbachg@gmail.com

---

*Built with precision. Verified with rigor. Secured by design.*
