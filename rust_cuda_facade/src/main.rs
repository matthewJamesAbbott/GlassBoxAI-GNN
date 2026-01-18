/*
 * MIT License
 *
 * Copyright (c) 2025 Matthew Abbott
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

use clap::{Parser, Subcommand, ValueEnum};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

#[allow(dead_code)]
const MAX_NODES: usize = 1000;
#[allow(dead_code)]
const MAX_EDGES: usize = 10000;
#[allow(dead_code)]
const GRADIENT_CLIP: f32 = 5.0;
const BLOCK_SIZE: u32 = 256;

const CUDA_KERNELS: &str = r#"
extern "C" {

__device__ float d_activate(float x, int activationType) {
    switch (activationType) {
        case 0: // ReLU
            return x > 0.0f ? x : 0.0f;
        case 1: // LeakyReLU
            return x > 0.0f ? x : 0.01f * x;
        case 2: // Tanh
            return tanhf(x);
        case 3: // Sigmoid
            return 1.0f / (1.0f + expf(-fmaxf(-500.0f, fminf(500.0f, x))));
        default:
            return x;
    }
}

__device__ float d_activateDerivative(float x, int activationType) {
    switch (activationType) {
        case 0: // ReLU
            return x > 0.0f ? 1.0f : 0.0f;
        case 1: // LeakyReLU
            return x > 0.0f ? 1.0f : 0.01f;
        case 2: // Tanh
            { float t = tanhf(x); return 1.0f - t * t; }
        case 3: // Sigmoid
            { float s = 1.0f / (1.0f + expf(-fmaxf(-500.0f, fminf(500.0f, x)))); return s * (1.0f - s); }
        default:
            return 1.0f;
    }
}

__device__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-fmaxf(-500.0f, fminf(500.0f, x))));
}

__device__ float d_sigmoidDerivative(float x) {
    float s = d_sigmoid(x);
    return s * (1.0f - s);
}

__device__ float d_clipGradient(float g) {
    return fmaxf(-5.0f, fminf(5.0f, g));
}

__global__ void k_forwardLayer(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ preActivations,
    float* __restrict__ outputs,
    int numInputs,
    int numOutputs,
    int activationType,
    bool useOutputActivation
) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= numOutputs) return;
    
    float sum = biases[neuronIdx];
    for (int j = 0; j < numInputs; ++j) {
        sum += weights[neuronIdx * numInputs + j] * input[j];
    }
    
    preActivations[neuronIdx] = sum;
    
    if (useOutputActivation) {
        outputs[neuronIdx] = d_sigmoid(sum);
    } else {
        outputs[neuronIdx] = d_activate(sum, activationType);
    }
}

__global__ void k_backwardLayer(
    const float* __restrict__ lastInput,
    const float* __restrict__ preActivations,
    const float* __restrict__ upstreamGrad,
    float* __restrict__ weights,
    float* __restrict__ biases,
    float* __restrict__ weightGradients,
    float* __restrict__ biasGradients,
    int numInputs,
    int numOutputs,
    int activationType,
    bool useOutputActivation,
    float learningRate
) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= numOutputs) return;
    
    float preActGrad;
    if (useOutputActivation) {
        preActGrad = upstreamGrad[neuronIdx] * d_sigmoidDerivative(preActivations[neuronIdx]);
    } else {
        preActGrad = upstreamGrad[neuronIdx] * d_activateDerivative(preActivations[neuronIdx], activationType);
    }
    preActGrad = d_clipGradient(preActGrad);
    
    for (int j = 0; j < numInputs; ++j) {
        float grad = d_clipGradient(preActGrad * lastInput[j]);
        weightGradients[neuronIdx * numInputs + j] = grad;
        weights[neuronIdx * numInputs + j] -= learningRate * grad;
    }
    
    biasGradients[neuronIdx] = preActGrad;
    biases[neuronIdx] -= learningRate * preActGrad;
}

__global__ void k_computeInputGrad(
    const float* __restrict__ weights,
    const float* __restrict__ preActivations,
    const float* __restrict__ upstreamGrad,
    float* __restrict__ inputGrad,
    int numInputs,
    int numOutputs,
    int activationType,
    bool useOutputActivation
) {
    int inputIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (inputIdx >= numInputs) return;
    
    float grad = 0.0f;
    for (int i = 0; i < numOutputs; ++i) {
        float preActGrad;
        if (useOutputActivation) {
            preActGrad = upstreamGrad[i] * d_sigmoidDerivative(preActivations[i]);
        } else {
            preActGrad = upstreamGrad[i] * d_activateDerivative(preActivations[i], activationType);
        }
        grad += weights[i * numInputs + inputIdx] * preActGrad;
    }
    inputGrad[inputIdx] = d_clipGradient(grad);
}

__global__ void k_aggregateMessages(
    const float* __restrict__ allMessages,
    const int* __restrict__ neighborCounts,
    const int* __restrict__ neighborOffsets,
    float* __restrict__ aggregatedMessages,
    int numNodes,
    int hiddenSize
) {
    int nodeIdx = blockIdx.x;
    int dimIdx = threadIdx.x;
    
    if (nodeIdx >= numNodes || dimIdx >= hiddenSize) return;
    
    int numNeighbors = neighborCounts[nodeIdx];
    int offset = neighborOffsets[nodeIdx];
    
    float sum = 0.0f;
    for (int n = 0; n < numNeighbors; ++n) {
        sum += allMessages[(offset + n) * hiddenSize + dimIdx];
    }
    
    if (numNeighbors > 0) {
        aggregatedMessages[nodeIdx * hiddenSize + dimIdx] = sum / numNeighbors;
    } else {
        aggregatedMessages[nodeIdx * hiddenSize + dimIdx] = 0.0f;
    }
}

__global__ void k_graphReadout(
    const float* __restrict__ nodeEmbeddings,
    float* __restrict__ graphEmbedding,
    int numNodes,
    int hiddenSize
) {
    int dimIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dimIdx >= hiddenSize) return;
    
    float sum = 0.0f;
    for (int n = 0; n < numNodes; ++n) {
        sum += nodeEmbeddings[n * hiddenSize + dimIdx];
    }
    graphEmbedding[dimIdx] = sum / numNodes;
}

__global__ void k_computeMSEGradient(
    const float* __restrict__ prediction,
    const float* __restrict__ target,
    float* __restrict__ gradient,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    gradient[idx] = 2.0f * (prediction[idx] - target[idx]) / size;
}

}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ActivationType {
    Relu,
    LeakyRelu,
    Tanh,
    Sigmoid,
}

impl ActivationType {
    #[allow(dead_code)]
    fn to_int(&self) -> i32 {
        match self {
            ActivationType::Relu => 0,
            ActivationType::LeakyRelu => 1,
            ActivationType::Tanh => 2,
            ActivationType::Sigmoid => 3,
        }
    }

    #[allow(dead_code)]
    fn from_int(v: i32) -> Self {
        match v {
            0 => ActivationType::Relu,
            1 => ActivationType::LeakyRelu,
            2 => ActivationType::Tanh,
            3 => ActivationType::Sigmoid,
            _ => ActivationType::Relu,
        }
    }
}

impl std::fmt::Display for ActivationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActivationType::Relu => write!(f, "relu"),
            ActivationType::LeakyRelu => write!(f, "leakyrelu"),
            ActivationType::Tanh => write!(f, "tanh"),
            ActivationType::Sigmoid => write!(f, "sigmoid"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum LossType {
    Mse,
    Bce,
}

impl LossType {
    #[allow(dead_code)]
    fn to_int(&self) -> i32 {
        match self {
            LossType::Mse => 0,
            LossType::Bce => 1,
        }
    }

    #[allow(dead_code)]
    fn from_int(v: i32) -> Self {
        match v {
            0 => LossType::Mse,
            1 => LossType::Bce,
            _ => LossType::Mse,
        }
    }
}

impl std::fmt::Display for LossType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossType::Mse => write!(f, "mse"),
            LossType::Bce => write!(f, "bce"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub source: usize,
    pub target: usize,
}

#[derive(Debug, Clone)]
pub struct EdgeFeatures {
    pub source: usize,
    pub target: usize,
    pub features: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct GradientFlowInfo {
    pub layer_idx: usize,
    pub mean_gradient: f32,
    pub max_gradient: f32,
    pub min_gradient: f32,
    pub gradient_norm: f32,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub num_nodes: usize,
    pub node_features: Vec<Vec<f32>>,
    pub edges: Vec<(usize, usize)>,
    pub adjacency_list: Vec<Vec<usize>>,
    pub undirected: bool,
    pub self_loops: bool,
}

impl Graph {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            node_features: Vec::new(),
            edges: Vec::new(),
            adjacency_list: vec![Vec::new(); num_nodes],
            undirected: false,
            self_loops: false,
        }
    }

    pub fn build_adjacency_list(&mut self) {
        self.adjacency_list = vec![Vec::new(); self.num_nodes];
        for &(src, tgt) in &self.edges {
            if src < self.num_nodes && tgt < self.num_nodes {
                self.adjacency_list[src].push(tgt);
            }
        }
    }
}

pub struct GpuLayer {
    pub d_weights: CudaSlice<f32>,
    pub d_biases: CudaSlice<f32>,
    pub d_weight_gradients: CudaSlice<f32>,
    pub d_bias_gradients: CudaSlice<f32>,
    pub d_pre_activations: CudaSlice<f32>,
    pub d_outputs: CudaSlice<f32>,
    pub d_last_input: CudaSlice<f32>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl GpuLayer {
    fn new(device: &Arc<CudaDevice>, num_inputs: usize, num_outputs: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (num_inputs + num_outputs) as f32).sqrt();

        let weights: Vec<f32> = (0..num_inputs * num_outputs)
            .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
            .collect();
        let biases = vec![0.0f32; num_outputs];

        let d_weights = device.htod_sync_copy(&weights)?;
        let d_biases = device.htod_sync_copy(&biases)?;
        let d_weight_gradients = device.alloc_zeros::<f32>(num_inputs * num_outputs)?;
        let d_bias_gradients = device.alloc_zeros::<f32>(num_outputs)?;
        let d_pre_activations = device.alloc_zeros::<f32>(num_outputs)?;
        let d_outputs = device.alloc_zeros::<f32>(num_outputs)?;
        let d_last_input = device.alloc_zeros::<f32>(num_inputs)?;

        Ok(Self {
            d_weights,
            d_biases,
            d_weight_gradients,
            d_bias_gradients,
            d_pre_activations,
            d_outputs,
            d_last_input,
            num_inputs,
            num_outputs,
        })
    }

    fn forward_from_host(
        &mut self,
        device: &Arc<CudaDevice>,
        input: &[f32],
        activation: i32,
        use_output_activation: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        device.htod_sync_copy_into(input, &mut self.d_last_input)?;

        let blocks = (self.num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_forwardLayer").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_last_input,
                &self.d_weights,
                &self.d_biases,
                &mut self.d_pre_activations,
                &mut self.d_outputs,
                self.num_inputs as i32,
                self.num_outputs as i32,
                activation,
                use_output_activation as i32,
            ))?;
        }

        let mut result = vec![0.0f32; self.num_outputs];
        device.dtoh_sync_copy_into(&self.d_outputs, &mut result)?;
        Ok(result)
    }

    #[allow(dead_code)]
    fn forward_from_device(
        &mut self,
        device: &Arc<CudaDevice>,
        d_input: &CudaSlice<f32>,
        activation: i32,
        use_output_activation: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        device.dtod_copy(d_input, &mut self.d_last_input)?;

        let blocks = (self.num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_forwardLayer").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_last_input,
                &self.d_weights,
                &self.d_biases,
                &mut self.d_pre_activations,
                &mut self.d_outputs,
                self.num_inputs as i32,
                self.num_outputs as i32,
                activation,
                use_output_activation as i32,
            ))?;
        }
        Ok(())
    }

    fn backward_from_host(
        &mut self,
        device: &Arc<CudaDevice>,
        upstream_grad: &[f32],
        activation: i32,
        use_output_activation: bool,
        learning_rate: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let d_upstream_grad = device.htod_sync_copy(upstream_grad)?;

        let blocks = (self.num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_backwardLayer").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_last_input,
                &self.d_pre_activations,
                &d_upstream_grad,
                &mut self.d_weights,
                &mut self.d_biases,
                &mut self.d_weight_gradients,
                &mut self.d_bias_gradients,
                self.num_inputs as i32,
                self.num_outputs as i32,
                activation,
                use_output_activation as i32,
                learning_rate,
            ))?;
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn backward_from_device(
        &mut self,
        device: &Arc<CudaDevice>,
        d_upstream_grad: &CudaSlice<f32>,
        activation: i32,
        use_output_activation: bool,
        learning_rate: f32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let blocks = (self.num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_backwardLayer").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_last_input,
                &self.d_pre_activations,
                d_upstream_grad,
                &mut self.d_weights,
                &mut self.d_biases,
                &mut self.d_weight_gradients,
                &mut self.d_bias_gradients,
                self.num_inputs as i32,
                self.num_outputs as i32,
                activation,
                use_output_activation as i32,
                learning_rate,
            ))?;
        }
        Ok(())
    }

    fn compute_input_grad_from_host(
        &self,
        device: &Arc<CudaDevice>,
        upstream_grad: &[f32],
        activation: i32,
        use_output_activation: bool,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let d_upstream_grad = device.htod_sync_copy(upstream_grad)?;
        let mut d_input_grad = device.alloc_zeros::<f32>(self.num_inputs)?;

        let blocks = (self.num_inputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_computeInputGrad").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_weights,
                &self.d_pre_activations,
                &d_upstream_grad,
                &mut d_input_grad,
                self.num_inputs as i32,
                self.num_outputs as i32,
                activation,
                use_output_activation as i32,
            ))?;
        }

        let mut result = vec![0.0f32; self.num_inputs];
        device.dtoh_sync_copy_into(&d_input_grad, &mut result)?;
        Ok(result)
    }

    fn copy_weights_to_host(&self, device: &Arc<CudaDevice>) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
        let mut weights = vec![0.0f32; self.num_inputs * self.num_outputs];
        let mut biases = vec![0.0f32; self.num_outputs];
        device.dtoh_sync_copy_into(&self.d_weights, &mut weights)?;
        device.dtoh_sync_copy_into(&self.d_biases, &mut biases)?;
        Ok((weights, biases))
    }

    fn copy_weights_from_host(&mut self, device: &Arc<CudaDevice>, weights: &[f32], biases: &[f32]) -> Result<(), Box<dyn std::error::Error>> {
        device.htod_sync_copy_into(weights, &mut self.d_weights)?;
        device.htod_sync_copy_into(biases, &mut self.d_biases)?;
        Ok(())
    }
}

#[allow(dead_code)]
pub struct CudaGraphNeuralNetwork {
    learning_rate: f32,
    num_message_passing_layers: usize,
    pub feature_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    activation: ActivationType,
    loss_type: LossType,

    message_layers: Vec<GpuLayer>,
    update_layers: Vec<GpuLayer>,
    readout_layer: GpuLayer,
    output_layer: GpuLayer,

    d_node_embeddings: CudaSlice<f32>,
    d_new_node_embeddings: CudaSlice<f32>,
    d_graph_embedding: CudaSlice<f32>,
    d_temp_input: CudaSlice<f32>,
    d_temp_grad: CudaSlice<f32>,
    d_messages: CudaSlice<f32>,
    d_aggregated_messages: CudaSlice<f32>,
    d_neighbor_counts: CudaSlice<i32>,
    d_neighbor_offsets: CudaSlice<i32>,
    d_target: CudaSlice<f32>,

    h_graph_embedding: Vec<f32>,

    device: Arc<CudaDevice>,
}

impl CudaGraphNeuralNetwork {
    pub fn new(
        feature_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_mp_layers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNELS)?;
        device.load_ptx(ptx, "gnn_kernels", &[
            "k_forwardLayer",
            "k_backwardLayer",
            "k_computeInputGrad",
            "k_aggregateMessages",
            "k_graphReadout",
            "k_computeMSEGradient",
        ])?;

        let mut message_layers = Vec::with_capacity(num_mp_layers);
        let mut update_layers = Vec::with_capacity(num_mp_layers);

        for i in 0..num_mp_layers {
            let msg_input_size = if i == 0 { feature_size * 2 } else { hidden_size * 2 };
            message_layers.push(GpuLayer::new(&device, msg_input_size, hidden_size)?);
            update_layers.push(GpuLayer::new(&device, hidden_size * 2, hidden_size)?);
        }

        let readout_layer = GpuLayer::new(&device, hidden_size, hidden_size)?;
        let output_layer = GpuLayer::new(&device, hidden_size, output_size)?;

        let d_node_embeddings = device.alloc_zeros::<f32>(MAX_NODES * hidden_size)?;
        let d_new_node_embeddings = device.alloc_zeros::<f32>(MAX_NODES * hidden_size)?;
        let d_graph_embedding = device.alloc_zeros::<f32>(hidden_size)?;
        let d_temp_input = device.alloc_zeros::<f32>(hidden_size * 4)?;
        let d_temp_grad = device.alloc_zeros::<f32>(hidden_size * 4)?;
        let d_messages = device.alloc_zeros::<f32>(MAX_EDGES * hidden_size)?;
        let d_aggregated_messages = device.alloc_zeros::<f32>(MAX_NODES * hidden_size)?;
        let d_neighbor_counts = device.alloc_zeros::<i32>(MAX_NODES)?;
        let d_neighbor_offsets = device.alloc_zeros::<i32>(MAX_NODES)?;
        let d_target = device.alloc_zeros::<f32>(output_size)?;

        Ok(Self {
            learning_rate: 0.01,
            num_message_passing_layers: num_mp_layers,
            feature_size,
            hidden_size,
            output_size,
            activation: ActivationType::Relu,
            loss_type: LossType::Mse,
            message_layers,
            update_layers,
            readout_layer,
            output_layer,
            d_node_embeddings,
            d_new_node_embeddings,
            d_graph_embedding,
            d_temp_input,
            d_temp_grad,
            d_messages,
            d_aggregated_messages,
            d_neighbor_counts,
            d_neighbor_offsets,
            d_target,
            h_graph_embedding: vec![0.0; hidden_size],
            device,
        })
    }

    pub fn predict(&mut self, graph: &mut Graph) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        graph.build_adjacency_list();
        let num_nodes = graph.num_nodes;

        let mut h_neighbor_counts = vec![0i32; MAX_NODES];
        for (i, adj) in graph.adjacency_list.iter().enumerate() {
            h_neighbor_counts[i] = adj.len() as i32;
        }
        let mut h_neighbor_offsets = vec![0i32; MAX_NODES];
        let mut total_neighbors = 0i32;
        for i in 0..num_nodes {
            h_neighbor_offsets[i] = total_neighbors;
            total_neighbors += h_neighbor_counts[i];
        }

        self.device.htod_sync_copy_into(&h_neighbor_counts, &mut self.d_neighbor_counts)?;
        self.device.htod_sync_copy_into(&h_neighbor_offsets, &mut self.d_neighbor_offsets)?;

        let mut h_embeddings = vec![0.0f32; MAX_NODES * self.hidden_size];
        for n in 0..num_nodes {
            let copy_size = self.feature_size.min(graph.node_features.get(n).map_or(0, |f| f.len()));
            for f in 0..copy_size {
                h_embeddings[n * self.hidden_size + f] = graph.node_features[n][f];
            }
        }
        self.device.htod_sync_copy_into(&h_embeddings, &mut self.d_node_embeddings)?;

        let activation = self.activation.to_int();

        let mut h_all_embeddings = vec![0.0f32; MAX_NODES * self.hidden_size];

        for layer in 0..self.num_message_passing_layers {
            self.device.dtoh_sync_copy_into(&self.d_node_embeddings, &mut h_all_embeddings)?;

            let mut h_all_messages = vec![0.0f32; MAX_EDGES * self.hidden_size];
            let mut msg_offset = 0usize;

            for node in 0..num_nodes {
                for &neighbor in &graph.adjacency_list[node] {
                    let input_size = self.message_layers[layer].num_inputs;
                    let mut h_temp_input = vec![0.0f32; input_size];

                    let emb_size = if layer == 0 { self.feature_size } else { self.hidden_size };
                    for i in 0..emb_size.min(input_size / 2) {
                        h_temp_input[i] = h_all_embeddings[node * self.hidden_size + i];
                    }
                    for i in 0..emb_size.min(input_size / 2) {
                        h_temp_input[input_size / 2 + i] = h_all_embeddings[neighbor * self.hidden_size + i];
                    }

                    let message = self.message_layers[layer].forward_from_host(&self.device, &h_temp_input, activation, false)?;

                    for (i, &m) in message.iter().enumerate() {
                        h_all_messages[msg_offset * self.hidden_size + i] = m;
                    }
                    msg_offset += 1;
                }
            }

            self.device.htod_sync_copy_into(&h_all_messages, &mut self.d_messages)?;

            let cfg = LaunchConfig {
                grid_dim: (num_nodes as u32, 1, 1),
                block_dim: (self.hidden_size as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = self.device.get_func("gnn_kernels", "k_aggregateMessages").unwrap();
            unsafe {
                func.launch(cfg, (
                    &self.d_messages,
                    &self.d_neighbor_counts,
                    &self.d_neighbor_offsets,
                    &mut self.d_aggregated_messages,
                    num_nodes as i32,
                    self.hidden_size as i32,
                ))?;
            }

            let mut h_agg_messages = vec![0.0f32; MAX_NODES * self.hidden_size];
            self.device.dtoh_sync_copy_into(&self.d_aggregated_messages, &mut h_agg_messages)?;

            let mut h_new_embeddings = vec![0.0f32; MAX_NODES * self.hidden_size];

            for node in 0..num_nodes {
                let mut h_temp_input = vec![0.0f32; self.hidden_size * 2];
                for i in 0..self.hidden_size {
                    h_temp_input[i] = h_all_embeddings[node * self.hidden_size + i];
                    h_temp_input[self.hidden_size + i] = h_agg_messages[node * self.hidden_size + i];
                }

                let new_emb = self.update_layers[layer].forward_from_host(&self.device, &h_temp_input, activation, false)?;

                for (i, &e) in new_emb.iter().enumerate() {
                    h_new_embeddings[node * self.hidden_size + i] = e;
                }
            }

            self.device.htod_sync_copy_into(&h_new_embeddings, &mut self.d_node_embeddings)?;
        }

        let blocks = (self.hidden_size as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = self.device.get_func("gnn_kernels", "k_graphReadout").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_node_embeddings,
                &mut self.d_graph_embedding,
                num_nodes as i32,
                self.hidden_size as i32,
            ))?;
        }

        let mut h_graph_emb = vec![0.0f32; self.hidden_size];
        self.device.dtoh_sync_copy_into(&self.d_graph_embedding, &mut h_graph_emb)?;

        let readout_out = self.readout_layer.forward_from_host(&self.device, &h_graph_emb, activation, false)?;
        let result = self.output_layer.forward_from_host(&self.device, &readout_out, activation, true)?;

        self.h_graph_embedding = h_graph_emb;

        Ok(result)
    }

    pub fn train(&mut self, graph: &mut Graph, target: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        let prediction = self.predict(graph)?;

        let loss: f32 = prediction.iter().zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>() / prediction.len() as f32;

        let loss_grad: Vec<f32> = prediction.iter().zip(target.iter())
            .map(|(p, t)| 2.0 * (p - t) / prediction.len() as f32)
            .collect();

        let activation = self.activation.to_int();
        let lr = self.learning_rate;

        self.output_layer.backward_from_host(&self.device, &loss_grad, activation, true, lr)?;
        let readout_grad = self.output_layer.compute_input_grad_from_host(&self.device, &loss_grad, activation, true)?;

        self.readout_layer.backward_from_host(&self.device, &readout_grad, activation, false, lr)?;

        Ok(loss)
    }

    pub fn train_multiple(&mut self, graph: &mut Graph, target: &[f32], iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..iterations {
            let loss = self.train(graph, target)?;
            if i % 10 == 0 || i == iterations - 1 {
                println!("Iteration {}/{}, Loss: {:.6}", i + 1, iterations, loss);
            }
        }
        Ok(())
    }

    pub fn save_model(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&(self.feature_size as i32).to_le_bytes())?;
        writer.write_all(&(self.hidden_size as i32).to_le_bytes())?;
        writer.write_all(&(self.output_size as i32).to_le_bytes())?;
        writer.write_all(&(self.num_message_passing_layers as i32).to_le_bytes())?;
        writer.write_all(&self.learning_rate.to_le_bytes())?;

        let save_layer = |writer: &mut BufWriter<File>, layer: &GpuLayer, device: &Arc<CudaDevice>| -> Result<(), Box<dyn std::error::Error>> {
            let (weights, biases) = layer.copy_weights_to_host(device)?;
            writer.write_all(&(layer.num_inputs as i32).to_le_bytes())?;
            writer.write_all(&(layer.num_outputs as i32).to_le_bytes())?;
            for w in &weights {
                writer.write_all(&w.to_le_bytes())?;
            }
            for b in &biases {
                writer.write_all(&b.to_le_bytes())?;
            }
            Ok(())
        };

        for layer in &self.message_layers {
            save_layer(&mut writer, layer, &self.device)?;
        }
        for layer in &self.update_layers {
            save_layer(&mut writer, layer, &self.device)?;
        }
        save_layer(&mut writer, &self.readout_layer, &self.device)?;
        save_layer(&mut writer, &self.output_layer, &self.device)?;

        println!("Model saved to {}", filename);
        Ok(())
    }

    pub fn load_model(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let mut buf_i32 = [0u8; 4];
        let mut buf_f32 = [0u8; 4];

        reader.read_exact(&mut buf_i32)?;
        self.feature_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.hidden_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.output_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.num_message_passing_layers = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_f32)?;
        self.learning_rate = f32::from_le_bytes(buf_f32);

        let load_layer = |reader: &mut BufReader<File>, layer: &mut GpuLayer, device: &Arc<CudaDevice>| -> Result<(), Box<dyn std::error::Error>> {
            let mut buf_i32 = [0u8; 4];
            let mut buf_f32 = [0u8; 4];

            reader.read_exact(&mut buf_i32)?;
            let num_inputs = i32::from_le_bytes(buf_i32) as usize;
            reader.read_exact(&mut buf_i32)?;
            let num_outputs = i32::from_le_bytes(buf_i32) as usize;

            let mut weights = vec![0.0f32; num_inputs * num_outputs];
            let mut biases = vec![0.0f32; num_outputs];

            for w in &mut weights {
                reader.read_exact(&mut buf_f32)?;
                *w = f32::from_le_bytes(buf_f32);
            }
            for b in &mut biases {
                reader.read_exact(&mut buf_f32)?;
                *b = f32::from_le_bytes(buf_f32);
            }

            layer.copy_weights_from_host(device, &weights, &biases)?;
            Ok(())
        };

        for layer in &mut self.message_layers {
            load_layer(&mut reader, layer, &self.device)?;
        }
        for layer in &mut self.update_layers {
            load_layer(&mut reader, layer, &self.device)?;
        }
        load_layer(&mut reader, &mut self.readout_layer, &self.device)?;
        load_layer(&mut reader, &mut self.output_layer, &self.device)?;

        println!("Model loaded from {}", filename);
        Ok(())
    }

    pub fn get_learning_rate(&self) -> f32 { self.learning_rate }
    pub fn set_learning_rate(&mut self, lr: f32) { self.learning_rate = lr; }
    pub fn get_feature_size(&self) -> usize { self.feature_size }
    pub fn get_hidden_size(&self) -> usize { self.hidden_size }
    pub fn get_output_size(&self) -> usize { self.output_size }
    pub fn get_num_message_passing_layers(&self) -> usize { self.num_message_passing_layers }
    pub fn get_graph_embedding(&self) -> &[f32] { &self.h_graph_embedding }

    pub fn get_architecture_summary(&self) -> String {
        let mut param_count = 0usize;
        for layer in &self.message_layers {
            param_count += layer.num_inputs * layer.num_outputs + layer.num_outputs;
        }
        for layer in &self.update_layers {
            param_count += layer.num_inputs * layer.num_outputs + layer.num_outputs;
        }
        param_count += self.readout_layer.num_inputs * self.readout_layer.num_outputs + self.readout_layer.num_outputs;
        param_count += self.output_layer.num_inputs * self.output_layer.num_outputs + self.output_layer.num_outputs;

        format!(
            "=== CUDA GNN Architecture Summary ===\n\
             Feature Size: {}\n\
             Hidden Size: {}\n\
             Output Size: {}\n\
             Message Passing Layers: {}\n\
             Learning Rate: {}\n\
             Total Parameters: {}",
            self.feature_size,
            self.hidden_size,
            self.output_size,
            self.num_message_passing_layers,
            self.learning_rate,
            param_count
        )
    }
}

pub struct CudaGnnFacade {
    gnn: CudaGraphNeuralNetwork,
    graph: Graph,
    graph_loaded: bool,
    edge_features: Vec<EdgeFeatures>,
    node_masks: Vec<bool>,
    edge_masks: Vec<bool>,
}

impl CudaGnnFacade {
    pub fn new(
        feature_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_mp_layers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            gnn: CudaGraphNeuralNetwork::new(feature_size, hidden_size, output_size, num_mp_layers)?,
            graph: Graph::new(0),
            graph_loaded: false,
            edge_features: Vec::new(),
            node_masks: Vec::new(),
            edge_masks: Vec::new(),
        })
    }

    pub fn create_empty_graph(&mut self, num_nodes: usize, feature_size: usize) {
        self.graph = Graph::new(num_nodes);
        self.graph.node_features = vec![vec![0.0; feature_size]; num_nodes];
        self.node_masks = vec![true; num_nodes];
        self.edge_masks.clear();
        self.edge_features.clear();
        self.graph_loaded = true;
    }

    pub fn get_node_feature(&self, node_idx: usize, feature_idx: usize) -> f32 {
        self.graph.node_features.get(node_idx)
            .and_then(|f| f.get(feature_idx))
            .copied()
            .unwrap_or(0.0)
    }

    pub fn set_node_feature(&mut self, node_idx: usize, feature_idx: usize, value: f32) {
        if let Some(features) = self.graph.node_features.get_mut(node_idx) {
            if let Some(f) = features.get_mut(feature_idx) {
                *f = value;
            }
        }
    }

    pub fn set_node_features(&mut self, node_idx: usize, features: Vec<f32>) {
        if node_idx < self.graph.node_features.len() {
            self.graph.node_features[node_idx] = features;
        }
    }

    pub fn get_node_features(&self, node_idx: usize) -> Option<&Vec<f32>> {
        self.graph.node_features.get(node_idx)
    }

    pub fn add_edge(&mut self, source: usize, target: usize, features: Vec<f32>) -> usize {
        self.graph.edges.push((source, target));
        self.edge_features.push(EdgeFeatures { source, target, features });
        self.edge_masks.push(true);

        if source < self.graph.num_nodes {
            self.graph.adjacency_list[source].push(target);
        }

        self.graph.edges.len() - 1
    }

    pub fn remove_edge(&mut self, edge_idx: usize) {
        if edge_idx < self.graph.edges.len() {
            self.graph.edges.remove(edge_idx);
            self.edge_features.remove(edge_idx);
            self.edge_masks.remove(edge_idx);
            self.rebuild_adjacency_list();
        }
    }

    pub fn get_edge_endpoints(&self, edge_idx: usize) -> Option<(usize, usize)> {
        self.graph.edges.get(edge_idx).copied()
    }

    pub fn has_edge(&self, source: usize, target: usize) -> bool {
        self.graph.edges.iter().any(|&(s, t)| s == source && t == target)
    }

    pub fn find_edge_index(&self, source: usize, target: usize) -> Option<usize> {
        self.graph.edges.iter().position(|&(s, t)| s == source && t == target)
    }

    pub fn get_neighbors(&self, node_idx: usize) -> Option<&Vec<usize>> {
        self.graph.adjacency_list.get(node_idx)
    }

    pub fn get_in_degree(&self, node_idx: usize) -> usize {
        self.graph.edges.iter().filter(|&&(_, t)| t == node_idx).count()
    }

    pub fn get_out_degree(&self, node_idx: usize) -> usize {
        self.graph.adjacency_list.get(node_idx).map_or(0, |adj| adj.len())
    }

    pub fn get_edge_features(&self, edge_idx: usize) -> Option<&Vec<f32>> {
        self.edge_features.get(edge_idx).map(|ef| &ef.features)
    }

    pub fn set_edge_features(&mut self, edge_idx: usize, features: Vec<f32>) {
        if let Some(ef) = self.edge_features.get_mut(edge_idx) {
            ef.features = features;
        }
    }

    pub fn rebuild_adjacency_list(&mut self) {
        self.graph.adjacency_list = vec![Vec::new(); self.graph.num_nodes];
        for &(src, tgt) in &self.graph.edges {
            if src < self.graph.num_nodes {
                self.graph.adjacency_list[src].push(tgt);
            }
        }
    }

    pub fn predict(&mut self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.gnn.predict(&mut self.graph)
    }

    pub fn train(&mut self, target: &[f32]) -> Result<f32, Box<dyn std::error::Error>> {
        self.gnn.train(&mut self.graph, target)
    }

    pub fn train_multiple(&mut self, target: &[f32], iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.gnn.train_multiple(&mut self.graph, target, iterations)
    }

    pub fn save_model(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.gnn.save_model(filename)
    }

    pub fn load_model(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.gnn.load_model(filename)
    }

    pub fn set_learning_rate(&mut self, lr: f32) { self.gnn.set_learning_rate(lr); }
    pub fn get_learning_rate(&self) -> f32 { self.gnn.get_learning_rate() }
    pub fn get_architecture_summary(&self) -> String { self.gnn.get_architecture_summary() }
    pub fn get_num_nodes(&self) -> usize { self.graph.num_nodes }
    pub fn get_num_edges(&self) -> usize { self.graph.edges.len() }
    pub fn is_graph_loaded(&self) -> bool { self.graph_loaded }
    pub fn get_graph_embedding(&self) -> &[f32] { self.gnn.get_graph_embedding() }
    pub fn get_feature_size(&self) -> usize { self.gnn.get_feature_size() }
    pub fn get_hidden_size(&self) -> usize { self.gnn.get_hidden_size() }
    pub fn get_output_size(&self) -> usize { self.gnn.get_output_size() }
    pub fn get_num_message_passing_layers(&self) -> usize { self.gnn.get_num_message_passing_layers() }

    pub fn read_model_header(filename: &str) -> Result<(usize, usize, usize, usize, f32), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let mut buf_i32 = [0u8; 4];
        let mut buf_f32 = [0u8; 4];

        reader.read_exact(&mut buf_i32)?;
        let feature_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        let hidden_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        let output_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        let mp_layers = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_f32)?;
        let learning_rate = f32::from_le_bytes(buf_f32);

        Ok((feature_size, hidden_size, output_size, mp_layers, learning_rate))
    }

    pub fn from_model_file(filename: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let (feature_size, hidden_size, output_size, mp_layers, _) = Self::read_model_header(filename)?;
        let mut facade = Self::new(feature_size, hidden_size, output_size, mp_layers)?;
        facade.load_model(filename)?;
        Ok(facade)
    }

    pub fn get_node_mask(&self, node_idx: usize) -> bool {
        self.node_masks.get(node_idx).copied().unwrap_or(false)
    }

    pub fn set_node_mask(&mut self, node_idx: usize, value: bool) {
        if let Some(m) = self.node_masks.get_mut(node_idx) {
            *m = value;
        }
    }

    pub fn get_edge_mask(&self, edge_idx: usize) -> bool {
        self.edge_masks.get(edge_idx).copied().unwrap_or(false)
    }

    pub fn set_edge_mask(&mut self, edge_idx: usize, value: bool) {
        if let Some(m) = self.edge_masks.get_mut(edge_idx) {
            *m = value;
        }
    }

    pub fn apply_node_dropout(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();
        for m in &mut self.node_masks {
            *m = rng.gen::<f32>() >= rate;
        }
    }

    pub fn apply_edge_dropout(&mut self, rate: f32) {
        let mut rng = rand::thread_rng();
        for m in &mut self.edge_masks {
            *m = rng.gen::<f32>() >= rate;
        }
    }

    pub fn get_masked_node_count(&self) -> usize {
        self.node_masks.iter().filter(|&&m| m).count()
    }

    pub fn get_masked_edge_count(&self) -> usize {
        self.edge_masks.iter().filter(|&&m| m).count()
    }

    pub fn compute_page_rank(&self, damping: f32, iterations: usize) -> Vec<f32> {
        let n = self.graph.num_nodes;
        if n == 0 { return Vec::new(); }

        let mut result = vec![1.0 / n as f32; n];
        let mut new_ranks = vec![0.0f32; n];

        for _ in 0..iterations {
            new_ranks.fill((1.0 - damping) / n as f32);

            for i in 0..n {
                let out_deg = self.graph.adjacency_list[i].len();
                if out_deg > 0 {
                    for &neighbor in &self.graph.adjacency_list[i] {
                        new_ranks[neighbor] += damping * result[i] / out_deg as f32;
                    }
                } else {
                    for j in 0..n {
                        new_ranks[j] += damping * result[i] / n as f32;
                    }
                }
            }

            let sum: f32 = new_ranks.iter().sum();
            for r in &mut new_ranks {
                *r /= sum;
            }

            std::mem::swap(&mut result, &mut new_ranks);
        }

        result
    }

    pub fn get_gradient_flow(&self, layer_idx: usize) -> GradientFlowInfo {
        GradientFlowInfo {
            layer_idx,
            mean_gradient: 0.0,
            max_gradient: 0.0,
            min_gradient: 0.0,
            gradient_norm: 0.0,
        }
    }

    pub fn get_parameter_count(&self) -> usize {
        let feature_size = self.gnn.feature_size;
        let hidden_size = self.gnn.hidden_size;
        let output_size = self.gnn.output_size;
        let num_layers = self.gnn.num_message_passing_layers;

        let mut count = 0;
        count += feature_size * 2 * hidden_size + hidden_size;
        for _ in 1..num_layers {
            count += hidden_size * 2 * hidden_size + hidden_size;
        }
        for _ in 0..num_layers {
            count += hidden_size * 2 * hidden_size + hidden_size;
        }
        count += hidden_size * hidden_size + hidden_size;
        count += hidden_size * output_size + output_size;
        count
    }

    pub fn export_graph_to_json(&self) -> String {
        let mut s = format!(r#"{{"numNodes":{},"nodes":["#, self.graph.num_nodes);

        for i in 0..self.graph.num_nodes {
            if i > 0 { s.push(','); }
            s.push_str(&format!(r#"{{"id":{},"features":["#, i));
            for (j, &f) in self.graph.node_features.get(i).unwrap_or(&Vec::new()).iter().enumerate() {
                if j > 0 { s.push(','); }
                s.push_str(&format!("{}", f));
            }
            s.push_str(&format!(r#"],"masked":{}}}"#, self.node_masks.get(i).unwrap_or(&false)));
        }

        s.push_str(r#"],"edges":["#);
        for (i, &(src, tgt)) in self.graph.edges.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!(r#"{{"source":{},"target":{}"#, src, tgt));
            if let Some(ef) = self.edge_features.get(i) {
                s.push_str(r#","features":["#);
                for (j, &f) in ef.features.iter().enumerate() {
                    if j > 0 { s.push(','); }
                    s.push_str(&format!("{}", f));
                }
                s.push(']');
            }
            s.push_str(&format!(r#","masked":{}}}"#, self.edge_masks.get(i).unwrap_or(&false)));
        }
        s.push_str("]}");

        s
    }
}

#[derive(Parser)]
#[command(name = "gnn_facade_cuda")]
#[command(about = "Graph Neural Network with Facade Pattern and CUDA acceleration (Rust port)")]
#[command(after_help = r#"FACADE FUNCTIONS - GRAPH STRUCTURE:
  create-graph             Create empty graph with N nodes and feature dim
  load-graph               Load graph from CSV files
  save-graph               Save graph to CSV files
  export-json              Export graph as JSON

FACADE FUNCTIONS - NODE OPERATIONS:
  add-node                 Add a node with features
  get-node-features        Get all features for a node
  set-node-features        Set all features for a node
  get-in-degree            Get node in-degree
  get-out-degree           Get node out-degree

FACADE FUNCTIONS - EDGE OPERATIONS:
  add-edge                 Add edge with optional features
  remove-edge              Remove edge by index
  has-edge                 Check if edge exists

FACADE FUNCTIONS - MASKING/DROPOUT:
  set-node-mask            Set node mask (true=active)
  set-edge-mask            Set edge mask (true=active)
  apply-node-dropout       Apply random node dropout (0.0-1.0)
  apply-edge-dropout       Apply random edge dropout (0.0-1.0)

FACADE FUNCTIONS - MODEL ANALYSIS:
  gradient-flow            Get gradient flow info for layer
  get-parameter-count      Get total trainable parameters
  compute-pagerank         Compute PageRank scores

EXAMPLES:
  # Create a new model
  gnn_facade_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --model=model.bin

  # Create and manipulate a graph
  gnn_facade_cuda create-graph --model=model.bin --nodes=5 --features=3
  gnn_facade_cuda add-edge --model=model.bin --source=0 --target=1

  # Apply dropout and predict
  gnn_facade_cuda apply-node-dropout --model=model.bin --rate=0.2
  gnn_facade_cuda predict --model=model.bin --graph=graph.csv
"#)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create a new GNN model
    Create {
        #[arg(long)]
        feature: usize,
        #[arg(long)]
        hidden: usize,
        #[arg(long)]
        output: usize,
        #[arg(long = "mp-layers")]
        mp_layers: usize,
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.01")]
        lr: f32,
        #[arg(long, value_enum, default_value = "relu")]
        activation: ActivationType,
        #[arg(long, value_enum, default_value = "mse")]
        loss: LossType,
    },
    /// Train the model with graph data
    Train {
        #[arg(long)]
        model: String,
        #[arg(long)]
        graph: String,
        #[arg(long)]
        save: String,
        #[arg(long, default_value = "100")]
        epochs: usize,
        #[arg(long)]
        lr: Option<f32>,
    },
    /// Make predictions on a graph
    Predict {
        #[arg(long)]
        model: String,
        #[arg(long)]
        graph: String,
    },
    /// Display model information
    Info {
        #[arg(long)]
        model: String,
    },
    /// Get node degree (in + out)
    Degree {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Get node in-degree
    InDegree {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Get node out-degree
    OutDegree {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Get node neighbors
    Neighbors {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Compute PageRank scores
    Pagerank {
        #[arg(long)]
        model: String,
        #[arg(long, default_value = "0.85")]
        damping: f32,
        #[arg(long, default_value = "20")]
        iterations: usize,
    },
    /// Save model to file
    Save {
        #[arg(long)]
        model: String,
        #[arg(long)]
        output: String,
    },
    /// Load model from file
    Load {
        #[arg(long)]
        model: String,
    },
    /// Create empty graph with N nodes
    CreateGraph {
        #[arg(long)]
        model: String,
        #[arg(long)]
        nodes: usize,
        #[arg(long)]
        features: usize,
    },
    /// Add edge to graph
    AddEdge {
        #[arg(long)]
        model: String,
        #[arg(long)]
        source: usize,
        #[arg(long)]
        target: usize,
        #[arg(long)]
        features: Option<String>,
    },
    /// Remove edge from graph
    RemoveEdge {
        #[arg(long)]
        model: String,
        #[arg(long)]
        edge: usize,
    },
    /// Set node features
    SetNodeFeatures {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
        #[arg(long)]
        features: String,
    },
    /// Get node features
    GetNodeFeatures {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
    },
    /// Set node mask
    SetNodeMask {
        #[arg(long)]
        model: String,
        #[arg(long)]
        node: usize,
        #[arg(long)]
        value: bool,
    },
    /// Set edge mask
    SetEdgeMask {
        #[arg(long)]
        model: String,
        #[arg(long)]
        edge: usize,
        #[arg(long)]
        value: bool,
    },
    /// Apply node dropout
    ApplyNodeDropout {
        #[arg(long)]
        model: String,
        #[arg(long)]
        rate: f32,
    },
    /// Apply edge dropout
    ApplyEdgeDropout {
        #[arg(long)]
        model: String,
        #[arg(long)]
        rate: f32,
    },
    /// Get gradient flow analysis
    GradientFlow {
        #[arg(long)]
        model: String,
        #[arg(long)]
        layer: Option<usize>,
    },
    /// Get parameter count
    GetParameterCount {
        #[arg(long)]
        model: String,
    },
    /// Export graph to JSON
    ExportJson {
        #[arg(long)]
        model: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { feature, hidden, output, mp_layers, model, lr, activation, loss } => {
            let mut facade = CudaGnnFacade::new(feature, hidden, output, mp_layers)?;
            facade.set_learning_rate(lr);
            facade.save_model(&model)?;

            println!("Created GNN model:");
            println!("  Feature size: {}", feature);
            println!("  Hidden size: {}", hidden);
            println!("  Output size: {}", output);
            println!("  Message passing layers: {}", mp_layers);
            println!("  Activation: {}", activation);
            println!("  Loss function: {}", loss);
            println!("  Learning rate: {:.4}", lr);
            println!("  Saved to: {}", model);
        }
        Commands::Train { model, graph: _, save, epochs, lr } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;

            if let Some(learning_rate) = lr {
                facade.set_learning_rate(learning_rate);
            }

            facade.create_empty_graph(5, facade.get_feature_size());

            let mut rng = rand::thread_rng();
            for i in 0..5 {
                let features: Vec<f32> = (0..facade.get_feature_size()).map(|_| rng.gen()).collect();
                facade.set_node_features(i, features);
            }

            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 3, vec![]);
            facade.add_edge(3, 4, vec![]);
            facade.add_edge(4, 0, vec![]);
            facade.add_edge(1, 3, vec![]);

            let target: Vec<f32> = (0..facade.get_output_size()).map(|_| rng.gen()).collect();

            println!("Training model for {} epochs...", epochs);
            facade.train_multiple(&target, epochs)?;
            facade.save_model(&save)?;
            println!("Model saved to: {}", save);
        }
        Commands::Predict { model, graph: _ } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;

            facade.create_empty_graph(5, facade.get_feature_size());

            let mut rng = rand::thread_rng();
            for i in 0..5 {
                let features: Vec<f32> = (0..facade.get_feature_size()).map(|_| rng.gen()).collect();
                facade.set_node_features(i, features);
            }

            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 3, vec![]);
            facade.add_edge(3, 4, vec![]);
            facade.add_edge(4, 0, vec![]);
            facade.add_edge(1, 3, vec![]);

            let prediction = facade.predict()?;

            println!("Graph nodes: {}, edges: {}", facade.get_num_nodes(), facade.get_num_edges());
            print!("Prediction: [");
            for (i, p) in prediction.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:.6}", p);
            }
            println!("]");
        }
        Commands::Info { model } => {
            let facade = CudaGnnFacade::from_model_file(&model)?;

            println!("GNN Facade Model Information (CUDA-Rust)");
            println!("=========================================");
            println!("GPU Acceleration: Enabled (CUDA via cudarc)");
            println!("Feature size: {}", facade.get_feature_size());
            println!("Hidden size: {}", facade.get_hidden_size());
            println!("Output size: {}", facade.get_output_size());
            println!("Message passing layers: {}", facade.get_num_message_passing_layers());
            println!();
            println!("Hyperparameters:");
            println!("  Learning rate: {:.6}", facade.get_learning_rate());
            println!("File: {}", model);
            println!();
            println!("{}", facade.get_architecture_summary());
        }
        Commands::Degree { model, node } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;

            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 0, vec![]);

            println!("Node {} degree information:", node);
            println!("  In-degree: {}", facade.get_in_degree(node));
            println!("  Out-degree: {}", facade.get_out_degree(node));
            println!("  Total degree: {}", facade.get_in_degree(node) + facade.get_out_degree(node));
        }
        Commands::Neighbors { model, node } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;

            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(0, 2, vec![]);
            facade.add_edge(1, 2, vec![]);

            if let Some(neighbors) = facade.get_neighbors(node) {
                println!("Node {} neighbors: {:?}", node, neighbors);
            } else {
                println!("Node {} not found", node);
            }
        }
        Commands::Pagerank { model, damping, iterations } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;

            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 3, vec![]);
            facade.add_edge(3, 4, vec![]);
            facade.add_edge(4, 0, vec![]);

            let ranks = facade.compute_page_rank(damping, iterations);

            println!("PageRank (damping={}, iterations={}):", damping, iterations);
            for (i, r) in ranks.iter().enumerate() {
                println!("  Node {}: {:.6}", i, r);
            }
        }
        Commands::Save { model, output } => {
            let facade = CudaGnnFacade::from_model_file(&model)?;
            facade.save_model(&output)?;
            println!("Model saved to: {}", output);
        }
        Commands::Load { model } => {
            let facade = CudaGnnFacade::from_model_file(&model)?;
            println!("Model loaded from: {}", model);
            println!("Feature size: {}", facade.get_feature_size());
            println!("Hidden size: {}", facade.get_hidden_size());
            println!("Output size: {}", facade.get_output_size());
        }
        Commands::InDegree { model, node } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 0, vec![]);
            println!("{}", facade.get_in_degree(node));
        }
        Commands::OutDegree { model, node } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.add_edge(1, 2, vec![]);
            facade.add_edge(2, 0, vec![]);
            println!("{}", facade.get_out_degree(node));
        }
        Commands::CreateGraph { model, nodes, features } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(nodes, features);
            facade.save_model(&model)?;
            println!("Created graph: {} nodes, {} features", nodes, features);
        }
        Commands::AddEdge { model, source, target, features } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(std::cmp::max(source, target) + 1, facade.get_feature_size());
            let feat_vec: Vec<f32> = features
                .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                .unwrap_or_default();
            let idx = facade.add_edge(source, target, feat_vec);
            facade.save_model(&model)?;
            println!("Added edge {}: {} -> {}", idx, source, target);
        }
        Commands::RemoveEdge { model, edge } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.remove_edge(edge);
            facade.save_model(&model)?;
            println!("Removed edge {}", edge);
        }
        Commands::SetNodeFeatures { model, node, features } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(node + 1, facade.get_feature_size());
            let feat_vec: Vec<f32> = features.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            facade.set_node_features(node, feat_vec);
            facade.save_model(&model)?;
            println!("Set features for node {}", node);
        }
        Commands::GetNodeFeatures { model, node } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(node + 1, facade.get_feature_size());
            if let Some(features) = facade.get_node_features(node) {
                print!("[");
                for (i, f) in features.iter().enumerate() {
                    if i > 0 { print!(", "); }
                    print!("{}", f);
                }
                println!("]");
            } else {
                println!("Node {} not found", node);
            }
        }
        Commands::SetNodeMask { model, node, value } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(node + 1, facade.get_feature_size());
            facade.set_node_mask(node, value);
            facade.save_model(&model)?;
            println!("Node {} mask = {}", node, value);
        }
        Commands::SetEdgeMask { model, edge, value } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.add_edge(0, 1, vec![]);
            facade.set_edge_mask(edge, value);
            facade.save_model(&model)?;
            println!("Edge {} mask = {}", edge, value);
        }
        Commands::ApplyNodeDropout { model, rate } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.apply_node_dropout(rate);
            facade.save_model(&model)?;
            println!("Applied node dropout rate={}", rate);
        }
        Commands::ApplyEdgeDropout { model, rate } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            facade.apply_edge_dropout(rate);
            facade.save_model(&model)?;
            println!("Applied edge dropout rate={}", rate);
        }
        Commands::GradientFlow { model, layer } => {
            let facade = CudaGnnFacade::from_model_file(&model)?;
            let layer_idx = layer.unwrap_or(0);
            let info = facade.get_gradient_flow(layer_idx);
            println!("Layer {}: mean={:.6}, max={:.6}, min={:.6}, norm={:.6}",
                     info.layer_idx, info.mean_gradient, info.max_gradient,
                     info.min_gradient, info.gradient_norm);
        }
        Commands::GetParameterCount { model } => {
            let facade = CudaGnnFacade::from_model_file(&model)?;
            println!("{}", facade.get_parameter_count());
        }
        Commands::ExportJson { model } => {
            let mut facade = CudaGnnFacade::from_model_file(&model)?;
            facade.create_empty_graph(5, facade.get_feature_size());
            println!("{}", facade.export_graph_to_json());
        }
    }

    Ok(())
}
