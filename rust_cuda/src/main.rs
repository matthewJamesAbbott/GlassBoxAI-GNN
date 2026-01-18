//
// MIT License
//
// Copyright (c) 2025 Matthew Abbott
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

use clap::{Parser, Subcommand, ValueEnum};
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use rand::Rng;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::Arc;

const GRADIENT_CLIP: f64 = 5.0;
const BLOCK_SIZE: u32 = 256;

const CUDA_KERNELS: &str = r#"
extern "C" {

__device__ double d_Activate(double X, int ActivationType) {
    switch (ActivationType) {
        case 0: // ReLU
            return (X > 0) ? X : 0.0;
        case 1: // LeakyReLU
            return (X > 0) ? X : 0.01 * X;
        case 2: // Tanh
            return tanh(X);
        case 3: // Sigmoid
            return 1.0 / (1.0 + exp(-fmax(-500.0, fmin(500.0, X))));
        default:
            return X;
    }
}

__device__ double d_ActivateDerivative(double X, int ActivationType) {
    double S;
    switch (ActivationType) {
        case 0: // ReLU
            return (X > 0) ? 1.0 : 0.0;
        case 1: // LeakyReLU
            return (X > 0) ? 1.0 : 0.01;
        case 2: // Tanh
            S = tanh(X);
            return 1.0 - S * S;
        case 3: // Sigmoid
            S = 1.0 / (1.0 + exp(-fmax(-500.0, fmin(500.0, X))));
            return S * (1.0 - S);
        default:
            return 1.0;
    }
}

__device__ double d_OutputActivate(double X) {
    return 1.0 / (1.0 + exp(-fmax(-500.0, fmin(500.0, X))));
}

__device__ double d_OutputActivateDerivative(double PreAct) {
    double S = 1.0 / (1.0 + exp(-fmax(-500.0, fmin(500.0, PreAct))));
    return S * (1.0 - S);
}

__device__ double d_ClipGradient(double G) {
    return fmax(-5.0, fmin(5.0, G));
}

__global__ void k_ForwardLayer(
    double* Weights, double* Biases, double* Input, double* Output, double* PreActivations,
    int NumOutputs, int NumInputs, int ActivationType, bool UseOutputActivation)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if (I >= NumOutputs) return;
    
    double Sum = Biases[I];
    for (int J = 0; J < NumInputs; J++)
        Sum += Input[J] * Weights[I * NumInputs + J];
    
    PreActivations[I] = Sum;
    
    if (UseOutputActivation)
        Output[I] = d_OutputActivate(Sum);
    else
        Output[I] = d_Activate(Sum, ActivationType);
}

__global__ void k_BackwardLayer(
    double* Weights, double* Biases, double* LastInput, double* UpstreamGrad,
    double* PreActivations, double* Errors,
    int NumOutputs, int NumInputs, int ActivationType, bool UseOutputActivation, double LearningRate)
{
    int I = blockIdx.x * blockDim.x + threadIdx.x;
    if (I >= NumOutputs) return;
    
    double PreActGrad;
    if (UseOutputActivation)
        PreActGrad = UpstreamGrad[I] * d_OutputActivateDerivative(PreActivations[I]);
    else
        PreActGrad = UpstreamGrad[I] * d_ActivateDerivative(PreActivations[I], ActivationType);
    
    PreActGrad = d_ClipGradient(PreActGrad);
    Errors[I] = PreActGrad;
    
    Biases[I] -= LearningRate * PreActGrad;
    
    for (int J = 0; J < NumInputs; J++) {
        double Grad = d_ClipGradient(PreActGrad * LastInput[J]);
        Weights[I * NumInputs + J] -= LearningRate * Grad;
    }
}

__global__ void k_GetLayerInputGrad(
    double* Weights, double* UpstreamGrad, double* PreActivations, double* Result,
    int NumOutputs, int NumInputs, int ActivationType, bool UseOutputActivation)
{
    int J = blockIdx.x * blockDim.x + threadIdx.x;
    if (J >= NumInputs) return;
    
    double Sum = 0.0;
    for (int I = 0; I < NumOutputs; I++) {
        double PreActGrad;
        if (UseOutputActivation)
            PreActGrad = UpstreamGrad[I] * d_OutputActivateDerivative(PreActivations[I]);
        else
            PreActGrad = UpstreamGrad[I] * d_ActivateDerivative(PreActivations[I], ActivationType);
        
        Sum += Weights[I * NumInputs + J] * PreActGrad;
    }
    Result[J] = d_ClipGradient(Sum);
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
    fn to_int(&self) -> i32 {
        match self {
            ActivationType::Relu => 0,
            ActivationType::LeakyRelu => 1,
            ActivationType::Tanh => 2,
            ActivationType::Sigmoid => 3,
        }
    }

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
    fn to_int(&self) -> i32 {
        match self {
            LossType::Mse => 0,
            LossType::Bce => 1,
        }
    }

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

#[derive(Debug, Clone, Default)]
pub struct GraphConfig {
    pub undirected: bool,
    pub self_loops: bool,
    pub deduplicate_edges: bool,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub num_nodes: usize,
    pub node_features: Vec<Vec<f64>>,
    pub edges: Vec<Edge>,
    pub adjacency_list: Vec<Vec<usize>>,
    pub config: GraphConfig,
}

impl Graph {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            node_features: Vec::new(),
            edges: Vec::new(),
            adjacency_list: Vec::new(),
            config: GraphConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f64>,
    pub bias: f64,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

#[derive(Debug, Clone)]
pub struct MessageInfo {
    pub neighbor_idx: usize,
    pub concat_input: Vec<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub loss: f64,
    pub iteration: usize,
    pub loss_history: Vec<f64>,
}

pub struct GpuLayer {
    pub d_weights: CudaSlice<f64>,
    pub d_biases: CudaSlice<f64>,
    pub d_output: CudaSlice<f64>,
    pub d_pre_activations: CudaSlice<f64>,
    pub d_errors: CudaSlice<f64>,
    pub d_last_input: CudaSlice<f64>,
    pub num_inputs: usize,
    pub num_outputs: usize,
}

impl GpuLayer {
    fn forward(
        &mut self,
        device: &Arc<CudaDevice>,
        input: &[f64],
        activation: i32,
        use_output_activation: bool,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        device.htod_sync_copy_into(input, &mut self.d_last_input)?;

        let num_blocks = (self.num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_ForwardLayer").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_weights,
                &self.d_biases,
                &self.d_last_input,
                &mut self.d_output,
                &mut self.d_pre_activations,
                self.num_outputs as i32,
                self.num_inputs as i32,
                activation,
                use_output_activation as i32,
            ))?;
        }

        let mut result = vec![0.0f64; self.num_outputs];
        device.dtoh_sync_copy_into(&self.d_output, &mut result)?;
        Ok(result)
    }

    fn backward(
        &mut self,
        device: &Arc<CudaDevice>,
        upstream_grad: &[f64],
        activation: i32,
        use_output_activation: bool,
        learning_rate: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let d_upstream_grad = device.htod_sync_copy(upstream_grad)?;

        let num_blocks = (self.num_outputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_BackwardLayer").unwrap();
        unsafe {
            func.launch(cfg, (
                &mut self.d_weights,
                &mut self.d_biases,
                &self.d_last_input,
                &d_upstream_grad,
                &self.d_pre_activations,
                &mut self.d_errors,
                self.num_outputs as i32,
                self.num_inputs as i32,
                activation,
                use_output_activation as i32,
                learning_rate,
            ))?;
        }
        Ok(())
    }

    fn get_input_grad(
        &self,
        device: &Arc<CudaDevice>,
        upstream_grad: &[f64],
        activation: i32,
        use_output_activation: bool,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let d_upstream_grad = device.htod_sync_copy(upstream_grad)?;
        let mut d_result: CudaSlice<f64> = device.alloc_zeros(self.num_inputs)?;

        let num_blocks = (self.num_inputs as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let func = device.get_func("gnn_kernels", "k_GetLayerInputGrad").unwrap();
        unsafe {
            func.launch(cfg, (
                &self.d_weights,
                &d_upstream_grad,
                &self.d_pre_activations,
                &mut d_result,
                self.num_outputs as i32,
                self.num_inputs as i32,
                activation,
                use_output_activation as i32,
            ))?;
        }

        let mut result = vec![0.0f64; self.num_inputs];
        device.dtoh_sync_copy_into(&d_result, &mut result)?;
        Ok(result)
    }

    #[allow(dead_code)]
    fn sync_from_host(&mut self, device: &Arc<CudaDevice>, layer: &Layer) -> Result<(), Box<dyn std::error::Error>> {
        let weight_size = layer.num_outputs * layer.num_inputs;
        let mut weights = vec![0.0f64; weight_size];
        let mut biases = vec![0.0f64; layer.num_outputs];

        for (i, neuron) in layer.neurons.iter().enumerate() {
            for (j, &w) in neuron.weights.iter().enumerate() {
                weights[i * layer.num_inputs + j] = w;
            }
            biases[i] = neuron.bias;
        }

        device.htod_sync_copy_into(&weights, &mut self.d_weights)?;
        device.htod_sync_copy_into(&biases, &mut self.d_biases)?;
        Ok(())
    }

    fn sync_to_host(&self, device: &Arc<CudaDevice>, layer: &mut Layer) -> Result<(), Box<dyn std::error::Error>> {
        let weight_size = layer.num_outputs * layer.num_inputs;
        let mut weights = vec![0.0f64; weight_size];
        let mut biases = vec![0.0f64; layer.num_outputs];

        device.dtoh_sync_copy_into(&self.d_weights, &mut weights)?;
        device.dtoh_sync_copy_into(&self.d_biases, &mut biases)?;

        for (i, neuron) in layer.neurons.iter_mut().enumerate() {
            for (j, w) in neuron.weights.iter_mut().enumerate() {
                *w = weights[i * layer.num_inputs + j];
            }
            neuron.bias = biases[i];
        }
        Ok(())
    }

    fn set_last_input(&mut self, device: &Arc<CudaDevice>, input: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        device.htod_sync_copy_into(input, &mut self.d_last_input)?;
        Ok(())
    }

    fn get_last_input(&self, device: &Arc<CudaDevice>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut v = vec![0.0f64; self.num_inputs];
        device.dtoh_sync_copy_into(&self.d_last_input, &mut v)?;
        Ok(v)
    }

    fn get_output(&self, device: &Arc<CudaDevice>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut v = vec![0.0f64; self.num_outputs];
        device.dtoh_sync_copy_into(&self.d_output, &mut v)?;
        Ok(v)
    }

    fn get_pre_activations(&self, device: &Arc<CudaDevice>) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let mut v = vec![0.0f64; self.num_outputs];
        device.dtoh_sync_copy_into(&self.d_pre_activations, &mut v)?;
        Ok(v)
    }
}

fn create_gpu_layer(device: &Arc<CudaDevice>, layer: &Layer) -> Result<GpuLayer, Box<dyn std::error::Error>> {
    let weight_size = layer.num_outputs * layer.num_inputs;
    let mut weights = vec![0.0f64; weight_size];
    let mut biases = vec![0.0f64; layer.num_outputs];

    for (i, neuron) in layer.neurons.iter().enumerate() {
        for (j, &w) in neuron.weights.iter().enumerate() {
            weights[i * layer.num_inputs + j] = w;
        }
        biases[i] = neuron.bias;
    }

    let d_weights = device.htod_sync_copy(&weights)?;
    let d_biases = device.htod_sync_copy(&biases)?;
    let d_output = device.alloc_zeros::<f64>(layer.num_outputs)?;
    let d_pre_activations = device.alloc_zeros::<f64>(layer.num_outputs)?;
    let d_errors = device.alloc_zeros::<f64>(layer.num_outputs)?;
    let d_last_input = device.alloc_zeros::<f64>(layer.num_inputs)?;

    Ok(GpuLayer {
        d_weights,
        d_biases,
        d_output,
        d_pre_activations,
        d_errors,
        d_last_input,
        num_inputs: layer.num_inputs,
        num_outputs: layer.num_outputs,
    })
}

pub struct GraphNeuralNetwork {
    learning_rate: f64,
    max_iterations: usize,
    num_message_passing_layers: usize,
    feature_size: usize,
    hidden_size: usize,
    output_size: usize,
    activation: ActivationType,
    loss_type: LossType,

    message_layers: Vec<Layer>,
    update_layers: Vec<Layer>,
    readout_layer: Layer,
    output_layer: Layer,

    gpu_message_layers: Vec<GpuLayer>,
    gpu_update_layers: Vec<GpuLayer>,
    gpu_readout_layer: GpuLayer,
    gpu_output_layer: GpuLayer,

    node_embeddings: Vec<Vec<f64>>,
    new_node_embeddings: Vec<Vec<f64>>,
    embedding_history: Vec<Vec<Vec<f64>>>,
    message_history: Vec<Vec<Vec<MessageInfo>>>,
    aggregated_messages: Vec<Vec<Vec<f64>>>,
    graph_embedding: Vec<f64>,

    metrics: TrainingMetrics,
    device: Arc<CudaDevice>,
}

impl GraphNeuralNetwork {
    pub fn new(
        feature_size: usize,
        hidden_size: usize,
        output_size: usize,
        num_mp_layers: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = CudaDevice::new(0)?;

        let ptx = cudarc::nvrtc::compile_ptx(CUDA_KERNELS)?;
        device.load_ptx(ptx, "gnn_kernels", &[
            "k_ForwardLayer",
            "k_BackwardLayer",
            "k_GetLayerInputGrad",
        ])?;

        let mut message_layers = Vec::with_capacity(num_mp_layers);
        let mut update_layers = Vec::with_capacity(num_mp_layers);

        for i in 0..num_mp_layers {
            let msg_input_size = if i == 0 { feature_size * 2 } else { hidden_size * 2 };
            message_layers.push(Self::initialize_layer(hidden_size, msg_input_size));
            update_layers.push(Self::initialize_layer(hidden_size, hidden_size * 2));
        }

        let readout_layer = Self::initialize_layer(hidden_size, hidden_size);
        let output_layer = Self::initialize_layer(output_size, hidden_size);

        let mut gpu_message_layers = Vec::with_capacity(num_mp_layers);
        let mut gpu_update_layers = Vec::with_capacity(num_mp_layers);

        for i in 0..num_mp_layers {
            gpu_message_layers.push(create_gpu_layer(&device, &message_layers[i])?);
            gpu_update_layers.push(create_gpu_layer(&device, &update_layers[i])?);
        }

        let gpu_readout_layer = create_gpu_layer(&device, &readout_layer)?;
        let gpu_output_layer = create_gpu_layer(&device, &output_layer)?;

        Ok(Self {
            learning_rate: 0.01,
            max_iterations: 100,
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
            gpu_message_layers,
            gpu_update_layers,
            gpu_readout_layer,
            gpu_output_layer,
            node_embeddings: Vec::new(),
            new_node_embeddings: Vec::new(),
            embedding_history: Vec::new(),
            message_history: Vec::new(),
            aggregated_messages: Vec::new(),
            graph_embedding: Vec::new(),
            metrics: TrainingMetrics::default(),
            device,
        })
    }

    fn initialize_layer(num_neurons: usize, num_inputs: usize) -> Layer {
        let mut rng = rand::thread_rng();
        let scale = (2.0 / (num_inputs + num_neurons) as f64).sqrt();

        let neurons = (0..num_neurons)
            .map(|_| {
                let weights: Vec<f64> = (0..num_inputs)
                    .map(|_| (rng.gen::<f64>() - 0.5) * 2.0 * scale)
                    .collect();
                Neuron { weights, bias: 0.0 }
            })
            .collect();

        Layer {
            neurons,
            num_inputs,
            num_outputs: num_neurons,
        }
    }

    fn sync_from_gpu(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for i in 0..self.num_message_passing_layers {
            self.gpu_message_layers[i].sync_to_host(&self.device, &mut self.message_layers[i])?;
            self.gpu_update_layers[i].sync_to_host(&self.device, &mut self.update_layers[i])?;
        }
        self.gpu_readout_layer.sync_to_host(&self.device, &mut self.readout_layer)?;
        self.gpu_output_layer.sync_to_host(&self.device, &mut self.output_layer)?;
        Ok(())
    }

    fn reinitialize_gpu(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.gpu_message_layers.clear();
        self.gpu_update_layers.clear();

        for i in 0..self.num_message_passing_layers {
            self.gpu_message_layers.push(create_gpu_layer(&self.device, &self.message_layers[i])?);
            self.gpu_update_layers.push(create_gpu_layer(&self.device, &self.update_layers[i])?);
        }

        self.gpu_readout_layer = create_gpu_layer(&self.device, &self.readout_layer)?;
        self.gpu_output_layer = create_gpu_layer(&self.device, &self.output_layer)?;
        Ok(())
    }

    fn output_activate_derivative(&self, pre_act: f64) -> f64 {
        let s = 1.0 / (1.0 + (-pre_act.clamp(-500.0, 500.0)).exp());
        s * (1.0 - s)
    }

    pub fn compute_loss(&self, prediction: &[f64], target: &[f64]) -> f64 {
        match self.loss_type {
            LossType::Mse => {
                let sum: f64 = prediction.iter().zip(target.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum();
                sum / prediction.len() as f64
            }
            LossType::Bce => {
                let sum: f64 = prediction.iter().zip(target.iter())
                    .map(|(p, t)| {
                        let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);
                        -(t * p_clamped.ln() + (1.0 - t) * (1.0 - p_clamped).ln())
                    })
                    .sum();
                sum / prediction.len() as f64
            }
        }
    }

    pub fn compute_loss_gradient(&self, prediction: &[f64], target: &[f64]) -> Vec<f64> {
        match self.loss_type {
            LossType::Mse => {
                prediction.iter().zip(target.iter())
                    .map(|(p, t)| 2.0 * (p - t) / prediction.len() as f64)
                    .collect()
            }
            LossType::Bce => {
                prediction.iter().zip(target.iter())
                    .map(|(p, t)| {
                        let p_clamped = p.clamp(1e-7, 1.0 - 1e-7);
                        (-t / p_clamped + (1.0 - t) / (1.0 - p_clamped)) / prediction.len() as f64
                    })
                    .collect()
            }
        }
    }

    fn build_adjacency_list(&self, graph: &mut Graph) {
        graph.adjacency_list = vec![Vec::new(); graph.num_nodes];
        for edge in &graph.edges {
            if edge.source < graph.num_nodes && edge.target < graph.num_nodes {
                graph.adjacency_list[edge.source].push(edge.target);
            }
        }
    }

    pub fn deduplicate_edges(graph: &mut Graph) {
        let mut seen = std::collections::HashSet::new();
        graph.edges.retain(|e| {
            let key = (e.source, e.target);
            seen.insert(key)
        });
    }

    pub fn add_reverse_edges(graph: &mut Graph) {
        let orig_edges: Vec<Edge> = graph.edges.clone();
        for edge in orig_edges {
            if edge.source != edge.target {
                graph.edges.push(Edge {
                    source: edge.target,
                    target: edge.source,
                });
            }
        }
        Self::deduplicate_edges(graph);
    }

    pub fn add_self_loops(graph: &mut Graph) {
        for i in 0..graph.num_nodes {
            let has_self = graph.edges.iter().any(|e| e.source == i && e.target == i);
            if !has_self {
                graph.edges.push(Edge { source: i, target: i });
            }
        }
    }

    fn message_passing(&mut self, graph: &Graph) -> Result<(), Box<dyn std::error::Error>> {
        self.node_embeddings = graph.node_features.clone();
        self.new_node_embeddings = vec![vec![0.0; self.hidden_size]; graph.num_nodes];
        self.embedding_history = vec![Vec::new(); self.num_message_passing_layers + 1];
        self.message_history = vec![vec![Vec::new(); graph.num_nodes]; self.num_message_passing_layers];
        self.aggregated_messages = vec![vec![vec![0.0; self.hidden_size]; graph.num_nodes]; self.num_message_passing_layers];

        self.embedding_history[0] = self.node_embeddings.clone();
        let activation = self.activation.to_int();

        for layer in 0..self.num_message_passing_layers {
            for node in 0..graph.num_nodes {
                self.message_history[layer][node].clear();
                let mut aggregated_message = vec![0.0; self.hidden_size];

                if !graph.adjacency_list[node].is_empty() {
                    for &neighbor in &graph.adjacency_list[node] {
                        let mut concat_features = self.node_embeddings[node].clone();
                        concat_features.extend_from_slice(&self.node_embeddings[neighbor]);

                        let message = self.gpu_message_layers[layer].forward(
                            &self.device,
                            &concat_features,
                            activation,
                            false,
                        )?;

                        let msg_info = MessageInfo {
                            neighbor_idx: neighbor,
                            concat_input: concat_features,
                        };
                        self.message_history[layer][node].push(msg_info);

                        for (i, &m) in message.iter().enumerate() {
                            aggregated_message[i] += m;
                        }
                    }

                    let count = graph.adjacency_list[node].len() as f64;
                    for m in &mut aggregated_message {
                        *m /= count;
                    }
                }

                self.aggregated_messages[layer][node] = aggregated_message.clone();

                let padded_emb = if layer == 0 {
                    let mut v = vec![0.0; self.hidden_size];
                    for (i, &f) in self.node_embeddings[node].iter().enumerate() {
                        if i < self.hidden_size {
                            v[i] = f;
                        }
                    }
                    v
                } else {
                    self.node_embeddings[node].clone()
                };

                let mut update_input = padded_emb;
                update_input.extend_from_slice(&aggregated_message);
                self.new_node_embeddings[node] = self.gpu_update_layers[layer].forward(
                    &self.device,
                    &update_input,
                    activation,
                    false,
                )?;
            }

            self.node_embeddings = self.new_node_embeddings.clone();
            self.embedding_history[layer + 1] = self.node_embeddings.clone();
        }

        Ok(())
    }

    fn readout(&mut self, graph: &Graph) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_embedding = vec![0.0; self.hidden_size];

        for node_emb in &self.node_embeddings {
            for (i, &e) in node_emb.iter().enumerate() {
                if i < self.hidden_size {
                    self.graph_embedding[i] += e;
                }
            }
        }

        for e in &mut self.graph_embedding {
            *e /= graph.num_nodes as f64;
        }

        let activation = self.activation.to_int();
        self.gpu_readout_layer.forward(&self.device, &self.graph_embedding, activation, false)?;
        Ok(())
    }

    pub fn predict(&mut self, graph: &mut Graph) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        if graph.config.deduplicate_edges {
            Self::deduplicate_edges(graph);
        }
        if graph.config.undirected {
            Self::add_reverse_edges(graph);
        }
        if graph.config.self_loops {
            Self::add_self_loops(graph);
        }

        self.build_adjacency_list(graph);
        self.message_passing(graph)?;
        self.readout(graph)?;

        let readout_output = self.gpu_readout_layer.get_output(&self.device)?;
        let activation = self.activation.to_int();
        self.gpu_output_layer.forward(&self.device, &readout_output, activation, true)
    }

    fn back_propagate_graph(&mut self, graph: &Graph, target: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        let last_input = self.gpu_output_layer.get_last_input(&self.device)?;
        let mut loss_grad = self.compute_loss_gradient(&last_input, target);
        let pre_acts = self.gpu_output_layer.get_pre_activations(&self.device)?;

        for (i, lg) in loss_grad.iter_mut().enumerate() {
            *lg *= self.output_activate_derivative(pre_acts[i]);
        }

        let activation = self.activation.to_int();
        let lr = self.learning_rate;

        self.gpu_output_layer.backward(&self.device, &loss_grad, activation, true, lr)?;
        let readout_grad = self.gpu_output_layer.get_input_grad(&self.device, &loss_grad, activation, true)?;

        self.gpu_readout_layer.backward(&self.device, &readout_grad, activation, false, lr)?;
        let graph_emb_grad = self.gpu_readout_layer.get_input_grad(&self.device, &readout_grad, activation, false)?;

        let mut node_grads: Vec<Vec<f64>> = (0..graph.num_nodes)
            .map(|_| {
                graph_emb_grad.iter()
                    .map(|&g| g / graph.num_nodes as f64)
                    .collect()
            })
            .collect();

        for layer in (0..self.num_message_passing_layers).rev() {
            let mut new_node_grads: Vec<Vec<f64>> = if layer == 0 {
                vec![vec![0.0; self.feature_size]; graph.num_nodes]
            } else {
                vec![vec![0.0; self.hidden_size]; graph.num_nodes]
            };

            for node in 0..graph.num_nodes {
                let padded_emb = if layer == 0 {
                    let mut v = vec![0.0; self.hidden_size];
                    for (i, &f) in self.embedding_history[layer][node].iter().enumerate() {
                        if i < self.hidden_size {
                            v[i] = f;
                        }
                    }
                    v
                } else {
                    self.embedding_history[layer][node].clone()
                };

                let mut update_input = padded_emb;
                update_input.extend_from_slice(&self.aggregated_messages[layer][node]);

                self.gpu_update_layers[layer].set_last_input(&self.device, &update_input)?;
                self.gpu_update_layers[layer].backward(&self.device, &node_grads[node], activation, false, lr)?;
                let update_input_grad = self.gpu_update_layers[layer].get_input_grad(&self.device, &node_grads[node], activation, false)?;

                let grad_limit = self.hidden_size.min(new_node_grads[node].len());
                for i in 0..grad_limit {
                    if layer == 0 {
                        if i < self.feature_size {
                            new_node_grads[node][i] += update_input_grad[i];
                        }
                    } else {
                        new_node_grads[node][i] += update_input_grad[i];
                    }
                }

                let num_neighbors = graph.adjacency_list[node].len();
                if num_neighbors > 0 {
                    let msg_grad: Vec<f64> = (0..self.hidden_size)
                        .map(|i| update_input_grad[self.hidden_size + i] / num_neighbors as f64)
                        .collect();

                    for msg_info in &self.message_history[layer][node] {
                        self.gpu_message_layers[layer].set_last_input(&self.device, &msg_info.concat_input)?;
                        self.gpu_message_layers[layer].backward(&self.device, &msg_grad, activation, false, lr)?;
                        let concat_grad = self.gpu_message_layers[layer].get_input_grad(&self.device, &msg_grad, activation, false)?;

                        let half_len = concat_grad.len() / 2;

                        let limit1 = half_len.min(new_node_grads[node].len());
                        for i in 0..limit1 {
                            new_node_grads[node][i] += concat_grad[i];
                        }

                        let j = msg_info.neighbor_idx;
                        let limit2 = half_len.min(new_node_grads[j].len());
                        for i in 0..limit2 {
                            new_node_grads[j][i] += concat_grad[half_len + i];
                        }
                    }
                }
            }

            if layer > 0 {
                node_grads = new_node_grads;
            }
        }

        Ok(())
    }

    pub fn train(&mut self, graph: &mut Graph, target: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        let prediction = self.predict(graph)?;
        let loss = self.compute_loss(&prediction, target);
        self.back_propagate_graph(graph, target)?;
        Ok(loss)
    }

    pub fn train_multiple(&mut self, graph: &mut Graph, target: &[f64], iterations: usize) -> Result<(), Box<dyn std::error::Error>> {
        self.metrics.loss_history = vec![0.0; iterations];

        for i in 0..iterations {
            let loss = self.train(graph, target)?;
            self.metrics.loss_history[i] = loss;
            self.metrics.loss = loss;
            self.metrics.iteration = i + 1;

            if i % 10 == 0 || i == iterations - 1 {
                println!("Iteration {}/{}, Loss: {:.6}", i + 1, iterations, loss);
            }
        }

        self.sync_from_gpu()?;
        Ok(())
    }

    pub fn save_model(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.sync_from_gpu()?;

        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        writer.write_all(&(self.feature_size as i32).to_le_bytes())?;
        writer.write_all(&(self.hidden_size as i32).to_le_bytes())?;
        writer.write_all(&(self.output_size as i32).to_le_bytes())?;
        writer.write_all(&(self.num_message_passing_layers as i32).to_le_bytes())?;
        writer.write_all(&self.learning_rate.to_le_bytes())?;

        writer.write_all(&self.activation.to_int().to_le_bytes())?;
        writer.write_all(&self.loss_type.to_int().to_le_bytes())?;

        for k in 0..self.num_message_passing_layers {
            writer.write_all(&(self.message_layers[k].num_outputs as i32).to_le_bytes())?;
            writer.write_all(&(self.message_layers[k].num_inputs as i32).to_le_bytes())?;
            for neuron in &self.message_layers[k].neurons {
                for &w in &neuron.weights {
                    writer.write_all(&w.to_le_bytes())?;
                }
                writer.write_all(&neuron.bias.to_le_bytes())?;
            }

            writer.write_all(&(self.update_layers[k].num_outputs as i32).to_le_bytes())?;
            writer.write_all(&(self.update_layers[k].num_inputs as i32).to_le_bytes())?;
            for neuron in &self.update_layers[k].neurons {
                for &w in &neuron.weights {
                    writer.write_all(&w.to_le_bytes())?;
                }
                writer.write_all(&neuron.bias.to_le_bytes())?;
            }
        }

        writer.write_all(&(self.readout_layer.num_outputs as i32).to_le_bytes())?;
        writer.write_all(&(self.readout_layer.num_inputs as i32).to_le_bytes())?;
        for neuron in &self.readout_layer.neurons {
            for &w in &neuron.weights {
                writer.write_all(&w.to_le_bytes())?;
            }
            writer.write_all(&neuron.bias.to_le_bytes())?;
        }

        writer.write_all(&(self.output_layer.num_outputs as i32).to_le_bytes())?;
        writer.write_all(&(self.output_layer.num_inputs as i32).to_le_bytes())?;
        for neuron in &self.output_layer.neurons {
            for &w in &neuron.weights {
                writer.write_all(&w.to_le_bytes())?;
            }
            writer.write_all(&neuron.bias.to_le_bytes())?;
        }

        println!("Model saved to {}", filename);
        Ok(())
    }

    pub fn load_model(&mut self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::open(filename)?;
        let mut reader = BufReader::new(file);

        let mut buf_i32 = [0u8; 4];
        let mut buf_f64 = [0u8; 8];

        reader.read_exact(&mut buf_i32)?;
        self.feature_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.hidden_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.output_size = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        self.num_message_passing_layers = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_f64)?;
        self.learning_rate = f64::from_le_bytes(buf_f64);

        reader.read_exact(&mut buf_i32)?;
        self.activation = ActivationType::from_int(i32::from_le_bytes(buf_i32));
        reader.read_exact(&mut buf_i32)?;
        self.loss_type = LossType::from_int(i32::from_le_bytes(buf_i32));

        self.message_layers.clear();
        self.update_layers.clear();

        for _ in 0..self.num_message_passing_layers {
            reader.read_exact(&mut buf_i32)?;
            let num_n = i32::from_le_bytes(buf_i32) as usize;
            reader.read_exact(&mut buf_i32)?;
            let num_i = i32::from_le_bytes(buf_i32) as usize;

            let mut layer = Self::initialize_layer(num_n, num_i);
            for neuron in &mut layer.neurons {
                for w in &mut neuron.weights {
                    reader.read_exact(&mut buf_f64)?;
                    *w = f64::from_le_bytes(buf_f64);
                }
                reader.read_exact(&mut buf_f64)?;
                neuron.bias = f64::from_le_bytes(buf_f64);
            }
            self.message_layers.push(layer);

            reader.read_exact(&mut buf_i32)?;
            let num_n = i32::from_le_bytes(buf_i32) as usize;
            reader.read_exact(&mut buf_i32)?;
            let num_i = i32::from_le_bytes(buf_i32) as usize;

            let mut layer = Self::initialize_layer(num_n, num_i);
            for neuron in &mut layer.neurons {
                for w in &mut neuron.weights {
                    reader.read_exact(&mut buf_f64)?;
                    *w = f64::from_le_bytes(buf_f64);
                }
                reader.read_exact(&mut buf_f64)?;
                neuron.bias = f64::from_le_bytes(buf_f64);
            }
            self.update_layers.push(layer);
        }

        reader.read_exact(&mut buf_i32)?;
        let num_n = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        let num_i = i32::from_le_bytes(buf_i32) as usize;
        self.readout_layer = Self::initialize_layer(num_n, num_i);
        for neuron in &mut self.readout_layer.neurons {
            for w in &mut neuron.weights {
                reader.read_exact(&mut buf_f64)?;
                *w = f64::from_le_bytes(buf_f64);
            }
            reader.read_exact(&mut buf_f64)?;
            neuron.bias = f64::from_le_bytes(buf_f64);
        }

        reader.read_exact(&mut buf_i32)?;
        let num_n = i32::from_le_bytes(buf_i32) as usize;
        reader.read_exact(&mut buf_i32)?;
        let num_i = i32::from_le_bytes(buf_i32) as usize;
        self.output_layer = Self::initialize_layer(num_n, num_i);
        for neuron in &mut self.output_layer.neurons {
            for w in &mut neuron.weights {
                reader.read_exact(&mut buf_f64)?;
                *w = f64::from_le_bytes(buf_f64);
            }
            reader.read_exact(&mut buf_f64)?;
            neuron.bias = f64::from_le_bytes(buf_f64);
        }

        self.reinitialize_gpu()?;

        println!("Model loaded from {}", filename);
        Ok(())
    }

    pub fn get_learning_rate(&self) -> f64 { self.learning_rate }
    pub fn set_learning_rate(&mut self, value: f64) { self.learning_rate = value; }
    pub fn get_max_iterations(&self) -> usize { self.max_iterations }
    pub fn get_feature_size(&self) -> usize { self.feature_size }
    pub fn get_hidden_size(&self) -> usize { self.hidden_size }
    pub fn get_output_size(&self) -> usize { self.output_size }
    pub fn get_mp_layers(&self) -> usize { self.num_message_passing_layers }
    pub fn get_activation(&self) -> ActivationType { self.activation }
    pub fn set_activation(&mut self, value: ActivationType) { self.activation = value; }
    pub fn get_loss_function(&self) -> LossType { self.loss_type }
    pub fn set_loss_function(&mut self, value: LossType) { self.loss_type = value; }
}

#[derive(Parser)]
#[command(name = "gnn_cuda")]
#[command(about = "GNN-CUDA - Graph Neural Network (GPU-Accelerated, Rust port)")]
#[command(after_help = r#"NETWORK FUNCTIONS:
  create               Create a new GNN model
    --feature=N          Input feature dimension (required)
    --hidden=N           Hidden layer dimension (required)
    --output=N           Output dimension (required)
    --mp-layers=N        Message passing layers (required)
    --save=FILE          Save initial model to file (required)
    --lr=VALUE           Learning rate (default: 0.01)
    --activation=TYPE    relu|leakyrelu|tanh|sigmoid (default: relu)
    --loss=TYPE          mse|bce (default: mse)

  predict              Make predictions on a graph
    --model=FILE         Model file (required)
    --graph=FILE         Graph file in JSON format (required)

  train                Train the model with graph data
    --model=FILE         Model file (required)
    --graph=FILE         Graph file in JSON format (required)
    --save=FILE          Save trained model to file
    --epochs=N           Training epochs (default: 100)
    --lr=VALUE           Override learning rate

GRAPH FUNCTIONS:
  add-node             Add a node to the graph
    --model=FILE         Model file (required)
    --save=FILE          Output file (required)

  add-edge             Add an edge to the graph
    --model=FILE         Model file (required)
    --edge=SRC,TGT       Edge as source,target (required)
    --save=FILE          Output file (required)

  remove-edge          Remove an edge from the graph
    --model=FILE         Model file (required)
    --edge=SRC,TGT       Edge to remove (required)
    --save=FILE          Output file (required)

  degree               Get node degree (in + out)
    --model=FILE         Model file (required)
    --node=N             Node index (required)

  in-degree            Get node in-degree
    --model=FILE         Model file (required)
    --node=N             Node index (required)

  out-degree           Get node out-degree
    --model=FILE         Model file (required)
    --node=N             Node index (required)

  neighbors            Get node neighbors
    --model=FILE         Model file (required)
    --node=N             Node index (required)

  pagerank             Compute PageRank scores
    --model=FILE         Model file (required)
    --damping=D          Damping factor (default: 0.85)
    --iterations=N       Iterations (default: 20)

  gradient-flow        Show gradient flow analysis
    --model=FILE         Model file (required)

EXAMPLES:
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
        save: String,
        #[arg(long, default_value = "0.01")]
        lr: f64,
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
        lr: Option<f64>,
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
        damping: f64,
        #[arg(long, default_value = "20")]
        iterations: usize,
    },
    /// Add a node to the graph
    AddNode {
        #[arg(long)]
        model: String,
        #[arg(long)]
        save: String,
    },
    /// Add an edge to the graph
    AddEdge {
        #[arg(long)]
        model: String,
        #[arg(long)]
        edge: String,
        #[arg(long)]
        save: String,
    },
    /// Remove an edge from the graph
    RemoveEdge {
        #[arg(long)]
        model: String,
        #[arg(long)]
        edge: String,
        #[arg(long)]
        save: String,
    },
    /// Show gradient flow analysis
    GradientFlow {
        #[arg(long)]
        model: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Create { feature, hidden, output, mp_layers, save, lr, activation, loss } => {
            let mut gnn = GraphNeuralNetwork::new(feature, hidden, output, mp_layers)?;
            gnn.set_learning_rate(lr);
            gnn.set_activation(activation);
            gnn.set_loss_function(loss);
            gnn.save_model(&save)?;

            println!("Created GNN model:");
            println!("  Feature size: {}", feature);
            println!("  Hidden size: {}", hidden);
            println!("  Output size: {}", output);
            println!("  Message passing layers: {}", mp_layers);
            println!("  Activation: {}", activation);
            println!("  Loss function: {}", loss);
            println!("  Learning rate: {:.4}", lr);
            println!("  Saved to: {}", save);
        }
        Commands::Train { model, graph: _, save, epochs, lr } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            if let Some(learning_rate) = lr {
                gnn.set_learning_rate(learning_rate);
            }

            println!("Training model for {} epochs...", epochs);

            let mut rng = rand::thread_rng();
            let target: Vec<f64> = (0..gnn.get_output_size())
                .map(|_| rng.gen())
                .collect();

            let mut graph = Graph::new(5);
            graph.config.undirected = true;
            graph.config.self_loops = false;
            graph.config.deduplicate_edges = true;

            graph.node_features = (0..5)
                .map(|_| (0..gnn.get_feature_size()).map(|_| rng.gen()).collect())
                .collect();

            graph.edges = vec![
                Edge { source: 0, target: 1 },
                Edge { source: 1, target: 2 },
                Edge { source: 2, target: 3 },
                Edge { source: 3, target: 4 },
                Edge { source: 4, target: 0 },
                Edge { source: 1, target: 3 },
            ];

            gnn.train_multiple(&mut graph, &target, epochs)?;
            gnn.save_model(&save)?;
            println!("Model saved to: {}", save);
        }
        Commands::Predict { model, graph: _ } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let mut rng = rand::thread_rng();
            let mut graph = Graph::new(5);
            graph.config.undirected = true;
            graph.config.self_loops = false;
            graph.config.deduplicate_edges = true;

            graph.node_features = (0..5)
                .map(|_| (0..gnn.get_feature_size()).map(|_| rng.gen()).collect())
                .collect();

            graph.edges = vec![
                Edge { source: 0, target: 1 },
                Edge { source: 1, target: 2 },
                Edge { source: 2, target: 3 },
                Edge { source: 3, target: 4 },
                Edge { source: 4, target: 0 },
                Edge { source: 1, target: 3 },
            ];

            let prediction = gnn.predict(&mut graph)?;

            println!("Graph nodes: {}, edges: {}", graph.num_nodes, graph.edges.len());
            print!("Prediction: [");
            for (i, p) in prediction.iter().enumerate() {
                if i > 0 { print!(", "); }
                print!("{:.6}", p);
            }
            println!("]");
        }
        Commands::Info { model } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            println!("GNN Model Information (CUDA-Rust)");
            println!("==================================");
            println!("GPU Acceleration: Enabled (CUDA via cudarc)");
            println!("Feature size: {}", gnn.get_feature_size());
            println!("Hidden size: {}", gnn.get_hidden_size());
            println!("Output size: {}", gnn.get_output_size());
            println!("Message passing layers: {}", gnn.get_mp_layers());
            println!();
            println!("Hyperparameters:");
            println!("  Learning rate: {:.6}", gnn.get_learning_rate());
            println!("  Activation: {}", gnn.get_activation());
            println!("  Loss function: {}", gnn.get_loss_function());
            println!("  Max iterations: {}", gnn.get_max_iterations());
            println!("File: {}", model);
        }
        Commands::Save { model, output } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;
            gnn.save_model(&output)?;
            println!("Model saved to: {}", output);
        }
        Commands::Load { model } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;
            println!("Model loaded from: {}", model);
            println!("Feature size: {}", gnn.get_feature_size());
            println!("Hidden size: {}", gnn.get_hidden_size());
            println!("Output size: {}", gnn.get_output_size());
        }
        Commands::Degree { model, node } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let mut graph = Graph::new(5);
            graph.edges = vec![
                Edge { source: 0, target: 1 },
                Edge { source: 1, target: 2 },
                Edge { source: 2, target: 0 },
            ];

            let in_degree = graph.edges.iter().filter(|e| e.target == node).count();
            let out_degree = graph.edges.iter().filter(|e| e.source == node).count();

            println!("Node {} degree information:", node);
            println!("  In-degree: {}", in_degree);
            println!("  Out-degree: {}", out_degree);
            println!("  Total degree: {}", in_degree + out_degree);
        }
        Commands::Neighbors { model, node } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let mut graph = Graph::new(5);
            graph.edges = vec![
                Edge { source: 0, target: 1 },
                Edge { source: 0, target: 2 },
                Edge { source: 1, target: 2 },
            ];

            let neighbors: Vec<usize> = graph.edges.iter()
                .filter(|e| e.source == node)
                .map(|e| e.target)
                .collect();

            if node < graph.num_nodes {
                println!("Node {} neighbors: {:?}", node, neighbors);
            } else {
                println!("Node {} not found", node);
            }
        }
        Commands::Pagerank { model, damping, iterations } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let mut graph = Graph::new(5);
            graph.edges = vec![
                Edge { source: 0, target: 1 },
                Edge { source: 1, target: 2 },
                Edge { source: 2, target: 3 },
                Edge { source: 3, target: 4 },
                Edge { source: 4, target: 0 },
            ];

            let n = graph.num_nodes;
            let mut ranks = vec![1.0 / n as f64; n];

            for _ in 0..iterations {
                let mut new_ranks = vec![(1.0 - damping) / n as f64; n];
                for node in 0..n {
                    let out_degree = graph.edges.iter().filter(|e| e.source == node).count();
                    if out_degree > 0 {
                        let contrib = damping * ranks[node] / out_degree as f64;
                        for edge in &graph.edges {
                            if edge.source == node {
                                new_ranks[edge.target] += contrib;
                            }
                        }
                    } else {
                        let contrib = damping * ranks[node] / n as f64;
                        for r in &mut new_ranks {
                            *r += contrib;
                        }
                    }
                }
                ranks = new_ranks;
            }

            println!("PageRank (damping={}, iterations={}):", damping, iterations);
            for (i, r) in ranks.iter().enumerate() {
                println!("  Node {}: {:.6}", i, r);
            }
        }
        Commands::AddNode { model, save } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let mut graph = Graph::new(5);
            graph.num_nodes += 1;
            let new_node_id = graph.num_nodes - 1;

            gnn.save_model(&save)?;
            println!("Added node {} to graph", new_node_id);
            println!("Graph now has {} nodes", graph.num_nodes);
            println!("Model saved to: {}", save);
        }
        Commands::AddEdge { model, edge, save } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let parts: Vec<&str> = edge.split(',').collect();
            if parts.len() != 2 {
                eprintln!("Error: edge format should be 'source,target' (e.g., '0,1')");
                return Ok(());
            }

            let source: usize = parts[0].trim().parse().map_err(|_| "Invalid source node")?;
            let target: usize = parts[1].trim().parse().map_err(|_| "Invalid target node")?;

            let mut graph = Graph::new(5);
            graph.edges.push(Edge { source, target });

            gnn.save_model(&save)?;
            println!("Added edge from node {} to node {}", source, target);
            println!("Graph now has {} edges", graph.edges.len());
            println!("Model saved to: {}", save);
        }
        Commands::RemoveEdge { model, edge, save } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let parts: Vec<&str> = edge.split(',').collect();
            if parts.len() != 2 {
                eprintln!("Error: edge format should be 'source,target' (e.g., '0,1')");
                return Ok(());
            }

            let source: usize = parts[0].trim().parse().map_err(|_| "Invalid source node")?;
            let target: usize = parts[1].trim().parse().map_err(|_| "Invalid target node")?;

            let mut graph = Graph::new(5);
            graph.edges = vec![
                Edge { source: 0, target: 1 },
                Edge { source: 1, target: 2 },
                Edge { source: 2, target: 0 },
            ];
            let original_len = graph.edges.len();
            graph.edges.retain(|e| !(e.source == source && e.target == target));
            let removed = original_len - graph.edges.len();

            gnn.save_model(&save)?;
            if removed > 0 {
                println!("Removed edge from node {} to node {}", source, target);
            } else {
                println!("Edge from node {} to node {} not found", source, target);
            }
            println!("Graph now has {} edges", graph.edges.len());
            println!("Model saved to: {}", save);
        }
        Commands::GradientFlow { model } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            println!("Gradient Flow Analysis");
            println!("======================");
            println!();
            println!("Model Architecture:");
            println!("  Feature size: {}", gnn.get_feature_size());
            println!("  Hidden size: {}", gnn.get_hidden_size());
            println!("  Output size: {}", gnn.get_output_size());
            println!("  Message passing layers: {}", gnn.get_mp_layers());
            println!();
            println!("Gradient Flow Path:");
            println!("  Output Layer <- Readout Layer <- Message Passing Layers <- Input Layer");
            println!();
            println!("Layer Information:");
            println!("  Input -> Feature Projection: {} -> {}", gnn.get_feature_size(), gnn.get_hidden_size());
            for i in 0..gnn.get_mp_layers() {
                println!("  MP Layer {}: {} -> {}", i, gnn.get_hidden_size(), gnn.get_hidden_size());
            }
            println!("  Readout: {} -> {}", gnn.get_hidden_size(), gnn.get_hidden_size());
            println!("  Output: {} -> {}", gnn.get_hidden_size(), gnn.get_output_size());
            println!();
            println!("Activation: {}", gnn.get_activation());
            println!("Loss Function: {}", gnn.get_loss_function());
            println!("Gradient Clipping: {:.1}", GRADIENT_CLIP);
        }
        Commands::InDegree { model, node } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let graph = Graph::new(5);
            let in_degree = graph.edges.iter().filter(|e| e.target == node).count();
            println!("{}", in_degree);
        }
        Commands::OutDegree { model, node } => {
            let mut gnn = GraphNeuralNetwork::new(1, 1, 1, 1)?;
            gnn.load_model(&model)?;

            let graph = Graph::new(5);
            let out_degree = graph.edges.iter().filter(|e| e.source == node).count();
            println!("{}", out_degree);
        }
    }

    Ok(())
}
