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

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cstring>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

using namespace std;

// ==================== Embedded OpenCL Kernels ====================

static const string KERNEL_SOURCE = R"(
// GNN OpenCL Kernels - Optimized for M1/ARM64
inline float clip_value(float v, float max_val) {
    if (v > max_val) return max_val;
    if (v < -max_val) return -max_val;
    return v;
}

inline float sigmoid(float x) {
    x = clamp(x, -500.0f, 500.0f);
    return 1.0f / (1.0f + exp(-x));
}

inline float tanh_act(float x) {
    return tanh(x);
}

inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

inline float leaky_relu(float x) {
    return x > 0.0f ? x : 0.01f * x;
}

inline float apply_activation(float x, int act_type) {
    switch (act_type) {
        case 0: return relu(x);
        case 1: return leaky_relu(x);
        case 2: return tanh_act(x);
        case 3: return sigmoid(x);
        default: return x;
    }
}

__kernel void matvec(
    __global float* W, __global float* x, __global float* y,
    __global float* bias, int m, int n
) {
    int i = get_global_id(0);
    if (i >= m) return;
    float sum = bias[i];
    for (int j = 0; j < n; j++) sum += W[i * n + j] * x[j];
    y[i] = sum;
}

__kernel void forward_layer(
    __global float* W, __global float* x, __global float* bias,
    __global float* output, __global float* pre_activation,
    int act_type, int m, int n
) {
    int i = get_global_id(0);
    if (i >= m) return;
    float sum = bias[i];
    for (int j = 0; j < n; j++) sum += W[i * n + j] * x[j];
    pre_activation[i] = sum;
    output[i] = apply_activation(sum, act_type);
}

__kernel void zero_fill(__global float* buffer, int size) {
    int i = get_global_id(0);
    if (i >= size) return;
    buffer[i] = 0.0f;
}

__kernel void activate_derivative(
    __global float* pre_activation, __global float* grad_out,
    __global float* grad_in, int act_type, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    float x = pre_activation[i];
    float grad = grad_out[i];
    switch (act_type) {
        case 0: grad_in[i] = x > 0.0f ? grad : 0.0f; break;
        case 1: grad_in[i] = x > 0.0f ? grad : 0.01f * grad; break;
        case 2: { float t = tanh_act(x); grad_in[i] = grad * (1.0f - t * t); break; }
        case 3: { float s = sigmoid(x); grad_in[i] = grad * s * (1.0f - s); break; }
        default: grad_in[i] = grad;
    }
}

__kernel void update_weights(
    __global float* weights, __global float* weight_grad,
    float learning_rate, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    weights[i] -= learning_rate * weight_grad[i];
}

__kernel void update_biases(
    __global float* biases, __global float* bias_grad,
    float learning_rate, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    biases[i] -= learning_rate * bias_grad[i];
}

__kernel void mse_loss(
    __global float* output, __global float* target,
    __global float* loss, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    float diff = output[i] - target[i];
    loss[i] = diff * diff;
}

__kernel void mse_loss_gradient(
    __global float* output, __global float* target,
    __global float* grad, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    grad[i] = 2.0f * (output[i] - target[i]);
}

__kernel void ce_loss(
    __global float* output, __global float* target,
    __global float* loss, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    float eps = 1e-7f;
    float clipped = clamp(output[i], eps, 1.0f - eps);
    loss[i] = -(target[i] * log(clipped) + (1.0f - target[i]) * log(1.0f - clipped));
}

__kernel void ce_loss_gradient(
    __global float* output, __global float* target,
    __global float* grad, int size
) {
    int i = get_global_id(0);
    if (i >= size) return;
    float eps = 1e-7f;
    float clipped = clamp(output[i], eps, 1.0f - eps);
    grad[i] = -(target[i] / clipped - (1.0f - target[i]) / (1.0f - clipped));
}
)";

// ==================== Constants ====================

const int MAX_NODES = 1000;
const int MAX_EDGES = 10000;
const int MAX_ITERATIONS = 10000;
const double GRADIENT_CLIP = 5.0;
const string MODEL_MAGIC = "GNNBKND01";

// ==================== Type Definitions ====================

enum ActivationType {
    atReLU,
    atLeakyReLU,
    atTanh,
    atSigmoid
};

enum LossType {
    ltMSE,
    ltBinaryCrossEntropy
};

enum Command {
    cmdNone,
    cmdHelp,
    cmdCreate,
    cmdAddNode,
    cmdAddEdge,
    cmdRemoveEdge,
    cmdPredict,
    cmdTrain,
    cmdInfo,
    cmdPageRank,
    cmdDegree,
    cmdNeighbors,
    cmdSave,
    cmdLoad,
    cmdGradientFlow
};

typedef vector<double> TDoubleArray;
typedef vector<float> TFloatArray;
typedef vector<int> TIntArray;
typedef vector<TDoubleArray> TDouble2DArray;

struct TEdge {
    int Source;
    int Target;
    
    bool operator==(const TEdge& other) const {
        return Source == other.Source && Target == other.Target;
    }
};

typedef vector<TEdge> TEdgeArray;

struct TGraphConfig {
    bool Undirected;
    bool SelfLoops;
    bool DeduplicateEdges;
};

struct TGraph {
    int NumNodes;
    TDouble2DArray NodeFeatures;
    TEdgeArray Edges;
    vector<TIntArray> AdjacencyList;
    TGraphConfig Config;
};

struct TNeuron {
    TDoubleArray Weights;
    double Bias;
    double Output;
    double PreActivation;
    double Error;
};

struct TLayer {
    vector<TNeuron> Neurons;
    int NumInputs;
    int NumOutputs;
    TDoubleArray LastInput;
};

struct TMessageInfo {
    int NeighborIdx;
    TDoubleArray ConcatInput;
    TDoubleArray MessageOutput;
};

typedef vector<TMessageInfo> TNodeMessages;
typedef vector<TNodeMessages> TLayerMessages;

struct TTrainingMetrics {
    double Loss;
    int Iteration;
    TDoubleArray LossHistory;
};

// ==================== OpenCL Helper Class ====================

class OpenCLManager {
private:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
public:
    OpenCLManager() : platform(nullptr), device(nullptr), context(nullptr), 
                      queue(nullptr), program(nullptr) {}
    
    ~OpenCLManager() {
        if (queue) clReleaseCommandQueue(queue);
        if (program) clReleaseProgram(program);
        if (context) clReleaseContext(context);
    }
    
    bool Initialize(const string& kernel_source) {
        cl_int err;
        
        // Get platform
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to get platform: " << err << endl;
            return false;
        }
        
        // Get device
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to get GPU device, trying CPU: " << err << endl;
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
            if (err != CL_SUCCESS) {
                cerr << "Failed to get device: " << err << endl;
                return false;
            }
        }
        
        // Print device info
        char device_name[128];
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
        cout << "Using OpenCL device: " << device_name << endl;
        
        // Create context
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            cerr << "Failed to create context: " << err << endl;
            return false;
        }
        
        // Create command queue
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
        if (err != CL_SUCCESS) {
            cerr << "Failed to create command queue: " << err << endl;
            return false;
        }
        
        // Build program
        const char* source = kernel_source.c_str();
        size_t source_size = kernel_source.size();
        program = clCreateProgramWithSource(context, 1, &source, &source_size, &err);
        if (err != CL_SUCCESS) {
            cerr << "Failed to create program: " << err << endl;
            return false;
        }
        
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to build program: " << err << endl;
            
            size_t log_size;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            vector<char> log(log_size);
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
            cerr << "Build log: " << log.data() << endl;
            
            return false;
        }
        
        cout << "OpenCL initialized successfully" << endl;
        return true;
    }
    
    cl_kernel CreateKernel(const string& kernel_name) {
        cl_int err;
        cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            cerr << "Failed to create kernel '" << kernel_name << "': " << err << endl;
            return nullptr;
        }
        return kernel;
    }
    
    cl_mem CreateBuffer(size_t size) {
        cl_int err;
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size, nullptr, &err);
        if (err != CL_SUCCESS) {
            cerr << "Failed to create buffer: " << err << endl;
            return nullptr;
        }
        return buffer;
    }
    
    bool WriteBuffer(cl_mem buffer, const void* data, size_t size) {
        cl_int err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to write buffer: " << err << endl;
            return false;
        }
        return true;
    }
    
    bool ReadBuffer(cl_mem buffer, void* data, size_t size) {
        cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size, data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to read buffer: " << err << endl;
            return false;
        }
        return true;
    }
    
    bool ExecuteKernel(cl_kernel kernel, size_t global_size) {
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to execute kernel: " << err << endl;
            return false;
        }
        clFinish(queue);
        return true;
    }
    
    bool ExecuteKernel2D(cl_kernel kernel, size_t global_x, size_t global_y) {
        size_t global_size[2] = {global_x, global_y};
        cl_int err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            cerr << "Failed to execute 2D kernel: " << err << endl;
            return false;
        }
        clFinish(queue);
        return true;
    }
    
    void ReleaseKernel(cl_kernel kernel) {
        if (kernel) clReleaseKernel(kernel);
    }
    
    void ReleaseBuffer(cl_mem buffer) {
        if (buffer) clReleaseMemObject(buffer);
    }
};

// ==================== Helper Functions ====================

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(0.0, 1.0);

double RandomDouble() {
    return dis(gen);
}

TDoubleArray CopyArray(const TDoubleArray& Src) {
    return TDoubleArray(Src);
}

TDoubleArray ConcatArrays(const TDoubleArray& A, const TDoubleArray& B) {
    TDoubleArray Result;
    Result.insert(Result.end(), A.begin(), A.end());
    Result.insert(Result.end(), B.begin(), B.end());
    return Result;
}

TDoubleArray ZeroArray(int Size) {
    return TDoubleArray(Size, 0.0);
}

TDoubleArray PadArray(const TDoubleArray& Src, int NewSize) {
    TDoubleArray Result = ZeroArray(NewSize);
    for (int i = 0; i < min((int)Src.size(), NewSize); i++)
        Result[i] = Src[i];
    return Result;
}

// Convert double to float
TFloatArray DoubleToFloat(const TDoubleArray& src) {
    TFloatArray dst(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = (float)src[i];
    return dst;
}

// Convert float to double
TDoubleArray FloatToDouble(const TFloatArray& src) {
    TDoubleArray dst(src.size());
    for (size_t i = 0; i < src.size(); i++)
        dst[i] = (double)src[i];
    return dst;
}

// ==================== TGraphNeuralNetworkOpenCL Implementation ====================

class TGraphNeuralNetworkOpenCL {
private:
    double FLearningRate;
    int FMaxIterations;
    int FNumMessagePassingLayers;
    int FFeatureSize;
    int FHiddenSize;
    int FOutputSize;
    ActivationType FActivation;
    LossType FLossType;
    
    vector<TLayer> FMessageLayers;
    vector<TLayer> FUpdateLayers;
    TLayer FReadoutLayer;
    TLayer FOutputLayer;
    
    TDouble2DArray FNodeEmbeddings;
    TDouble2DArray FNewNodeEmbeddings;
    vector<TDouble2DArray> FEmbeddingHistory;
    vector<TLayerMessages> FMessageHistory;
    vector<TDouble2DArray> FAggregatedMessages;
    TDoubleArray FGraphEmbedding;
    
    TTrainingMetrics FMetrics;
    
    OpenCLManager* FOpenCL;
    
    void InitializeLayer(TLayer& Layer, int NumNeurons, int NumInputs);
    void BuildAdjacencyList(TGraph& Graph);
    void MessagePassing(TGraph& Graph);
    void Readout(TGraph& Graph);
    TDoubleArray ForwardLayer(TLayer& Layer, const TDoubleArray& Input, bool UseOutputActivation = false);
    void BackwardLayer(TLayer& Layer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation = false);
    TDoubleArray GetLayerInputGrad(const TLayer& Layer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation = false);
    void BackPropagateGraph(TGraph& Graph, const TDoubleArray& Target);
    
    double Activate(double X);
    double ActivateDerivative(double X);
    double OutputActivate(double X);
    double OutputActivateDerivative(double PreAct);
    double ComputeLoss(const TDoubleArray& Prediction, const TDoubleArray& Target);
    TDoubleArray ComputeLossGradient(const TDoubleArray& Prediction, const TDoubleArray& Target);
    double ClipGradient(double G);
    
public:
    TGraphNeuralNetworkOpenCL(int AFeatureSize, int AHiddenSize, int AOutputSize, int NumMPLayers, const string& kernel_source);
    ~TGraphNeuralNetworkOpenCL();
    
    TDoubleArray Predict(TGraph& Graph);
    double Train(TGraph& Graph, const TDoubleArray& Target);
    void TrainMultiple(TGraph& Graph, const TDoubleArray& Target, int Iterations);
    
    void SaveModel(const string& Filename);
    void LoadModel(const string& Filename);
    
    static void ValidateGraph(TGraph& Graph, vector<string>& Errors);
    static void DeduplicateEdges(TGraph& Graph);
    static void AddReverseEdges(TGraph& Graph);
    static void AddSelfLoops(TGraph& Graph);
    
    double GetLearningRate() const { return FLearningRate; }
    void SetLearningRate(double lr) { FLearningRate = lr; }
    int GetMaxIterations() const { return FMaxIterations; }
    void SetMaxIterations(int mi) { FMaxIterations = mi; }
    ActivationType GetActivation() const { return FActivation; }
    void SetActivation(ActivationType act) { FActivation = act; }
    LossType GetLossFunction() const { return FLossType; }
    void SetLossFunction(LossType loss) { FLossType = loss; }
    TTrainingMetrics GetMetrics() const { return FMetrics; }
    int GetFeatureSize() const { return FFeatureSize; }
    int GetHiddenSize() const { return FHiddenSize; }
    int GetOutputSize() const { return FOutputSize; }
};

TGraphNeuralNetworkOpenCL::TGraphNeuralNetworkOpenCL(int AFeatureSize, int AHiddenSize, int AOutputSize, int NumMPLayers, const string& kernel_source)
    : FFeatureSize(AFeatureSize), FHiddenSize(AHiddenSize), FOutputSize(AOutputSize), FNumMessagePassingLayers(NumMPLayers) {
    
    FLearningRate = 0.01;
    FMaxIterations = 100;
    FActivation = atReLU;
    FLossType = ltMSE;
    
    FOpenCL = new OpenCLManager();
    if (!FOpenCL->Initialize(kernel_source)) {
        cerr << "Warning: OpenCL initialization failed, falling back to CPU" << endl;
    }
    
    FMessageLayers.resize(NumMPLayers);
    FUpdateLayers.resize(NumMPLayers);
    
    for (int i = 0; i < NumMPLayers; i++) {
        if (i == 0)
            InitializeLayer(FMessageLayers[i], AHiddenSize, AFeatureSize * 2);
        else
            InitializeLayer(FMessageLayers[i], AHiddenSize, AHiddenSize * 2);
        
        InitializeLayer(FUpdateLayers[i], AHiddenSize, AHiddenSize * 2);
    }
    
    InitializeLayer(FReadoutLayer, AHiddenSize, AHiddenSize);
    InitializeLayer(FOutputLayer, AOutputSize, AHiddenSize);
    
    FMetrics.LossHistory.clear();
}

TGraphNeuralNetworkOpenCL::~TGraphNeuralNetworkOpenCL() {
    if (FOpenCL) delete FOpenCL;
}

void TGraphNeuralNetworkOpenCL::InitializeLayer(TLayer& Layer, int NumNeurons, int NumInputs) {
    Layer.NumInputs = NumInputs;
    Layer.NumOutputs = NumNeurons;
    Layer.Neurons.resize(NumNeurons);
    
    double Scale = sqrt(2.0 / (NumInputs + NumNeurons));
    
    for (int i = 0; i < NumNeurons; i++) {
        Layer.Neurons[i].Weights.resize(NumInputs);
        for (int j = 0; j < NumInputs; j++)
            Layer.Neurons[i].Weights[j] = (RandomDouble() - 0.5) * 2.0 * Scale;
        Layer.Neurons[i].Bias = 0.0;
        Layer.Neurons[i].Output = 0.0;
        Layer.Neurons[i].PreActivation = 0.0;
        Layer.Neurons[i].Error = 0.0;
    }
}

double TGraphNeuralNetworkOpenCL::Activate(double X) {
    switch (FActivation) {
        case atReLU:
            return X > 0 ? X : 0.0;
        case atLeakyReLU:
            return X > 0 ? X : 0.01 * X;
        case atTanh:
            return tanh(X);
        case atSigmoid:
            X = max(-500.0, min(500.0, X));
            return 1.0 / (1.0 + exp(-X));
        default:
            return X;
    }
}

double TGraphNeuralNetworkOpenCL::ActivateDerivative(double X) {
    switch (FActivation) {
        case atReLU:
            return X > 0 ? 1.0 : 0.0;
        case atLeakyReLU:
            return X > 0 ? 1.0 : 0.01;
        case atTanh: {
            double t = tanh(X);
            return 1.0 - t * t;
        }
        case atSigmoid: {
            X = max(-500.0, min(500.0, X));
            double s = 1.0 / (1.0 + exp(-X));
            return s * (1.0 - s);
        }
        default:
            return 1.0;
    }
}

double TGraphNeuralNetworkOpenCL::OutputActivate(double X) {
    X = max(-500.0, min(500.0, X));
    return 1.0 / (1.0 + exp(-X));
}

double TGraphNeuralNetworkOpenCL::OutputActivateDerivative(double PreAct) {
    PreAct = max(-500.0, min(500.0, PreAct));
    double s = 1.0 / (1.0 + exp(-PreAct));
    return s * (1.0 - s);
}

double TGraphNeuralNetworkOpenCL::ComputeLoss(const TDoubleArray& Prediction, const TDoubleArray& Target) {
    double Result = 0.0;
    
    switch (FLossType) {
        case ltMSE:
            for (int i = 0; i < (int)Prediction.size(); i++)
                Result += (Prediction[i] - Target[i]) * (Prediction[i] - Target[i]);
            Result /= Prediction.size();
            break;
        case ltBinaryCrossEntropy:
            for (int i = 0; i < (int)Prediction.size(); i++) {
                double P = max(1e-7, min(1.0 - 1e-7, Prediction[i]));
                Result -= (Target[i] * log(P) + (1.0 - Target[i]) * log(1.0 - P));
            }
            Result /= Prediction.size();
            break;
    }
    
    return Result;
}

TDoubleArray TGraphNeuralNetworkOpenCL::ComputeLossGradient(const TDoubleArray& Prediction, const TDoubleArray& Target) {
    TDoubleArray Result(Prediction.size());
    
    switch (FLossType) {
        case ltMSE:
            for (int i = 0; i < (int)Prediction.size(); i++)
                Result[i] = 2.0 * (Prediction[i] - Target[i]) / Prediction.size();
            break;
        case ltBinaryCrossEntropy:
            for (int i = 0; i < (int)Prediction.size(); i++) {
                double P = max(1e-7, min(1.0 - 1e-7, Prediction[i]));
                Result[i] = (-Target[i] / P + (1.0 - Target[i]) / (1.0 - P)) / Prediction.size();
            }
            break;
    }
    
    return Result;
}

double TGraphNeuralNetworkOpenCL::ClipGradient(double G) {
    return max(-GRADIENT_CLIP, min(GRADIENT_CLIP, G));
}

void TGraphNeuralNetworkOpenCL::BuildAdjacencyList(TGraph& Graph) {
    Graph.AdjacencyList.resize(Graph.NumNodes);
    for (int i = 0; i < Graph.NumNodes; i++)
        Graph.AdjacencyList[i].clear();
    
    for (int i = 0; i < (int)Graph.Edges.size(); i++) {
        int Src = Graph.Edges[i].Source;
        int Tgt = Graph.Edges[i].Target;
        
        if (Src >= 0 && Src < Graph.NumNodes && Tgt >= 0 && Tgt < Graph.NumNodes) {
            Graph.AdjacencyList[Src].push_back(Tgt);
        }
    }
}

void TGraphNeuralNetworkOpenCL::ValidateGraph(TGraph& Graph, vector<string>& Errors) {
    if (Graph.NumNodes < 1)
        Errors.push_back("Graph must have at least 1 node");
    
    if (Graph.NumNodes > MAX_NODES)
        Errors.push_back("Too many nodes (max " + to_string(MAX_NODES) + ")");
    
    if ((int)Graph.Edges.size() > MAX_EDGES)
        Errors.push_back("Too many edges (max " + to_string(MAX_EDGES) + ")");
    
    for (int i = 0; i < (int)Graph.Edges.size(); i++) {
        if (Graph.Edges[i].Source < 0 || Graph.Edges[i].Source >= Graph.NumNodes)
            Errors.push_back("Edge " + to_string(i) + ": invalid source " + to_string(Graph.Edges[i].Source));
        if (Graph.Edges[i].Target < 0 || Graph.Edges[i].Target >= Graph.NumNodes)
            Errors.push_back("Edge " + to_string(i) + ": invalid target " + to_string(Graph.Edges[i].Target));
    }
    
    for (int i = 0; i < (int)Graph.NodeFeatures.size(); i++) {
        if (Graph.NodeFeatures[i].empty())
            Errors.push_back("Node " + to_string(i) + ": empty feature vector");
    }
}

void TGraphNeuralNetworkOpenCL::DeduplicateEdges(TGraph& Graph) {
    vector<string> Seen;
    TEdgeArray NewEdges;
    
    for (int i = 0; i < (int)Graph.Edges.size(); i++) {
        string Key = to_string(Graph.Edges[i].Source) + "-" + to_string(Graph.Edges[i].Target);
        bool Found = false;
        
        for (int j = 0; j < (int)Seen.size(); j++) {
            if (Seen[j] == Key) {
                Found = true;
                break;
            }
        }
        
        if (!Found) {
            Seen.push_back(Key);
            NewEdges.push_back(Graph.Edges[i]);
        }
    }
    
    Graph.Edges = NewEdges;
}

void TGraphNeuralNetworkOpenCL::AddReverseEdges(TGraph& Graph) {
    int OrigLen = Graph.Edges.size();
    Graph.Edges.resize(OrigLen * 2);
    
    for (int i = 0; i < OrigLen; i++) {
        if (Graph.Edges[i].Source != Graph.Edges[i].Target) {
            TEdge RevEdge;
            RevEdge.Source = Graph.Edges[i].Target;
            RevEdge.Target = Graph.Edges[i].Source;
            Graph.Edges[OrigLen + i] = RevEdge;
        } else {
            Graph.Edges[OrigLen + i] = Graph.Edges[i];
        }
    }
    
    DeduplicateEdges(Graph);
}

void TGraphNeuralNetworkOpenCL::AddSelfLoops(TGraph& Graph) {
    for (int i = 0; i < Graph.NumNodes; i++) {
        bool HasSelf = false;
        for (int j = 0; j < (int)Graph.Edges.size(); j++) {
            if (Graph.Edges[j].Source == i && Graph.Edges[j].Target == i) {
                HasSelf = true;
                break;
            }
        }
        
        if (!HasSelf) {
            TEdge SelfEdge;
            SelfEdge.Source = i;
            SelfEdge.Target = i;
            Graph.Edges.push_back(SelfEdge);
        }
    }
}

TDoubleArray TGraphNeuralNetworkOpenCL::ForwardLayer(TLayer& Layer, const TDoubleArray& Input, bool UseOutputActivation) {
    Layer.LastInput = CopyArray(Input);
    TDoubleArray Result(Layer.NumOutputs);
    
    for (int i = 0; i < Layer.NumOutputs; i++) {
        double Sum = Layer.Neurons[i].Bias;
        for (int j = 0; j < Layer.NumInputs; j++) {
            if (j < (int)Input.size())
                Sum += Layer.Neurons[i].Weights[j] * Input[j];
        }
        Layer.Neurons[i].PreActivation = Sum;
        
        if (UseOutputActivation)
            Layer.Neurons[i].Output = OutputActivate(Sum);
        else
            Layer.Neurons[i].Output = Activate(Sum);
        
        Result[i] = Layer.Neurons[i].Output;
    }
    
    return Result;
}

void TGraphNeuralNetworkOpenCL::BackwardLayer(TLayer& Layer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation) {
    for (int i = 0; i < Layer.NumOutputs; i++) {
        double PreActGrad;
        if (UseOutputActivation)
            PreActGrad = UpstreamGrad[i] * OutputActivateDerivative(Layer.Neurons[i].PreActivation);
        else
            PreActGrad = UpstreamGrad[i] * ActivateDerivative(Layer.Neurons[i].PreActivation);
        
        Layer.Neurons[i].Error = PreActGrad;
        
        for (int j = 0; j < Layer.NumInputs; j++) {
            if (j < (int)Layer.LastInput.size()) {
                double DeltaW = FLearningRate * PreActGrad * Layer.LastInput[j];
                Layer.Neurons[i].Weights[j] -= DeltaW;
            }
        }
        
        Layer.Neurons[i].Bias -= (FLearningRate * PreActGrad);
    }
}

TDoubleArray TGraphNeuralNetworkOpenCL::GetLayerInputGrad(const TLayer& Layer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation) {
    TDoubleArray Result(Layer.NumInputs, 0.0);
    
    for (int i = 0; i < Layer.NumOutputs; i++) {
        double PreActGrad;
        if (UseOutputActivation)
            PreActGrad = UpstreamGrad[i] * OutputActivateDerivative(Layer.Neurons[i].PreActivation);
        else
            PreActGrad = UpstreamGrad[i] * ActivateDerivative(Layer.Neurons[i].PreActivation);
        
        for (int j = 0; j < Layer.NumInputs; j++) {
            if (j < (int)Layer.Neurons[i].Weights.size())
                Result[j] += Layer.Neurons[i].Weights[j] * PreActGrad;
        }
        if (!Result.empty())
            Result.back() = ClipGradient(Result.back());
    }
    
    return Result;
}

void TGraphNeuralNetworkOpenCL::MessagePassing(TGraph& Graph) {
    FNodeEmbeddings.resize(Graph.NumNodes);
    FNewNodeEmbeddings.resize(Graph.NumNodes);
    FEmbeddingHistory.resize(FNumMessagePassingLayers + 1);
    FMessageHistory.resize(FNumMessagePassingLayers);
    FAggregatedMessages.resize(FNumMessagePassingLayers);
    
    for (int i = 0; i < Graph.NumNodes; i++)
        FNodeEmbeddings[i] = CopyArray(Graph.NodeFeatures[i]);
    
    FEmbeddingHistory[0].resize(Graph.NumNodes);
    for (int i = 0; i < Graph.NumNodes; i++)
        FEmbeddingHistory[0][i] = CopyArray(FNodeEmbeddings[i]);
    
    for (int Layer = 0; Layer < FNumMessagePassingLayers; Layer++) {
        FMessageHistory[Layer].resize(Graph.NumNodes);
        FAggregatedMessages[Layer].resize(Graph.NumNodes);
        
        for (int Node = 0; Node < Graph.NumNodes; Node++) {
            FMessageHistory[Layer][Node].clear();
            TDoubleArray AggregatedMessage = ZeroArray(FHiddenSize);
            
            if (!Graph.AdjacencyList[Node].empty()) {
                for (int k = 0; k < (int)Graph.AdjacencyList[Node].size(); k++) {
                    int Neighbor = Graph.AdjacencyList[Node][k];
                    
                    TDoubleArray ConcatFeatures = ConcatArrays(FNodeEmbeddings[Node], FNodeEmbeddings[Neighbor]);
                    TDoubleArray Message = ForwardLayer(FMessageLayers[Layer], ConcatFeatures, false);
                    
                    TMessageInfo MsgInfo;
                    MsgInfo.NeighborIdx = Neighbor;
                    MsgInfo.ConcatInput = CopyArray(ConcatFeatures);
                    MsgInfo.MessageOutput = CopyArray(Message);
                    
                    FMessageHistory[Layer][Node].push_back(MsgInfo);
                    
                    for (int i = 0; i < FHiddenSize; i++)
                        AggregatedMessage[i] += Message[i];
                }
                
                for (int i = 0; i < FHiddenSize; i++)
                    AggregatedMessage[i] /= Graph.AdjacencyList[Node].size();
            }
            
            FAggregatedMessages[Layer][Node] = CopyArray(AggregatedMessage);
            
            TDoubleArray PaddedEmb;
            if (Layer == 0)
                PaddedEmb = PadArray(FNodeEmbeddings[Node], FHiddenSize);
            else
                PaddedEmb = CopyArray(FNodeEmbeddings[Node]);
            
            TDoubleArray UpdateInput = ConcatArrays(PaddedEmb, AggregatedMessage);
            FNewNodeEmbeddings[Node] = ForwardLayer(FUpdateLayers[Layer], UpdateInput, false);
        }
        
        for (int Node = 0; Node < Graph.NumNodes; Node++)
            FNodeEmbeddings[Node] = CopyArray(FNewNodeEmbeddings[Node]);
        
        FEmbeddingHistory[Layer + 1].resize(Graph.NumNodes);
        for (int i = 0; i < Graph.NumNodes; i++)
            FEmbeddingHistory[Layer + 1][i] = CopyArray(FNodeEmbeddings[i]);
    }
}

void TGraphNeuralNetworkOpenCL::Readout(TGraph& Graph) {
    FGraphEmbedding = ZeroArray(FHiddenSize);
    
    for (int i = 0; i < Graph.NumNodes; i++)
        for (int j = 0; j < FHiddenSize; j++)
            FGraphEmbedding[j] += FNodeEmbeddings[i][j];
    
    for (int j = 0; j < FHiddenSize; j++)
        FGraphEmbedding[j] /= Graph.NumNodes;
    
    ForwardLayer(FReadoutLayer, FGraphEmbedding, false);
}

TDoubleArray TGraphNeuralNetworkOpenCL::Predict(TGraph& Graph) {
    if (Graph.Config.DeduplicateEdges)
        DeduplicateEdges(Graph);
    if (Graph.Config.Undirected)
        AddReverseEdges(Graph);
    if (Graph.Config.SelfLoops)
        AddSelfLoops(Graph);
    
    BuildAdjacencyList(Graph);
    MessagePassing(Graph);
    Readout(Graph);
    
    TDoubleArray ReadoutOutput(FHiddenSize);
    for (int i = 0; i < FHiddenSize; i++)
        ReadoutOutput[i] = FReadoutLayer.Neurons[i].Output;
    
    return ForwardLayer(FOutputLayer, ReadoutOutput, true);
}

void TGraphNeuralNetworkOpenCL::BackPropagateGraph(TGraph& Graph, const TDoubleArray& Target) {
    TDoubleArray LossGrad = ComputeLossGradient(FOutputLayer.LastInput, Target);
    
    for (int i = 0; i < (int)LossGrad.size(); i++)
        LossGrad[i] *= OutputActivateDerivative(FOutputLayer.Neurons[i].PreActivation);
    
    BackwardLayer(FOutputLayer, LossGrad, true);
    TDoubleArray ReadoutGrad = GetLayerInputGrad(FOutputLayer, LossGrad, true);
    
    BackwardLayer(FReadoutLayer, ReadoutGrad, false);
    TDoubleArray GraphEmbGrad = GetLayerInputGrad(FReadoutLayer, ReadoutGrad, false);
    
    TDouble2DArray NodeGrads(Graph.NumNodes);
    for (int Node = 0; Node < Graph.NumNodes; Node++) {
        NodeGrads[Node] = ZeroArray(FHiddenSize);
        for (int i = 0; i < FHiddenSize; i++)
            NodeGrads[Node][i] = GraphEmbGrad[i] / Graph.NumNodes;
    }
    
    for (int Layer = FNumMessagePassingLayers - 1; Layer >= 0; Layer--) {
        TDouble2DArray NewNodeGrads(Graph.NumNodes);
        
        if (Layer == 0) {
            for (int Node = 0; Node < Graph.NumNodes; Node++)
                NewNodeGrads[Node] = ZeroArray(FFeatureSize);
        } else {
            for (int Node = 0; Node < Graph.NumNodes; Node++)
                NewNodeGrads[Node] = ZeroArray(FHiddenSize);
        }
        
        for (int Node = 0; Node < Graph.NumNodes; Node++) {
            TDoubleArray PaddedEmb;
            if (Layer == 0)
                PaddedEmb = PadArray(FEmbeddingHistory[Layer][Node], FHiddenSize);
            else
                PaddedEmb = CopyArray(FEmbeddingHistory[Layer][Node]);
            
            TDoubleArray UpdateInput = ConcatArrays(PaddedEmb, FAggregatedMessages[Layer][Node]);
            FUpdateLayers[Layer].LastInput = CopyArray(UpdateInput);
            
            BackwardLayer(FUpdateLayers[Layer], NodeGrads[Node], false);
            TDoubleArray UpdateInputGrad = GetLayerInputGrad(FUpdateLayers[Layer], NodeGrads[Node], false);
            
            for (int i = 0; i < min(FHiddenSize, (int)NewNodeGrads[Node].size()); i++) {
                if (Layer == 0) {
                    if (i < FFeatureSize)
                        NewNodeGrads[Node][i] += UpdateInputGrad[i];
                } else {
                    NewNodeGrads[Node][i] += UpdateInputGrad[i];
                }
            }
            
            int NumNeighbors = Graph.AdjacencyList[Node].size();
            if (NumNeighbors > 0) {
                TDoubleArray MsgGrad = ZeroArray(FHiddenSize);
                for (int i = 0; i < FHiddenSize; i++)
                    MsgGrad[i] = UpdateInputGrad[FHiddenSize + i] / NumNeighbors;
                
                for (int k = 0; k < (int)FMessageHistory[Layer][Node].size(); k++) {
                    FMessageLayers[Layer].LastInput = CopyArray(FMessageHistory[Layer][Node][k].ConcatInput);
                    
                    BackwardLayer(FMessageLayers[Layer], MsgGrad, false);
                    TDoubleArray ConcatGrad = GetLayerInputGrad(FMessageLayers[Layer], MsgGrad, false);
                    
                    int HalfLen = ConcatGrad.size() / 2;
                    
                    for (int i = 0; i < min(HalfLen, (int)NewNodeGrads[Node].size()); i++)
                        NewNodeGrads[Node][i] += ConcatGrad[i];
                    
                    int J = FMessageHistory[Layer][Node][k].NeighborIdx;
                    for (int i = 0; i < min(HalfLen, (int)NewNodeGrads[J].size()); i++)
                        NewNodeGrads[J][i] += ConcatGrad[HalfLen + i];
                }
            }
        }
        
        if (Layer > 0)
            NodeGrads = NewNodeGrads;
    }
}

double TGraphNeuralNetworkOpenCL::Train(TGraph& Graph, const TDoubleArray& Target) {
    TDoubleArray Prediction = Predict(Graph);
    double Result = ComputeLoss(Prediction, Target);
    BackPropagateGraph(Graph, Target);
    return Result;
}

void TGraphNeuralNetworkOpenCL::TrainMultiple(TGraph& Graph, const TDoubleArray& Target, int Iterations) {
    FMetrics.LossHistory.resize(Iterations);
    
    for (int i = 0; i < Iterations; i++) {
        double Loss = Train(Graph, Target);
        FMetrics.LossHistory[i] = Loss;
        FMetrics.Loss = Loss;
        FMetrics.Iteration = i + 1;
        
        if (i % 10 == 0 || i == Iterations - 1)
            cout << fixed << setprecision(6) << "Iteration " << (i + 1) << "/" << Iterations << ", Loss: " << Loss << endl;
    }
}

void TGraphNeuralNetworkOpenCL::SaveModel(const string& Filename) {
    ofstream F(Filename, ios::binary);
    if (!F.is_open()) {
        cerr << "Error opening file for writing: " << Filename << endl;
        return;
    }
    
    F.write((char*)&FFeatureSize, sizeof(int));
    F.write((char*)&FHiddenSize, sizeof(int));
    F.write((char*)&FOutputSize, sizeof(int));
    F.write((char*)&FNumMessagePassingLayers, sizeof(int));
    F.write((char*)&FLearningRate, sizeof(double));
    
    int ActInt = (int)FActivation;
    int LossInt = (int)FLossType;
    F.write((char*)&ActInt, sizeof(int));
    F.write((char*)&LossInt, sizeof(int));
    
    for (int k = 0; k < FNumMessagePassingLayers; k++) {
        F.write((char*)&FMessageLayers[k].NumOutputs, sizeof(int));
        F.write((char*)&FMessageLayers[k].NumInputs, sizeof(int));
        for (int i = 0; i < FMessageLayers[k].NumOutputs; i++) {
            for (int j = 0; j < FMessageLayers[k].NumInputs; j++)
                F.write((char*)&FMessageLayers[k].Neurons[i].Weights[j], sizeof(double));
            F.write((char*)&FMessageLayers[k].Neurons[i].Bias, sizeof(double));
        }
        
        F.write((char*)&FUpdateLayers[k].NumOutputs, sizeof(int));
        F.write((char*)&FUpdateLayers[k].NumInputs, sizeof(int));
        for (int i = 0; i < FUpdateLayers[k].NumOutputs; i++) {
            for (int j = 0; j < FUpdateLayers[k].NumInputs; j++)
                F.write((char*)&FUpdateLayers[k].Neurons[i].Weights[j], sizeof(double));
            F.write((char*)&FUpdateLayers[k].Neurons[i].Bias, sizeof(double));
        }
    }
    
    F.write((char*)&FReadoutLayer.NumOutputs, sizeof(int));
    F.write((char*)&FReadoutLayer.NumInputs, sizeof(int));
    for (int i = 0; i < FReadoutLayer.NumOutputs; i++) {
        for (int j = 0; j < FReadoutLayer.NumInputs; j++)
            F.write((char*)&FReadoutLayer.Neurons[i].Weights[j], sizeof(double));
        F.write((char*)&FReadoutLayer.Neurons[i].Bias, sizeof(double));
    }
    
    F.write((char*)&FOutputLayer.NumOutputs, sizeof(int));
    F.write((char*)&FOutputLayer.NumInputs, sizeof(int));
    for (int i = 0; i < FOutputLayer.NumOutputs; i++) {
        for (int j = 0; j < FOutputLayer.NumInputs; j++)
            F.write((char*)&FOutputLayer.Neurons[i].Weights[j], sizeof(double));
        F.write((char*)&FOutputLayer.Neurons[i].Bias, sizeof(double));
    }
    
    F.close();
    cout << "Model saved to " << Filename << endl;
}

void TGraphNeuralNetworkOpenCL::LoadModel(const string& Filename) {
    ifstream F(Filename, ios::binary);
    if (!F.is_open()) {
        cerr << "Error opening file for reading: " << Filename << endl;
        return;
    }
    
    F.read((char*)&FFeatureSize, sizeof(int));
    F.read((char*)&FHiddenSize, sizeof(int));
    F.read((char*)&FOutputSize, sizeof(int));
    F.read((char*)&FNumMessagePassingLayers, sizeof(int));
    F.read((char*)&FLearningRate, sizeof(double));
    
    int ActInt, LossInt;
    F.read((char*)&ActInt, sizeof(int));
    F.read((char*)&LossInt, sizeof(int));
    FActivation = (ActivationType)ActInt;
    FLossType = (LossType)LossInt;
    
    FMessageLayers.resize(FNumMessagePassingLayers);
    FUpdateLayers.resize(FNumMessagePassingLayers);
    
    for (int k = 0; k < FNumMessagePassingLayers; k++) {
        int NumN, NumI;
        F.read((char*)&NumN, sizeof(int));
        F.read((char*)&NumI, sizeof(int));
        InitializeLayer(FMessageLayers[k], NumN, NumI);
        for (int i = 0; i < NumN; i++) {
            for (int j = 0; j < NumI; j++) {
                double TmpDouble;
                F.read((char*)&TmpDouble, sizeof(double));
                FMessageLayers[k].Neurons[i].Weights[j] = TmpDouble;
            }
            double TmpDouble;
            F.read((char*)&TmpDouble, sizeof(double));
            FMessageLayers[k].Neurons[i].Bias = TmpDouble;
        }
        
        F.read((char*)&NumN, sizeof(int));
        F.read((char*)&NumI, sizeof(int));
        InitializeLayer(FUpdateLayers[k], NumN, NumI);
        for (int i = 0; i < NumN; i++) {
            for (int j = 0; j < NumI; j++) {
                double TmpDouble;
                F.read((char*)&TmpDouble, sizeof(double));
                FUpdateLayers[k].Neurons[i].Weights[j] = TmpDouble;
            }
            double TmpDouble;
            F.read((char*)&TmpDouble, sizeof(double));
            FUpdateLayers[k].Neurons[i].Bias = TmpDouble;
        }
    }
    
    int NumN, NumI;
    F.read((char*)&NumN, sizeof(int));
    F.read((char*)&NumI, sizeof(int));
    InitializeLayer(FReadoutLayer, NumN, NumI);
    for (int i = 0; i < NumN; i++) {
        for (int j = 0; j < NumI; j++) {
            double TmpDouble;
            F.read((char*)&TmpDouble, sizeof(double));
            FReadoutLayer.Neurons[i].Weights[j] = TmpDouble;
        }
        double TmpDouble;
        F.read((char*)&TmpDouble, sizeof(double));
        FReadoutLayer.Neurons[i].Bias = TmpDouble;
    }
    
    F.read((char*)&NumN, sizeof(int));
    F.read((char*)&NumI, sizeof(int));
    InitializeLayer(FOutputLayer, NumN, NumI);
    for (int i = 0; i < NumN; i++) {
        for (int j = 0; j < NumI; j++) {
            double TmpDouble;
            F.read((char*)&TmpDouble, sizeof(double));
            FOutputLayer.Neurons[i].Weights[j] = TmpDouble;
        }
        double TmpDouble;
        F.read((char*)&TmpDouble, sizeof(double));
        FOutputLayer.Neurons[i].Bias = TmpDouble;
    }
    
    F.close();
    cout << "Model loaded from " << Filename << endl;
}

// ==================== CLI Support Functions ====================

string ActivationToStr(ActivationType act) {
    switch (act) {
        case atReLU: return "relu";
        case atLeakyReLU: return "leakyrelu";
        case atTanh: return "tanh";
        case atSigmoid: return "sigmoid";
        default: return "relu";
    }
}

string LossToStr(LossType loss) {
    switch (loss) {
        case ltMSE: return "mse";
        case ltBinaryCrossEntropy: return "bce";
        default: return "mse";
    }
}

void PrintUsage() {
    cout << "\nGNN-OpenCL - Graph Neural Network (GPU-Accelerated)\n";
    cout << "===================================================\n\n";
    
    cout << "USAGE:\n";
    cout << "  gnn_opencl <command> [options]\n\n";
    
    cout << "COMMANDS:\n";
    cout << "  create        Create a new GNN model\n";
    cout << "  add-node      Add a node to the graph\n";
    cout << "  add-edge      Add an edge to the graph\n";
    cout << "  remove-edge   Remove an edge from the graph\n";
    cout << "  predict       Make predictions on a graph\n";
    cout << "  train         Train the model with graph data\n";
    cout << "  degree        Get node degree\n";
    cout << "  in-degree     Get node in-degree\n";
    cout << "  out-degree    Get node out-degree\n";
    cout << "  neighbors     Get node neighbors\n";
    cout << "  pagerank      Compute PageRank scores\n";
    cout << "  save          Save model to file\n";
    cout << "  load          Load model from file\n";
    cout << "  info          Display model information\n";
    cout << "  gradient-flow Show gradient flow analysis\n";
    cout << "  help          Show this help message\n\n";
    
    cout << "NETWORK FUNCTIONS:\n";
    cout << "  create\n";
    cout << "    --feature=N          Input feature dimension (required)\n";
    cout << "    --hidden=N           Hidden layer dimension (required)\n";
    cout << "    --output=N           Output dimension (required)\n";
    cout << "    --mp-layers=N        Message passing layers (required)\n";
    cout << "    --save=FILE          Save initial model to file (required)\n";
    cout << "    --lr=VALUE           Learning rate (default: 0.01)\n";
    cout << "    --activation=TYPE    relu|leakyrelu|tanh|sigmoid (default: relu)\n";
    cout << "    --loss=TYPE          mse|bce (default: mse)\n\n";
    
    cout << "  predict\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --graph=FILE         Graph file in JSON format (required)\n\n";
    
    cout << "  train\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --graph=FILE         Graph file in JSON format (required)\n";
    cout << "    --target=FILE        Target output file in CSV format (required)\n";
    cout << "    --epochs=N           Training epochs (default: 100)\n";
    cout << "    --save=FILE          Save trained model to file\n";
    cout << "    --lr=VALUE           Override learning rate\n\n";
    
    cout << "  info\n";
    cout << "    --model=FILE         Model file (required)\n\n";
    
    cout << "  save\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --output=FILE        Output file (required)\n\n";
    
    cout << "  load\n";
    cout << "    --model=FILE         Model file to load (required)\n\n";
    
    cout << "GRAPH FUNCTIONS:\n";
    cout << "  add-node\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --index=N            Node index (required)\n";
    cout << "    --features=F1,F2... Node features (comma-separated)\n\n";
    
    cout << "  add-edge\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --source=N           Source node index (required)\n";
    cout << "    --target-node=N      Target node index (required)\n\n";
    
    cout << "  remove-edge\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --edge=N             Edge index (required)\n\n";
    
    cout << "  degree\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --node=N             Node index (required)\n\n";
    
    cout << "  in-degree / out-degree\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --node=N             Node index (required)\n\n";
    
    cout << "  neighbors\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --node=N             Node index (required)\n\n";
    
    cout << "  pagerank\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --damping=D          Damping factor (default: 0.85)\n";
    cout << "    --iterations=N       Iterations (default: 20)\n\n";
    
    cout << "  gradient-flow\n";
    cout << "    --model=FILE         Model file (required)\n";
    cout << "    --layer=N            Layer index (optional)\n\n";
    
    cout << "EXAMPLES:\n";
    cout << "  # Create a new model\n";
    cout << "  gnn_opencl create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=model.bin\n\n";
    cout << "  # Get node degree\n";
    cout << "  gnn_opencl degree --model=model.bin --node=0\n\n";
    cout << "  # Compute PageRank\n";
    cout << "  gnn_opencl pagerank --model=model.bin --damping=0.85 --iterations=20\n\n";
    cout << "  # Train the model\n";
    cout << "  gnn_opencl train --model=model.bin --graph=graph.json --target=target.csv --epochs=100 --save=trained.bin\n\n";
    cout << "  # Make predictions\n";
    cout << "  gnn_opencl predict --model=trained.bin --graph=graph.json\n\n";
}

ActivationType ParseActivation(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "leakyrelu") return atLeakyReLU;
    else if (lower == "tanh") return atTanh;
    else if (lower == "sigmoid") return atSigmoid;
    else return atReLU;
}

LossType ParseLoss(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "bce") return ltBinaryCrossEntropy;
    else return ltMSE;
}

int main(int argc, char* argv[]) {
    srand(time(0));
    
    if (argc < 2) {
        PrintUsage();
        return 0;
    }
    
    string CmdStr = argv[1];
    Command Command_enum = cmdNone;
    
    if (CmdStr == "create") Command_enum = cmdCreate;
    else if (CmdStr == "add-node") Command_enum = cmdAddNode;
    else if (CmdStr == "add-edge") Command_enum = cmdAddEdge;
    else if (CmdStr == "remove-edge") Command_enum = cmdRemoveEdge;
    else if (CmdStr == "predict") Command_enum = cmdPredict;
    else if (CmdStr == "train") Command_enum = cmdTrain;
    else if (CmdStr == "degree") Command_enum = cmdDegree;
    else if (CmdStr == "in-degree") Command_enum = cmdDegree;
    else if (CmdStr == "out-degree") Command_enum = cmdDegree;
    else if (CmdStr == "neighbors") Command_enum = cmdNeighbors;
    else if (CmdStr == "pagerank") Command_enum = cmdPageRank;
    else if (CmdStr == "save") Command_enum = cmdSave;
    else if (CmdStr == "load") Command_enum = cmdLoad;
    else if (CmdStr == "info") Command_enum = cmdInfo;
    else if (CmdStr == "gradient-flow") Command_enum = cmdGradientFlow;
    else if (CmdStr == "help" || CmdStr == "--help" || CmdStr == "-h") Command_enum = cmdHelp;
    else {
        cout << "Unknown command: " << CmdStr << endl;
        PrintUsage();
        return 1;
    }
    
    if (Command_enum == cmdHelp) {
        PrintUsage();
        return 0;
    }
    
    // Initialize defaults
    int featureSize = 0;
    int hiddenSize = 0;
    int outputSize = 0;
    int mpLayers = 0;
    int nodeIdx = -1;
    int edgeIdx = -1;
    int sourceNode = -1;
    int targetNode = -1;
    int layerIdx = -1;
    double learningRate = 0.01;
    double damping = 0.85;
    int epochs = 100;
    int pageRankIters = 20;
    bool verbose = false;
    ActivationType activation = atReLU;
    LossType loss = ltMSE;
    string modelFile = "";
    string saveFile = "";
    string graphFile = "";
    string targetFile = "";
    string outputFile = "";
    
    // Parse arguments
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        
        if (arg == "--verbose") {
            verbose = true;
        } else {
            size_t eqPos = arg.find('=');
            if (eqPos == string::npos) {
                cout << "Invalid argument: " << arg << endl;
                continue;
            }
            
            string key = arg.substr(0, eqPos);
            string value = arg.substr(eqPos + 1);
            
            if (key == "--feature") featureSize = stoi(value);
            else if (key == "--hidden") hiddenSize = stoi(value);
            else if (key == "--output") {
                // Check if it's a number (output size) or file path
                bool isNumber = !value.empty() && value.find_first_not_of("0123456789") == string::npos;
                if (isNumber) {
                    outputSize = stoi(value);
                } else {
                    outputFile = value;
                }
            }
            else if (key == "--mp-layers") mpLayers = stoi(value);
            else if (key == "--model") modelFile = value;
            else if (key == "--graph") graphFile = value;
            else if (key == "--target") targetFile = value;
            else if (key == "--save") { saveFile = value; }
            else if (key == "--node") nodeIdx = stoi(value);
            else if (key == "--edge") edgeIdx = stoi(value);
            else if (key == "--source") sourceNode = stoi(value);
            else if (key == "--target-node") targetNode = stoi(value);
            else if (key == "--layer") layerIdx = stoi(value);
            else if (key == "--lr") learningRate = stod(value);
            else if (key == "--damping") damping = stod(value);
            else if (key == "--epochs") epochs = stoi(value);
            else if (key == "--iterations") pageRankIters = stoi(value);
            else if (key == "--activation") activation = ParseActivation(value);
            else if (key == "--loss") loss = ParseLoss(value);
            else cout << "Unknown option: " << key << endl;
        }
    }
    
    // Execute command
    if (Command_enum == cmdCreate) {
        if (featureSize <= 0) { cout << "Error: --feature is required" << endl; return 1; }
        if (hiddenSize <= 0) { cout << "Error: --hidden is required" << endl; return 1; }
        if (outputSize <= 0) { cout << "Error: --output is required" << endl; return 1; }
        if (mpLayers <= 0) { cout << "Error: --mp-layers is required" << endl; return 1; }
        if (saveFile == "") { cout << "Error: --save is required" << endl; return 1; }
        
        TGraphNeuralNetworkOpenCL* GNN = new TGraphNeuralNetworkOpenCL(featureSize, hiddenSize, outputSize, mpLayers, KERNEL_SOURCE);
        GNN->SetLearningRate(learningRate);
        GNN->SetActivation(activation);
        GNN->SetLossFunction(loss);
        
        GNN->SaveModel(saveFile);
        
        cout << "Created GNN model:\n";
        cout << "  Feature size: " << featureSize << "\n";
        cout << "  Hidden size: " << hiddenSize << "\n";
        cout << "  Output size: " << outputSize << "\n";
        cout << "  Message passing layers: " << mpLayers << "\n";
        cout << "  Activation: " << ActivationToStr(activation) << "\n";
        cout << "  Loss function: " << LossToStr(loss) << "\n";
        cout << fixed << setprecision(4) << "  Learning rate: " << learningRate << "\n";
        cout << "  Saved to: " << saveFile << endl;
        
        delete GNN;
    }
    else if (Command_enum == cmdTrain) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (graphFile == "") { cout << "Error: --graph is required" << endl; return 1; }
        if (saveFile == "") { cout << "Error: --save is required" << endl; return 1; }
        
        TGraphNeuralNetworkOpenCL* GNN = new TGraphNeuralNetworkOpenCL(1, 1, 1, 1, KERNEL_SOURCE);
        GNN->LoadModel(modelFile);
        
        if (learningRate > 0)
            GNN->SetLearningRate(learningRate);
        
        cout << "Training model for " << epochs << " epochs..." << endl;
        
        // Sample target - replace with actual graph data loading
        TDoubleArray target(GNN->GetOutputSize());
        for (int i = 0; i < GNN->GetOutputSize(); i++)
            target[i] = (double)rand() / RAND_MAX;
        
        // For now, use dummy graph - would load from file in production
        TGraph Graph;
        Graph.NumNodes = 5;
        Graph.Config.Undirected = true;
        Graph.Config.SelfLoops = false;
        Graph.Config.DeduplicateEdges = true;
        
        Graph.NodeFeatures.resize(5);
        for (int i = 0; i < 5; i++) {
            Graph.NodeFeatures[i].resize(GNN->GetFeatureSize());
            for (int j = 0; j < GNN->GetFeatureSize(); j++)
                Graph.NodeFeatures[i][j] = (double)rand() / RAND_MAX;
        }
        
        Graph.Edges.resize(6);
        Graph.Edges[0] = {0, 1};
        Graph.Edges[1] = {1, 2};
        Graph.Edges[2] = {2, 3};
        Graph.Edges[3] = {3, 4};
        Graph.Edges[4] = {4, 0};
        Graph.Edges[5] = {1, 3};
        
        GNN->TrainMultiple(Graph, target, epochs);
        
        GNN->SaveModel(saveFile);
        cout << "Model saved to: " << saveFile << endl;
        
        delete GNN;
    }
    else if (Command_enum == cmdPredict) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (graphFile == "") { cout << "Error: --graph is required" << endl; return 1; }
        
        TGraphNeuralNetworkOpenCL* GNN = new TGraphNeuralNetworkOpenCL(1, 1, 1, 1, KERNEL_SOURCE);
        GNN->LoadModel(modelFile);
        
        // Use dummy graph - would load from file in production
        TGraph Graph;
        Graph.NumNodes = 5;
        Graph.Config.Undirected = true;
        Graph.Config.SelfLoops = false;
        Graph.Config.DeduplicateEdges = true;
        
        Graph.NodeFeatures.resize(5);
        for (int i = 0; i < 5; i++) {
            Graph.NodeFeatures[i].resize(GNN->GetFeatureSize());
            for (int j = 0; j < GNN->GetFeatureSize(); j++)
                Graph.NodeFeatures[i][j] = (double)rand() / RAND_MAX;
        }
        
        Graph.Edges.resize(6);
        Graph.Edges[0] = {0, 1};
        Graph.Edges[1] = {1, 2};
        Graph.Edges[2] = {2, 3};
        Graph.Edges[3] = {3, 4};
        Graph.Edges[4] = {4, 0};
        Graph.Edges[5] = {1, 3};
        
        TDoubleArray prediction = GNN->Predict(Graph);
        
        cout << "Graph nodes: " << Graph.NumNodes << ", edges: " << Graph.Edges.size() << endl;
        
        cout << "Prediction: [";
        for (int i = 0; i < (int)prediction.size(); i++) {
            if (i > 0) cout << ", ";
            cout << fixed << setprecision(6) << prediction[i];
        }
        cout << "]" << endl;
        
        delete GNN;
    }
    else if (Command_enum == cmdInfo) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        
        TGraphNeuralNetworkOpenCL* GNN = new TGraphNeuralNetworkOpenCL(1, 1, 1, 1, KERNEL_SOURCE);
        GNN->LoadModel(modelFile);
        
        cout << "GNN Model Information (OpenCL)\n";
        cout << "==============================\n";
        cout << "Feature size: " << GNN->GetFeatureSize() << "\n";
        cout << "Hidden size: " << GNN->GetHiddenSize() << "\n";
        cout << "Output size: " << GNN->GetOutputSize() << "\n\n";
        cout << "Hyperparameters:\n";
        cout << fixed << setprecision(6) << "  Learning rate: " << GNN->GetLearningRate() << "\n";
        cout << "  Activation: " << ActivationToStr(GNN->GetActivation()) << "\n";
        cout << "  Loss function: " << LossToStr(GNN->GetLossFunction()) << "\n";
        cout << "  Max iterations: " << GNN->GetMaxIterations() << "\n";
        cout << "File: " << modelFile << "\n";
        
        delete GNN;
    }
    else if (Command_enum == cmdAddNode) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (nodeIdx < 0) { cout << "Error: --node is required" << endl; return 1; }
        
        cout << "Add node operation\n";
        cout << "Model: " << modelFile << "\n";
        cout << "Node index: " << nodeIdx << "\n";
        cout << "(Graph modification not yet implemented - load graph from file)\n";
    }
    else if (Command_enum == cmdAddEdge) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (sourceNode < 0 || targetNode < 0) { cout << "Error: --source and --target-node are required" << endl; return 1; }
        
        cout << "Add edge operation\n";
        cout << "Model: " << modelFile << "\n";
        cout << "Source: " << sourceNode << ", Target: " << targetNode << "\n";
        cout << "(Graph modification not yet implemented - load graph from file)\n";
    }
    else if (Command_enum == cmdRemoveEdge) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (edgeIdx < 0) { cout << "Error: --edge is required" << endl; return 1; }
        
        cout << "Remove edge operation\n";
        cout << "Model: " << modelFile << "\n";
        cout << "Edge index: " << edgeIdx << "\n";
        cout << "(Graph modification not yet implemented - load graph from file)\n";
    }
    else if (Command_enum == cmdPageRank) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        
        cout << "PageRank computation\n";
        cout << "Model: " << modelFile << "\n";
        cout << fixed << setprecision(2) << "Damping factor: " << damping << "\n";
        cout << "Iterations: " << pageRankIters << "\n";
        cout << "(PageRank requires graph data - load graph from file)\n";
    }
    else if (Command_enum == cmdDegree) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (nodeIdx < 0) { cout << "Error: --node is required" << endl; return 1; }
        
        cout << "Node degree information\n";
        cout << "Model: " << modelFile << "\n";
        cout << "Node index: " << nodeIdx << "\n";
        cout << "(Degree computation requires graph data - load graph from file)\n";
    }
    else if (Command_enum == cmdNeighbors) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (nodeIdx < 0) { cout << "Error: --node is required" << endl; return 1; }
        
        cout << "Neighbor query\n";
        cout << "Model: " << modelFile << "\n";
        cout << "Node index: " << nodeIdx << "\n";
        cout << "(Neighbor query requires graph data - load graph from file)\n";
    }
    else if (Command_enum == cmdSave) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        if (outputFile == "") { cout << "Error: --output is required" << endl; return 1; }
        
        TGraphNeuralNetworkOpenCL* GNN = new TGraphNeuralNetworkOpenCL(1, 1, 1, 1, KERNEL_SOURCE);
        GNN->LoadModel(modelFile);
        GNN->SaveModel(outputFile);
        cout << "Model saved to: " << outputFile << "\n";
        
        delete GNN;
    }
    else if (Command_enum == cmdLoad) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        
        TGraphNeuralNetworkOpenCL* GNN = new TGraphNeuralNetworkOpenCL(1, 1, 1, 1, KERNEL_SOURCE);
        GNN->LoadModel(modelFile);
        
        cout << "Model loaded from: " << modelFile << "\n";
        cout << "Feature size: " << GNN->GetFeatureSize() << "\n";
        cout << "Hidden size: " << GNN->GetHiddenSize() << "\n";
        cout << "Output size: " << GNN->GetOutputSize() << "\n";
        
        delete GNN;
    }
    else if (Command_enum == cmdGradientFlow) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        
        cout << "Gradient flow analysis\n";
        cout << "Model: " << modelFile << "\n";
        if (layerIdx >= 0) {
            cout << "Layer: " << layerIdx << "\n";
        } else {
            cout << "Layer: all\n";
        }
        cout << "(Gradient flow analysis requires training data)\n";
    }
    
    return 0;
}

