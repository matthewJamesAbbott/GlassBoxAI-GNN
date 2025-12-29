//
// Matthew Abbott
// Graph Neural Network
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <cuda_runtime.h>

using namespace std;

const int MAX_NODES = 1000;
const int MAX_EDGES = 10000;
const int MAX_ITERATIONS = 10000;
const double GRADIENT_CLIP = 5.0;
const string MODEL_MAGIC = "GNNBKND01";
const int BLOCK_SIZE = 256;

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
    cmdCreate,
    cmdTrain,
    cmdPredict,
    cmdInfo,
    cmdHelp
};

typedef vector<double> TDoubleArray;
typedef vector<float> TFloatArray;
typedef vector<int> TIntArray;
typedef vector<TDoubleArray> TDouble2DArray;

#define CUDA_CHECK(call) \
    do { \
       cudaError_t err = call; \
       if (err != cudaSuccess) { \
          cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
          exit(1); \
       } \
    } while(0)

struct TGPULayer {
    double* d_Weights;
    double* d_Biases;
    double* d_Output;
    double* d_PreActivations;
    double* d_Errors;
    double* d_LastInput;
    int NumInputs;
    int NumOutputs;
};

// ==================== CUDA Kernels ====================

__device__ double d_Activate(double X, int ActivationType) {
   switch (ActivationType) {
      case 0: // atReLU
         return (X > 0) ? X : 0.0;
      case 1: // atLeakyReLU
         return (X > 0) ? X : 0.01 * X;
      case 2: // atTanh
         return tanh(X);
      case 3: // atSigmoid
         return 1.0 / (1.0 + exp(-fmax(-500.0, fmin(500.0, X))));
      default:
         return X;
   }
}

__device__ double d_ActivateDerivative(double X, int ActivationType) {
   double S;
   switch (ActivationType) {
      case 0: // atReLU
         return (X > 0) ? 1.0 : 0.0;
      case 1: // atLeakyReLU
         return (X > 0) ? 1.0 : 0.01;
      case 2: // atTanh
         S = tanh(X);
         return 1.0 - S * S;
      case 3: // atSigmoid
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
   return fmax(-GRADIENT_CLIP, fmin(GRADIENT_CLIP, G));
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

__global__ void k_AggregateMessages(
   double* Messages, int* NeighborCounts, double* AggregatedMessages,
   int NumNodes, int HiddenSize, int MaxNeighbors)
{
   int Node = blockIdx.x;
   int Dim = threadIdx.x;
   
   if (Node >= NumNodes || Dim >= HiddenSize) return;
   
   int Count = NeighborCounts[Node];
   if (Count == 0) {
      AggregatedMessages[Node * HiddenSize + Dim] = 0.0;
      return;
   }
   
   double Sum = 0.0;
   for (int K = 0; K < Count; K++)
      Sum += Messages[(Node * MaxNeighbors + K) * HiddenSize + Dim];
   
   AggregatedMessages[Node * HiddenSize + Dim] = Sum / Count;
}

__global__ void k_ComputeGraphEmbedding(
   double* NodeEmbeddings, double* GraphEmbedding, int NumNodes, int HiddenSize)
{
   int J = blockIdx.x * blockDim.x + threadIdx.x;
   if (J >= HiddenSize) return;
   
   double Sum = 0.0;
   for (int I = 0; I < NumNodes; I++)
      Sum += NodeEmbeddings[I * HiddenSize + J];
   
   GraphEmbedding[J] = Sum / NumNodes;
}

// ==================== Host Structures ====================

struct TEdge {
   int Source;
   int Target;
   
   bool operator==(const TEdge& Other) const {
      return (Source == Other.Source) && (Target == Other.Target);
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

// ==================== Helper Functions ====================

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

// ==================== TGraphNeuralNetwork ====================

class TGraphNeuralNetwork {
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
    
    vector<TGPULayer> FGPUMessageLayers;
    vector<TGPULayer> FGPUUpdateLayers;
    TGPULayer FGPUReadoutLayer;
    TGPULayer FGPUOutputLayer;
    
    TDouble2DArray FNodeEmbeddings;
    TDouble2DArray FNewNodeEmbeddings;
    vector<TDouble2DArray> FEmbeddingHistory;
    vector<TLayerMessages> FMessageHistory;
    vector<TDouble2DArray> FAggregatedMessages;
    TDoubleArray FGraphEmbedding;
    
    TTrainingMetrics FMetrics;
    bool FGPUInitialized;
    
    void InitializeLayer(TLayer& Layer, int NumNeurons, int NumInputs);
    void InitializeGPULayer(TGPULayer& GPULayer, const TLayer& Layer);
    void FreeGPULayer(TGPULayer& GPULayer);
    void SyncLayerToGPU(TGPULayer& GPULayer, const TLayer& Layer);
    void SyncLayerFromGPU(TLayer& Layer, const TGPULayer& GPULayer);
    
    void BuildAdjacencyList(TGraph& Graph);
    void MessagePassing(TGraph& Graph);
    void Readout(TGraph& Graph);
    TDoubleArray ForwardLayerGPU(TGPULayer& GPULayer, const TDoubleArray& Input, bool UseOutputActivation);
    void BackwardLayerGPU(TGPULayer& GPULayer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation);
    TDoubleArray GetLayerInputGradGPU(TGPULayer& GPULayer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation);
    void BackPropagateGraph(TGraph& Graph, const TDoubleArray& Target);
    
    double Activate(double X);
    double ActivateDerivative(double X);
    double OutputActivate(double X);
    double OutputActivateDerivative(double PreAct);
    double ClipGradient(double G);
   
public:
    TGraphNeuralNetwork(int AFeatureSize, int AHiddenSize, int AOutputSize, int NumMPLayers);
    ~TGraphNeuralNetwork();
    
    void InitializeGPU();
    void FreeGPU();
    void SyncToGPU();
    void SyncFromGPU();
    
    TDoubleArray Predict(TGraph& Graph);
    double Train(TGraph& Graph, const TDoubleArray& Target);
    void TrainMultiple(TGraph& Graph, const TDoubleArray& Target, int Iterations);
    
    double ComputeLoss(const TDoubleArray& Prediction, const TDoubleArray& Target);
    TDoubleArray ComputeLossGradient(const TDoubleArray& Prediction, const TDoubleArray& Target);
    
    void SaveModel(const string& Filename);
    void LoadModel(const string& Filename);
    
    static void ValidateGraph(TGraph& Graph, vector<string>& Errors);
    static void DeduplicateEdges(TGraph& Graph);
    static void AddReverseEdges(TGraph& Graph);
    static void AddSelfLoops(TGraph& Graph);
    
    double GetLearningRate() const { return FLearningRate; }
    void SetLearningRate(double Value) { FLearningRate = Value; }
    int GetMaxIterations() const { return FMaxIterations; }
    void SetMaxIterations(int Value) { FMaxIterations = Value; }
    int GetFeatureSize() const { return FFeatureSize; }
    int GetHiddenSize() const { return FHiddenSize; }
    int GetOutputSize() const { return FOutputSize; }
    ActivationType GetActivation() const { return FActivation; }
    void SetActivation(ActivationType Value) { FActivation = Value; }
    LossType GetLossFunction() const { return FLossType; }
    void SetLossFunction(LossType Value) { FLossType = Value; }
    TTrainingMetrics GetMetrics() const { return FMetrics; }
};

// ==================== TGraphNeuralNetwork Implementation ====================

TGraphNeuralNetwork::TGraphNeuralNetwork(int AFeatureSize, int AHiddenSize, int AOutputSize, int NumMPLayers) {
    FLearningRate = 0.01;
    FMaxIterations = 100;
    FFeatureSize = AFeatureSize;
    FHiddenSize = AHiddenSize;
    FOutputSize = AOutputSize;
    FNumMessagePassingLayers = NumMPLayers;
    FActivation = atReLU;
    FLossType = ltMSE;
    FGPUInitialized = false;
    
    FMessageLayers.resize(NumMPLayers);
    FUpdateLayers.resize(NumMPLayers);
    
    for (int I = 0; I < NumMPLayers; I++) {
       if (I == 0)
          InitializeLayer(FMessageLayers[I], AHiddenSize, AFeatureSize * 2);
       else
          InitializeLayer(FMessageLayers[I], AHiddenSize, AHiddenSize * 2);
       
       InitializeLayer(FUpdateLayers[I], AHiddenSize, AHiddenSize * 2);
    }
    
    InitializeLayer(FReadoutLayer, AHiddenSize, AHiddenSize);
    InitializeLayer(FOutputLayer, AOutputSize, AHiddenSize);
    
    FMetrics.LossHistory.clear();
    
    InitializeGPU();
}

TGraphNeuralNetwork::~TGraphNeuralNetwork() {
    FreeGPU();
}

void TGraphNeuralNetwork::InitializeLayer(TLayer& Layer, int NumNeurons, int NumInputs) {
    Layer.NumInputs = NumInputs;
    Layer.NumOutputs = NumNeurons;
    Layer.Neurons.resize(NumNeurons);
    
    double Scale = sqrt(2.0 / (NumInputs + NumNeurons));
    
    for (int I = 0; I < NumNeurons; I++) {
       Layer.Neurons[I].Weights.resize(NumInputs);
       for (int J = 0; J < NumInputs; J++)
          Layer.Neurons[I].Weights[J] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * Scale;
       Layer.Neurons[I].Bias = 0.0;
       Layer.Neurons[I].Output = 0.0;
       Layer.Neurons[I].PreActivation = 0.0;
       Layer.Neurons[I].Error = 0.0;
    }
}

void TGraphNeuralNetwork::InitializeGPULayer(TGPULayer& GPULayer, const TLayer& Layer) {
    GPULayer.NumInputs = Layer.NumInputs;
    GPULayer.NumOutputs = Layer.NumOutputs;
    
    int WeightSize = Layer.NumOutputs * Layer.NumInputs;
    
    CUDA_CHECK(cudaMalloc(&GPULayer.d_Weights, WeightSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&GPULayer.d_Biases, Layer.NumOutputs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&GPULayer.d_Output, Layer.NumOutputs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&GPULayer.d_PreActivations, Layer.NumOutputs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&GPULayer.d_Errors, Layer.NumOutputs * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&GPULayer.d_LastInput, Layer.NumInputs * sizeof(double)));
    
    SyncLayerToGPU(GPULayer, Layer);
}

void TGraphNeuralNetwork::FreeGPULayer(TGPULayer& GPULayer) {
    if (GPULayer.d_Weights) cudaFree(GPULayer.d_Weights);
    if (GPULayer.d_Biases) cudaFree(GPULayer.d_Biases);
    if (GPULayer.d_Output) cudaFree(GPULayer.d_Output);
    if (GPULayer.d_PreActivations) cudaFree(GPULayer.d_PreActivations);
    if (GPULayer.d_Errors) cudaFree(GPULayer.d_Errors);
    if (GPULayer.d_LastInput) cudaFree(GPULayer.d_LastInput);
    GPULayer.d_Weights = nullptr;
    GPULayer.d_Biases = nullptr;
    GPULayer.d_Output = nullptr;
    GPULayer.d_PreActivations = nullptr;
    GPULayer.d_Errors = nullptr;
    GPULayer.d_LastInput = nullptr;
}

void TGraphNeuralNetwork::SyncLayerToGPU(TGPULayer& GPULayer, const TLayer& Layer) {
    vector<double> Weights(Layer.NumOutputs * Layer.NumInputs);
    vector<double> Biases(Layer.NumOutputs);
    
    for (int I = 0; I < Layer.NumOutputs; I++) {
       for (int J = 0; J < Layer.NumInputs; J++)
          Weights[I * Layer.NumInputs + J] = Layer.Neurons[I].Weights[J];
       Biases[I] = Layer.Neurons[I].Bias;
    }
    
    CUDA_CHECK(cudaMemcpy(GPULayer.d_Weights, Weights.data(), Weights.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(GPULayer.d_Biases, Biases.data(), Biases.size() * sizeof(double), cudaMemcpyHostToDevice));
}

void TGraphNeuralNetwork::SyncLayerFromGPU(TLayer& Layer, const TGPULayer& GPULayer) {
    vector<double> Weights(Layer.NumOutputs * Layer.NumInputs);
    vector<double> Biases(Layer.NumOutputs);
    
    CUDA_CHECK(cudaMemcpy(Weights.data(), GPULayer.d_Weights, Weights.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Biases.data(), GPULayer.d_Biases, Biases.size() * sizeof(double), cudaMemcpyDeviceToHost));
    
    for (int I = 0; I < Layer.NumOutputs; I++) {
       for (int J = 0; J < Layer.NumInputs; J++)
          Layer.Neurons[I].Weights[J] = Weights[I * Layer.NumInputs + J];
       Layer.Neurons[I].Bias = Biases[I];
    }
}

void TGraphNeuralNetwork::InitializeGPU() {
    if (FGPUInitialized) return;
    
    FGPUMessageLayers.resize(FNumMessagePassingLayers);
    FGPUUpdateLayers.resize(FNumMessagePassingLayers);
    
    for (int I = 0; I < FNumMessagePassingLayers; I++) {
       InitializeGPULayer(FGPUMessageLayers[I], FMessageLayers[I]);
       InitializeGPULayer(FGPUUpdateLayers[I], FUpdateLayers[I]);
    }
    
    InitializeGPULayer(FGPUReadoutLayer, FReadoutLayer);
    InitializeGPULayer(FGPUOutputLayer, FOutputLayer);
    
    FGPUInitialized = true;
}

void TGraphNeuralNetwork::FreeGPU() {
    if (!FGPUInitialized) return;
    
    for (int I = 0; I < FNumMessagePassingLayers; I++) {
       FreeGPULayer(FGPUMessageLayers[I]);
       FreeGPULayer(FGPUUpdateLayers[I]);
    }
    
    FreeGPULayer(FGPUReadoutLayer);
    FreeGPULayer(FGPUOutputLayer);
    
    FGPUInitialized = false;
}

void TGraphNeuralNetwork::SyncToGPU() {
    for (int I = 0; I < FNumMessagePassingLayers; I++) {
       SyncLayerToGPU(FGPUMessageLayers[I], FMessageLayers[I]);
       SyncLayerToGPU(FGPUUpdateLayers[I], FUpdateLayers[I]);
    }
    SyncLayerToGPU(FGPUReadoutLayer, FReadoutLayer);
    SyncLayerToGPU(FGPUOutputLayer, FOutputLayer);
}

void TGraphNeuralNetwork::SyncFromGPU() {
    for (int I = 0; I < FNumMessagePassingLayers; I++) {
       SyncLayerFromGPU(FMessageLayers[I], FGPUMessageLayers[I]);
       SyncLayerFromGPU(FUpdateLayers[I], FGPUUpdateLayers[I]);
    }
    SyncLayerFromGPU(FReadoutLayer, FGPUReadoutLayer);
    SyncLayerFromGPU(FOutputLayer, FGPUOutputLayer);
}

double TGraphNeuralNetwork::Activate(double X) {
   switch (FActivation) {
      case atReLU:
         return (X > 0) ? X : 0.0;
      case atLeakyReLU:
         return (X > 0) ? X : 0.01 * X;
      case atTanh:
         return tanh(X);
      case atSigmoid:
         return 1.0 / (1.0 + exp(-max(-500.0, min(500.0, X))));
      default:
         return X;
   }
}

double TGraphNeuralNetwork::ActivateDerivative(double X) {
   double S;
   switch (FActivation) {
      case atReLU:
         return (X > 0) ? 1.0 : 0.0;
      case atLeakyReLU:
         return (X > 0) ? 1.0 : 0.01;
      case atTanh:
         return 1.0 - tanh(X) * tanh(X);
      case atSigmoid:
         S = 1.0 / (1.0 + exp(-max(-500.0, min(500.0, X))));
         return S * (1.0 - S);
      default:
         return 1.0;
   }
}

double TGraphNeuralNetwork::OutputActivate(double X) {
   return 1.0 / (1.0 + exp(-max(-500.0, min(500.0, X))));
}

double TGraphNeuralNetwork::OutputActivateDerivative(double PreAct) {
   double S = 1.0 / (1.0 + exp(-max(-500.0, min(500.0, PreAct))));
   return S * (1.0 - S);
}

double TGraphNeuralNetwork::ComputeLoss(const TDoubleArray& Prediction, const TDoubleArray& Target) {
   double Result = 0.0;
   double P;
   
   switch (FLossType) {
      case ltMSE:
         for (size_t I = 0; I < Prediction.size(); I++)
            Result += (Prediction[I] - Target[I]) * (Prediction[I] - Target[I]);
         Result /= Prediction.size();
         break;
      case ltBinaryCrossEntropy:
         for (size_t I = 0; I < Prediction.size(); I++) {
            P = max(1e-7, min(1.0 - 1e-7, Prediction[I]));
            Result -= (Target[I] * log(P) + (1.0 - Target[I]) * log(1.0 - P));
         }
         Result /= Prediction.size();
         break;
   }
   return Result;
}

TDoubleArray TGraphNeuralNetwork::ComputeLossGradient(const TDoubleArray& Prediction, const TDoubleArray& Target) {
   TDoubleArray Result(Prediction.size());
   double P;
   
   switch (FLossType) {
      case ltMSE:
         for (size_t I = 0; I < Prediction.size(); I++)
            Result[I] = 2.0 * (Prediction[I] - Target[I]) / Prediction.size();
         break;
      case ltBinaryCrossEntropy:
         for (size_t I = 0; I < Prediction.size(); I++) {
            P = max(1e-7, min(1.0 - 1e-7, Prediction[I]));
            Result[I] = (-Target[I] / P + (1.0 - Target[I]) / (1.0 - P)) / Prediction.size();
         }
         break;
   }
   return Result;
}

double TGraphNeuralNetwork::ClipGradient(double G) {
   return max(-GRADIENT_CLIP, min(GRADIENT_CLIP, G));
}

void TGraphNeuralNetwork::BuildAdjacencyList(TGraph& Graph) {
   Graph.AdjacencyList.resize(Graph.NumNodes);
   for (int I = 0; I < Graph.NumNodes; I++)
      Graph.AdjacencyList[I].clear();
   
   for (size_t I = 0; I < Graph.Edges.size(); I++) {
      int Src = Graph.Edges[I].Source;
      int Tgt = Graph.Edges[I].Target;
      
      if ((Src >= 0) && (Src < Graph.NumNodes) && 
          (Tgt >= 0) && (Tgt < Graph.NumNodes)) {
         Graph.AdjacencyList[Src].push_back(Tgt);
      }
   }
}

void TGraphNeuralNetwork::ValidateGraph(TGraph& Graph, vector<string>& Errors) {
   Errors.clear();
   
   if (Graph.NumNodes < 1)
      Errors.push_back("Graph must have at least 1 node");
   
   if (Graph.NumNodes > MAX_NODES)
      Errors.push_back("Too many nodes (max " + to_string(MAX_NODES) + ")");
   
   if ((int)Graph.Edges.size() > MAX_EDGES)
      Errors.push_back("Too many edges (max " + to_string(MAX_EDGES) + ")");
   
   for (size_t I = 0; I < Graph.Edges.size(); I++) {
      if ((Graph.Edges[I].Source < 0) || (Graph.Edges[I].Source >= Graph.NumNodes))
         Errors.push_back("Edge " + to_string(I) + ": invalid source " + to_string(Graph.Edges[I].Source));
      if ((Graph.Edges[I].Target < 0) || (Graph.Edges[I].Target >= Graph.NumNodes))
         Errors.push_back("Edge " + to_string(I) + ": invalid target " + to_string(Graph.Edges[I].Target));
   }
   
   for (size_t I = 0; I < Graph.NodeFeatures.size(); I++) {
      if (Graph.NodeFeatures[I].size() == 0)
         Errors.push_back("Node " + to_string(I) + ": empty feature vector");
   }
}

void TGraphNeuralNetwork::DeduplicateEdges(TGraph& Graph) {
   vector<string> Seen;
   TEdgeArray NewEdges;
   
   for (size_t I = 0; I < Graph.Edges.size(); I++) {
      string Key = to_string(Graph.Edges[I].Source) + "-" + to_string(Graph.Edges[I].Target);
      bool Found = false;
      
      for (size_t J = 0; J < Seen.size(); J++) {
         if (Seen[J] == Key) {
            Found = true;
            break;
         }
      }
      
      if (!Found) {
         Seen.push_back(Key);
         NewEdges.push_back(Graph.Edges[I]);
      }
   }
   
   Graph.Edges = NewEdges;
}

void TGraphNeuralNetwork::AddReverseEdges(TGraph& Graph) {
   size_t OrigLen = Graph.Edges.size();
   Graph.Edges.resize(OrigLen * 2);
   
   for (size_t I = 0; I < OrigLen; I++) {
      if (Graph.Edges[I].Source != Graph.Edges[I].Target) {
         TEdge RevEdge;
         RevEdge.Source = Graph.Edges[I].Target;
         RevEdge.Target = Graph.Edges[I].Source;
         Graph.Edges[OrigLen + I] = RevEdge;
      }
      else {
         Graph.Edges[OrigLen + I] = Graph.Edges[I];
      }
   }
   
   DeduplicateEdges(Graph);
}

void TGraphNeuralNetwork::AddSelfLoops(TGraph& Graph) {
   for (int I = 0; I < Graph.NumNodes; I++) {
      bool HasSelf = false;
      for (size_t J = 0; J < Graph.Edges.size(); J++) {
         if ((Graph.Edges[J].Source == I) && (Graph.Edges[J].Target == I)) {
            HasSelf = true;
            break;
         }
      }
      
      if (!HasSelf) {
         TEdge SelfEdge;
         SelfEdge.Source = I;
         SelfEdge.Target = I;
         Graph.Edges.push_back(SelfEdge);
      }
   }
}

TDoubleArray TGraphNeuralNetwork::ForwardLayerGPU(TGPULayer& GPULayer, const TDoubleArray& Input, bool UseOutputActivation) {
   CUDA_CHECK(cudaMemcpy(GPULayer.d_LastInput, Input.data(), Input.size() * sizeof(double), cudaMemcpyHostToDevice));
   
   int NumBlocks = (GPULayer.NumOutputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
   k_ForwardLayer<<<NumBlocks, BLOCK_SIZE>>>(
      GPULayer.d_Weights, GPULayer.d_Biases, GPULayer.d_LastInput, GPULayer.d_Output, GPULayer.d_PreActivations,
      GPULayer.NumOutputs, GPULayer.NumInputs, (int)FActivation, UseOutputActivation);
   CUDA_CHECK(cudaDeviceSynchronize());
   
   TDoubleArray Result(GPULayer.NumOutputs);
   CUDA_CHECK(cudaMemcpy(Result.data(), GPULayer.d_Output, Result.size() * sizeof(double), cudaMemcpyDeviceToHost));
   
   return Result;
}

void TGraphNeuralNetwork::BackwardLayerGPU(TGPULayer& GPULayer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation) {
   double* d_UpstreamGrad;
   CUDA_CHECK(cudaMalloc(&d_UpstreamGrad, UpstreamGrad.size() * sizeof(double)));
   CUDA_CHECK(cudaMemcpy(d_UpstreamGrad, UpstreamGrad.data(), UpstreamGrad.size() * sizeof(double), cudaMemcpyHostToDevice));
   
   int NumBlocks = (GPULayer.NumOutputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
   k_BackwardLayer<<<NumBlocks, BLOCK_SIZE>>>(
      GPULayer.d_Weights, GPULayer.d_Biases, GPULayer.d_LastInput, d_UpstreamGrad,
      GPULayer.d_PreActivations, GPULayer.d_Errors,
      GPULayer.NumOutputs, GPULayer.NumInputs, (int)FActivation, UseOutputActivation, FLearningRate);
   CUDA_CHECK(cudaDeviceSynchronize());
   
   cudaFree(d_UpstreamGrad);
}

TDoubleArray TGraphNeuralNetwork::GetLayerInputGradGPU(TGPULayer& GPULayer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation) {
   double* d_UpstreamGrad;
   double* d_Result;
   
   CUDA_CHECK(cudaMalloc(&d_UpstreamGrad, UpstreamGrad.size() * sizeof(double)));
   CUDA_CHECK(cudaMalloc(&d_Result, GPULayer.NumInputs * sizeof(double)));
   CUDA_CHECK(cudaMemcpy(d_UpstreamGrad, UpstreamGrad.data(), UpstreamGrad.size() * sizeof(double), cudaMemcpyHostToDevice));
   
   int NumBlocks = (GPULayer.NumInputs + BLOCK_SIZE - 1) / BLOCK_SIZE;
   k_GetLayerInputGrad<<<NumBlocks, BLOCK_SIZE>>>(
      GPULayer.d_Weights, d_UpstreamGrad, GPULayer.d_PreActivations, d_Result,
      GPULayer.NumOutputs, GPULayer.NumInputs, (int)FActivation, UseOutputActivation);
   CUDA_CHECK(cudaDeviceSynchronize());
   
   TDoubleArray Result(GPULayer.NumInputs);
   CUDA_CHECK(cudaMemcpy(Result.data(), d_Result, Result.size() * sizeof(double), cudaMemcpyDeviceToHost));
   
   cudaFree(d_UpstreamGrad);
   cudaFree(d_Result);
   
   return Result;
}

void TGraphNeuralNetwork::MessagePassing(TGraph& Graph) {
   FNodeEmbeddings.resize(Graph.NumNodes);
   FNewNodeEmbeddings.resize(Graph.NumNodes);
   FEmbeddingHistory.resize(FNumMessagePassingLayers + 1);
   FMessageHistory.resize(FNumMessagePassingLayers);
   FAggregatedMessages.resize(FNumMessagePassingLayers);
   
   for (int I = 0; I < Graph.NumNodes; I++)
      FNodeEmbeddings[I] = CopyArray(Graph.NodeFeatures[I]);
   
   FEmbeddingHistory[0].resize(Graph.NumNodes);
   for (int I = 0; I < Graph.NumNodes; I++)
      FEmbeddingHistory[0][I] = CopyArray(FNodeEmbeddings[I]);
   
   for (int Layer = 0; Layer < FNumMessagePassingLayers; Layer++) {
      FMessageHistory[Layer].resize(Graph.NumNodes);
      FAggregatedMessages[Layer].resize(Graph.NumNodes);
      
      for (int Node = 0; Node < Graph.NumNodes; Node++) {
         FMessageHistory[Layer][Node].clear();
         TDoubleArray AggregatedMessage = ZeroArray(FHiddenSize);
         
         if (Graph.AdjacencyList[Node].size() > 0) {
            for (size_t K = 0; K < Graph.AdjacencyList[Node].size(); K++) {
               int Neighbor = Graph.AdjacencyList[Node][K];
               
               TDoubleArray ConcatFeatures = ConcatArrays(FNodeEmbeddings[Node], FNodeEmbeddings[Neighbor]);
               TDoubleArray Message = ForwardLayerGPU(FGPUMessageLayers[Layer], ConcatFeatures, false);
               
               TMessageInfo MsgInfo;
               MsgInfo.NeighborIdx = Neighbor;
               MsgInfo.ConcatInput = CopyArray(ConcatFeatures);
               MsgInfo.MessageOutput = CopyArray(Message);
               
               FMessageHistory[Layer][Node].push_back(MsgInfo);
               
               for (int I = 0; I < FHiddenSize; I++)
                  AggregatedMessage[I] += Message[I];
            }
            
            for (int I = 0; I < FHiddenSize; I++)
               AggregatedMessage[I] /= Graph.AdjacencyList[Node].size();
         }
         
         FAggregatedMessages[Layer][Node] = CopyArray(AggregatedMessage);
         
         TDoubleArray PaddedEmb;
         if (Layer == 0)
            PaddedEmb = PadArray(FNodeEmbeddings[Node], FHiddenSize);
         else
            PaddedEmb = CopyArray(FNodeEmbeddings[Node]);
         
         TDoubleArray UpdateInput = ConcatArrays(PaddedEmb, AggregatedMessage);
         FNewNodeEmbeddings[Node] = ForwardLayerGPU(FGPUUpdateLayers[Layer], UpdateInput, false);
      }
      
      for (int Node = 0; Node < Graph.NumNodes; Node++)
         FNodeEmbeddings[Node] = CopyArray(FNewNodeEmbeddings[Node]);
      
      FEmbeddingHistory[Layer + 1].resize(Graph.NumNodes);
      for (int I = 0; I < Graph.NumNodes; I++)
         FEmbeddingHistory[Layer + 1][I] = CopyArray(FNodeEmbeddings[I]);
   }
}

void TGraphNeuralNetwork::Readout(TGraph& Graph) {
   FGraphEmbedding = ZeroArray(FHiddenSize);
   
   for (int I = 0; I < Graph.NumNodes; I++)
      for (int J = 0; J < FHiddenSize; J++)
         FGraphEmbedding[J] += FNodeEmbeddings[I][J];
   
   for (int J = 0; J < FHiddenSize; J++)
      FGraphEmbedding[J] /= Graph.NumNodes;
   
   ForwardLayerGPU(FGPUReadoutLayer, FGraphEmbedding, false);
}

TDoubleArray TGraphNeuralNetwork::Predict(TGraph& Graph) {
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
   CUDA_CHECK(cudaMemcpy(ReadoutOutput.data(), FGPUReadoutLayer.d_Output, ReadoutOutput.size() * sizeof(double), cudaMemcpyDeviceToHost));
   
   return ForwardLayerGPU(FGPUOutputLayer, ReadoutOutput, true);
}

void TGraphNeuralNetwork::BackPropagateGraph(TGraph& Graph, const TDoubleArray& Target) {
   TDoubleArray LastInput(FGPUOutputLayer.NumInputs);
   CUDA_CHECK(cudaMemcpy(LastInput.data(), FGPUOutputLayer.d_LastInput, LastInput.size() * sizeof(double), cudaMemcpyDeviceToHost));
   
   TDoubleArray LossGrad = ComputeLossGradient(LastInput, Target);
   
   TDoubleArray PreActs(FOutputSize);
   CUDA_CHECK(cudaMemcpy(PreActs.data(), FGPUOutputLayer.d_PreActivations, PreActs.size() * sizeof(double), cudaMemcpyDeviceToHost));
   
   for (int I = 0; I < FOutputSize; I++)
      LossGrad[I] *= OutputActivateDerivative(PreActs[I]);
   
   BackwardLayerGPU(FGPUOutputLayer, LossGrad, true);
   TDoubleArray ReadoutGrad = GetLayerInputGradGPU(FGPUOutputLayer, LossGrad, true);
   
   BackwardLayerGPU(FGPUReadoutLayer, ReadoutGrad, false);
   TDoubleArray GraphEmbGrad = GetLayerInputGradGPU(FGPUReadoutLayer, ReadoutGrad, false);
   
   TDouble2DArray NodeGrads(Graph.NumNodes);
   for (int Node = 0; Node < Graph.NumNodes; Node++) {
      NodeGrads[Node] = ZeroArray(FHiddenSize);
      for (int I = 0; I < FHiddenSize; I++)
         NodeGrads[Node][I] = GraphEmbGrad[I] / Graph.NumNodes;
   }
   
   for (int Layer = FNumMessagePassingLayers - 1; Layer >= 0; Layer--) {
      TDouble2DArray NewNodeGrads(Graph.NumNodes);
      
      if (Layer == 0) {
         for (int Node = 0; Node < Graph.NumNodes; Node++)
            NewNodeGrads[Node] = ZeroArray(FFeatureSize);
      }
      else {
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
         CUDA_CHECK(cudaMemcpy(FGPUUpdateLayers[Layer].d_LastInput, UpdateInput.data(), UpdateInput.size() * sizeof(double), cudaMemcpyHostToDevice));
         
         BackwardLayerGPU(FGPUUpdateLayers[Layer], NodeGrads[Node], false);
         TDoubleArray UpdateInputGrad = GetLayerInputGradGPU(FGPUUpdateLayers[Layer], NodeGrads[Node], false);
         
         int GradLimit = min(FHiddenSize, (int)NewNodeGrads[Node].size());
         for (int I = 0; I < GradLimit; I++) {
            if (Layer == 0) {
               if (I < FFeatureSize)
                  NewNodeGrads[Node][I] += UpdateInputGrad[I];
            }
            else {
               NewNodeGrads[Node][I] += UpdateInputGrad[I];
            }
         }
         
         int NumNeighbors = Graph.AdjacencyList[Node].size();
         if (NumNeighbors > 0) {
            TDoubleArray MsgGrad = ZeroArray(FHiddenSize);
            for (int I = 0; I < FHiddenSize; I++)
               MsgGrad[I] = UpdateInputGrad[FHiddenSize + I] / NumNeighbors;
            
            for (size_t K = 0; K < FMessageHistory[Layer][Node].size(); K++) {
               CUDA_CHECK(cudaMemcpy(FGPUMessageLayers[Layer].d_LastInput, 
                  FMessageHistory[Layer][Node][K].ConcatInput.data(),
                  FMessageHistory[Layer][Node][K].ConcatInput.size() * sizeof(double), cudaMemcpyHostToDevice));
               
               BackwardLayerGPU(FGPUMessageLayers[Layer], MsgGrad, false);
               TDoubleArray ConcatGrad = GetLayerInputGradGPU(FGPUMessageLayers[Layer], MsgGrad, false);
               
               int HalfLen = ConcatGrad.size() / 2;
               
               int Limit1 = min(HalfLen, (int)NewNodeGrads[Node].size());
               for (int I = 0; I < Limit1; I++)
                  NewNodeGrads[Node][I] += ConcatGrad[I];
               
               int J = FMessageHistory[Layer][Node][K].NeighborIdx;
               int Limit2 = min(HalfLen, (int)NewNodeGrads[J].size());
               for (int I = 0; I < Limit2; I++)
                  NewNodeGrads[J][I] += ConcatGrad[HalfLen + I];
            }
         }
      }
      
      if (Layer > 0)
         NodeGrads = NewNodeGrads;
   }
}

double TGraphNeuralNetwork::Train(TGraph& Graph, const TDoubleArray& Target) {
   TDoubleArray Prediction = Predict(Graph);
   double Loss = ComputeLoss(Prediction, Target);
   BackPropagateGraph(Graph, Target);
   return Loss;
}

void TGraphNeuralNetwork::TrainMultiple(TGraph& Graph, const TDoubleArray& Target, int Iterations) {
   FMetrics.LossHistory.resize(Iterations);
   
   for (int I = 0; I < Iterations; I++) {
      double Loss = Train(Graph, Target);
      FMetrics.LossHistory[I] = Loss;
      FMetrics.Loss = Loss;
      FMetrics.Iteration = I + 1;
      
      if ((I % 10 == 0) || (I == Iterations - 1))
         cout << "Iteration " << (I + 1) << "/" << Iterations << ", Loss: " << fixed << setprecision(6) << Loss << endl;
   }
   
   SyncFromGPU();
}

void TGraphNeuralNetwork::SaveModel(const string& Filename) {
   SyncFromGPU();
   
   ofstream F(Filename, ios::binary);
   if (!F) {
      cerr << "Error: Cannot create file " << Filename << endl;
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
   
   for (int K = 0; K < FNumMessagePassingLayers; K++) {
      F.write((char*)&FMessageLayers[K].NumOutputs, sizeof(int));
      F.write((char*)&FMessageLayers[K].NumInputs, sizeof(int));
      for (int I = 0; I < FMessageLayers[K].NumOutputs; I++) {
         for (int J = 0; J < FMessageLayers[K].NumInputs; J++)
            F.write((char*)&FMessageLayers[K].Neurons[I].Weights[J], sizeof(double));
         F.write((char*)&FMessageLayers[K].Neurons[I].Bias, sizeof(double));
      }
      
      F.write((char*)&FUpdateLayers[K].NumOutputs, sizeof(int));
      F.write((char*)&FUpdateLayers[K].NumInputs, sizeof(int));
      for (int I = 0; I < FUpdateLayers[K].NumOutputs; I++) {
         for (int J = 0; J < FUpdateLayers[K].NumInputs; J++)
            F.write((char*)&FUpdateLayers[K].Neurons[I].Weights[J], sizeof(double));
         F.write((char*)&FUpdateLayers[K].Neurons[I].Bias, sizeof(double));
      }
   }
   
   F.write((char*)&FReadoutLayer.NumOutputs, sizeof(int));
   F.write((char*)&FReadoutLayer.NumInputs, sizeof(int));
   for (int I = 0; I < FReadoutLayer.NumOutputs; I++) {
      for (int J = 0; J < FReadoutLayer.NumInputs; J++)
         F.write((char*)&FReadoutLayer.Neurons[I].Weights[J], sizeof(double));
      F.write((char*)&FReadoutLayer.Neurons[I].Bias, sizeof(double));
   }
   
   F.write((char*)&FOutputLayer.NumOutputs, sizeof(int));
   F.write((char*)&FOutputLayer.NumInputs, sizeof(int));
   for (int I = 0; I < FOutputLayer.NumOutputs; I++) {
      for (int J = 0; J < FOutputLayer.NumInputs; J++)
         F.write((char*)&FOutputLayer.Neurons[I].Weights[J], sizeof(double));
      F.write((char*)&FOutputLayer.Neurons[I].Bias, sizeof(double));
   }
   
   F.close();
   cout << "Model saved to " << Filename << endl;
}

void TGraphNeuralNetwork::LoadModel(const string& Filename) {
   ifstream F(Filename, ios::binary);
   if (!F) {
      cerr << "Error: Cannot open file " << Filename << endl;
      return;
   }
   
   FreeGPU();
   
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
   
   for (int K = 0; K < FNumMessagePassingLayers; K++) {
      int NumN, NumI;
      F.read((char*)&NumN, sizeof(int));
      F.read((char*)&NumI, sizeof(int));
      InitializeLayer(FMessageLayers[K], NumN, NumI);
      for (int I = 0; I < NumN; I++) {
         for (int J = 0; J < NumI; J++)
            F.read((char*)&FMessageLayers[K].Neurons[I].Weights[J], sizeof(double));
         F.read((char*)&FMessageLayers[K].Neurons[I].Bias, sizeof(double));
      }
      
      F.read((char*)&NumN, sizeof(int));
      F.read((char*)&NumI, sizeof(int));
      InitializeLayer(FUpdateLayers[K], NumN, NumI);
      for (int I = 0; I < NumN; I++) {
         for (int J = 0; J < NumI; J++)
            F.read((char*)&FUpdateLayers[K].Neurons[I].Weights[J], sizeof(double));
         F.read((char*)&FUpdateLayers[K].Neurons[I].Bias, sizeof(double));
      }
   }
   
   int NumN, NumI;
   F.read((char*)&NumN, sizeof(int));
   F.read((char*)&NumI, sizeof(int));
   InitializeLayer(FReadoutLayer, NumN, NumI);
   for (int I = 0; I < NumN; I++) {
      for (int J = 0; J < NumI; J++)
         F.read((char*)&FReadoutLayer.Neurons[I].Weights[J], sizeof(double));
      F.read((char*)&FReadoutLayer.Neurons[I].Bias, sizeof(double));
   }
   
   F.read((char*)&NumN, sizeof(int));
   F.read((char*)&NumI, sizeof(int));
   InitializeLayer(FOutputLayer, NumN, NumI);
   for (int I = 0; I < NumN; I++) {
      for (int J = 0; J < NumI; J++)
         F.read((char*)&FOutputLayer.Neurons[I].Weights[J], sizeof(double));
      F.read((char*)&FOutputLayer.Neurons[I].Bias, sizeof(double));
   }
   
   F.close();
   
   InitializeGPU();
   
   cout << "Model loaded from " << Filename << endl;
}

// ==================== Command Line Parsing ====================

void PrintUsage() {
   cout << "GNN - Graph Neural Network (CUDA)" << endl;
   cout << endl;
   cout << "Usage: gnn_cuda --nodes FILE --edges FILE [options]" << endl;
   cout << "       gnn_cuda --load MODEL [options]" << endl;
   cout << endl;
   cout << "Input (required for new model):" << endl;
   cout << "  --nodes FILE            CSV file with node features (one node per row)" << endl;
   cout << "  --edges FILE            CSV file with edges (source,target per line)" << endl;
   cout << endl;
   cout << "Training:" << endl;
   cout << "  --target VALUES         Comma-separated target values (e.g., 1.0,0.0)" << endl;
   cout << "  --target-file FILE      File with target values (one per line)" << endl;
   cout << "  -i, --iterations N      Training iterations (default: 500)" << endl;
   cout << "  -lr, --learning-rate N  Learning rate (default: 0.05)" << endl;
   cout << "  --no-train              Skip training (inference only)" << endl;
   cout << endl;
   cout << "Model:" << endl;
   cout << "  --load FILE             Load model from file" << endl;
   cout << "  -o, --output FILE       Save model to file (default: gnn_model.bin)" << endl;
   cout << "  -hs, --hidden-size N    Hidden layer size (default: 16)" << endl;
   cout << "  -os, --output-size N    Output size (default: 2)" << endl;
   cout << "  -mp, --mp-layers N      Message passing layers (default: 2)" << endl;
   cout << "  -a, --activation TYPE   Activation: relu, leakyrelu, tanh, sigmoid (default: leakyrelu)" << endl;
   cout << "  -l, --loss TYPE         Loss function: mse, bce (default: mse)" << endl;
   cout << endl;
   cout << "Graph options:" << endl;
   cout << "  --undirected            Treat graph as undirected (default)" << endl;
   cout << "  --directed              Treat graph as directed" << endl;
   cout << "  --self-loops            Add self-loops to nodes" << endl;
   cout << endl;
   cout << "Other:" << endl;
   cout << "  -q, --quiet             Reduce output verbosity" << endl;
   cout << "  -h, --help              Show this help message" << endl;
   cout << endl;
   cout << "Examples:" << endl;
   cout << "  gnn_cuda --nodes graph_nodes.csv --edges graph_edges.csv --target 1.0,0.0 -i 1000" << endl;
   cout << "  gnn_cuda --load model.bin --nodes test.csv --edges test_edges.csv --no-train" << endl;
   cout << "  gnn_cuda --nodes data.csv --edges edges.csv --target-file targets.txt -o trained.bin" << endl;
}

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

ActivationType ParseActivation(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    
    if (lower == "relu") return atReLU;
    else if (lower == "leakyrelu") return atLeakyReLU;
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

bool FileExists(const string& Filename) {
   ifstream F(Filename);
   return F.good();
}

bool LoadNodesCSV(const string& Filename, TDouble2DArray& NodeFeatures, int& FeatureSize) {
   ifstream F(Filename);
   if (!F) {
      cerr << "Error: Cannot open nodes file: " << Filename << endl;
      return false;
   }
   
   NodeFeatures.clear();
   string Line;
   FeatureSize = 0;
   
   while (getline(F, Line)) {
      if (Line.empty() || Line[0] == '#') continue;
      
      TDoubleArray Features;
      size_t Pos = 0;
      string Token;
      
      while ((Pos = Line.find(',')) != string::npos || !Line.empty()) {
         if (Pos == string::npos) {
            Token = Line;
            Line = "";
         } else {
            Token = Line.substr(0, Pos);
            Line.erase(0, Pos + 1);
         }
         
         if (!Token.empty()) {
            Features.push_back(atof(Token.c_str()));
         }
         
         if (Pos == string::npos) break;
      }
      
      if (Features.size() > 0) {
         if (FeatureSize == 0)
            FeatureSize = Features.size();
         NodeFeatures.push_back(Features);
      }
   }
   
   F.close();
   return NodeFeatures.size() > 0;
}

bool LoadEdgesCSV(const string& Filename, TEdgeArray& Edges) {
   ifstream F(Filename);
   if (!F) {
      cerr << "Error: Cannot open edges file: " << Filename << endl;
      return false;
   }
   
   Edges.clear();
   string Line;
   
   while (getline(F, Line)) {
      if (Line.empty() || Line[0] == '#') continue;
      
      size_t Pos = Line.find(',');
      if (Pos != string::npos) {
         TEdge Edge;
         Edge.Source = atoi(Line.substr(0, Pos).c_str());
         Edge.Target = atoi(Line.substr(Pos + 1).c_str());
         Edges.push_back(Edge);
      }
   }
   
   F.close();
   return Edges.size() > 0;
}

bool ParseTargetValues(const string& Values, TDoubleArray& Target) {
   Target.clear();
   string S = Values;
   size_t Pos;
   
   while ((Pos = S.find(',')) != string::npos || !S.empty()) {
      string Token;
      if (Pos == string::npos) {
         Token = S;
         S = "";
      } else {
         Token = S.substr(0, Pos);
         S.erase(0, Pos + 1);
      }
      
      if (!Token.empty()) {
         Target.push_back(atof(Token.c_str()));
      }
      
      if (Pos == string::npos) break;
   }
   
   return Target.size() > 0;
}

bool LoadTargetFile(const string& Filename, TDoubleArray& Target) {
   ifstream F(Filename);
   if (!F) {
      cerr << "Error: Cannot open target file: " << Filename << endl;
      return false;
   }
   
   Target.clear();
   string Line;
   
   while (getline(F, Line)) {
      if (Line.empty() || Line[0] == '#') continue;
      Target.push_back(atof(Line.c_str()));
   }
   
   F.close();
   return Target.size() > 0;
}

// ==================== Main Program ====================

int main(int argc, char* argv[]) {
   srand(time(NULL));
   
   if (argc < 2) {
      PrintUsage();
      return 1;
   }
   
   int DeviceCount;
   cudaGetDeviceCount(&DeviceCount);
   if (DeviceCount == 0) {
      cerr << "No CUDA devices found!" << endl;
      return 1;
   }
   
   cudaDeviceProp Prop;
   cudaGetDeviceProperties(&Prop, 0);
   
   double ArgLearningRate = 0.05;
   int ArgIterations = 500;
   int ArgHiddenSize = 16;
   int ArgOutputSize = 2;
   int ArgMPLayers = 2;
   ActivationType ArgActivation = atLeakyReLU;
   LossType ArgLoss = ltMSE;
   string ArgModelFile = "gnn_model.bin";
   string ArgLoadFile = "";
   string ArgNodesFile = "";
   string ArgEdgesFile = "";
   string ArgTargetValues = "";
   string ArgTargetFile = "";
   bool ArgNoTrain = false;
   bool ArgUndirected = true;
   bool ArgSelfLoops = false;
   bool ArgQuiet = false;
   bool ArgNoSave = false;
   
   int I = 1;
   while (I < argc) {
      string Arg = argv[I];
      
      if ((Arg == "-h") || (Arg == "--help")) {
         PrintUsage();
         return 0;
      }
      else if ((Arg == "-lr") || (Arg == "--learning-rate")) {
         I++;
         if (I < argc) ArgLearningRate = atof(argv[I]);
      }
      else if ((Arg == "-i") || (Arg == "--iterations")) {
         I++;
         if (I < argc) ArgIterations = atoi(argv[I]);
      }
      else if ((Arg == "-hs") || (Arg == "--hidden-size")) {
         I++;
         if (I < argc) ArgHiddenSize = atoi(argv[I]);
      }
      else if ((Arg == "-os") || (Arg == "--output-size")) {
         I++;
         if (I < argc) ArgOutputSize = atoi(argv[I]);
      }
      else if ((Arg == "-mp") || (Arg == "--mp-layers")) {
         I++;
         if (I < argc) ArgMPLayers = atoi(argv[I]);
      }
      else if (Arg == "--nodes") {
         I++;
         if (I < argc) ArgNodesFile = argv[I];
      }
      else if (Arg == "--edges") {
         I++;
         if (I < argc) ArgEdgesFile = argv[I];
      }
      else if (Arg == "--target") {
         I++;
         if (I < argc) ArgTargetValues = argv[I];
      }
      else if (Arg == "--target-file") {
         I++;
         if (I < argc) ArgTargetFile = argv[I];
      }
      else if ((Arg == "-a") || (Arg == "--activation")) {
         I++;
         if (I < argc) ArgActivation = ParseActivation(argv[I]);
      }
      else if ((Arg == "-l") || (Arg == "--loss")) {
         I++;
         if (I < argc) ArgLoss = ParseLoss(argv[I]);
      }
      else if ((Arg == "-o") || (Arg == "--output")) {
         I++;
         if (I < argc) ArgModelFile = argv[I];
      }
      else if (Arg == "--load") {
         I++;
         if (I < argc) ArgLoadFile = argv[I];
      }
      else if (Arg == "--no-train")
         ArgNoTrain = true;
      else if (Arg == "--no-save")
         ArgNoSave = true;
      else if (Arg == "--undirected")
         ArgUndirected = true;
      else if (Arg == "--directed")
         ArgUndirected = false;
      else if (Arg == "--self-loops")
         ArgSelfLoops = true;
      else if ((Arg == "-q") || (Arg == "--quiet"))
         ArgQuiet = true;
      else {
         cerr << "Unknown argument: " << Arg << endl;
         return 1;
      }
      
      I++;
   }
   
   if (ArgNodesFile.empty() && ArgLoadFile.empty()) {
      cerr << "Error: Must specify --nodes FILE or --load MODEL" << endl;
      cerr << "Run with --help for usage information." << endl;
      return 1;
   }
   
   if (!ArgNodesFile.empty() && ArgEdgesFile.empty()) {
      cerr << "Error: Must specify --edges FILE with --nodes" << endl;
      return 1;
   }
   
   if (!ArgNoTrain && ArgTargetValues.empty() && ArgTargetFile.empty()) {
      cerr << "Error: Must specify --target VALUES or --target-file FILE for training" << endl;
      cerr << "Use --no-train for inference only." << endl;
      return 1;
   }
   
   if (!ArgQuiet) {
      cout << "=== Graph Neural Network (CUDA) ===" << endl;
      cout << "GPU: " << Prop.name << endl;
      cout << endl;
   }
   
   TGraph Graph;
   int FeatureSize = 0;
   
   if (!ArgNodesFile.empty()) {
      if (!LoadNodesCSV(ArgNodesFile, Graph.NodeFeatures, FeatureSize)) {
         cerr << "Error: Failed to load nodes from " << ArgNodesFile << endl;
         return 1;
      }
      Graph.NumNodes = Graph.NodeFeatures.size();
      
      if (!LoadEdgesCSV(ArgEdgesFile, Graph.Edges)) {
         cerr << "Error: Failed to load edges from " << ArgEdgesFile << endl;
         return 1;
      }
   }
   
   Graph.Config.Undirected = ArgUndirected;
   Graph.Config.SelfLoops = ArgSelfLoops;
   Graph.Config.DeduplicateEdges = true;
   
   TDoubleArray Target;
   if (!ArgTargetValues.empty()) {
      if (!ParseTargetValues(ArgTargetValues, Target)) {
         cerr << "Error: Failed to parse target values" << endl;
         return 1;
      }
      ArgOutputSize = Target.size();
   }
   else if (!ArgTargetFile.empty()) {
      if (!LoadTargetFile(ArgTargetFile, Target)) {
         cerr << "Error: Failed to load target file" << endl;
         return 1;
      }
      ArgOutputSize = Target.size();
   }
   else if (ArgNoTrain) {
      Target.resize(ArgOutputSize, 0.0);
   }
   
   vector<string> Errors;
   TGraphNeuralNetwork::ValidateGraph(Graph, Errors);
   if (Errors.size() > 0) {
      cerr << "Graph validation errors:" << endl;
      for (size_t I = 0; I < Errors.size(); I++)
         cerr << "  - " << Errors[I] << endl;
      return 1;
   }
   
   if (!ArgQuiet) {
      cout << "Graph: " << Graph.NumNodes << " nodes, " << Graph.Edges.size() << " edges" << endl;
      cout << "Config: Undirected=" << (Graph.Config.Undirected ? "true" : "false")
           << ", SelfLoops=" << (Graph.Config.SelfLoops ? "true" : "false") << endl;
      cout << "Network: Features=" << FeatureSize << ", Hidden=" << ArgHiddenSize
           << ", Output=" << ArgOutputSize << ", MPLayers=" << ArgMPLayers << endl;
      if (!ArgNoTrain)
         cout << "Training: LR=" << fixed << setprecision(4) << ArgLearningRate 
              << ", Iterations=" << ArgIterations << endl;
      cout << endl;
   }
   
   TGraphNeuralNetwork* Net;
   
   if (!ArgLoadFile.empty() && FileExists(ArgLoadFile)) {
      Net = new TGraphNeuralNetwork(FeatureSize > 0 ? FeatureSize : 1, ArgHiddenSize, ArgOutputSize, ArgMPLayers);
      Net->LoadModel(ArgLoadFile);
      if (!ArgQuiet)
         cout << "Loaded model from " << ArgLoadFile << endl;
   }
   else {
      Net = new TGraphNeuralNetwork(FeatureSize, ArgHiddenSize, ArgOutputSize, ArgMPLayers);
   }
   
   Net->SetLearningRate(ArgLearningRate);
   Net->SetActivation(ArgActivation);
   Net->SetLossFunction(ArgLoss);
   
   TDoubleArray Prediction;
   double InitialLoss, FinalLoss;
   
   if (!ArgQuiet) {
      cout << "Initial prediction:" << endl;
      Prediction = Net->Predict(Graph);
      cout << "  [";
      for (int I = 0; I < ArgOutputSize; I++) {
         cout << fixed << setprecision(4) << Prediction[I];
         if (I < ArgOutputSize - 1) cout << ", ";
      }
      cout << "]" << endl;
      InitialLoss = Net->ComputeLoss(Prediction, Target);
      cout << "  Loss: " << fixed << setprecision(6) << InitialLoss << endl;
      cout << endl;
   }
   else {
      Prediction = Net->Predict(Graph);
      InitialLoss = Net->ComputeLoss(Prediction, Target);
   }
   
   if (!ArgNoTrain) {
      if (!ArgQuiet)
         cout << "Training..." << endl;
      Net->TrainMultiple(Graph, Target, ArgIterations);
      if (!ArgQuiet)
         cout << endl;
   }
   
   cout << "Final prediction:" << endl;
   Prediction = Net->Predict(Graph);
   cout << "  [";
   for (int I = 0; I < ArgOutputSize; I++) {
      cout << fixed << setprecision(4) << Prediction[I];
      if (I < ArgOutputSize - 1) cout << ", ";
   }
   cout << "]" << endl;
   
   cout << "  Target: [";
   for (int I = 0; I < ArgOutputSize; I++) {
      cout << fixed << setprecision(4) << Target[I];
      if (I < ArgOutputSize - 1) cout << ", ";
   }
   cout << "]" << endl;
   
   FinalLoss = Net->ComputeLoss(Prediction, Target);
   cout << "  Final Loss: " << fixed << setprecision(6) << FinalLoss << endl;
   if (InitialLoss > 0)
      cout << "  Loss reduction: " << fixed << setprecision(2) << ((1 - FinalLoss/InitialLoss) * 100) << "%" << endl;
   cout << endl;
   
   if (!ArgNoSave) {
      Net->SaveModel(ArgModelFile);
   }
   
   delete Net;
   if (!ArgQuiet)
      cout << "Done!" << endl;
   
   return 0;
}
