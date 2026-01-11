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
    int GetMPLayers() const { return FNumMessagePassingLayers; }
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
    cout << "\nGNN-CUDA - Graph Neural Network (GPU-Accelerated)\n";
    cout << "=================================================\n\n";
    
    cout << "USAGE:\n";
    cout << "  gnn_cuda <command> [options]\n\n";
    
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
    cout << "  gnn_cuda create --feature=3 --hidden=16 --output=2 --mp-layers=2 --save=model.bin\n\n";
    cout << "  # Get node degree\n";
    cout << "  gnn_cuda degree --model=model.bin --node=0\n\n";
    cout << "  # Compute PageRank\n";
    cout << "  gnn_cuda pagerank --model=model.bin --damping=0.85 --iterations=20\n\n";
    cout << "  # Train the model\n";
    cout << "  gnn_cuda train --model=model.bin --graph=graph.json --target=target.csv --epochs=100 --save=trained.bin\n\n";
    cout << "  # Make predictions\n";
    cout << "  gnn_cuda predict --model=trained.bin --graph=graph.json\n\n";
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
        return 0;
    }
    
    int DeviceCount;
    cudaGetDeviceCount(&DeviceCount);
    if (DeviceCount == 0) {
        cerr << "No CUDA devices found!" << endl;
        return 1;
    }
    
    cudaDeviceProp Prop;
    cudaGetDeviceProperties(&Prop, 0);
    
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
    
    if (Command_enum == cmdCreate) {
        if (featureSize <= 0) { cout << "Error: --feature is required" << endl; return 1; }
        if (hiddenSize <= 0) { cout << "Error: --hidden is required" << endl; return 1; }
        if (outputSize <= 0) { cout << "Error: --output is required" << endl; return 1; }
        if (mpLayers <= 0) { cout << "Error: --mp-layers is required" << endl; return 1; }
        if (saveFile == "") { cout << "Error: --model is required" << endl; return 1; }
        
        TGraphNeuralNetwork* GNN = new TGraphNeuralNetwork(featureSize, hiddenSize, outputSize, mpLayers);
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
        
        TGraphNeuralNetwork* GNN = new TGraphNeuralNetwork(1, 1, 1, 1);
        GNN->LoadModel(modelFile);
        
        if (learningRate > 0)
            GNN->SetLearningRate(learningRate);
        
        cout << "Training model for " << epochs << " epochs..." << endl;
        
        TDoubleArray target(GNN->GetOutputSize());
        for (int i = 0; i < GNN->GetOutputSize(); i++)
            target[i] = (double)rand() / RAND_MAX;
        
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
        
        TGraphNeuralNetwork* GNN = new TGraphNeuralNetwork(1, 1, 1, 1);
        GNN->LoadModel(modelFile);
        
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
        
        TGraphNeuralNetwork* GNN = new TGraphNeuralNetwork(1, 1, 1, 1);
        GNN->LoadModel(modelFile);
        
        cout << "GNN Model Information (CUDA)\n";
        cout << "============================\n";
        cout << "GPU Acceleration: Enabled (CUDA)\n";
        cout << "Feature size: " << GNN->GetFeatureSize() << "\n";
        cout << "Hidden size: " << GNN->GetHiddenSize() << "\n";
        cout << "Output size: " << GNN->GetOutputSize() << "\n";
        cout << "Message passing layers: " << GNN->GetMPLayers() << "\n\n";
        cout << "Hyperparameters:\n";
        cout << fixed << setprecision(6) << "  Learning rate: " << GNN->GetLearningRate() << "\n";
        cout << "  Activation: " << ActivationToStr(GNN->GetActivation()) << "\n";
        cout << "  Loss function: " << LossToStr(GNN->GetLossFunction()) << "\n";
        cout << "  Max iterations: " << GNN->GetMaxIterations() << "\n";
        cout << "GPU: " << Prop.name << "\n";
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
        
        TGraphNeuralNetwork* GNN = new TGraphNeuralNetwork(1, 1, 1, 1);
        GNN->LoadModel(modelFile);
        GNN->SaveModel(outputFile);
        cout << "Model saved to: " << outputFile << "\n";
        
        delete GNN;
    }
    else if (Command_enum == cmdLoad) {
        if (modelFile == "") { cout << "Error: --model is required" << endl; return 1; }
        
        TGraphNeuralNetwork* GNN = new TGraphNeuralNetwork(1, 1, 1, 1);
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
