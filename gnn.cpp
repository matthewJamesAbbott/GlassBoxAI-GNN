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

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <memory>
#include <cstring>

using namespace std;

const int MAX_NODES = 1000;
const int MAX_EDGES = 10000;
const int MAX_ITERATIONS = 10000;
const double GRADIENT_CLIP = 5.0;
const char* MODEL_MAGIC = "GNNBKND01";

enum TActivationType {
   atReLU,
   atLeakyReLU,
   atTanh,
   atSigmoid
};

enum TLossType {
   ltMSE,
   ltBinaryCrossEntropy
};

enum TCommand {
   cmdNone,
   cmdCreate,
   cmdTrain,
   cmdPredict,
   cmdInfo,
   cmdHelp
};

typedef vector<double> TDoubleArray;
typedef vector<int> TIntArray;
typedef vector<TDoubleArray> TDouble2DArray;

struct TEdge {
   int Source;
   int Target;

   bool operator==(const TEdge& other) const {
      return (Source == other.Source) && (Target == other.Target);
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

/* TGraphNeuralNetwork */
class TGraphNeuralNetwork {
private:
   double FLearningRate;
   int FMaxIterations;
   int FNumMessagePassingLayers;
   int FFeatureSize;
   int FHiddenSize;
   int FOutputSize;
   TActivationType FActivation;
   TLossType FLossType;

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
   TGraphNeuralNetwork(int AFeatureSize, int AHiddenSize, int AOutputSize, int NumMPLayers);
   ~TGraphNeuralNetwork();

   TDoubleArray Predict(TGraph& Graph);
   double Train(TGraph& Graph, const TDoubleArray& Target);
   void TrainMultiple(TGraph& Graph, const TDoubleArray& Target, int Iterations);

   void SaveModel(const string& Filename);
   void LoadModel(const string& Filename);

   static void ValidateGraph(TGraph& Graph, vector<string>& Errors);
   static void DeduplicateEdges(TGraph& Graph);
   static void AddReverseEdges(TGraph& Graph);
   static void AddSelfLoops(TGraph& Graph);

   // Property accessors
   double GetLearningRate() const { return FLearningRate; }
   void SetLearningRate(double value) { FLearningRate = value; }
   int GetMaxIterations() const { return FMaxIterations; }
   void SetMaxIterations(int value) { FMaxIterations = value; }
   TActivationType GetActivation() const { return FActivation; }
   void SetActivation(TActivationType value) { FActivation = value; }
   TLossType GetLossFunction() const { return FLossType; }
   void SetLossFunction(TLossType value) { FLossType = value; }
   TTrainingMetrics GetMetrics() const { return FMetrics; }
   int GetFeatureSize();
   int GetHiddenSize();
   int GetOutputSize();

   /* JSON serialization methods */
   void SaveModelToJSON(const string& Filename);
   void LoadModelFromJSON(const string& Filename);

   /* JSON serialization helper functions */
   string Array1DToJSON(const TDoubleArray& Arr);
   string Array2DToJSON(const TDouble2DArray& Arr);
};

// ==================== Forward Declarations ====================

string ActivationToStr(TActivationType act);
string LossToStr(TLossType loss);
TActivationType ParseActivation(const string& s);
TLossType ParseLoss(const string& s);

// ==================== Helper Functions ====================

TDoubleArray CopyArray(const TDoubleArray& Src) {
   TDoubleArray Result;
   Result.resize(Src.size());
   for (size_t I = 0; I < Src.size(); I++)
      Result[I] = Src[I];
   return Result;
}

TDoubleArray ConcatArrays(const TDoubleArray& A, const TDoubleArray& B) {
   TDoubleArray Result;
   Result.resize(A.size() + B.size());
   for (size_t I = 0; I < A.size(); I++)
      Result[I] = A[I];
   for (size_t I = 0; I < B.size(); I++)
      Result[A.size() + I] = B[I];
   return Result;
}

TDoubleArray ZeroArray(int Size) {
   TDoubleArray Result;
   Result.resize(Size);
   for (int I = 0; I < Size; I++)
      Result[I] = 0.0;
   return Result;
}

TDoubleArray PadArray(const TDoubleArray& Src, int NewSize) {
   TDoubleArray Result = ZeroArray(NewSize);
   for (size_t I = 0; I < min(Src.size(), (size_t)NewSize); I++)
      Result[I] = Src[I];
   return Result;
}

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

   FMetrics.LossHistory.resize(0);
}

TGraphNeuralNetwork:: ~TGraphNeuralNetwork() {
   // Destructor
}

int TGraphNeuralNetwork::GetFeatureSize() {
   return FFeatureSize;
}

int TGraphNeuralNetwork::GetHiddenSize() {
   return FHiddenSize;
}

int TGraphNeuralNetwork::GetOutputSize() {
   return FOutputSize;
}

void TGraphNeuralNetwork::InitializeLayer(TLayer& Layer, int NumNeurons, int NumInputs) {
   int I, J;
   double Scale;

   Layer.NumInputs = NumInputs;
   Layer.NumOutputs = NumNeurons;
   Layer. Neurons.resize(NumNeurons);

   Scale = sqrt(2.0 / (NumInputs + NumNeurons));

   static random_device rd;
   static mt19937 gen(rd());
   static uniform_real_distribution<> dis(0.0, 1.0);

   for (I = 0; I < NumNeurons; I++) {
      Layer.Neurons[I]. Weights.resize(NumInputs);
      for (J = 0; J < NumInputs; J++)
         Layer.Neurons[I].Weights[J] = (dis(gen) - 0.5) * 2.0 * Scale;
      Layer.Neurons[I]. Bias = 0.0;
      Layer.Neurons[I].Output = 0.0;
      Layer.Neurons[I].PreActivation = 0.0;
      Layer.Neurons[I].Error = 0.0;
   }
}

double TGraphNeuralNetwork:: Activate(double X) {
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

double TGraphNeuralNetwork:: ActivateDerivative(double X) {
   double S;

   switch (FActivation) {
      case atReLU:
         return (X > 0) ? 1.0 : 0.0;
      case atLeakyReLU:
         return (X > 0) ? 1.0 : 0.01;
      case atTanh:
         return 1.0 - pow(tanh(X), 2);
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
   double S;
   S = 1.0 / (1.0 + exp(-max(-500.0, min(500.0, PreAct))));
   return S * (1.0 - S);
}

double TGraphNeuralNetwork:: ComputeLoss(const TDoubleArray& Prediction, const TDoubleArray& Target) {
   int I;
   double P;
   double Result = 0.0;

   switch (FLossType) {
      case ltMSE:
         for (I = 0; I < (int)Prediction.size(); I++)
            Result = Result + pow(Prediction[I] - Target[I], 2);
         Result = Result / Prediction.size();
         break;
      case ltBinaryCrossEntropy:
         for (I = 0; I < (int)Prediction.size(); I++) {
            P = max(1e-7, min(1.0 - 1e-7, Prediction[I]));
            Result = Result - (Target[I] * log(P) + (1.0 - Target[I]) * log(1.0 - P));
         }
         Result = Result / Prediction.size();
         break;
   }
   return Result;
}

TDoubleArray TGraphNeuralNetwork::ComputeLossGradient(const TDoubleArray& Prediction, const TDoubleArray& Target) {
   int I;
   double P;
   TDoubleArray Result;
   Result.resize(Prediction.size());

   switch (FLossType) {
      case ltMSE:
         for (I = 0; I < (int)Prediction.size(); I++)
            Result[I] = 2.0 * (Prediction[I] - Target[I]) / Prediction.size();
         break;
      case ltBinaryCrossEntropy:
         for (I = 0; I < (int)Prediction.size(); I++) {
            P = max(1e-7, min(1.0 - 1e-7, Prediction[I]));
            Result[I] = (-Target[I] / P + (1.0 - Target[I]) / (1.0 - P)) / Prediction.size();
         }
         break;
   }
   return Result;
}

double TGraphNeuralNetwork:: ClipGradient(double G) {
   return max(-GRADIENT_CLIP, min(GRADIENT_CLIP, G));
}

void TGraphNeuralNetwork::BuildAdjacencyList(TGraph& Graph) {
   int I, Src, Tgt;

   Graph.AdjacencyList.resize(Graph.NumNodes);
   for (I = 0; I < Graph.NumNodes; I++)
      Graph.AdjacencyList[I].resize(0);

   for (I = 0; I < (int)Graph.Edges.size(); I++) {
      Src = Graph.Edges[I].Source;
      Tgt = Graph.Edges[I].Target;

      if ((Src >= 0) && (Src < Graph.NumNodes) &&
          (Tgt >= 0) && (Tgt < Graph.NumNodes)) {
         Graph.AdjacencyList[Src].push_back(Tgt);
      }
   }
}

void TGraphNeuralNetwork::ValidateGraph(TGraph& Graph, vector<string>& Errors) {
   int I;

   Errors.clear();

   if (Graph.NumNodes < 1)
      Errors.push_back("Graph must have at least 1 node");

   if (Graph.NumNodes > MAX_NODES) {
      ostringstream oss;
      oss << "Too many nodes (max " << MAX_NODES << ")";
      Errors.push_back(oss.str());
   }

   if ((int)Graph.Edges.size() > MAX_EDGES) {
      ostringstream oss;
      oss << "Too many edges (max " << MAX_EDGES << ")";
      Errors.push_back(oss.str());
   }

   for (I = 0; I < (int)Graph.Edges.size(); I++) {
      if ((Graph.Edges[I].Source < 0) || (Graph.Edges[I].Source >= Graph.NumNodes)) {
         ostringstream oss;
         oss << "Edge " << I << ": invalid source " << Graph.Edges[I]. Source;
         Errors.push_back(oss.str());
      }
      if ((Graph.Edges[I]. Target < 0) || (Graph.Edges[I].Target >= Graph.NumNodes)) {
         ostringstream oss;
         oss << "Edge " << I << ": invalid target " << Graph.Edges[I].Target;
         Errors.push_back(oss.str());
      }
   }

   for (I = 0; I < (int)Graph.NodeFeatures.size(); I++) {
      if (Graph.NodeFeatures[I].size() == 0) {
         ostringstream oss;
         oss << "Node " << I << ": empty feature vector";
         Errors. push_back(oss.str());
      }
   }
}

void TGraphNeuralNetwork::DeduplicateEdges(TGraph& Graph) {
   int I, J;
   vector<string> Seen;
   string Key;
   bool Found;
   TEdgeArray NewEdges;

   Seen.resize(0);
   NewEdges.resize(0);

   for (I = 0; I < (int)Graph.Edges.size(); I++) {
      ostringstream oss;
      oss << Graph.Edges[I].Source << "-" << Graph.Edges[I].Target;
      Key = oss.str();
      Found = false;

      for (J = 0; J < (int)Seen.size(); J++) {
         if (Seen[J] == Key) {
            Found = true;
            break;
         }
      }

      if (!Found) {
         Seen.push_back(Key);
         NewEdges.push_back(Graph. Edges[I]);
      }
   }

   Graph.Edges = NewEdges;
}

void TGraphNeuralNetwork::AddReverseEdges(TGraph& Graph) {
   int I, OrigLen;
   TEdge RevEdge;

   OrigLen = Graph.Edges.size();
   Graph.Edges.resize(OrigLen * 2);

   for (I = 0; I < OrigLen; I++) {
      if (Graph.Edges[I]. Source != Graph.Edges[I].Target) {
         RevEdge. Source = Graph.Edges[I].Target;
         RevEdge.Target = Graph.Edges[I].Source;
         Graph. Edges[OrigLen + I] = RevEdge;
      }
      else
         Graph.Edges[OrigLen + I] = Graph. Edges[I];
   }

   DeduplicateEdges(Graph);
}

void TGraphNeuralNetwork::AddSelfLoops(TGraph& Graph) {
   int I, J;
   bool HasSelf;
   TEdge SelfEdge;

   for (I = 0; I < Graph.NumNodes; I++) {
      HasSelf = false;
      for (J = 0; J < (int)Graph.Edges.size(); J++) {
         if ((Graph.Edges[J].Source == I) && (Graph.Edges[J].Target == I)) {
            HasSelf = true;
            break;
         }
      }

      if (! HasSelf) {
         SelfEdge.Source = I;
         SelfEdge.Target = I;
         Graph. Edges.push_back(SelfEdge);
      }
   }
}

TDoubleArray TGraphNeuralNetwork::ForwardLayer(TLayer& Layer, const TDoubleArray& Input, bool UseOutputActivation) {
   int I, J;
   double Sum;
   TDoubleArray Result;

   Layer.LastInput = CopyArray(Input);
   Result.resize(Layer.NumOutputs);

   for (I = 0; I < Layer.NumOutputs; I++) {
      Sum = Layer. Neurons[I]. Bias;
      for (J = 0; J < Layer.NumInputs; J++) {
         if (J < (int)Input.size())
            Sum = Sum + Layer.Neurons[I].Weights[J] * Input[J];
      }
      Layer.Neurons[I].PreActivation = Sum;

      if (UseOutputActivation)
         Layer.Neurons[I].Output = OutputActivate(Sum);
      else
         Layer. Neurons[I].Output = Activate(Sum);

      Result[I] = Layer.Neurons[I].Output;
   }

   return Result;
}

void TGraphNeuralNetwork::BackwardLayer(TLayer& Layer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation) {
   int I, J;
   double PreActGrad, DeltaW;

   for (I = 0; I < Layer.NumOutputs; I++) {
      if (UseOutputActivation)
         PreActGrad = UpstreamGrad[I] * OutputActivateDerivative(Layer.Neurons[I].PreActivation);
      else
         PreActGrad = UpstreamGrad[I] * ActivateDerivative(Layer. Neurons[I].PreActivation);

      Layer.Neurons[I].Error = PreActGrad;

      for (J = 0; J < Layer.NumInputs; J++) {
         if (J < (int)Layer.LastInput.size()) {
            DeltaW = FLearningRate * PreActGrad * Layer.LastInput[J];
            Layer. Neurons[I].Weights[J] = Layer.Neurons[I].Weights[J] - DeltaW;
         }
      }

      Layer.Neurons[I].Bias = Layer.Neurons[I]. Bias - (FLearningRate * PreActGrad);
   }
}

TDoubleArray TGraphNeuralNetwork::GetLayerInputGrad(const TLayer& Layer, const TDoubleArray& UpstreamGrad, bool UseOutputActivation) {
   int I, J;
   double PreActGrad;
   TDoubleArray Result;

   Result.resize(Layer.NumInputs);
   for (I = 0; I < Layer.NumInputs; I++)
      Result[I] = 0.0;

   for (I = 0; I < Layer.NumOutputs; I++) {
      if (UseOutputActivation)
         PreActGrad = UpstreamGrad[I] * OutputActivateDerivative(Layer. Neurons[I].PreActivation);
      else
         PreActGrad = UpstreamGrad[I] * ActivateDerivative(Layer.Neurons[I].PreActivation);

      for (J = 0; J < Layer.NumInputs; J++) {
         if (J < (int)Layer.Neurons[I]. Weights.size())
            Result[J] = Result[J] + Layer.Neurons[I].Weights[J] * PreActGrad;
      }
      Result[J] = ClipGradient(Result[J]);
   }

   return Result;
}

void TGraphNeuralNetwork::MessagePassing(TGraph& Graph) {
   int Layer, Node, K, I, Neighbor;
   TDoubleArray ConcatFeatures, Message, AggregatedMessage, UpdateInput, PaddedEmb;
   TMessageInfo MsgInfo;

   FNodeEmbeddings.resize(Graph.NumNodes);
   FNewNodeEmbeddings.resize(Graph.NumNodes);
   FEmbeddingHistory.resize(FNumMessagePassingLayers + 1);
   FMessageHistory.resize(FNumMessagePassingLayers);
   FAggregatedMessages.resize(FNumMessagePassingLayers);

   for (I = 0; I < Graph. NumNodes; I++)
      FNodeEmbeddings[I] = CopyArray(Graph.NodeFeatures[I]);

   FEmbeddingHistory[0]. resize(Graph.NumNodes);
   for (I = 0; I < Graph.NumNodes; I++)
      FEmbeddingHistory[0][I] = CopyArray(FNodeEmbeddings[I]);

   for (Layer = 0; Layer < FNumMessagePassingLayers; Layer++) {
      FMessageHistory[Layer].resize(Graph.NumNodes);
      FAggregatedMessages[Layer]. resize(Graph.NumNodes);

      for (Node = 0; Node < Graph.NumNodes; Node++) {
         FMessageHistory[Layer][Node]. resize(0);
         AggregatedMessage = ZeroArray(FHiddenSize);

         if (Graph.AdjacencyList[Node].size() > 0) {
            for (K = 0; K < (int)Graph.AdjacencyList[Node].size(); K++) {
               Neighbor = Graph. AdjacencyList[Node][K];

               ConcatFeatures = ConcatArrays(FNodeEmbeddings[Node], FNodeEmbeddings[Neighbor]);
               Message = ForwardLayer(FMessageLayers[Layer], ConcatFeatures, false);

               MsgInfo. NeighborIdx = Neighbor;
               MsgInfo. ConcatInput = CopyArray(ConcatFeatures);
               MsgInfo.MessageOutput = CopyArray(Message);

               FMessageHistory[Layer][Node].push_back(MsgInfo);

               for (I = 0; I < FHiddenSize; I++)
                  AggregatedMessage[I] = AggregatedMessage[I] + Message[I];
            }

            for (I = 0; I < FHiddenSize; I++)
               AggregatedMessage[I] = AggregatedMessage[I] / Graph.AdjacencyList[Node].size();
         }

         FAggregatedMessages[Layer][Node] = CopyArray(AggregatedMessage);

         if (Layer == 0)
            PaddedEmb = PadArray(FNodeEmbeddings[Node], FHiddenSize);
         else
            PaddedEmb = CopyArray(FNodeEmbeddings[Node]);

         UpdateInput = ConcatArrays(PaddedEmb, AggregatedMessage);
         FNewNodeEmbeddings[Node] = ForwardLayer(FUpdateLayers[Layer], UpdateInput, false);
      }

      for (Node = 0; Node < Graph.NumNodes; Node++)
         FNodeEmbeddings[Node] = CopyArray(FNewNodeEmbeddings[Node]);

      FEmbeddingHistory[Layer + 1].resize(Graph.NumNodes);
      for (I = 0; I < Graph.NumNodes; I++)
         FEmbeddingHistory[Layer + 1][I] = CopyArray(FNodeEmbeddings[I]);
   }
}

void TGraphNeuralNetwork:: Readout(TGraph& Graph) {
   int I, J;

   FGraphEmbedding = ZeroArray(FHiddenSize);

   for (I = 0; I < Graph.NumNodes; I++)
      for (J = 0; J < FHiddenSize; J++)
         FGraphEmbedding[J] = FGraphEmbedding[J] + FNodeEmbeddings[I][J];

   for (J = 0; J < FHiddenSize; J++)
      FGraphEmbedding[J] = FGraphEmbedding[J] / Graph.NumNodes;

   ForwardLayer(FReadoutLayer, FGraphEmbedding, false);
}

TDoubleArray TGraphNeuralNetwork:: Predict(TGraph& Graph) {
   int I;
   TDoubleArray ReadoutOutput;

   if (Graph.Config.DeduplicateEdges)
      DeduplicateEdges(Graph);
   if (Graph.Config. Undirected)
      AddReverseEdges(Graph);
   if (Graph.Config.SelfLoops)
      AddSelfLoops(Graph);

   BuildAdjacencyList(Graph);
   MessagePassing(Graph);
   Readout(Graph);

   ReadoutOutput.resize(FHiddenSize);
   for (I = 0; I < FHiddenSize; I++)
      ReadoutOutput[I] = FReadoutLayer. Neurons[I]. Output;

   return ForwardLayer(FOutputLayer, ReadoutOutput, true);
}

void TGraphNeuralNetwork::BackPropagateGraph(TGraph& Graph, const TDoubleArray& Target) {
   int Layer, Node, I, J, K, HalfLen;
   TDoubleArray LossGrad, ReadoutGrad, GraphEmbGrad, MsgGrad, ConcatGrad;
   TDouble2DArray NodeGrads, NewNodeGrads;
   TDoubleArray UpdateInputGrad, PaddedEmb, UpdateInput;
   int NumNeighbors;

   LossGrad = ComputeLossGradient(FOutputLayer. LastInput, Target);

   for (I = 0; I < FOutputSize; I++)
      LossGrad[I] = LossGrad[I] * OutputActivateDerivative(FOutputLayer. Neurons[I].PreActivation);

   BackwardLayer(FOutputLayer, LossGrad, true);
   ReadoutGrad = GetLayerInputGrad(FOutputLayer, LossGrad, true);

   BackwardLayer(FReadoutLayer, ReadoutGrad, false);
   GraphEmbGrad = GetLayerInputGrad(FReadoutLayer, ReadoutGrad, false);

   NodeGrads. resize(Graph.NumNodes);
   for (Node = 0; Node < Graph.NumNodes; Node++) {
      NodeGrads[Node] = ZeroArray(FHiddenSize);
      for (I = 0; I < FHiddenSize; I++)
         NodeGrads[Node][I] = GraphEmbGrad[I] / Graph.NumNodes;
   }

      for (Layer = FNumMessagePassingLayers - 1; Layer >= 0; Layer--) {
      NewNodeGrads. resize(Graph.NumNodes);

      if (Layer == 0) {
         for (Node = 0; Node < Graph.NumNodes; Node++)
            NewNodeGrads[Node] = ZeroArray(FFeatureSize);
      }
      else {
         for (Node = 0; Node < Graph.NumNodes; Node++)
            NewNodeGrads[Node] = ZeroArray(FHiddenSize);
      }

      for (Node = 0; Node < Graph.NumNodes; Node++) {
         if (Layer == 0)
            PaddedEmb = PadArray(FEmbeddingHistory[Layer][Node], FHiddenSize);
         else
            PaddedEmb = CopyArray(FEmbeddingHistory[Layer][Node]);

         UpdateInput = ConcatArrays(PaddedEmb, FAggregatedMessages[Layer][Node]);
         FUpdateLayers[Layer].LastInput = CopyArray(UpdateInput);

         BackwardLayer(FUpdateLayers[Layer], NodeGrads[Node], false);
         UpdateInputGrad = GetLayerInputGrad(FUpdateLayers[Layer], NodeGrads[Node], false);

         for (I = 0; I < min(FHiddenSize, (int)NewNodeGrads[Node].size()); I++) {
            if (Layer == 0) {
               if (I < FFeatureSize)
                  NewNodeGrads[Node][I] = NewNodeGrads[Node][I] + UpdateInputGrad[I];
            }
            else
               NewNodeGrads[Node][I] = NewNodeGrads[Node][I] + UpdateInputGrad[I];
         }

         NumNeighbors = Graph.AdjacencyList[Node].size();
         if (NumNeighbors > 0) {
            MsgGrad = ZeroArray(FHiddenSize);
            for (I = 0; I < FHiddenSize; I++)
               MsgGrad[I] = UpdateInputGrad[FHiddenSize + I] / NumNeighbors;

            for (K = 0; K < (int)FMessageHistory[Layer][Node].size(); K++) {
               FMessageLayers[Layer]. LastInput = CopyArray(FMessageHistory[Layer][Node][K]. ConcatInput);

               BackwardLayer(FMessageLayers[Layer], MsgGrad, false);
               ConcatGrad = GetLayerInputGrad(FMessageLayers[Layer], MsgGrad, false);

               HalfLen = ConcatGrad.size() / 2;

               for (I = 0; I < min(HalfLen, (int)NewNodeGrads[Node].size()); I++)
                  NewNodeGrads[Node][I] = NewNodeGrads[Node][I] + ConcatGrad[I];

               J = FMessageHistory[Layer][Node][K]. NeighborIdx;
               for (I = 0; I < min(HalfLen, (int)NewNodeGrads[J].size()); I++)
                  NewNodeGrads[J][I] = NewNodeGrads[J][I] + ConcatGrad[HalfLen + I];
            }
         }
      }

      if (Layer > 0)
         NodeGrads = NewNodeGrads;
   }
}

double TGraphNeuralNetwork::Train(TGraph& Graph, const TDoubleArray& Target) {
   TDoubleArray Prediction;

   Prediction = Predict(Graph);
   double Result = ComputeLoss(Prediction, Target);
   BackPropagateGraph(Graph, Target);
   return Result;
}

void TGraphNeuralNetwork::TrainMultiple(TGraph& Graph, const TDoubleArray& Target, int Iterations) {
   int I;
   double Loss;

   FMetrics.LossHistory.resize(Iterations);

   for (I = 0; I < Iterations; I++) {
      Loss = Train(Graph, Target);
      FMetrics.LossHistory[I] = Loss;
      FMetrics.Loss = Loss;
      FMetrics.Iteration = I + 1;

      if ((I % 10 == 0) || (I == Iterations - 1)) {
         cout << "Iteration " << (I + 1) << "/" << Iterations << ", Loss: "
              << fixed << setprecision(6) << Loss << endl;
      }
   }
}

void TGraphNeuralNetwork::SaveModel(const string& Filename) {
   ofstream F(Filename, ios::binary);
   int I, J, K;
   int ActInt, LossInt;

   if (! F.is_open()) {
      cerr << "Error: Could not open file for writing:  " << Filename << endl;
      return;
   }

   F. write((char*)&FFeatureSize, sizeof(int));
   F.write((char*)&FHiddenSize, sizeof(int));
   F.write((char*)&FOutputSize, sizeof(int));
   F.write((char*)&FNumMessagePassingLayers, sizeof(int));
   F.write((char*)&FLearningRate, sizeof(double));

   ActInt = (int)FActivation;
   LossInt = (int)FLossType;
   F.write((char*)&ActInt, sizeof(int));
   F.write((char*)&LossInt, sizeof(int));

   for (K = 0; K < FNumMessagePassingLayers; K++) {
      F.write((char*)&FMessageLayers[K].NumOutputs, sizeof(int));
      F.write((char*)&FMessageLayers[K].NumInputs, sizeof(int));
      for (I = 0; I < FMessageLayers[K].NumOutputs; I++) {
         for (J = 0; J < FMessageLayers[K].NumInputs; J++)
            F.write((char*)&FMessageLayers[K]. Neurons[I]. Weights[J], sizeof(double));
         F.write((char*)&FMessageLayers[K].Neurons[I].Bias, sizeof(double));
      }

      F.write((char*)&FUpdateLayers[K].NumOutputs, sizeof(int));
      F.write((char*)&FUpdateLayers[K].NumInputs, sizeof(int));
      for (I = 0; I < FUpdateLayers[K].NumOutputs; I++) {
         for (J = 0; J < FUpdateLayers[K].NumInputs; J++)
            F.write((char*)&FUpdateLayers[K]. Neurons[I].Weights[J], sizeof(double));
         F.write((char*)&FUpdateLayers[K].Neurons[I].Bias, sizeof(double));
      }
   }

   F.write((char*)&FReadoutLayer.NumOutputs, sizeof(int));
   F.write((char*)&FReadoutLayer.NumInputs, sizeof(int));
   for (I = 0; I < FReadoutLayer. NumOutputs; I++) {
      for (J = 0; J < FReadoutLayer.NumInputs; J++)
         F.write((char*)&FReadoutLayer.Neurons[I]. Weights[J], sizeof(double));
      F.write((char*)&FReadoutLayer.Neurons[I].Bias, sizeof(double));
   }

   F.write((char*)&FOutputLayer.NumOutputs, sizeof(int));
   F.write((char*)&FOutputLayer.NumInputs, sizeof(int));
   for (I = 0; I < FOutputLayer.NumOutputs; I++) {
      for (J = 0; J < FOutputLayer.NumInputs; J++)
         F.write((char*)&FOutputLayer.Neurons[I].Weights[J], sizeof(double));
      F.write((char*)&FOutputLayer.Neurons[I]. Bias, sizeof(double));
   }

   F.close();
   cout << "Model saved to " << Filename << endl;
}

void TGraphNeuralNetwork::LoadModel(const string& Filename) {
   ifstream F(Filename, ios:: binary);
   int I, J, K, NumN, NumI;
   int ActInt, LossInt;
   double TmpDouble;

   if (!F. is_open()) {
      cerr << "Error: Could not open file for reading:  " << Filename << endl;
      return;
   }

   F.read((char*)&FFeatureSize, sizeof(int));
   F.read((char*)&FHiddenSize, sizeof(int));
   F.read((char*)&FOutputSize, sizeof(int));
   F.read((char*)&FNumMessagePassingLayers, sizeof(int));
   F.read((char*)&FLearningRate, sizeof(double));

   F.read((char*)&ActInt, sizeof(int));
   F.read((char*)&LossInt, sizeof(int));
   FActivation = (TActivationType)ActInt;
   FLossType = (TLossType)LossInt;

   FMessageLayers.resize(FNumMessagePassingLayers);
   FUpdateLayers.resize(FNumMessagePassingLayers);

   for (K = 0; K < FNumMessagePassingLayers; K++) {
      F.read((char*)&NumN, sizeof(int));
      F.read((char*)&NumI, sizeof(int));
      InitializeLayer(FMessageLayers[K], NumN, NumI);
      for (I = 0; I < NumN; I++) {
         for (J = 0; J < NumI; J++) {
            F.read((char*)&TmpDouble, sizeof(double));
            FMessageLayers[K].Neurons[I]. Weights[J] = TmpDouble;
         }
         F.read((char*)&TmpDouble, sizeof(double));
         FMessageLayers[K]. Neurons[I].Bias = TmpDouble;
      }

      F.read((char*)&NumN, sizeof(int));
      F.read((char*)&NumI, sizeof(int));
      InitializeLayer(FUpdateLayers[K], NumN, NumI);
      for (I = 0; I < NumN; I++) {
         for (J = 0; J < NumI; J++) {
            F.read((char*)&TmpDouble, sizeof(double));
            FUpdateLayers[K].Neurons[I].Weights[J] = TmpDouble;
         }
         F.read((char*)&TmpDouble, sizeof(double));
         FUpdateLayers[K].Neurons[I].Bias = TmpDouble;
      }
   }

   F.read((char*)&NumN, sizeof(int));
   F.read((char*)&NumI, sizeof(int));
   InitializeLayer(FReadoutLayer, NumN, NumI);
   for (I = 0; I < NumN; I++) {
      for (J = 0; J < NumI; J++) {
         F.read((char*)&TmpDouble, sizeof(double));
         FReadoutLayer.Neurons[I].Weights[J] = TmpDouble;
      }
      F.read((char*)&TmpDouble, sizeof(double));
      FReadoutLayer.Neurons[I]. Bias = TmpDouble;
   }

   F. read((char*)&NumN, sizeof(int));
   F.read((char*)&NumI, sizeof(int));
   InitializeLayer(FOutputLayer, NumN, NumI);
   for (I = 0; I < NumN; I++) {
      for (J = 0; J < NumI; J++) {
         F.read((char*)&TmpDouble, sizeof(double));
         FOutputLayer. Neurons[I].Weights[J] = TmpDouble;
      }
      F.read((char*)&TmpDouble, sizeof(double));
      FOutputLayer. Neurons[I].Bias = TmpDouble;
   }

   F.close();
   cout << "Model loaded from " << Filename << endl;
}

string TGraphNeuralNetwork::Array1DToJSON(const TDoubleArray& Arr) {
   ostringstream oss;
   oss << "[";
   for (size_t i = 0; i < Arr.size(); i++) {
      oss << fixed << setprecision(6) << Arr[i];
      if (i < Arr. size() - 1)
         oss << ",";
   }
   oss << "]";
   return oss.str();
}

string TGraphNeuralNetwork::Array2DToJSON(const TDouble2DArray& Arr) {
   ostringstream oss;
   oss << "[";
   for (size_t i = 0; i < Arr. size(); i++) {
      oss << Array1DToJSON(Arr[i]);
      if (i < Arr.size() - 1)
         oss << ",";
   }
   oss << "]";
   return oss. str();
}

void TGraphNeuralNetwork::SaveModelToJSON(const string& Filename) {
   ofstream F(Filename);
   int I, J, K;

   if (!F. is_open()) {
      cerr << "Error: Could not open file for writing:  " << Filename << endl;
      return;
   }

   F << "{" << endl;
   F << "  \"feature_size\": " << FFeatureSize << "," << endl;
   F << "  \"hidden_size\": " << FHiddenSize << "," << endl;
   F << "  \"output_size\": " << FOutputSize << "," << endl;
   F << "  \"num_message_passing_layers\": " << FNumMessagePassingLayers << "," << endl;
   F << "  \"learning_rate\": " << fixed << setprecision(6) << FLearningRate << "," << endl;
   F << "  \"activation\": \"" << ActivationToStr(FActivation) << "\"," << endl;
   F << "  \"loss_function\": \"" << LossToStr(FLossType) << "\"," << endl;

   F << "  \"message_layers\": [" << endl;
   for (K = 0; K < FNumMessagePassingLayers; K++) {
      F << "    {" << endl;
      F << "      \"num_outputs\": " << FMessageLayers[K].NumOutputs << "," << endl;
      F << "      \"num_inputs\":  " << FMessageLayers[K].NumInputs << "," << endl;
      F << "      \"neurons\": [" << endl;
      for (I = 0; I < FMessageLayers[K].NumOutputs; I++) {
         F << "        {" << endl;
         F << "          \"weights\": " << Array1DToJSON(FMessageLayers[K].Neurons[I].Weights) << "," << endl;
         F << "          \"bias\": " << fixed << setprecision(6) << FMessageLayers[K]. Neurons[I].Bias << endl;
         F << "        }";
         if (I < FMessageLayers[K].NumOutputs - 1) F << ",";
         F << endl;
      }
      F << "      ]" << endl;
      F << "    }";
      if (K < FNumMessagePassingLayers - 1) F << ",";
      F << endl;
   }
   F << "  ]," << endl;

   F << "  \"update_layers\": [" << endl;
   for (K = 0; K < FNumMessagePassingLayers; K++) {
      F << "    {" << endl;
      F << "      \"num_outputs\": " << FUpdateLayers[K].NumOutputs << "," << endl;
      F << "      \"num_inputs\": " << FUpdateLayers[K].NumInputs << "," << endl;
      F << "      \"neurons\": [" << endl;
      for (I = 0; I < FUpdateLayers[K].NumOutputs; I++) {
         F << "        {" << endl;
         F << "          \"weights\": " << Array1DToJSON(FUpdateLayers[K].Neurons[I].Weights) << "," << endl;
         F << "          \"bias\": " << fixed << setprecision(6) << FUpdateLayers[K]. Neurons[I].Bias << endl;
         F << "        }";
         if (I < FUpdateLayers[K].NumOutputs - 1) F << ",";
         F << endl;
      }
      F << "      ]" << endl;
      F << "    }";
      if (K < FNumMessagePassingLayers - 1) F << ",";
      F << endl;
   }
   F << "  ]," << endl;

   F << "  \"readout_layer\": {" << endl;
   F << "    \"num_outputs\": " << FReadoutLayer.NumOutputs << "," << endl;
   F << "    \"num_inputs\": " << FReadoutLayer.NumInputs << "," << endl;
   F << "    \"neurons\": [" << endl;
   for (I = 0; I < FReadoutLayer.NumOutputs; I++) {
      F << "      {" << endl;
      F << "        \"weights\": " << Array1DToJSON(FReadoutLayer.Neurons[I].Weights) << "," << endl;
      F << "        \"bias\": " << fixed << setprecision(6) << FReadoutLayer.Neurons[I].Bias << endl;
      F << "      }";
      if (I < FReadoutLayer.NumOutputs - 1) F << ",";
      F << endl;
   }
   F << "    ]" << endl;
   F << "  }," << endl;

   F << "  \"output_layer\": {" << endl;
   F << "    \"num_outputs\": " << FOutputLayer.NumOutputs << "," << endl;
   F << "    \"num_inputs\": " << FOutputLayer.NumInputs << "," << endl;
   F << "    \"neurons\": [" << endl;
   for (I = 0; I < FOutputLayer.NumOutputs; I++) {
      F << "      {" << endl;
      F << "        \"weights\": " << Array1DToJSON(FOutputLayer.Neurons[I]. Weights) << "," << endl;
      F << "        \"bias\": " << fixed << setprecision(6) << FOutputLayer.Neurons[I]. Bias << endl;
      F << "      }";
      if (I < FOutputLayer.NumOutputs - 1) F << ",";
      F << endl;
   }
   F << "    ]" << endl;
   F << "  }" << endl;
   F << "}" << endl;

   F.close();
   cout << "Model saved to JSON: " << Filename << endl;
}

void TGraphNeuralNetwork::LoadModelFromJSON(const string& Filename) {
   ifstream F(Filename);

   if (!F.is_open()) {
      cerr << "Error: Could not open file for reading: " << Filename << endl;
      return;
   }

   // Simple JSON parsing - reads line by line and extracts values
   string line;
   int K = -1;
   bool inMessageLayers = false;
   bool inUpdateLayers = false;
   bool inReadoutLayer = false;
   bool inOutputLayer = false;
   int neuronIdx = 0;
   TLayer* currentLayer = nullptr;

   while (getline(F, line)) {
      // Trim whitespace
      size_t start = line.find_first_not_of(" \t");
      if (start == string::npos) continue;
      line = line.substr(start);

      // Parse architecture parameters
      if (line.find("\"feature_size\": ") != string::npos) {
         sscanf(line.c_str(), "\"feature_size\": %d", &FFeatureSize);
      }
      else if (line.find("\"hidden_size\":") != string::npos) {
         sscanf(line.c_str(), "\"hidden_size\": %d", &FHiddenSize);
      }
      else if (line.find("\"output_size\":") != string::npos) {
         sscanf(line.c_str(), "\"output_size\": %d", &FOutputSize);
      }
      else if (line. find("\"num_message_passing_layers\":") != string::npos) {
         sscanf(line.c_str(), "\"num_message_passing_layers\": %d", &FNumMessagePassingLayers);
         FMessageLayers.resize(FNumMessagePassingLayers);
         FUpdateLayers.resize(FNumMessagePassingLayers);
      }
      else if (line.find("\"learning_rate\":") != string::npos) {
         sscanf(line.c_str(), "\"learning_rate\": %lf", &FLearningRate);
      }
      else if (line.find("\"activation\": ") != string::npos) {
         size_t pos1 = line.find("\"", line.find(": ") + 1);
         size_t pos2 = line.find("\"", pos1 + 1);
         string actStr = line.substr(pos1 + 1, pos2 - pos1 - 1);
         FActivation = ParseActivation(actStr);
      }
      else if (line.find("\"loss_function\":") != string::npos) {
         size_t pos1 = line. find("\"", line.find(":") + 1);
         size_t pos2 = line.find("\"", pos1 + 1);
         string lossStr = line.substr(pos1 + 1, pos2 - pos1 - 1);
         FLossType = ParseLoss(lossStr);
      }
      // Section markers
      else if (line.find("\"message_layers\":") != string::npos) {
         inMessageLayers = true;
         K = -1;
      }
      else if (line.find("\"update_layers\":") != string::npos) {
         inMessageLayers = false;
         inUpdateLayers = true;
         K = -1;
      }
      else if (line.find("\"readout_layer\":") != string::npos) {
         inUpdateLayers = false;
         inReadoutLayer = true;
         currentLayer = &FReadoutLayer;
         neuronIdx = 0;
      }
      else if (line. find("\"output_layer\":") != string::npos) {
         inReadoutLayer = false;
         inOutputLayer = true;
         currentLayer = &FOutputLayer;
         neuronIdx = 0;
      }
      // Layer properties
      else if (line.find("\"num_outputs\":") != string::npos) {
         int numOutputs;
         sscanf(line.c_str(), "\"num_outputs\": %d", &numOutputs);
         if (currentLayer != nullptr) {
            currentLayer->NumOutputs = numOutputs;
         }
      }
      else if (line.find("\"num_inputs\":") != string::npos) {
         int numInputs;
         sscanf(line.c_str(), "\"num_inputs\": %d", &numInputs);
         if (currentLayer != nullptr) {
            currentLayer->NumInputs = numInputs;
            currentLayer->Neurons.resize(currentLayer->NumOutputs);
            for (int i = 0; i < currentLayer->NumOutputs; i++) {
               currentLayer->Neurons[i].Weights.resize(numInputs);
            }
         }
      }
      // New layer object in array
      else if (line.find("{") != string::npos && line.find("\"num_outputs\"") == string::npos) {
         if (inMessageLayers || inUpdateLayers) {
            K++;
            if (inMessageLayers) {
               currentLayer = &FMessageLayers[K];
            } else {
               currentLayer = &FUpdateLayers[K];
            }
            neuronIdx = 0;
         }
      }
      // Weights
      else if (line.find("\"weights\":") != string::npos) {
         size_t pos1 = line.find("[");
         size_t pos2 = line.find("]");
         if (pos1 != string::npos && pos2 != string:: npos) {
            string weightsStr = line.substr(pos1 + 1, pos2 - pos1 - 1);
            stringstream ss(weightsStr);
            string token;
            int idx = 0;
            while (getline(ss, token, ',')) {
               if (currentLayer != nullptr && neuronIdx < (int)currentLayer->Neurons. size()) {
                  currentLayer->Neurons[neuronIdx]. Weights[idx++] = stod(token);
               }
            }
         }
      }
      // Bias
      else if (line.find("\"bias\":") != string::npos) {
         double bias;
         sscanf(line.c_str(), "\"bias\": %lf", &bias);
         if (currentLayer != nullptr && neuronIdx < (int)currentLayer->Neurons.size()) {
            currentLayer->Neurons[neuronIdx].Bias = bias;
            currentLayer->Neurons[neuronIdx].Output = 0.0;
            currentLayer->Neurons[neuronIdx].PreActivation = 0.0;
            currentLayer->Neurons[neuronIdx].Error = 0.0;
            neuronIdx++;
         }
      }
   }

   F.close();
   cout << "Model loaded from JSON: " << Filename << endl;
   cout << "  Feature size: " << FFeatureSize << endl;
   cout << "  Hidden size: " << FHiddenSize << endl;
   cout << "  Output size: " << FOutputSize << endl;
   cout << "  Message passing layers: " << FNumMessagePassingLayers << endl;
   cout << "  Activation:  " << ActivationToStr(FActivation) << endl;
   cout << "  Loss function: " << LossToStr(FLossType) << endl;
   cout << "  Learning rate: " << fixed << setprecision(6) << FLearningRate << endl;
}

// ==================== Utility Functions ====================

string ActivationToStr(TActivationType act) {
   switch (act) {
      case atReLU:
         return "ReLU";
      case atLeakyReLU:
         return "LeakyReLU";
      case atTanh:
         return "Tanh";
      case atSigmoid:
         return "Sigmoid";
      default:
         return "Unknown";
   }
}

string LossToStr(TLossType loss) {
   switch (loss) {
      case ltMSE:
         return "MSE";
      case ltBinaryCrossEntropy:
         return "BinaryCrossEntropy";
      default:
         return "Unknown";
   }
}

TActivationType ParseActivation(const string& s) {
   string lower = s;
   transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

   if (lower == "relu")
      return atReLU;
   else if (lower == "leakyrelu")
      return atLeakyReLU;
   else if (lower == "tanh")
      return atTanh;
   else if (lower == "sigmoid")
      return atSigmoid;
   else
      return atReLU; // default
}

TLossType ParseLoss(const string& s) {
   string lower = s;
   transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

   if (lower == "mse")
      return ltMSE;
   else if (lower == "binarycrossentropy")
      return ltBinaryCrossEntropy;
   else
      return ltMSE; // default
}

// ==================== Main Program ====================

int main(int argc, char* argv[]) {
   TCommand Command = cmdNone;
   string modelFile = "";
   string graphFile = "";
   string saveFile = "";
   int featureSize = 0;
   int hiddenSize = 0;
   int outputSize = 0;
   int mpLayers = 0;
   double learningRate = 0.01;
   TActivationType activation = atReLU;
   TLossType loss = ltMSE;
   int epochs = 100;

   TGraphNeuralNetwork* GNN = nullptr;

   // Parse command line arguments
   if (argc < 2) {
      Command = cmdHelp;
   }
   else {
      string cmd = argv[1];
      transform(cmd.begin(), cmd.end(), cmd.begin(), ::tolower);

      if (cmd == "create")
         Command = cmdCreate;
      else if (cmd == "train")
         Command = cmdTrain;
      else if (cmd == "predict")
         Command = cmdPredict;
      else if (cmd == "info")
         Command = cmdInfo;
      else if (cmd == "help")
         Command = cmdHelp;
      else
         Command = cmdHelp;
   }

   // Show help
   if (Command == cmdHelp) {
      cout << "GNN - Graph Neural Network" << endl;
      cout << endl;
      cout << "Usage:" << endl;
      cout << "  GNN create --feature <size> --hidden <size> --output <size> --mp-layers <n> --save <file>" << endl;
      cout << "  GNN train --model <file> --graph <file> --save <file> [--epochs <n>] [--lr <rate>]" << endl;
      cout << "  GNN predict --model <file> --graph <file>" << endl;
      cout << "  GNN info --model <file>" << endl;
      cout << "  GNN help" << endl;
      cout << endl;
      cout << "Commands:" << endl;
      cout << "  create   - Create a new GNN model" << endl;
      cout << "  train    - Train an existing model" << endl;
      cout << "  predict  - Make predictions with a model" << endl;
      cout << "  info     - Display model information" << endl;
      cout << "  help     - Show this help message" << endl;
      cout << endl;
      cout << "Options:" << endl;
      cout << "  --feature <size>    - Feature vector size" << endl;
      cout << "  --hidden <size>     - Hidden layer size" << endl;
      cout << "  --output <size>     - Output size" << endl;
      cout << "  --mp-layers <n>     - Number of message passing layers" << endl;
      cout << "  --model <file>      - Model file to load" << endl;
      cout << "  --graph <file>      - Graph data file" << endl;
      cout << "  --save <file>       - File to save model/results" << endl;
      cout << "  --lr <rate>         - Learning rate (default: 0.01)" << endl;
      cout << "  --activation <type> - Activation function (ReLU, LeakyReLU, Tanh, Sigmoid)" << endl;
      cout << "  --loss <type>       - Loss function (MSE, BinaryCrossEntropy)" << endl;
      cout << "  --epochs <n>        - Number of training epochs (default: 100)" << endl;
      return 0;
   }
}
