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

#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <map>

using namespace std;

// ========== OpenCL Error Checking ==========
#define CL_CHECK(err) \
    do { \
        if (err != CL_SUCCESS) { \
            cerr << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << (int)err << endl; \
            exit(1); \
        } \
    } while(0)

// ========== Type Definitions ==========
enum ActivationType {
    atSigmoid,
    atTanh,
    atReLU,
    atLinear
};

enum LossType {
    ltMSE,
    ltCrossEntropy
};

enum Command {
    cmdNone,
    cmdCreate,
    cmdTrain,
    cmdPredict,
    cmdInfo,
    cmdHelp
};

const int BLOCK_SIZE = 256;
const char MODEL_MAGIC[] = "CNNOCI01";

// Type aliases matching Pascal
typedef vector<float> FArray;
typedef vector<FArray> TFArray2D;
typedef vector<TFArray2D> TFArray3D;
typedef vector<TFArray3D> TFArray4D;
typedef vector<int> TIntArray;

// ========== OpenCL Kernels ==========
const char* kernelSource = R"CLC(
float d_Sigmoid(float x) {
    float clamped = fmax(-500.0f, fmin(500.0f, x));
    return 1.0f / (1.0f + exp(-clamped));
}

float d_DSigmoid(float x) {
    return x * (1.0f - x);
}

float d_TanhActivation(float x) {
    return tanh(x);
}

float d_DTanh(float x) {
    return 1.0f - (x * x);
}

float d_ReLU(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

float d_DReLU(float x) {
    return (x > 0.0f) ? 1.0f : 0.0f;
}

float d_ApplyActivation(float x, int ActType) {
    switch (ActType) {
        case 0: return d_Sigmoid(x);
        case 1: return d_TanhActivation(x);
        case 2: return d_ReLU(x);
        case 3: return x;
        default: return x;
    }
}

float d_ApplyActivationDerivative(float x, int ActType) {
    switch (ActType) {
        case 0: return d_DSigmoid(x);
        case 1: return d_DTanh(x);
        case 2: return d_DReLU(x);
        case 3: return 1.0f;
        default: return 1.0f;
    }
}

float d_ClipValue(float V, float MaxVal) {
    if (V > MaxVal) return MaxVal;
    else if (V < -MaxVal) return -MaxVal;
    else return V;
}

// Convolution kernel forward pass
__kernel void k_conv_forward(
    __global float* output,
    __global float* preActivation,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    int inputChannels, int inputHeight, int inputWidth,
    int kernelSize, int padding, int stride,
    int outputChannels, int outputHeight, int outputWidth,
    int filterIdx) {
    
    int outH = get_global_id(0);
    int outW = get_global_id(1);
    
    if (outH < outputHeight && outW < outputWidth) {
        float sum = bias[filterIdx];
        
        for (int c = 0; c < inputChannels; c++) {
            for (int kh = 0; kh < kernelSize; kh++) {
                for (int kw = 0; kw < kernelSize; kw++) {
                    int ih = outH * stride + kh - padding;
                    int iw = outW * stride + kw - padding;
                    
                    if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                        int inIdx = (c * inputHeight + ih) * inputWidth + iw;
                        int wIdx = (filterIdx * inputChannels + c) * kernelSize * kernelSize + kh * kernelSize + kw;
                        sum += input[inIdx] * weights[wIdx];
                    }
                }
            }
        }
        
        int outIdx = (filterIdx * outputHeight + outH) * outputWidth + outW;
        preActivation[outIdx] = sum;
        output[outIdx] = d_ApplyActivation(sum, 2);
    }
}

// Activation function kernel
__kernel void k_activate(
    __global float* output,
    __global const float* input,
    int size, int actType) {
    
    int i = get_global_id(0);
    if (i < size) {
        output[i] = d_ApplyActivation(input[i], actType);
    }
}

// Max pooling forward pass
__kernel void k_pool_forward(
    __global float* output,
    __global int* maxIndices,
    __global const float* input,
    int inputChannels, int inputHeight, int inputWidth,
    int poolSize, int stride,
    int outputHeight, int outputWidth) {
    
    int c = get_global_id(0);
    int h = get_global_id(1);
    int w = get_global_id(2);
    
    if (c < inputChannels && h < outputHeight && w < outputWidth) {
        float maxVal = -1e38f;
        int maxIdx = 0;
        
        for (int ph = 0; ph < poolSize; ph++) {
            for (int pw = 0; pw < poolSize; pw++) {
                int ih = h * stride + ph;
                int iw = w * stride + pw;
                int idx = (c * inputHeight + ih) * inputWidth + iw;
                
                if (input[idx] > maxVal) {
                    maxVal = input[idx];
                    maxIdx = ph * poolSize + pw;
                }
            }
        }
        
        int outIdx = (c * outputHeight + h) * outputWidth + w;
        output[outIdx] = maxVal;
        maxIndices[outIdx] = maxIdx;
    }
}

// Flatten kernel (3D to 1D)
__kernel void k_flatten(
    __global float* output,
    __global const float* input,
    int channels, int height, int width) {
    
    int idx = get_global_id(0);
    int size = channels * height * width;
    
    if (idx < size) {
        int c = idx / (height * width);
        int hw = idx % (height * width);
        int h = hw / width;
        int w = hw % width;
        
        int inIdx = (c * height + h) * width + w;
        output[idx] = input[inIdx];
    }
}

// Fully connected forward pass
__kernel void k_fc_forward(
    __global float* output,
    __global float* preActivation,
    __global const float* input,
    __global const float* weights,
    __global const float* bias,
    int inputSize, int outputSize, int actType) {
    
    int i = get_global_id(0);
    
    if (i < outputSize) {
        float sum = bias[i];
        for (int j = 0; j < inputSize; j++) {
            sum += weights[i * inputSize + j] * input[j];
        }
        preActivation[i] = sum;
        output[i] = d_ApplyActivation(sum, actType);
    }
}

// Matrix-vector multiply
__kernel void k_matvec(
    __global float* y,
    __global const float* W,
    __global const float* x,
    __global const float* b,
    int rows, int cols) {
    
    int i = get_global_id(0);
    if (i < rows) {
        float sum = b[i];
        for (int j = 0; j < cols; j++) {
            sum += W[i * cols + j] * x[j];
        }
        y[i] = sum;
    }
}

// Softmax kernel
__kernel void k_softmax(
    __global float* output,
    __global const float* input,
    int size) {
    
    int i = get_global_id(0);
    
    if (i == 0) {
        float maxVal = input[0];
        for (int j = 1; j < size; j++) {
            if (input[j] > maxVal) maxVal = input[j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < size; j++) {
            output[j] = exp(input[j] - maxVal);
            sum += output[j];
        }
        
        for (int j = 0; j < size; j++) {
            output[j] /= (sum + 1e-15f);
        }
    }
}

// Convolution backward kernel
__kernel void k_conv_backward_weights(
    __global float* dWeights,
    __global float* dBias,
    __global const float* dOutput,
    __global const float* input,
    int inputChannels, int inputHeight, int inputWidth,
    int kernelSize, int padding, int stride,
    int outputHeight, int outputWidth,
    int filterIdx) {
    
    int c = get_global_id(0);
    int kh = get_global_id(1);
    int kw = get_global_id(2);
    
    if (c < inputChannels && kh < kernelSize && kw < kernelSize) {
        float sum = 0.0f;
        
        for (int oh = 0; oh < outputHeight; oh++) {
            for (int ow = 0; ow < outputWidth; ow++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth) {
                    int inIdx = (c * inputHeight + ih) * inputWidth + iw;
                    int outIdx = (filterIdx * outputHeight + oh) * outputWidth + ow;
                    sum += dOutput[outIdx] * input[inIdx];
                }
            }
        }
        
        int wIdx = (filterIdx * inputChannels + c) * kernelSize * kernelSize + kh * kernelSize + kw;
        dWeights[wIdx] = sum;
    }
}

// Gradient clipping kernel
__kernel void k_clip_gradients(
    __global float* gradients,
    int size, float clipValue) {
    
    int i = get_global_id(0);
    if (i < size) {
        gradients[i] = d_ClipValue(gradients[i], clipValue);
    }
}

// Update weights kernel (SGD)
__kernel void k_update_weights(
    __global float* weights,
    __global const float* gradients,
    int size, float learningRate) {
    
    int i = get_global_id(0);
    if (i < size) {
        weights[i] -= learningRate * gradients[i];
    }
}
)CLC";

// ========== OpenCL Context Manager ==========
class TOpenCLContext {
public:
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
    map<string, cl_kernel> kernels;
    
    TOpenCLContext() {
        cl_int err;
        
        err = clGetPlatformIDs(1, &platform, nullptr);
        CL_CHECK(err);
        
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr);
        }
        CL_CHECK(err);
        
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        CL_CHECK(err);
        
        queue = clCreateCommandQueue(context, device, 0, &err);
        CL_CHECK(err);
        
        program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
        CL_CHECK(err);
        
        err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t len;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            char* log = new char[len];
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log, nullptr);
            cerr << "Kernel build error:\n" << log << endl;
            delete[] log;
            exit(1);
        }
    }
    
    ~TOpenCLContext() {
        for (auto& kv : kernels) {
            clReleaseKernel(kv.second);
        }
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
    }
    
    cl_kernel GetKernel(const string& name) {
        if (kernels.find(name) == kernels.end()) {
            cl_int err;
            kernels[name] = clCreateKernel(program, name.c_str(), &err);
            CL_CHECK(err);
        }
        return kernels[name];
    }
};

// ========== Utility Functions ==========
double RandomWeight(double Scale) {
    return (rand() / (double)RAND_MAX - 0.5) * 2.0 * Scale;
}

void InitMatrixF(TFArray2D& M, int Rows, int Cols, double Scale) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = RandomWeight(Scale);
        }
    }
}

void ZeroMatrixF(TFArray2D& M, int Rows, int Cols) {
    M.resize(Rows);
    for (int i = 0; i < Rows; i++) {
        M[i].resize(Cols);
        for (int j = 0; j < Cols; j++) {
            M[i][j] = 0.0f;
        }
    }
}

void ZeroArrayF(FArray& A, int Size) {
    A.resize(Size);
    for (int i = 0; i < Size; i++) {
        A[i] = 0.0f;
    }
}

void Zero3DArrayF(TFArray3D& A, int D1, int D2, int D3) {
    A.resize(D1);
    for (int i = 0; i < D1; i++) {
        A[i].resize(D2);
        for (int j = 0; j < D2; j++) {
            A[i][j].resize(D3);
            for (int k = 0; k < D3; k++) {
                A[i][j][k] = 0.0f;
            }
        }
    }
}

void Zero4DArrayF(TFArray4D& A, int D1, int D2, int D3, int D4) {
    A.resize(D1);
    for (int i = 0; i < D1; i++) {
        A[i].resize(D2);
        for (int j = 0; j < D2; j++) {
            A[i][j].resize(D3);
            for (int k = 0; k < D3; k++) {
                A[i][j][k].resize(D4);
                for (int l = 0; l < D4; l++) {
                    A[i][j][k][l] = 0.0f;
                }
            }
        }
    }
}

// ========== Helper Functions ==========
string ActivationToStr(ActivationType act) {
    switch (act) {
        case atSigmoid: return "sigmoid";
        case atTanh: return "tanh";
        case atReLU: return "relu";
        case atLinear: return "linear";
        default: return "sigmoid";
    }
}

string LossToStr(LossType loss) {
    switch (loss) {
        case ltMSE: return "mse";
        case ltCrossEntropy: return "crossentropy";
        default: return "mse";
    }
}

ActivationType ParseActivation(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "tanh") return atTanh;
    else if (lower == "relu") return atReLU;
    else if (lower == "linear") return atLinear;
    else return atSigmoid;
}

LossType ParseLoss(const string& s) {
    string lower = s;
    transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "crossentropy") return ltCrossEntropy;
    else return ltMSE;
}

void ParseIntArrayHelper(const string& s, TIntArray& result) {
    result.clear();
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        if (!token.empty()) {
            result.push_back(stoi(token));
        }
    }
}

TIntArray ParseIntList(const string& str) {
    TIntArray result;
    istringstream iss(str);
    string token;
    while (getline(iss, token, ',')) {
        result.push_back(stoi(token));
    }
    return result;
}

string GetArgValue(int argc, char* argv[], const string& arg, const string& defaultValue = "") {
    for (int i = 1; i < argc - 1; i++) {
        if (string(argv[i]) == arg) {
            return string(argv[i + 1]);
        }
    }
    return defaultValue;
}

bool HasArg(int argc, char* argv[], const string& arg) {
    for (int i = 1; i < argc; i++) {
        if (string(argv[i]) == arg) {
            return true;
        }
    }
    return false;
}

void PrintUsage() {
    cout << "\n=================================================================\n";
    cout << "  Advanced CNN OpenCL - GPU-accelerated Convolutional Neural Network\n";
    cout << "  Copyright (c) 2025 Matthew Abbott - MIT License\n";
    cout << "=================================================================\n\n";
    cout << "USAGE:\n";
    cout << "  cnn_opencl <command> [options]\n\n";
    cout << "COMMANDS:\n";
    cout << "  help                 Show this help message\n";
    cout << "  info <model.json>    Display model architecture information\n";
    cout << "  create <config>      Create and save a new model\n";
    cout << "  train <config>       Train a model\n";
    cout << "  predict <config>     Run prediction on input data\n\n";
    cout << "EXAMPLES:\n";
    cout << "  cnn_opencl help\n";
    cout << "  cnn_opencl info model.json\n";
    cout << "  cnn_opencl create --model model.json --input 28,28,1 --conv 16,32 --kernels 3,3 --pool 2,2 --fc 128 --output 10\n";
    cout << "  cnn_opencl train --model model.json --data train.json --epochs 10 --batch 32 --lr 0.001\n";
    cout << "  cnn_opencl predict --model model.json --input image.json\n\n";
    cout << "OPTIONS:\n";
    cout << "  --model <file>       Model file path (JSON)\n";
    cout << "  --input <W,H,C>      Input dimensions (width,height,channels)\n";
    cout << "  --conv <F1,F2,...>   Convolutional filter counts per layer\n";
    cout << "  --kernels <K1,K2>    Kernel sizes per conv layer\n";
    cout << "  --pool <P1,P2>       Pooling sizes per layer\n";
    cout << "  --fc <N1,N2,...>     Fully connected layer sizes\n";
    cout << "  --output <N>         Output size (number of classes)\n";
    cout << "  --activation <type>  Activation function: sigmoid, tanh, relu, linear (default: relu)\n";
    cout << "  --loss <type>        Loss function: mse, crossentropy (default: crossentropy)\n";
    cout << "  --lr <rate>          Learning rate (default: 0.001)\n";
    cout << "  --clip <value>       Gradient clipping value (default: 5.0)\n";
    cout << "  --data <file>        Training/test data file (JSON)\n";
    cout << "  --epochs <N>         Number of training epochs\n";
    cout << "  --batch <N>          Batch size for training\n";
    cout << "  --validation <rate>  Validation split ratio (default: 0.2)\n\n";
}

void PrintModelInfo(const string& modelFile) {
    ifstream file(modelFile);
    if (!file.is_open()) {
        cerr << "Error: Cannot open model file: " << modelFile << endl;
        return;
    }
    
    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    file.close();
    
    auto findValue = [&content](const string& key) -> string {
        string searchKey = "\"" + key + "\": ";
        size_t pos = content.find(searchKey);
        if (pos == string::npos) return "";
        pos += searchKey.length();
        while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\n')) pos++;
        size_t endPos = pos;
        while (endPos < content.length() && content[endPos] != ',' &&
               content[endPos] != '\n' && content[endPos] != '}') endPos++;
        return content.substr(pos, endPos - pos);
    };
    
    cout << "\n=================================================================\n";
    cout << "  Model Information: " << modelFile << "\n";
    cout << "=================================================================\n\n";
    cout << "Architecture:\n";
    cout << "  Input:   " << findValue("input_width") << " x "
         << findValue("input_height") << " x "
         << findValue("input_channels") << "\n";
    cout << "  Output:  " << findValue("output_size") << " classes\n\n";
    cout << "Training Parameters:\n";
    cout << "  Learning Rate:   " << findValue("learning_rate") << "\n";
    cout << "  Gradient Clip:   " << findValue("gradient_clip") << "\n";
    cout << "  Activation:      " << findValue("activation") << "\n";
    cout << "  Output Act:      " << findValue("output_activation") << "\n";
    cout << "  Loss Type:       " << findValue("loss_type") << "\n\n";
}

// ========== Convolutional Filter ==========
class TConvFilter {
public:
    int FInputChannels, FOutputChannels, FKernelSize;
    TFArray4D Weights;
    TFArray4D dWeights;
    float Bias;
    float dBias;
    
    cl_mem gpu_weights, gpu_dweights, gpu_bias, gpu_dbias;
    
    TConvFilter(int InputChannels, int OutputChannels, int KernelSize, TOpenCLContext* oclCtx) {
        FInputChannels = InputChannels;
        FOutputChannels = OutputChannels;
        FKernelSize = KernelSize;
        double Scale = sqrt(2.0 / (InputChannels * KernelSize * KernelSize));
        
        Zero4DArrayF(Weights, 1, InputChannels, KernelSize, KernelSize);
        Zero4DArrayF(dWeights, 1, InputChannels, KernelSize, KernelSize);
        
        for (int j = 0; j < InputChannels; j++) {
            for (int k = 0; k < KernelSize; k++) {
                for (int l = 0; l < KernelSize; l++) {
                    Weights[0][j][k][l] = RandomWeight(Scale);
                }
            }
        }
        
        Bias = 0.0f;
        dBias = 0.0f;
        
        // Allocate GPU memory
        cl_int err;
        int wsize = InputChannels * KernelSize * KernelSize;
        FArray flatWeights(wsize);
        int idx = 0;
        for (int j = 0; j < InputChannels; j++) {
            for (int k = 0; k < KernelSize; k++) {
                for (int l = 0; l < KernelSize; l++) {
                    flatWeights[idx++] = Weights[0][j][k][l];
                }
            }
        }
        
        gpu_weights = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(float) * wsize, flatWeights.data(), &err);
        CL_CHECK(err);
        
        gpu_dweights = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                      sizeof(float) * wsize, nullptr, &err);
        CL_CHECK(err);
        
        gpu_bias = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float), &Bias, &err);
        CL_CHECK(err);
        
        gpu_dbias = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                   sizeof(float), nullptr, &err);
        CL_CHECK(err);
    }
    
    ~TConvFilter() {
        clReleaseMemObject(gpu_weights);
        clReleaseMemObject(gpu_dweights);
        clReleaseMemObject(gpu_bias);
        clReleaseMemObject(gpu_dbias);
    }
    
    void ResetGradients() {
        for (size_t j = 0; j < dWeights[0].size(); j++) {
            for (size_t k = 0; k < dWeights[0][j].size(); k++) {
                for (size_t l = 0; l < dWeights[0][j][k].size(); l++) {
                    dWeights[0][j][k][l] = 0.0f;
                }
            }
        }
        dBias = 0.0f;
    }
};

// ========== Convolutional Layer ==========
class TConvLayer {
private:
    int FInputChannels, FOutputChannels, FKernelSize, FStride, FPadding;
    ActivationType FActivation;
    
public:
    vector<TConvFilter*> Filters;
    TFArray3D InputCache;
    TFArray3D OutputCache;
    TFArray3D PreActivation;
    
    cl_mem gpu_input, gpu_output, gpu_preact;
    TOpenCLContext* oclCtx;
    
    TConvLayer(int InputChannels, int OutputChannels, int KernelSize, int Stride, int Padding,
               ActivationType Activation, TOpenCLContext* OclCtx) {
        FInputChannels = InputChannels;
        FOutputChannels = OutputChannels;
        FKernelSize = KernelSize;
        FStride = Stride;
        FPadding = Padding;
        FActivation = Activation;
        oclCtx = OclCtx;
        
        gpu_input = nullptr;
        gpu_output = nullptr;
        gpu_preact = nullptr;
        
        Filters.resize(OutputChannels);
        for (int i = 0; i < OutputChannels; i++) {
            Filters[i] = new TConvFilter(InputChannels, 1, KernelSize, OclCtx);
        }
    }
    
    ~TConvLayer() {
        for (auto& f : Filters) {
            delete f;
        }
        Filters.clear();
        if (gpu_input) clReleaseMemObject(gpu_input);
        if (gpu_output) clReleaseMemObject(gpu_output);
        if (gpu_preact) clReleaseMemObject(gpu_preact);
    }
    
    TFArray3D Pad3D(const TFArray3D& Input, int Padding) {
        TFArray3D Result;
        if (Padding == 0) {
            return Input;
        }
        
        int newH = Input[0].size() + 2 * Padding;
        int newW = Input[0][0].size() + 2 * Padding;
        Zero3DArrayF(Result, Input.size(), newH, newW);
        
        for (size_t c = 0; c < Input.size(); c++) {
            for (int h = 0; h < newH; h++) {
                for (int w = 0; w < newW; w++) {
                    int SrcH = h - Padding;
                    int SrcW = w - Padding;
                    if (SrcH >= 0 && SrcH < (int)Input[c].size() &&
                        SrcW >= 0 && SrcW < (int)Input[c][0].size()) {
                        Result[c][h][w] = Input[c][SrcH][SrcW];
                    } else {
                        Result[c][h][w] = 0;
                    }
                }
            }
        }
        return Result;
    }
    
    void Forward(const TFArray3D& Input, TFArray3D& Output) {
        InputCache = Input;
        
        TFArray3D Padded = FPadding > 0 ? Pad3D(Input, FPadding) : Input;
        
        int outH = (Padded[0].size() - FKernelSize) / FStride + 1;
        int outW = (Padded[0][0].size() - FKernelSize) / FStride + 1;
        
        Zero3DArrayF(Output, FOutputChannels, outH, outW);
        Zero3DArrayF(PreActivation, FOutputChannels, outH, outW);
        
        // Flatten padded input for GPU
        int padSize = Padded.size() * Padded[0].size() * Padded[0][0].size();
        FArray flatInput(padSize);
        int idx = 0;
        for (size_t c = 0; c < Padded.size(); c++) {
            for (size_t h = 0; h < Padded[c].size(); h++) {
                for (size_t w = 0; w < Padded[c][h].size(); w++) {
                    flatInput[idx++] = Padded[c][h][w];
                }
            }
        }
        
        cl_int err;
        if (gpu_input) clReleaseMemObject(gpu_input);
        gpu_input = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(float) * flatInput.size(), flatInput.data(), &err);
        CL_CHECK(err);
        
        int outSize = FOutputChannels * outH * outW;
        if (gpu_output) clReleaseMemObject(gpu_output);
        gpu_output = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                    sizeof(float) * outSize, nullptr, &err);
        CL_CHECK(err);
        
        if (gpu_preact) clReleaseMemObject(gpu_preact);
        gpu_preact = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                                    sizeof(float) * outSize, nullptr, &err);
        CL_CHECK(err);
        
        // Run convolution for each filter on GPU
        for (int f = 0; f < FOutputChannels; f++) {
            cl_kernel kern = oclCtx->GetKernel("k_conv_forward");
            
            int padH = Padded[0].size();
            int padW = Padded[0][0].size();
            
            err = clSetKernelArg(kern, 0, sizeof(cl_mem), &gpu_output);
            err |= clSetKernelArg(kern, 1, sizeof(cl_mem), &gpu_preact);
            err |= clSetKernelArg(kern, 2, sizeof(cl_mem), &gpu_input);
            err |= clSetKernelArg(kern, 3, sizeof(cl_mem), &Filters[f]->gpu_weights);
            err |= clSetKernelArg(kern, 4, sizeof(cl_mem), &Filters[f]->gpu_bias);
            err |= clSetKernelArg(kern, 5, sizeof(int), &FInputChannels);
            err |= clSetKernelArg(kern, 6, sizeof(int), &padH);
            err |= clSetKernelArg(kern, 7, sizeof(int), &padW);
            err |= clSetKernelArg(kern, 8, sizeof(int), &FKernelSize);
            err |= clSetKernelArg(kern, 9, sizeof(int), &FPadding);
            err |= clSetKernelArg(kern, 10, sizeof(int), &FStride);
            err |= clSetKernelArg(kern, 11, sizeof(int), &FOutputChannels);
            err |= clSetKernelArg(kern, 12, sizeof(int), &outH);
            err |= clSetKernelArg(kern, 13, sizeof(int), &outW);
            err |= clSetKernelArg(kern, 14, sizeof(int), &f);
            CL_CHECK(err);
            
            size_t globalSize[2] = {(size_t)outH, (size_t)outW};
            err = clEnqueueNDRangeKernel(oclCtx->queue, kern, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
            CL_CHECK(err);
        }
        
        clFinish(oclCtx->queue);
        
        // Read output back to host
        FArray flatOutput(outSize);
        err = clEnqueueReadBuffer(oclCtx->queue, gpu_output, CL_TRUE, 0, sizeof(float) * outSize, flatOutput.data(), 0, nullptr, nullptr);
        CL_CHECK(err);
        
        idx = 0;
        for (int f = 0; f < FOutputChannels; f++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    Output[f][h][w] = flatOutput[idx++];
                }
            }
        }
        
        OutputCache = Output;
    }
    
    void Backward(const TFArray3D& dOutput, TFArray3D& dInput) {
        // Simple implementation - gradient computation on host
        int outH = dOutput[0].size();
        int outW = dOutput[0][0].size();
        
        Zero3DArrayF(dInput, FInputChannels, InputCache[0].size(), InputCache[0][0].size());
        
        for (int f = 0; f < FOutputChannels; f++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    for (int c = 0; c < FInputChannels; c++) {
                        for (int kh = 0; kh < FKernelSize; kh++) {
                            for (int kw = 0; kw < FKernelSize; kw++) {
                                int ih = h * FStride + kh;
                                int iw = w * FStride + kw;
                                if (ih >= 0 && ih < (int)dInput[c].size() &&
                                    iw >= 0 && iw < (int)dInput[c][0].size()) {
                                    dInput[c][ih][iw] += dOutput[f][h][w] * 
                                                        Filters[f]->Weights[0][c][kh][kw];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    void ApplyGradients(double LR, double ClipVal) {
        for (int f = 0; f < FOutputChannels; f++) {
            Filters[f]->Bias -= LR * ClipVal;
            for (size_t i = 0; i < Filters[f]->Weights[0].size(); i++) {
                for (size_t j = 0; j < Filters[f]->Weights[0][i].size(); j++) {
                    for (size_t k = 0; k < Filters[f]->Weights[0][i][j].size(); k++) {
                        Filters[f]->Weights[0][i][j][k] -= LR * Filters[f]->dWeights[0][i][j][k];
                    }
                }
            }
        }
    }
    
    void ResetGradients() {
        for (int i = 0; i < FOutputChannels; i++) {
            Filters[i]->ResetGradients();
        }
    }
    
    int GetOutputChannels() const {
        return FOutputChannels;
    }
};

// ========== Pooling Layer ==========
class TPoolingLayer {
private:
    int FPoolSize, FStride;
    
public:
    TFArray3D InputCache;
    TFArray3D OutputCache;
    vector<vector<vector<pair<int,int>>>> MaxIndices;
    
    TPoolingLayer(int PoolSize, int Stride) {
        FPoolSize = PoolSize;
        FStride = Stride;
    }
    
    void Forward(const TFArray3D& Input, TFArray3D& Output) {
        InputCache = Input;
        int outH = (Input[0].size() - FPoolSize) / FStride + 1;
        int outW = (Input[0][0].size() - FPoolSize) / FStride + 1;
        
        Zero3DArrayF(Output, Input.size(), outH, outW);
        MaxIndices.assign(Input.size(), vector<vector<pair<int,int>>>(outH, vector<pair<int,int>>(outW)));
        
        for (size_t c = 0; c < Input.size(); c++) {
            for (int h = 0; h < outH; h++) {
                for (int w = 0; w < outW; w++) {
                    float MaxVal = -1e38f;
                    int MaxH = 0, MaxW = 0;
                    for (int kh = 0; kh < FPoolSize; kh++) {
                        for (int kw = 0; kw < FPoolSize; kw++) {
                            int ph = h * FStride + kh;
                            int pw = w * FStride + kw;
                            if (Input[c][ph][pw] > MaxVal) {
                                MaxVal = Input[c][ph][pw];
                                MaxH = kh;
                                MaxW = kw;
                            }
                        }
                    }
                    Output[c][h][w] = MaxVal;
                    MaxIndices[c][h][w] = make_pair(MaxW, MaxH);
                }
            }
        }
        OutputCache = Output;
    }
    
    void Backward(const TFArray3D& dOutput, TFArray3D& dInput) {
        Zero3DArrayF(dInput, InputCache.size(), InputCache[0].size(), InputCache[0][0].size());
        
        for (size_t c = 0; c < dOutput.size(); c++) {
            for (size_t h = 0; h < dOutput[c].size(); h++) {
                for (size_t w = 0; w < dOutput[c][h].size(); w++) {
                    int ph = h * FStride + MaxIndices[c][h][w].second;
                    int pw = w * FStride + MaxIndices[c][h][w].first;
                    dInput[c][ph][pw] = dOutput[c][h][w];
                }
            }
        }
    }
};

// ========== Fully Connected Layer ==========
class TFCLayer {
private:
    int FInputSize, FOutputSize;
    ActivationType FActivation;
    
public:
    TFArray2D W;
    FArray B;
    TFArray2D dW;
    FArray dB;
    FArray InputCache;
    FArray OutputCache;
    FArray PreActivation;
    
    cl_mem gpu_W, gpu_B, gpu_dW, gpu_dB, gpu_input, gpu_output, gpu_preact;
    TOpenCLContext* oclCtx;
    
    TFCLayer(int InputSize, int OutputSize, ActivationType Activation, TOpenCLContext* OclCtx) {
        FInputSize = InputSize;
        FOutputSize = OutputSize;
        FActivation = Activation;
        oclCtx = OclCtx;
        
        double Scale = sqrt(2.0 / InputSize);
        
        InitMatrixF(W, OutputSize, InputSize, Scale);
        ZeroArrayF(B, OutputSize);
        ZeroMatrixF(dW, OutputSize, InputSize);
        ZeroArrayF(dB, OutputSize);
        
        gpu_input = nullptr;
        gpu_output = nullptr;
        gpu_preact = nullptr;
        
        cl_int err;
        FArray flatW(W.size() * W[0].size());
        int idx = 0;
        for (size_t i = 0; i < W.size(); i++) {
            for (size_t j = 0; j < W[i].size(); j++) {
                flatW[idx++] = W[i][j];
            }
        }
        
        gpu_W = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * flatW.size(), flatW.data(), &err);
        CL_CHECK(err);
        
        gpu_B = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                              sizeof(float) * B.size(), B.data(), &err);
        CL_CHECK(err);
        
        gpu_dW = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                               sizeof(float) * flatW.size(), nullptr, &err);
        CL_CHECK(err);
        
        gpu_dB = clCreateBuffer(oclCtx->context, CL_MEM_READ_WRITE,
                               sizeof(float) * B.size(), nullptr, &err);
        CL_CHECK(err);
    }
    
    ~TFCLayer() {
        clReleaseMemObject(gpu_W);
        clReleaseMemObject(gpu_B);
        clReleaseMemObject(gpu_dW);
        clReleaseMemObject(gpu_dB);
        if (gpu_input) clReleaseMemObject(gpu_input);
        if (gpu_output) clReleaseMemObject(gpu_output);
        if (gpu_preact) clReleaseMemObject(gpu_preact);
    }
    
    void Forward(const FArray& Input, FArray& Output) {
        InputCache = Input;
        Output.resize(FOutputSize);
        PreActivation.resize(FOutputSize);
        
        for (int i = 0; i < FOutputSize; i++) {
            float Sum = B[i];
            for (int j = 0; j < FInputSize; j++) {
                Sum += W[i][j] * Input[j];
            }
            PreActivation[i] = Sum;
            Output[i] = (FActivation == atLinear) ? Sum : (FActivation == atReLU ? (Sum > 0 ? Sum : 0) : 
                       (FActivation == atSigmoid ? (1.0f / (1.0f + exp(-Sum))) : tanh(Sum)));
        }
        
        OutputCache = Output;
    }
    
    void Backward(const FArray& dOutput, FArray& dInput) {
        dInput.assign(FInputSize, 0.0f);
        
        for (int i = 0; i < FOutputSize; i++) {
            float deriv = (FActivation == atLinear) ? 1.0f : 
                         (FActivation == atReLU ? (OutputCache[i] > 0 ? 1.0f : 0.0f) :
                         (FActivation == atSigmoid ? (OutputCache[i] * (1.0f - OutputCache[i])) : 
                         (1.0f - OutputCache[i] * OutputCache[i])));
            
            float dRaw = dOutput[i] * deriv;
            dB[i] += dRaw;
            
            for (int j = 0; j < FInputSize; j++) {
                dW[i][j] += dRaw * InputCache[j];
                dInput[j] += dRaw * W[i][j];
            }
        }
    }
    
    void ApplyGradients(double LR, double ClipVal) {
        for (int i = 0; i < FOutputSize; i++) {
            B[i] -= LR * dB[i];
            dB[i] = 0;
            
            for (int j = 0; j < FInputSize; j++) {
                W[i][j] -= LR * dW[i][j];
                dW[i][j] = 0;
            }
        }
    }
    
    void ResetGradients() {
        for (int i = 0; i < FOutputSize; i++) {
            dB[i] = 0;
            for (int j = 0; j < FInputSize; j++) {
                dW[i][j] = 0;
            }
        }
    }
};

// ========== Main Advanced CNN (OpenCL version) ==========
class TAdvancedCNNOpenCL {
private:
    int FInputWidth, FInputHeight, FInputChannels, FOutputSize;
    ActivationType FActivation, FOutputActivation;
    LossType FLossType;
    double FLearningRate, FGradientClip;
    
    vector<TConvLayer*> FConvLayers;
    vector<TPoolingLayer*> FPoolLayers;
    vector<TFCLayer*> FFullyConnectedLayers;
    TFCLayer* FOutputLayer;
    int FFlattenedSize;
    
    TOpenCLContext* oclCtx;
    
public:
    TAdvancedCNNOpenCL(int InputWidth, int InputHeight, int InputChannels,
                       const TIntArray& ConvFilters,
                       const TIntArray& KernelSizes,
                       const TIntArray& PoolSizes,
                       const TIntArray& FCLayerSizes,
                       int OutputSize,
                       ActivationType Activation,
                       ActivationType OutputActivation,
                       LossType LossType,
                       double LearningRate,
                       double GradientClip) {
        
        oclCtx = new TOpenCLContext();
        
        FInputWidth = InputWidth;
        FInputHeight = InputHeight;
        FInputChannels = InputChannels;
        FOutputSize = OutputSize;
        FActivation = Activation;
        FOutputActivation = OutputActivation;
        FLossType = LossType;
        FLearningRate = LearningRate;
        FGradientClip = GradientClip;
        
        int CurrentChannels = InputChannels;
        int CurrentWidth = InputWidth;
        int CurrentHeight = InputHeight;
        
        FConvLayers.resize(ConvFilters.size());
        for (size_t i = 0; i < ConvFilters.size(); i++) {
            FConvLayers[i] = new TConvLayer(CurrentChannels, ConvFilters[i], 
                                            KernelSizes[i], 1, KernelSizes[i] / 2, 
                                            Activation, oclCtx);
            CurrentChannels = ConvFilters[i];
            
            if (i < PoolSizes.size()) {
                CurrentWidth /= PoolSizes[i];
                CurrentHeight /= PoolSizes[i];
            }
        }
        
        FPoolLayers.resize(PoolSizes.size());
        for (size_t i = 0; i < PoolSizes.size(); i++) {
            FPoolLayers[i] = new TPoolingLayer(PoolSizes[i], PoolSizes[i]);
        }
        
        FFlattenedSize = CurrentChannels * CurrentWidth * CurrentHeight;
        
        int NumInputs = FFlattenedSize;
        FFullyConnectedLayers.resize(FCLayerSizes.size());
        for (size_t i = 0; i < FCLayerSizes.size(); i++) {
            FFullyConnectedLayers[i] = new TFCLayer(NumInputs, FCLayerSizes[i], Activation, oclCtx);
            NumInputs = FCLayerSizes[i];
        }
        
        FOutputLayer = new TFCLayer(NumInputs, OutputSize, OutputActivation, oclCtx);
    }
    
    ~TAdvancedCNNOpenCL() {
        for (auto& layer : FConvLayers) {
            delete layer;
        }
        for (auto& layer : FPoolLayers) {
            delete layer;
        }
        for (auto& layer : FFullyConnectedLayers) {
            delete layer;
        }
        delete FOutputLayer;
        FConvLayers.clear();
        FPoolLayers.clear();
        FFullyConnectedLayers.clear();
        
        delete oclCtx;
    }
    
    FArray Flatten(const TFArray3D& Input) {
        FArray Result;
        Result.reserve(Input.size() * Input[0].size() * Input[0][0].size());
        for (size_t c = 0; c < Input.size(); c++) {
            for (size_t h = 0; h < Input[c].size(); h++) {
                for (size_t w = 0; w < Input[c][h].size(); w++) {
                    Result.push_back(Input[c][h][w]);
                }
            }
        }
        return Result;
    }
    
    TFArray3D Unflatten(const FArray& Input, int Channels, int Height, int Width) {
        TFArray3D Result;
        Zero3DArrayF(Result, Channels, Height, Width);
        size_t idx = 0;
        for (int c = 0; c < Channels; c++) {
            for (int h = 0; h < Height; h++) {
                for (int w = 0; w < Width; w++) {
                    Result[c][h][w] = Input[idx++];
                }
            }
        }
        return Result;
    }
    
    FArray ForwardPass(const TFArray3D& Input) {
        TFArray3D CurrentOutput = Input;
        
        for (size_t i = 0; i < FConvLayers.size(); i++) {
            FConvLayers[i]->Forward(CurrentOutput, CurrentOutput);
            
            if (i < FPoolLayers.size()) {
                FPoolLayers[i]->Forward(CurrentOutput, CurrentOutput);
            }
        }
        
        FArray FlatInput = Flatten(CurrentOutput);
        FArray LayerInput = FlatInput;
        
        for (auto& layer : FFullyConnectedLayers) {
            layer->Forward(LayerInput, LayerInput);
        }
        
        FArray Logits(FOutputSize);
        FOutputLayer->Forward(LayerInput, Logits);
        
        return Logits;
    }
    
    double GetLearningRate() const { return FLearningRate; }
    void SetLearningRate(double LR) { FLearningRate = LR; }
    double GetGradientClip() const { return FGradientClip; }
    void SetGradientClip(double GC) { FGradientClip = GC; }
    
    // JSON Serialization Helper Methods
    string Array1DToJSON(const FArray& Arr) {
        ostringstream oss;
        oss << setprecision(17);
        oss << "[";
        for (size_t i = 0; i < Arr.size(); i++) {
            oss << Arr[i];
            if (i < Arr.size() - 1) oss << ",";
        }
        oss << "]";
        return oss.str();
    }
    
    string Array2DToJSON(const TFArray2D& Arr) {
        ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < Arr.size(); i++) {
            oss << Array1DToJSON(Arr[i]);
            if (i < Arr.size() - 1) oss << ",";
        }
        oss << "]";
        return oss.str();
    }
    
    string Array3DToJSON(const TFArray3D& Arr) {
        ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < Arr.size(); i++) {
            oss << Array2DToJSON(Arr[i]);
            if (i < Arr.size() - 1) oss << ",";
        }
        oss << "]";
        return oss.str();
    }
    
    string Array4DToJSON(const TFArray4D& Arr) {
        ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < Arr.size(); i++) {
            oss << Array3DToJSON(Arr[i]);
            if (i < Arr.size() - 1) oss << ",";
        }
        oss << "]";
        return oss.str();
    }
    
    void SaveModelToJSON(const string& Filename) {
        ofstream file(Filename);
        if (!file.is_open()) {
            cerr << "Error: Cannot open file for writing: " << Filename << endl;
            return;
        }
        
        file << setprecision(17);
        file << "{\n";
        
        file << "  \"input_width\": " << FInputWidth << ",\n";
        file << "  \"input_height\": " << FInputHeight << ",\n";
        file << "  \"input_channels\": " << FInputChannels << ",\n";
        file << "  \"output_size\": " << FOutputSize << ",\n";
        file << "  \"learning_rate\": " << FLearningRate << ",\n";
        file << "  \"gradient_clip\": " << FGradientClip << ",\n";
        file << "  \"activation\": " << static_cast<int>(FActivation) << ",\n";
        file << "  \"output_activation\": " << static_cast<int>(FOutputActivation) << ",\n";
        file << "  \"loss_type\": " << static_cast<int>(FLossType) << ",\n";
        
        file << "  \"conv_layers\": [\n";
        for (size_t i = 0; i < FConvLayers.size(); i++) {
            file << "    {\n";
            file << "      \"filters\": [\n";
            for (size_t f = 0; f < FConvLayers[i]->Filters.size(); f++) {
                file << "        {\n";
                file << "          \"weights\": " << Array4DToJSON(FConvLayers[i]->Filters[f]->Weights) << ",\n";
                file << "          \"bias\": " << FConvLayers[i]->Filters[f]->Bias << "\n";
                file << "        }";
                if (f < FConvLayers[i]->Filters.size() - 1) file << ",";
                file << "\n";
            }
            file << "      ]\n";
            file << "    }";
            if (i < FConvLayers.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ],\n";
        
        file << "  \"fc_layers\": [\n";
        for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
            file << "    {\n";
            file << "      \"weights\": " << Array2DToJSON(FFullyConnectedLayers[i]->W) << ",\n";
            file << "      \"bias\": " << Array1DToJSON(FFullyConnectedLayers[i]->B) << "\n";
            file << "    }";
            if (i < FFullyConnectedLayers.size() - 1) file << ",";
            file << "\n";
        }
        file << "  ],\n";
        
        file << "  \"output_layer\": {\n";
        file << "    \"weights\": " << Array2DToJSON(FOutputLayer->W) << ",\n";
        file << "    \"bias\": " << Array1DToJSON(FOutputLayer->B) << "\n";
        file << "  }\n";
        
        file << "}\n";
        file.close();
    }
    
    void LoadModelFromJSON(const string& Filename) {
        ifstream file(Filename);
        if (!file.is_open()) {
            cerr << "Error: Cannot open file for reading: " << Filename << endl;
            return;
        }
        
        string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
        file.close();
        
        auto findValue = [&content](const string& key) -> string {
            string searchKey = "\"" + key + "\": ";
            size_t pos = content.find(searchKey);
            if (pos == string::npos) return "";
            pos += searchKey.length();
            while (pos < content.length() && (content[pos] == ' ' || content[pos] == '\n')) pos++;
            size_t endPos = pos;
            if (content[pos] == '"') {
                endPos = content.find('"', pos + 1);
                return content.substr(pos + 1, endPos - pos - 1);
            } else if (content[pos] == '[' || content[pos] == '{') {
                int depth = 0;
                char startChar = content[pos];
                char endChar = (startChar == '[') ? ']' : '}';
                do {
                    if (content[endPos] == startChar) depth++;
                    if (content[endPos] == endChar) depth--;
                    endPos++;
                } while (depth > 0 && endPos < content.length());
                return content.substr(pos, endPos - pos);
            } else {
                while (endPos < content.length() && content[endPos] != ',' &&
                       content[endPos] != '\n' && content[endPos] != '}') endPos++;
                return content.substr(pos, endPos - pos);
            }
        };
        
        auto parseArray1D = [](const string& str) -> FArray {
            FArray result;
            if (str.empty() || str[0] != '[') return result;
            string nums = str.substr(1, str.length() - 2);
            istringstream iss(nums);
            string token;
            while (getline(iss, token, ',')) {
                result.push_back(stof(token));
            }
            return result;
        };
        
        auto parseArray2D = [&parseArray1D](const string& str) -> TFArray2D {
            TFArray2D result;
            if (str.empty() || str[0] != '[') return result;
            size_t pos = 1;
            int depth = 0;
            size_t start = pos;
            while (pos < str.length() - 1) {
                if (str[pos] == '[') {
                    if (depth == 0) start = pos;
                    depth++;
                } else if (str[pos] == ']') {
                    depth--;
                    if (depth == 0) {
                        result.push_back(parseArray1D(str.substr(start, pos - start + 1)));
                    }
                }
                pos++;
            }
            return result;
        };
        
        auto parseArray4D = [&parseArray2D](const string& str) -> TFArray4D {
            TFArray4D result;
            if (str.empty() || str[0] != '[') return result;
            size_t pos = 1;
            int depth = 0;
            size_t start = pos;
            while (pos < str.length() - 1) {
                if (str[pos] == '[') {
                    if (depth == 0) start = pos;
                    depth++;
                } else if (str[pos] == ']') {
                    depth--;
                    if (depth == 0) {
                        TFArray3D arr3d;
                        string str3d = str.substr(start, pos - start + 1);
                        size_t pos3 = 1;
                        int depth3 = 0;
                        size_t start3 = pos3;
                        while (pos3 < str3d.length() - 1) {
                            if (str3d[pos3] == '[') {
                                if (depth3 == 0) start3 = pos3;
                                depth3++;
                            } else if (str3d[pos3] == ']') {
                                depth3--;
                                if (depth3 == 0) {
                                    arr3d.push_back(parseArray2D(str3d.substr(start3, pos3 - start3 + 1)));
                                }
                            }
                            pos3++;
                        }
                        result.push_back(arr3d);
                    }
                }
                pos++;
            }
            return result;
        };
        
        FLearningRate = stod(findValue("learning_rate"));
        FGradientClip = stod(findValue("gradient_clip"));
        
        string convLayersStr = findValue("conv_layers");
        size_t convPos = 0;
        int convDepth = 0;
        size_t convLayerIdx = 0;
        
        while (convPos < convLayersStr.length() && convLayerIdx < FConvLayers.size()) {
            if (convLayersStr[convPos] == '{' && convDepth == 1) {
                size_t filtersStart = convLayersStr.find("\"filters\":", convPos);
                if (filtersStart != string::npos) {
                    filtersStart = convLayersStr.find('[', filtersStart);
                    int filterDepth = 0;
                    size_t filtersEnd = filtersStart;
                    do {
                        if (convLayersStr[filtersEnd] == '[') filterDepth++;
                        if (convLayersStr[filtersEnd] == ']') filterDepth--;
                        filtersEnd++;
                    } while (filterDepth > 0);
                    
                    string filtersStr = convLayersStr.substr(filtersStart, filtersEnd - filtersStart);
                    size_t filterPos = 1;
                    size_t filterIdx = 0;
                    int fDepth = 0;
                    
                    while (filterPos < filtersStr.length() && filterIdx < FConvLayers[convLayerIdx]->Filters.size()) {
                        if (filtersStr[filterPos] == '{') {
                            if (fDepth == 0) {
                                size_t weightsStart = filtersStr.find("\"weights\":", filterPos);
                                size_t weightsValStart = filtersStr.find('[', weightsStart);
                                int wDepth = 0;
                                size_t weightsEnd = weightsValStart;
                                do {
                                    if (filtersStr[weightsEnd] == '[') wDepth++;
                                    if (filtersStr[weightsEnd] == ']') wDepth--;
                                    weightsEnd++;
                                } while (wDepth > 0);
                                
                                string weightsStr = filtersStr.substr(weightsValStart, weightsEnd - weightsValStart);
                                FConvLayers[convLayerIdx]->Filters[filterIdx]->Weights = parseArray4D(weightsStr);
                                
                                size_t biasStart = filtersStr.find("\"bias\":", filterPos);
                                size_t biasValStart = biasStart + 7;
                                while (filtersStr[biasValStart] == ' ') biasValStart++;
                                size_t biasEnd = filtersStr.find_first_of(",\n}", biasValStart);
                                FConvLayers[convLayerIdx]->Filters[filterIdx]->Bias =
                                    stof(filtersStr.substr(biasValStart, biasEnd - biasValStart));
                                
                                filterIdx++;
                            }
                            fDepth++;
                        } else if (filtersStr[filterPos] == '}') {
                            fDepth--;
                        }
                        filterPos++;
                    }
                    
                    convPos = filtersEnd + convPos;
                    convLayerIdx++;
                }
            }
            if (convLayersStr[convPos] == '{') convDepth++;
            if (convLayersStr[convPos] == '}') convDepth--;
            convPos++;
        }
        
        string fcLayersStr = findValue("fc_layers");
        size_t fcPos = 1;
        size_t fcLayerIdx = 0;
        int fcDepth = 0;
        
        while (fcPos < fcLayersStr.length() && fcLayerIdx < FFullyConnectedLayers.size()) {
            if (fcLayersStr[fcPos] == '{') {
                if (fcDepth == 0) {
                    size_t weightsStart = fcLayersStr.find("\"weights\":", fcPos);
                    size_t weightsValStart = fcLayersStr.find('[', weightsStart);
                    int wDepth = 0;
                    size_t weightsEnd = weightsValStart;
                    do {
                        if (fcLayersStr[weightsEnd] == '[') wDepth++;
                        if (fcLayersStr[weightsEnd] == ']') wDepth--;
                        weightsEnd++;
                    } while (wDepth > 0);
                    
                    string weightsStr = fcLayersStr.substr(weightsValStart, weightsEnd - weightsValStart);
                    FFullyConnectedLayers[fcLayerIdx]->W = parseArray2D(weightsStr);
                    
                    size_t biasStart = fcLayersStr.find("\"bias\":", fcPos);
                    size_t biasValStart = fcLayersStr.find('[', biasStart);
                    int bDepth = 0;
                    size_t biasEnd = biasValStart;
                    do {
                        if (fcLayersStr[biasEnd] == '[') bDepth++;
                        if (fcLayersStr[biasEnd] == ']') bDepth--;
                        biasEnd++;
                    } while (bDepth > 0);
                    
                    string biasStr = fcLayersStr.substr(biasValStart, biasEnd - biasValStart);
                    FFullyConnectedLayers[fcLayerIdx]->B = parseArray1D(biasStr);
                    
                    fcLayerIdx++;
                }
                fcDepth++;
            } else if (fcLayersStr[fcPos] == '}') {
                fcDepth--;
            }
            fcPos++;
        }
        
        string outputLayerStr = findValue("output_layer");
        size_t weightsStart = outputLayerStr.find("\"weights\":");
        size_t weightsValStart = outputLayerStr.find('[', weightsStart);
        int wDepth = 0;
        size_t weightsEnd = weightsValStart;
        do {
            if (outputLayerStr[weightsEnd] == '[') wDepth++;
            if (outputLayerStr[weightsEnd] == ']') wDepth--;
            weightsEnd++;
        } while (wDepth > 0);
        
        string weightsStr = outputLayerStr.substr(weightsValStart, weightsEnd - weightsValStart);
        FOutputLayer->W = parseArray2D(weightsStr);
        
        size_t biasStart = outputLayerStr.find("\"bias\":");
        size_t biasValStart = outputLayerStr.find('[', biasStart);
        int bDepth = 0;
        size_t biasEnd = biasValStart;
        do {
            if (outputLayerStr[biasEnd] == '[') bDepth++;
            if (outputLayerStr[biasEnd] == ']') bDepth--;
            biasEnd++;
        } while (bDepth > 0);
        
        string biasStr = outputLayerStr.substr(biasValStart, biasEnd - biasValStart);
        FOutputLayer->B = parseArray1D(biasStr);
    }
    
    // Training/Inference Methods
    double BackwardPass(const FArray& Target) {
        FArray OutputGrad(FOutputSize);
        for (int i = 0; i < FOutputSize; i++) {
            OutputGrad[i] = FOutputLayer->OutputCache[i] - Target[i];
        }
        
        FArray FCGrad;
        FOutputLayer->Backward(OutputGrad, FCGrad);
        
        for (int i = FFullyConnectedLayers.size() - 1; i >= 0; i--) {
            FFullyConnectedLayers[i]->Backward(FCGrad, FCGrad);
        }
        
        int LastConvIdx = FConvLayers.size() - 1;
        TFArray3D CurrentGrad = Unflatten(FCGrad,
                                          FConvLayers[LastConvIdx]->OutputCache.size(),
                                          FConvLayers[LastConvIdx]->OutputCache[0].size(),
                                          FConvLayers[LastConvIdx]->OutputCache[0][0].size());
        
        for (int i = FConvLayers.size() - 1; i >= 0; i--) {
            if (i < (int)FPoolLayers.size()) {
                FPoolLayers[i]->Backward(CurrentGrad, CurrentGrad);
            }
            FConvLayers[i]->Backward(CurrentGrad, CurrentGrad);
        }
        
        return ComputeLossSingle(FOutputLayer->OutputCache, Target);
    }
    
    double ComputeLossSingle(const FArray& Pred, const FArray& Target) {
        double Result = 0;
        if (FLossType == ltMSE) {
            for (size_t i = 0; i < Pred.size(); i++) {
                Result += (Pred[i] - Target[i]) * (Pred[i] - Target[i]);
            }
        } else {
            for (size_t i = 0; i < Pred.size(); i++) {
                float P = max(1e-15f, min(1.0f - 1e-15f, Pred[i]));
                Result += -(Target[i] * log(P) + (1.0f - Target[i]) * log(1.0f - P));
            }
        }
        return Result / Pred.size();
    }
    
    double TrainSample(const TFArray3D& Input, const FArray& Target) {
        ResetGradients();
        ForwardPass(Input);
        double Loss = BackwardPass(Target);
        ApplyGradients();
        return Loss;
    }
    
    double TrainBatch(const TFArray4D& BatchInputs, const vector<FArray>& BatchTargets) {
        ResetGradients();
        double BatchLoss = 0.0;
        
        for (size_t b = 0; b < BatchInputs.size(); b++) {
            ForwardPass(BatchInputs[b]);
            BatchLoss += BackwardPass(BatchTargets[b]);
        }
        
        ApplyGradients();
        return BatchLoss / BatchInputs.size();
    }
    
    FArray Predict(const TFArray3D& Input) {
        return ForwardPass(Input);
    }
    
    double ComputeLoss(const TFArray4D& Inputs, const vector<FArray>& Targets) {
        double Result = 0.0;
        for (size_t i = 0; i < Inputs.size(); i++) {
            FArray Output = ForwardPass(Inputs[i]);
            Result += ComputeLossSingle(Output, Targets[i]);
        }
        return Result / Inputs.size();
    }
    
    void ResetGradients() {
        for (size_t i = 0; i < FConvLayers.size(); i++) {
            FConvLayers[i]->ResetGradients();
        }
        for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
            FFullyConnectedLayers[i]->ResetGradients();
        }
        FOutputLayer->ResetGradients();
    }
    
    void ApplyGradients() {
        for (size_t i = 0; i < FConvLayers.size(); i++) {
            FConvLayers[i]->ApplyGradients(FLearningRate, FGradientClip);
        }
        for (size_t i = 0; i < FFullyConnectedLayers.size(); i++) {
            FFullyConnectedLayers[i]->ApplyGradients(FLearningRate, FGradientClip);
        }
        FOutputLayer->ApplyGradients(FLearningRate, FGradientClip);
    }
};

// ========== CLI Command Handlers ==========
void HandleCreate(int argc, char* argv[]) {
    string modelFile = GetArgValue(argc, argv, "--model", "model.json");
    string inputDims = GetArgValue(argc, argv, "--input", "28,28,1");
    string convFilters = GetArgValue(argc, argv, "--conv", "16,32");
    string kernels = GetArgValue(argc, argv, "--kernels", "3,3");
    string pools = GetArgValue(argc, argv, "--pool", "2,2");
    string fcLayers = GetArgValue(argc, argv, "--fc", "128");
    int outputSize = stoi(GetArgValue(argc, argv, "--output", "10"));
    
    string activationStr = GetArgValue(argc, argv, "--activation", "relu");
    string lossStr = GetArgValue(argc, argv, "--loss", "crossentropy");
    double lr = stod(GetArgValue(argc, argv, "--lr", "0.001"));
    double clip = stod(GetArgValue(argc, argv, "--clip", "5.0"));
    
    TIntArray inputDimVec = ParseIntList(inputDims);
    if (inputDimVec.size() != 3) {
        cerr << "Error: Input dimensions must be W,H,C" << endl;
        return;
    }
    
    TIntArray convFilterVec = ParseIntList(convFilters);
    TIntArray kernelVec = ParseIntList(kernels);
    TIntArray poolVec = ParseIntList(pools);
    TIntArray fcVec = ParseIntList(fcLayers);
    
    ActivationType activation = ParseActivation(activationStr);
    LossType lossType = ParseLoss(lossStr);
    
    cout << "Creating model...\n";
    cout << "  Input: " << inputDimVec[0] << "x" << inputDimVec[1] << "x" << inputDimVec[2] << "\n";
    cout << "  Conv layers: " << convFilters << "\n";
    cout << "  FC layers: " << fcLayers << "\n";
    cout << "  Output: " << outputSize << "\n";
    
    TAdvancedCNNOpenCL* cnn = new TAdvancedCNNOpenCL(
        inputDimVec[0], inputDimVec[1], inputDimVec[2],
        convFilterVec, kernelVec, poolVec, fcVec,
        outputSize, activation, atLinear,
        lossType, lr, clip
    );
    
    cnn->SaveModelToJSON(modelFile);
    cout << "Model saved to: " << modelFile << "\n";
    
    delete cnn;
}

void HandleTrain(int argc, char* argv[]) {
    string modelFile = GetArgValue(argc, argv, "--model", "model.json");
    string dataFile = GetArgValue(argc, argv, "--data", "");
    int epochs = stoi(GetArgValue(argc, argv, "--epochs", "10"));
    int batchSize = stoi(GetArgValue(argc, argv, "--batch", "32"));
    double lr = stod(GetArgValue(argc, argv, "--lr", "0.001"));
    
    if (dataFile.empty()) {
        cerr << "Error: --data argument required for training" << endl;
        return;
    }
    
    cout << "Training not fully implemented in this CLI demo.\n";
    cout << "To implement training:\n";
    cout << "  1. Load data from " << dataFile << "\n";
    cout << "  2. Load model from " << modelFile << "\n";
    cout << "  3. Run training loop with TrainBatch()\n";
    cout << "  4. Save updated model\n";
    cout << "\nSee the library API for complete training implementation.\n";
}

void HandlePredict(int argc, char* argv[]) {
    string modelFile = GetArgValue(argc, argv, "--model", "model.json");
    string inputFile = GetArgValue(argc, argv, "--input", "");
    
    if (inputFile.empty()) {
        cerr << "Error: --input argument required for prediction" << endl;
        return;
    }
    
    cout << "Prediction not fully implemented in this CLI demo.\n";
    cout << "To implement prediction:\n";
    cout << "  1. Load model from " << modelFile << "\n";
    cout << "  2. Load input data from " << inputFile << "\n";
    cout << "  3. Run Predict() on input\n";
    cout << "  4. Output predictions\n";
    cout << "\nSee the library API for complete prediction implementation.\n";
}

Command ParseCommand(const string& cmd) {
    if (cmd == "help") return cmdHelp;
    if (cmd == "info") return cmdInfo;
    if (cmd == "create") return cmdCreate;
    if (cmd == "train") return cmdTrain;
    if (cmd == "predict") return cmdPredict;
    return cmdNone;
}

// ========== Main Program ==========
int main(int argc, char* argv[]) {
    srand(time(0));
    
    if (argc < 2) {
        PrintUsage();
        return 0;
    }
    
    Command cmd = ParseCommand(argv[1]);
    
    try {
        switch (cmd) {
            case cmdHelp:
                PrintUsage();
                break;
            
            case cmdInfo:
                if (argc < 3) {
                    cerr << "Error: info command requires model file\n";
                    cerr << "Usage: cnn_opencl info <model.json>\n";
                    return 1;
                }
                PrintModelInfo(argv[2]);
                break;
            
            case cmdCreate:
                HandleCreate(argc, argv);
                break;
            
            case cmdTrain:
                HandleTrain(argc, argv);
                break;
            
            case cmdPredict:
                HandlePredict(argc, argv);
                break;
            
            case cmdNone:
            default:
                cerr << "Error: Unknown command '" << argv[1] << "'\n";
                cerr << "Run 'cnn_opencl help' for usage information\n";
                return 1;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}
