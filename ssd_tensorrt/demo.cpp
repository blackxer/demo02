///*
// * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
// *
// * NOTICE TO LICENSEE:
// *
// * This source code and/or documentation ("Licensed Deliverables") are
// * subject to NVIDIA intellectual property rights under U.S. and
// * international Copyright laws.
// *
// * These Licensed Deliverables contained herein is PROPRIETARY and
// * CONFIDENTIAL to NVIDIA and is being provided under the terms and
// * conditions of a form of NVIDIA software license agreement by and
// * between NVIDIA and Licensee ("License Agreement") or electronically
// * accepted by Licensee.  Notwithstanding any terms or conditions to
// * the contrary in the License Agreement, reproduction or disclosure
// * of the Licensed Deliverables to any third party without the express
// * written consent of NVIDIA is prohibited.
// *
// * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
// * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
// * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
// * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
// * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
// * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
// * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
// * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
// * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
// * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
// * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
// * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
// * OF THESE LICENSED DELIVERABLES.
// *
// * U.S. Government End Users.  These Licensed Deliverables are a
// * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
// * 1995), consisting of "commercial computer software" and "commercial
// * computer software documentation" as such terms are used in 48
// * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
// * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
// * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
// * U.S. Government End Users acquire the Licensed Deliverables with
// * only those rights set forth herein.
// *
// * Any use of the Licensed Deliverables in individual and commercial
// * software must include, in the user documentation and internal
// * comments to the code, the above Disclaimer and U.S. Government End
// * Users Notice.
// */

//#include "NvCaffeParser.h"
//#include "NvInfer.h"
//#include "logger.h"
//#include "common.h"
//#include "argsParser.h"
//#include "cuda_runtime_api.h"
//#include <algorithm>
//#include <cassert>
//#include <cmath>
//#include <cstring>
//#include <ctime>
//#include <fstream>
//#include <iostream>
//#include <map>
//#include <sstream>
//#ifdef _MSC_VER
//#include <direct.h>
//#else
//#include <sys/stat.h>
//#endif
//#include <vector>

//// stuff we know about the network and the input/output blobs
//static const int INPUT_H = 28;
//static const int INPUT_W = 28;
//static const int OUTPUT_SIZE = 10;
//samplesCommon::Args gArgs;

//const char* INPUT_BLOB_NAME = "data";
//const char* OUTPUT_BLOB_NAME = "prob";

//using namespace nvinfer1;
//using namespace nvcaffeparser1;

//const std::string gSampleName = "TensorRT.sample_mnist_api";

//// Load weights from files shared with TensorRT samples.
//// TensorRT weight files have a simple space delimited format:
//// [type] [size] <data x size in hex>
//std::map<std::string, Weights> loadWeights(const std::string file)
//{
//    gLogInfo << "Loading weights: " << file << std::endl;
//    std::map<std::string, Weights> weightMap;

//    // Open weights file
//    std::ifstream input(file);
//    assert(input.is_open() && "Unable to load weight file.");

//    // Read number of weight blobs
//    int32_t count;
//    input >> count;
//    assert(count > 0 && "Invalid weight map file.");

//    while (count--)
//    {
//        Weights wt{DataType::kFLOAT, nullptr, 0};
//        uint32_t type, size;

//        // Read name and type of blob
//        std::string name;
//        input >> name >> std::dec >> type >> size;
//        wt.type = static_cast<DataType>(type);

//        // Load blob
//        if (wt.type == DataType::kFLOAT)
//        {
//            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
//            for (uint32_t x = 0, y = size; x < y; ++x)
//            {
//                input >> std::hex >> val[x];
//            }
//            wt.values = val;
//        }
//        else if (wt.type == DataType::kHALF)
//        {
//            uint16_t* val = reinterpret_cast<uint16_t*>(malloc(sizeof(val) * size));
//            for (uint32_t x = 0, y = size; x < y; ++x)
//            {
//                input >> std::hex >> val[x];
//            }
//            wt.values = val;
//        }

//        wt.count = size;
//        weightMap[name] = wt;
//    }

//    return weightMap;
//}

//// simple PGM (portable greyscale map) reader
//void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
//{
//    readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
//}

//// Creat the engine using only the API and not any parser.
//ICudaEngine* createMNISTEngine(unsigned int maxBatchSize, IBuilder* builder, DataType dt)
//{
//    INetworkDefinition* network = builder->createNetwork();

//    // Create input tensor of shape { 1, 1, 28, 28 } with name INPUT_BLOB_NAME
//    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{1, INPUT_H, INPUT_W});
//    assert(data);

//    // Create scale layer with default power/shift and specified scale parameter.
//    const float scaleParam = 0.0125f;
//    const Weights power{DataType::kFLOAT, nullptr, 0};
//    const Weights shift{DataType::kFLOAT, nullptr, 0};
//    const Weights scale{DataType::kFLOAT, &scaleParam, 1};
//    IScaleLayer* scale_1 = network->addScale(*data, ScaleMode::kUNIFORM, shift, scale, power);
//    assert(scale_1);

//    // Add convolution layer with 20 outputs and a 5x5 filter.
//    std::map<std::string, Weights> weightMap = loadWeights(locateFile("mnistapi.wts", gArgs.dataDirs));
//    IConvolutionLayer* conv1 = network->addConvolution(*scale_1->getOutput(0), 20, DimsHW{5, 5}, weightMap["conv1filter"], weightMap["conv1bias"]);
//    assert(conv1);
//    conv1->setStride(DimsHW{1, 1});

//    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
//    IPoolingLayer* pool1 = network->addPooling(*conv1->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
//    assert(pool1);
//    pool1->setStride(DimsHW{2, 2});

//    // Add second convolution layer with 50 outputs and a 5x5 filter.
//    IConvolutionLayer* conv2 = network->addConvolution(*pool1->getOutput(0), 50, DimsHW{5, 5}, weightMap["conv2filter"], weightMap["conv2bias"]);
//    assert(conv2);
//    conv2->setStride(DimsHW{1, 1});

//    // Add second max pooling layer with stride of 2x2 and kernel size of 2x2
//    IPoolingLayer* pool2 = network->addPooling(*conv2->getOutput(0), PoolingType::kMAX, DimsHW{2, 2});
//    assert(pool2);
//    pool2->setStride(DimsHW{2, 2});

//    // Add fully connected layer with 500 outputs.
//    IFullyConnectedLayer* ip1 = network->addFullyConnected(*pool2->getOutput(0), 500, weightMap["ip1filter"], weightMap["ip1bias"]);
//    assert(ip1);

//    // Add activation layer using the ReLU algorithm.
//    IActivationLayer* relu1 = network->addActivation(*ip1->getOutput(0), ActivationType::kRELU);
//    assert(relu1);

//    // Add second fully connected layer with 10 outputs.
//    IFullyConnectedLayer* ip2 = network->addFullyConnected(*relu1->getOutput(0), OUTPUT_SIZE, weightMap["ip2filter"], weightMap["ip2bias"]);
//    assert(ip2);

//    // Add softmax layer to determine the probability.
//    ISoftMaxLayer* prob = network->addSoftMax(*ip2->getOutput(0));
//    assert(prob);
//    prob->getOutput(0)->setName(OUTPUT_BLOB_NAME);
//    network->markOutput(*prob->getOutput(0));

//    // Build engine
//    builder->setMaxBatchSize(maxBatchSize);
//    builder->setMaxWorkspaceSize(1 << 20);
//    builder->setFp16Mode(gArgs.runInFp16);
//    if (gArgs.runInInt8)
//    {
//        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
//    }
    
//    samplesCommon::enableDLA(builder, gArgs.useDLACore);
    
//    ICudaEngine* engine = builder->buildCudaEngine(*network);

//    // Don't need the network any more
//    network->destroy();

//    // Release host memory
//    for (auto& mem : weightMap)
//    {
//        free((void*) (mem.second.values));
//    }

//    return engine;
//}

//void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
//{
//    // Create builder
//    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
//    assert(builder != nullptr);

//    // Create model to populate the network, then set the outputs and create an engine
//    ICudaEngine* engine = createMNISTEngine(maxBatchSize, builder, DataType::kFLOAT);
//    assert(engine != nullptr);

//    // Serialize the engine
//    (*modelStream) = engine->serialize();

//    // Close everything down
//    engine->destroy();
//    builder->destroy();
//}

//void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
//{
//    const ICudaEngine& engine = context.getEngine();

//    // Pointers to input and output device buffers to pass to engine.
//    // Engine requires exactly IEngine::getNbBindings() number of buffers.
//    assert(engine.getNbBindings() == 2);
//    void* buffers[2];

//    // In order to bind the buffers, we need to know the names of the input and output tensors.
//    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
//    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
//    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

//    // Create GPU buffers on device
//    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
//    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

//    // Create stream
//    cudaStream_t stream;
//    CHECK(cudaStreamCreate(&stream));

//    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//    context.enqueue(batchSize, buffers, stream, nullptr);
//    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
//    cudaStreamSynchronize(stream);

//    // Release stream and buffers
//    cudaStreamDestroy(stream);
//    CHECK(cudaFree(buffers[inputIndex]));
//    CHECK(cudaFree(buffers[outputIndex]));
//}

////!
////! \brief This function prints the help information for running this sample
////!
//void printHelpInfo()
//{
//    std::cout << "Usage: ./sample_mnist_api [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
//    std::cout << "--help          Display help information\n";
//    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
//    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
//    std::cout << "--int8          Run in Int8 mode.\n";
//    std::cout << "--fp16          Run in FP16 mode.\n";
//}


//int main(int argc, char** argv)
//{
//    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
//    if (gArgs.help)
//    {
//        printHelpInfo();
//        return EXIT_SUCCESS;
//    }
//    if (!argsOK)
//    {
//        gLogError << "Invalid arguments" << std::endl;
//        printHelpInfo();
//        return EXIT_FAILURE;
//    }
//    if (gArgs.dataDirs.empty())
//    {
//        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
//    }

//    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

//    gLogger.reportTestStart(sampleTest);

//    // create a model using the API directly and serialize it to a stream
//    IHostMemory* modelStream{nullptr};
//    APIToModel(1, &modelStream);
//    assert(modelStream != nullptr);

//    // Read random digit file
//    srand(unsigned(time(nullptr)));
//    uint8_t fileData[INPUT_H * INPUT_W];
//    const int num = rand() % 10;
//    readPGMFile(std::to_string(num) + ".pgm", fileData);

//    // Print ASCII representation of digit image
//    gLogInfo << "Input:\n";
//    for (int i = 0; i < INPUT_H * INPUT_W; i++)
//        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
//    gLogInfo << std::endl;

//    // Parse mean file
//    ICaffeParser* parser = createCaffeParser();
//    assert(parser != nullptr);
//    IBinaryProtoBlob* meanBlob = parser->parseBinaryProto(locateFile("mnist_mean.binaryproto", gArgs.dataDirs).c_str());
//    parser->destroy();
//    const float* meanData = reinterpret_cast<const float*>(meanBlob->getData());

//    // Subtract mean from image
//    float data[INPUT_H * INPUT_W];
//    for (int i = 0; i < INPUT_H * INPUT_W; i++)
//    {
//        data[i] = float(fileData[i]) - meanData[i];
//    }
//    meanBlob->destroy();

//    IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
//    assert(runtime != nullptr);
//    if (gArgs.useDLACore >= 0)
//    {
//        runtime->setDLACore(gArgs.useDLACore);
//    }
//    ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
//    assert(engine != nullptr);
//    modelStream->destroy();
//    IExecutionContext* context = engine->createExecutionContext();
//    assert(context != nullptr);

//    // Run inference
//    float prob[OUTPUT_SIZE];
//    doInference(*context, data, prob, 1);

//    // Destroy the engine
//    context->destroy();
//    engine->destroy();
//    runtime->destroy();

//    // Print histogram of the output distribution
//    gLogInfo << "Output:\n";
//    float val{0.0f};
//    int idx{0};
//    for (unsigned int i = 0; i < 10; i++)
//    {
//        val = std::max(val, prob[i]);
//        if (val == prob[i])
//            idx = i;
//        gLogInfo << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
//    }
//    gLogInfo << std::endl;

//    bool pass{idx == num && val > 0.9f};

//    return gLogger.reportTest(sampleTest, pass);
//}
