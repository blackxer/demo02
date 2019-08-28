#include <torch/script.h> // One-stop header.
#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "argsParser.h"
#include "logger.h"
#include "common.h"

#include <opencv2/opencv.hpp>

#include "BoxUtils.h"


using namespace nvinfer1;

static const int INPUT_C = 3;
static const int INPUT_H = 300;
static const int INPUT_W = 300;
static const int OUTPUT_SIZE_SCORES = 1*8732*21;
static const int OUTPUT_SIZE_BOXES = 1*8732*4;

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

samplesCommon::Args gArgs;

void saveResults(std::string filename, at::Tensor picked_box_probs_tensor, std::vector<int> picked_labels, std::vector<std::string> labels){
    std::ofstream fout;
    for(int j=0; j<picked_labels.size(); j++){

        std::string cate = labels[picked_labels[j]];
        fout.open("result/"+cate+".txt",ios::app);
        fout << filename.substr(55,6) << " ";
        fout << picked_box_probs_tensor[j][4].item().to<float>() << " ";
        fout << picked_box_probs_tensor[j][0].item().to<float>() << " ";
        fout << picked_box_probs_tensor[j][1].item().to<float>() << " ";
        fout << picked_box_probs_tensor[j][2].item().to<float>() << " ";
        fout << picked_box_probs_tensor[j][3].item().to<float>() << "\n";
        fout.close();

    }

}

void imgList(std::vector<std::string>& filenames, std::vector<string>& labels, std::string test_txt, std::string labelname_txt)
{
    std::string filename;
    char line[1024]={0};
    std::ifstream fin(test_txt, std::ios::in);
    while(fin.getline(line,sizeof(line))){
        std::stringstream word(line);
        word >> filename;
        filenames.push_back(filename);
    }
    fin.close();

    fin.open(labelname_txt, std::ios::in);
    while(fin.getline(line,sizeof(line))){
        std::stringstream word(line);
        word >> filename;
        labels.push_back(filename);
    }
    fin.close();
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& fileName, uint8_t buffer[INPUT_C*INPUT_H*INPUT_W])
{
    cv::Mat img = cv::imread(fileName);
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    cv::resize(img,img,cv::Size(300,300));
    assert(img.channels() == 3);
    assert(img.rows == 300);
    assert(img.cols == 300);
//    vector<uint8_t> data(1 * 3 * 224 * 224);

    int predHeight = img.rows;
    int predWidth = img.cols;
    int size = predHeight * predWidth;
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            buffer[i * predWidth + j + 0*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 0];
            buffer[i * predWidth + j + 1*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 1];
            buffer[i * predWidth + j + 2*size] = (uint8_t)img.data[(i*predWidth + j) * 3 + 2];
        }
    }
//    readPGMFile(fileName, buffer, INPUT_H, INPUT_W);
}

bool onnxToTRTModel( const std::string& modelFile,        // name of the onnx model
                     unsigned int maxBatchSize,            // batch size - NB must be at least as large as the batch we want to run with
                     IHostMemory *&trtModelStream)      // output buffer for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());

    //Optional - uncomment below lines to view network layer information
//    parser->setPrintLayerInfo(true);
//    parser->reportParsingInfo();

    if ( !parser->parseFromFile( locateFile(modelFile, gArgs.dataDirs).c_str(), static_cast<int>(gLogger.getReportableSeverity()) ) )
   {
       gLogError << "Failure while parsing ONNX file" << std::endl;
       return false;
   }

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);

    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we can destroy the parser
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();

    // 设置保存文件的名称为cached_model.bin
    std::string cache_path = "../cached_model.bin";
    std::ofstream serialize_output_stream;

    // 将序列化的模型结果拷贝至serialize_str字符串
    std::string serialize_str;
    serialize_str.resize( trtModelStream->size() );
    memcpy((void*)serialize_str.data(), trtModelStream->data(), trtModelStream->size());

    // 将serialize_str字符串的内容输出至cached_model.bin文件
    serialize_output_stream.open(cache_path);
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();

    engine->destroy();
    network->destroy();
    builder->destroy();

    return true;
}

void doInference(IExecutionContext& context, float* input, float* scores, float* boxes, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 3);
    void* buffers[3];
    int inputIndex = 0;
    int scoresIndex = engine.getBindingIndex("scores");
    int boxesIndex = engine.getBindingIndex("boxes");


    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[scoresIndex], batchSize * OUTPUT_SIZE_SCORES * sizeof(float)));
    CHECK(cudaMalloc(&buffers[boxesIndex], batchSize * OUTPUT_SIZE_BOXES * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(scores, buffers[scoresIndex], batchSize * OUTPUT_SIZE_SCORES*sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(boxes, buffers[boxesIndex], batchSize * OUTPUT_SIZE_BOXES*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[scoresIndex]));
    CHECK(cudaFree(buffers[boxesIndex]));
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char** argv)
{

    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "/media/zw/DL/ly/workspace/project02/experiment/exp04/demo02/pytorch-ssd/models/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

        gLogger.reportTestStart(sampleTest);


    // 初始化测试图片列表
    std::vector<std::string> filenames;
    std::vector<std::string> labels;
    std::string test_txt = "/media/zw/DL/ly/workspace/project02/experiment/exp04/demo02/ssd_tensorrt/test.txt";
    std::string labelname_txt = "/media/zw/DL/ly/workspace/project02/experiment/exp04/demo02/ssd_tensorrt/voc-model-labels.txt";
    imgList(filenames, labels, test_txt, labelname_txt);
    cout << filenames.size() << endl;
    cout << labels.size() << endl;

//    // create a TensorRT model from the onnx model and serialize it to a stream
//    IHostMemory *trtModelStream{nullptr};
//    if (!onnxToTRTModel("vgg16-ssd.onnx", 1, trtModelStream))
//     gLogger.reportFail(sampleTest);
//    cout << "finished" << endl;
//    assert(trtModelStream != nullptr);


    char *trtModelStream{nullptr};
    size_t size{0};
    std::ifstream file("/media/zw/DL/ly/workspace/project02/experiment/exp04/demo02/ssd_tensorrt/cached_model.bin", std::ios::binary);
    if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
    }

    // deserialize the engine
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }

/*////////////////////////////////////////////////////////////////////////////*/
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    delete[] trtModelStream;
    IExecutionContext *context = engine->createExecutionContext();
    assert(context != nullptr);


    std::chrono::_V2::system_clock::time_point t_start, t_end;
    int t_sum = 0;
    for(int i=0; i<filenames.size(); i++){
        uint8_t fileData[INPUT_C*INPUT_H*INPUT_W];
//        filenames[i] = "/media/zw/DL/ly/data/data/VOCdevkit/VOC2007/JPEGImages/000002.jpg";
        readPGMFile(filenames[i], fileData);
        cv::Mat input = cv::imread(filenames[i]);
        float data[INPUT_C*INPUT_H*INPUT_W];
        for (int k = 0; k < INPUT_C*INPUT_H*INPUT_W; k++)
            data[k] = float(fileData[k]);

        // run inference
        float scores[OUTPUT_SIZE_SCORES];
        float boxes[OUTPUT_SIZE_BOXES];
        t_start = std::chrono::high_resolution_clock::now();
        doInference(*context, data, scores, boxes, 1);
        t_end = std::chrono::high_resolution_clock::now();
        t_sum += std::chrono::duration<float, std::milli>(t_end - t_start).count();

        at::Tensor scores1 = at::from_blob((void*)scores, {1, 8732, 21}, torch::kFloat);
        at::Tensor boxes1 = at::from_blob((void*)boxes, {1, 8732, 4}, torch::kFloat);

        at::Tensor picked_box_probs_tensor;
        std::vector<int> picked_labels;
        predict1(input.cols,input.rows, scores1, boxes1, picked_box_probs_tensor, picked_labels, 0.01, 0.45, -1);
        if(picked_labels.empty()){
            cout << "not object found" << endl;
            continue;
        }
        cout << picked_box_probs_tensor << " " << picked_labels.at(0) << endl;
//        for(int j=0; j<picked_labels.size(); j++){
//            cv::Point p1(picked_box_probs_tensor[j][0].item().to<int>(),picked_box_probs_tensor[j][1].item().to<int>());
//            cv::Point p2(picked_box_probs_tensor[j][2].item().to<int>(),picked_box_probs_tensor[j][3].item().to<int>());
//            cv::Scalar color(255,0,0);
//            cv::rectangle(input,p1,p2,color,2);

//            std::string cate = labels[picked_labels[j]];
//            cv::Point org(picked_box_probs_tensor[j][0].item().to<int>()+10,picked_box_probs_tensor[j][1].item().to<int>()+10);
//            cv::putText(input, cate, org, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, {0,0,225});
//        }
//        cv::namedWindow("demo");
//        cv::imshow("demo",input);
//        cv::waitKey(0);
//        break;
//        saveResults(filenames[i], picked_box_probs_tensor, picked_labels, labels);
    }


    std::cout << "time:" << t_sum/filenames.size() << "ms\n";


    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();





    return 0;
}
