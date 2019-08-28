#include <caffe2/core/flags.h>
#include <caffe2/core/init.h>
#include <caffe2/core/tensor.h>
#include <caffe2/core/workspace.h>
#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include <caffe2/core/context_gpu.h>
#include <gtest/gtest.h>
#include "caffe2/core/blob.h"
#include "caffe2/core/blob_serialization.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/proto/caffe2_pb.h"
#include <time.h>
#include <sys/time.h>
#include <torch/script.h> // One-stop header.

#include "Caffe2Net.h"


Caffe2Net::Caffe2Net(string initNet,string predictNet):workspace(nullptr)
{

#ifdef WITH_CUDA
    device = caffe2::CUDA;
    device_ops.set_device_type(caffe2::PROTO_CUDA);
    device_ops.set_device_id(0);
#endif
    device = caffe2::CPU;
    device_ops.set_device_type(caffe2::PROTO_CPU);
    device_ops.set_device_id(0);

    //载入部署模型
    CAFFE_ENFORCE(ReadProtoFromFile(initNet, &init_net));
    CAFFE_ENFORCE(ReadProtoFromFile(predictNet, &predict_net));
//    std::cout << "Init net: " << ProtoDebugString(init_net);
//    std::cout << "Predict net: " << ProtoDebugString(predict_net);
    init_net.mutable_device_option()->CopyFrom(device_ops);
    predict_net.mutable_device_option()->CopyFrom(device_ops);
    //网络初始化
    workspace.RunNetOnce(init_net);
    //创建判别器
    workspace.CreateNet(predict_net);
}

Caffe2Net::~Caffe2Net()
{
}

void Caffe2Net::predict(String img_path)
{
//    //create input blob
//#ifdef WITH_CUDA
//    TensorCUDA input = TensorCUDA(preProcess(img));
//    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCUDA>();
//#else
//    TensorCPU input = preProcess(img);
//    auto tensor = workspace.CreateBlob("data")->GetMutable<TensorCPU>();
//#endif
//    tensor->ResizeLike(input);
//    tensor->ShareData(input);
//    //predict
//    workspace.RunNet(predict_net.name());
//    //get output blob
//#ifdef WITH_CUDA
//    TensorCPU output = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCUDA>());
//#else
//    TensorCPU output = TensorCPU(workspace.GetBlob("softmax")->Get<TensorCPU>());
//#endif
//    return postProcess(output);

}

at::Tensor Caffe2Net::preProcess(String img_path)
{
}

void Caffe2Net::postProcess()
{
}
