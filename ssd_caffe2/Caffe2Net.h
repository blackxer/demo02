#ifndef CAFFE2NET_H
#define CAFFE2NET_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iosfwd>
#include <algorithm>
#include <cstdlib>
#include <time.h>
#include <sys/time.h>

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
#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>

using namespace std;
using namespace caffe2;
using namespace cv;

class Caffe2Net {
public:
    caffe2::DeviceType device = caffe2::CPU;
    caffe2::DeviceOption device_ops;
    Caffe2Net(string initNet,string predictNet);
    virtual ~Caffe2Net() = 0;
    virtual void predict(String img_path) = 0;

    virtual at::Tensor preProcess(String img_path) = 0;
    virtual void postProcess() = 0;

protected:
    Workspace workspace;
    caffe2::NetDef init_net, predict_net;
};
#endif
