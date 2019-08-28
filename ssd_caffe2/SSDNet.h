#ifndef SSDNET_H
#define SSDNET_H

#include "Caffe2Net.h"

using namespace std;
using namespace cv;
using namespace caffe2;

class SSDNet : public Caffe2Net {
public:
    // original input data
    cv::Mat input;
    // output
    at::Tensor scores;
    at::Tensor boxes;
    SSDNet(string initNet,string predictNet);
    virtual void predict(String img_path);
    virtual ~SSDNet();

    virtual at::Tensor preProcess(String img_path);
    virtual void postProcess();
};

#endif
