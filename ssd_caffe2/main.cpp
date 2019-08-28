#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <iosfwd>
#include <algorithm>
#include <cstdlib>
#include <opencv2/opencv.hpp>
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
//#include <torch/script.h> // One-stop header.

#include <opencv2/opencv.hpp>

#include "SSDNet.h"

using namespace std;
using namespace caffe2;
using namespace cv;




void imgList(std::vector<std::string>& filenames, std::vector<int>& labels, std::string test_txt)
{
    std:string filename;
    std::string idx;
    char line[1024]={0};
    std::ifstream fin(test_txt, std::ios::in);
    while(fin.getline(line,sizeof(line))){
        std::stringstream word(line);
        word >> filename;
        word >> idx;
        filenames.push_back(filename);
        labels.push_back(std::stoi(idx));

    }
    fin.close();
}


vector<float> preProcess(string img_path)
{
    cv::Mat img = cv::imread(img_path);
    cv::resize(img,img,cv::Size(300,300));
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    assert(img.channels() == 3);
    assert(img.rows == 300);
    assert(img.cols == 300);
    vector<int> dims({1, img.channels(), img.rows, img.cols});
    vector<float> data(1 * 3 * 300 * 300);

    int predHeight = img.rows;
    int predWidth = img.cols;
    int size = predHeight * predWidth;
    // 注意imread读入的图像格式是unsigned char，如果你的网络输入要求是float的话，下面的操作就不对了。
    for (auto i=0; i<predHeight; i++) {
        //printf("+\n");
        for (auto j=0; j<predWidth; j++) {
            data[i * predWidth + j + 0*size] = (float)img.data[(i*predWidth + j) * 3 + 0];
            data[i * predWidth + j + 1*size] = (float)img.data[(i*predWidth + j) * 3 + 1];
            data[i * predWidth + j + 2*size] = (float)img.data[(i*predWidth + j) * 3 + 2];
        }
    }


//    img.convertTo(img, CV_32FC1);
//     //输入是float格式
//    for (auto i = 0; i < predHeight; i++) {
//       //模版的输入格式是float
//        const float* inData = img.ptr<float>(i);
//        for (auto j = 0; j < predWidth; j++) {
//            data[i * predWidth + j + 0 * size] = (float)((inData[j]) + 0);
//            data[i * predWidth + j + 1 * size] = (float)((inData[j]) + 1);
//            data[i * predWidth + j + 2 * size] = (float)((inData[j]) + 2);
//        }
//    }


//    copy((float *)img.datastart, (float *)img.dataend,data.begin());
//    caffe2::DeviceType deviceType=caffe2::CPU;
//    input(dims, deviceType);
//    input->ShareExternalPointer(data.data());
//    const float* temp1 = input->data<float>();
//    cout << "middle: " << *temp1 << endl;
//    return input;
    return data;
}


int main(int argc, char** argv) {

    // 初始化测试图片列表
    std::vector<std::string> filenames;
    std::vector<int> labels;
    std::string test_txt = "/media/zw/DL/ly/workspace/project02/experiment/exp01/preData/test.txt";
    imgList(filenames, labels, test_txt);
    cout << filenames.size() << endl;
    cout << labels.size() << endl;


    // 定义初始化网络结构与权重值
    string initNet = "/media/zw/DL/ly/workspace/project02/experiment/exp04/demo02/pytorch-ssd/models/vgg16-ssd_init_net.pb";
    string predictNet = "/media/zw/DL/ly/workspace/project02/experiment/exp04/demo02/pytorch-ssd/models/vgg16-ssd_predict_net.pb";
    SSDNet net(initNet, predictNet);

    // inpute data
    int err=0, total = 0;
    std::chrono::_V2::system_clock::time_point t_start, t_end;
    float t_sum=0;
    for(int i=0; i<filenames.size(); i++){
        string img_path = "/media/zw/DL/ly/data/data/VOCdevkit/VOC2007/JPEGImages/000002.jpg";


        t_start = std::chrono::high_resolution_clock::now();
        net.predict(img_path);
        t_end = std::chrono::high_resolution_clock::now();
        t_sum += std::chrono::duration<float, std::milli>(t_end - t_start).count();

        net.postProcess();
        break;

    }
    return 0;
}
// 971/10152, err_ratio:0.0956462, time:64.4051ms
