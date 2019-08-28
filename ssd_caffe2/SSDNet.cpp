#include <algorithm>
#include "SSDNet.h"
#include "BoxUtils.h"

using namespace std;

SSDNet::SSDNet(string initNet,string predictNet):Caffe2Net(initNet,predictNet)
{
}

SSDNet::~SSDNet()
{
}
at::Tensor SSDNet::preProcess(String img_path)
{
    input = cv::imread(img_path);
    cv::Mat img;
    cv::resize(input,img,cv::Size(300,300));
    cv::cvtColor(img,img,cv::COLOR_BGR2RGB);
    at::Tensor input_transformed = at::from_blob((void*)img.data, {img.rows, img.cols, img.channels()}, torch::kByte); // HWC
    input_transformed = input_transformed.permute({2,0,1});  // CHW
    input_transformed = input_transformed.unsqueeze(0); // NCHW
    input_transformed = input_transformed.toType(at::kFloat);

    return input_transformed;

}

void SSDNet::predict(String img_path)
{
    at::Tensor input = preProcess(img_path);
    auto* blob = workspace.CreateBlob("0");
    blob->Reset();
    Tensor tensor_fixed(input);
    BlobGetMutableTensor(blob, device)->CopyFrom(tensor_fixed);
    workspace.RunNet(predict_net.name());
    Tensor  scores1(workspace.GetBlob("scores")->Get<Tensor>(), caffe2::CPU);
    Tensor  boxes1(workspace.GetBlob("boxes")->Get<Tensor>(), caffe2::CPU);
    at::Tensor scores2(scores1);
    at::Tensor boxes2(boxes1);
    scores = scores2;
    boxes = boxes2;


}
void SSDNet::postProcess()
{
    at::Tensor picked_box_probs_tensor;
    std::vector<int> picked_labels;
    predict1(input.cols,input.rows, scores, boxes, picked_box_probs_tensor, picked_labels, 0.5, 0.5, -1);
    cout << picked_box_probs_tensor << " " << picked_labels.at(0) << endl;
    for(int j=0; j<picked_labels.size(); j++){
        cv::Point p1(picked_box_probs_tensor[j][0].item().to<int>(),picked_box_probs_tensor[j][1].item().to<int>());
        cv::Point p2(picked_box_probs_tensor[j][2].item().to<int>(),picked_box_probs_tensor[j][3].item().to<int>());
        cv::Scalar color(255,0,0);
        cv::rectangle(input,p1,p2,color,5);
    }
    cv::namedWindow("demo");
    cv::imshow("demo",input);
    cv::waitKey(0);
}
