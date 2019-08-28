#ifndef BOXUTILS_H
#define BOXUTILS_H

#include <torch/script.h> // One-stop header.
extern "C" {
at::Tensor area_of(at::Tensor left_top, at::Tensor right_bottom);
at::Tensor iou_of(at::Tensor boxes0, at::Tensor boxes1, float eps=1e-5);
at::Tensor hard_nms(at::Tensor box_probs, float iou_threshold, int top_k=-1, int candidate_size=200);
void predict1(int width, int height, at::Tensor confidences, at::Tensor boxes, at::Tensor& picked_box_probs_tensor,
             std::vector<int>& picked_labels, float prob_threshold=0.5,
             float iou_threshold=0.5, int top_k=-1);
}
#endif
