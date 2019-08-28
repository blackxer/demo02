#include "BoxUtils.h"
using namespace std;

at::Tensor area_of(at::Tensor left_top, at::Tensor right_bottom){

    at::Tensor hw0 = right_bottom - left_top;
    at::Tensor hw = at::clamp_min_out(hw0, hw0, 0);
    auto overlap_area = hw.slice(1,0,1) * hw.slice(1,1);
    return overlap_area;
}

at::Tensor iou_of(at::Tensor boxes0, at::Tensor boxes1, float eps){

    at::Tensor overlap_left_top = boxes0.slice(1,0,2).clone();
    at::max_out(overlap_left_top, boxes0.slice(1,0,2), boxes1.slice(1,0,2));

    at::Tensor overlap_right_bottom = boxes0.slice(1,2).clone();
    at::min_out(overlap_right_bottom, boxes0.slice(1,2), boxes1.slice(1,2));


    auto overlap_area = area_of(overlap_left_top, overlap_right_bottom);
    auto area0 = area_of(boxes0.slice(1,0,2), boxes0.slice(1,2));
    auto area1 = area_of(boxes1.slice(1,0,2), boxes1.slice(1,2));

    return overlap_area / (area0 + area1 - overlap_area + eps);
}

at::Tensor hard_nms(at::Tensor box_probs, float iou_threshold, int top_k, int candidate_size){

    auto scores = box_probs.slice(1,4);
    auto boxes = box_probs.slice(1,0,4);
    auto indexes = scores.argsort(0);
    std::vector<int> picked;
    indexes = indexes.slice(0,-candidate_size);

    while(indexes.size(0) > 0){
        auto current = indexes[-1].item<int>();
        picked.push_back(current);
        if((0<(top_k==picked.size()))||(indexes.size(0)==1)){
            break;
        }
        auto current_box = boxes[current];
        indexes = indexes.slice(0,0,-1);
        auto rest_boxes = boxes.index_select(0,indexes.view(-1));
        cout << current_box.sizes() << " " << indexes.sizes() << " " << rest_boxes.sizes() << endl;
        auto iou = iou_of(rest_boxes, current_box.unsqueeze(0), 1e-5);

        indexes = indexes.masked_select(iou.le(iou_threshold));
        indexes = indexes.unsqueeze(-1);

    }
    at::Tensor select_index=at::from_blob((void*)picked.data(), {picked.size()}, at::kInt).toType(at::kLong);
    return box_probs.index_select(0, select_index);
}


void predict1(int width, int height, at::Tensor confidences, at::Tensor boxes, at::Tensor& picked_box_probs_tensor,
             std::vector<int>& picked_labels, float prob_threshold,
             float iou_threshold, int top_k){
    boxes = boxes[0];
    confidences = confidences[0];
    cout << boxes.sizes() << " " << confidences.sizes() << endl;

    std::vector<at::Tensor> picked_box_probs;
//    std::vector<int> picked_labels;
    for(int class_index=1; class_index<confidences.size(1); class_index++){
        auto probs = confidences.slice(1,class_index,class_index+1);

        auto mask = probs.gt(prob_threshold);
        probs = probs.masked_select(mask);    //probs[mask];
        if(probs.size(0) == 0) continue;
        auto subset_boxes = boxes.masked_select(mask);
//        cout << subset_boxes.sizes() << endl;
        at::Tensor a[2] = { subset_boxes.view({probs.size(0),4}), probs.view({probs.size(0),1}) };
        at::TensorList all_tensor(a);
        auto box_probs = at::cat(all_tensor,1);
        cout << box_probs.sizes() << endl;

        box_probs = hard_nms(box_probs, iou_threshold, top_k, 200);
        cout << class_index << endl;
        picked_box_probs.push_back(box_probs);
        for(int i=0; i<box_probs.size(0); i++){
            picked_labels.push_back(class_index);
        }

    }
    if(0==picked_box_probs.size()){
        return;
    }else{
        at::TensorList picked_box_probs_cat(picked_box_probs);
        picked_box_probs_tensor = at::cat(picked_box_probs_cat,0);
        picked_box_probs_tensor.slice(1,0,1) *= width;
        picked_box_probs_tensor.slice(1,1,2) *= height;
        picked_box_probs_tensor.slice(1,2,3) *= width;
        picked_box_probs_tensor.slice(1,3,4) *= height;
        cout << picked_box_probs_tensor << endl;
    }
}



