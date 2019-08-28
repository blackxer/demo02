import vision.utils.box_utils_numpy as box_utils
from vision.utils.misc import Timer
from vision.ssd.config.mobilenetv1_ssd_config import specs, center_variance, size_variance


import cv2
import sys
from caffe2.python import core, workspace, net_printer
from caffe2.proto import caffe2_pb2
import numpy as np

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
import torch

priors = box_utils.generate_ssd_priors(specs, 300)
print('priors.shape', priors.shape)


def load_model(init_net_path, predict_net_path, input_data):
    init_def = caffe2_pb2.NetDef()
    with open(init_net_path, "rb") as f:
        init_def.ParseFromString(f.read())
        init_def.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0))
        workspace.RunNetOnce(init_def.SerializeToString())

    predict_net = caffe2_pb2.NetDef()
    with open(predict_net_path, "rb") as f:
        predict_net.ParseFromString(f.read())
        predict_net.device_option.CopyFrom(core.DeviceOption(caffe2_pb2.CUDA, 0))
        workspace.CreateNet(predict_net.SerializeToString())

    name = predict_net.name
    input_name = predict_net.external_input[0]
    workspace.FeedBlob(input_name, input_data, core.DeviceOption(caffe2_pb2.CUDA, 0))  # device_opts：CPU/GPU 模式的选项
    workspace.RunNet(name, 1)
    scores = workspace.FetchBlob("scores")
    boxes = workspace.FetchBlob("boxes")
    return scores, boxes

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                  iou_threshold=iou_threshold,
                                  top_k=top_k,
                                  )
        print(class_index)
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


if len(sys.argv) < 2:
    print('Usage: python run_ssd_live_caffe2.py init_net predict_net')
    # sys.exit(0)
init_net_path = "models/vgg16-ssd_init_net.pb" #sys.argv[1]
predict_net_path = "models/vgg16-ssd_predict_net.pb" #sys.argv[2]
label_path = "models/voc-model-labels.txt" #sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]
# predictor = load_model(init_net_path, predict_net_path)


net = create_vgg_ssd(21, is_test=True).cuda()
net.load('models/vgg16-ssd-Epoch-199-Loss-3.338298067378537.pth')
net.eval()
# predictor = create_vgg_ssd_predictor(net, candidate_size=200)

timer = Timer()
while True:
    orig_image = cv2.imread("/media/zw/DL/ly/data/data/VOCdevkit/VOC2007/JPEGImages/000002.jpg")
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)


    image = cv2.resize(image, (300, 300))
    image = image.astype(np.float32)
    # image = (image - 127) / 128
    # image -= np.array([123, 117, 104])  # RGB layout
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    timer.start()
    # confidences, boxes = net(torch.from_numpy(image).cuda())
    # confidences = confidences.data.cpu().numpy()
    # boxes = boxes.data.cpu().numpy()
    confidences, boxes = load_model(init_net_path, predict_net_path, image)
    # np.testing.assert_almost_equal(confidences, confidences1, decimal=3)
    # np.testing.assert_almost_equal(boxes.data, boxes1, decimal=3)

    interval = timer.end()
    print('Inference Time: {:.2f}s.'.format(interval))
    timer.start()
    boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, 0.5)
    interval = timer.end()
    print('NMS Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.shape[0]))
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.4f}"

        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
