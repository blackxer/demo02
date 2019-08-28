from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

import sys
import torch.onnx
from caffe2.python.onnx.backend import Caffe2Backend as c2
from caffe2.python import core, workspace, net_printer
from caffe2.proto import caffe2_pb2
import onnx

import numpy as np


if len(sys.argv) < 3:
    print('Usage: python convert_to_caffe2_models.py <net type: mobilenet-v1-ssd|others>  <model path>')
    sys.argv[1] = 'vgg16-ssd'
    sys.argv[2] = 'models/vgg16-ssd-Epoch-199-Loss-3.338298067378537.pth'
    sys.argv[3] = 'models/voc-model-labels.txt'
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]

label_path = sys.argv[3]

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)
net.load(model_path)
net = net.cuda()
net.eval()

model_path = f"models/{net_type}.onnx"
init_net_path = f"models/{net_type}_init_net.pb"
init_net_txt_path = f"models/{net_type}_init_net.pbtxt"
predict_net_path = f"models/{net_type}_predict_net.pb"
predict_net_txt_path = f"models/{net_type}_predict_net.pbtxt"

dummy_input = torch.randn(1, 3, 300, 300).cuda()
ori_scores, ori_boxes = net(dummy_input)
scores, boxes = torch.onnx._export(net, dummy_input, model_path, verbose=True, output_names=['scores', 'boxes'])
np.testing.assert_almost_equal(ori_scores.data.cpu().numpy(), scores.data.cpu().numpy(), decimal=3)
np.testing.assert_almost_equal(ori_boxes.data.cpu().numpy(), boxes.data.cpu().numpy(), decimal=3)

model = onnx.load(model_path)
# init_net, predict_net = c2.onnx_graph_to_caffe2_net(model)
prepared_backend = c2.prepare(model,device="CUDA")
W = {model.graph.input[0].name: dummy_input.data.cpu().numpy()}
c2_out = prepared_backend.run(W)[0]

c2_workspace = prepared_backend.workspace
c2_model = prepared_backend.predict_net

# Now import the caffe2 mobile exporter
from caffe2.python.predictor import mobile_exporter

# call the Export to get the predict_net, init_net. These nets are needed for running things on mobile
init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)


print(f"Save the model in binary format to the files {init_net_path} and {predict_net_path}.")


with open(init_net_path, "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open(predict_net_path, "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

print(f"Save the model in txt format to the files {init_net_txt_path} and {predict_net_txt_path}. ")
with open(init_net_txt_path, 'w') as f:
    f.write(str(init_net))
with open(predict_net_txt_path, 'w') as f:
    f.write(str(predict_net))



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

# predictor = load_model(init_net_path, predict_net_path)
confidences1, boxes1 = load_model(init_net_path, predict_net_path, dummy_input.data.cpu().numpy())
np.testing.assert_almost_equal(ori_scores.data.cpu().numpy(), confidences1, decimal=3)
np.testing.assert_almost_equal(ori_boxes.data.cpu().numpy(), boxes1, decimal=3)

print("Exported model has been executed on Caffe2 backend, and the result looks good!")