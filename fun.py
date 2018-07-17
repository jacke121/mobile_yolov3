import network.mobile_yolo as mobile_yolo
import torch

#
# class config:
#     classes_num = 80
#     anchors = [[[116, 90], [156, 198], [373, 326]],
#                [[30, 61], [62, 45], [59, 119]],
#                [[10, 13], [16, 30], [33, 23]]]
#     backbone_pretrained = "weights/mobilenetv2_weights.pth"
#
#
# model = mobile_yolo.Mobile_YOLO(config=config, is_training=False)
# model = torch.nn.DataParallel(model.cuda())
# model_dict = model.state_dict()
#
# pre_dict = torch.load('weights/18_46.pth')
# pre_dict = {k: v for k, v in pre_dict.items() if '_adj' not in k}
#
# model_dict.update(pre_dict)
#
# model.load_state_dict(model_dict)
# print(model.state_dict()['module.embedding2.3.weight'])

a = -2 - 4
print(a)

print(10 ** a)
