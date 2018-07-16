# yolo
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes_num = 80
iou_thres = 0.5

# network
batch_size = 16
eval_path = "/home/wxrui/DATA/coco/coco/5k.txt"
img_w = 416
img_h = 416
device_ids = [0, 1, 2, 3]

# checkpoint
backbone_pretrained = ''
checkpoint = '../training/output/model138922.pth'
