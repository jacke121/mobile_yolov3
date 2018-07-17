# yolo
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes_num = 80
conf_thres = 0.5

# network
batch_size = 16
img_h = 416
img_w = 416
device_ids = [0, 1, 2, 3]

# path
classes_names_path = "/home/wxrui/DATA/coco/coco.names"

# checkpoint
backbone_pretrained = ''
# checkpoint = '../training/output/model138922.pth'
checkpoint = '../weights/18_46.pth'
