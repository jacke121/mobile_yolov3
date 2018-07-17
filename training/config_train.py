from tensorboardX import SummaryWriter

# log
writer = SummaryWriter('log')

# backbone
backbone_pretrained = "weights/mobilenetv2_weights.pth"

# checkpoints
start_epoch = 0
checkpoint = ''

# yolo
anchors = [[[116, 90], [156, 198], [373, 326]],
           [[30, 61], [62, 45], [59, 119]],
           [[10, 13], [16, 30], [33, 23]]]
classes_num = 80

# lr
decay_gamma = 0.1
decay_step = 20
lr = 10 ** (-2 - start_epoch // decay_step)
backbone_lr = 0.1 * lr

# optimizer
optim_type = "sgd"
weight_decay = 4e-05
epochs = 100

# network
batch_size = 16
train_path = "/home/wxrui/DATA/coco/coco/trainvalno5k.txt"
img_w = 416
img_h = 416
device_ids = [0, 1, 2, 3]

# save
save_path = 'output'
