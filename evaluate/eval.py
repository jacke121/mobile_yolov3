import torch.utils.data
import os
import evaluate.config_eval as config
import data.coco_dataset as coco_dataset
import network.mobile_yolo as mobile_yolo
import network.yolo_loss as yolo_loss
import network.utils as utils


def eval():
    # DataLoader
    dataloader = torch.utils.data.DataLoader(
        coco_dataset.COCODataset(config.eval_path, (config.img_w, config.img_h), is_training=False),
        batch_size=config.batch_size, shuffle=False, num_workers=16, pin_memory=False)

    # net
    net = mobile_yolo.Mobile_YOLO(config, is_training=False)
    net = torch.nn.DataParallel(net.cuda())

    # checkpoint
    net.load_state_dict(torch.load(config.checkpoint))

    yolo_losses = []
    for i in range(3):
        yolo_losses.append(
            yolo_loss.YOLOLoss(config.anchors[i], config.classes_num, (config.img_w, config.img_h)))

    print('Start eval...')
    net.eval()
    n_gt = 0
    correct = 0
    for step, samples in enumerate(dataloader):
        images, labels = samples["image"], samples["label"]
        labels = labels.cuda()
        with torch.no_grad():
            outputs = net(images)
            output_list = []
            for i in range(3):
                output_list.append(yolo_losses[i](outputs[i]))
            output = torch.cat(output_list, 1)
            output = utils.non_max_suppression(output, config.classes_num, conf_thres=0.2)
            #  calculate
            for sample_i in range(labels.size(0)):
                # Get labels for sample where width is not zero (dummies)
                target_sample = labels[sample_i, labels[sample_i, :, 3] != 0]
                for obj_cls, tx, ty, tw, th in target_sample:
                    # Get rescaled gt coordinates
                    tx1, tx2 = config.img_w * (tx - tw / 2), config.img_w * (tx + tw / 2)
                    ty1, ty2 = config.img_h * (ty - th / 2), config.img_h * (ty + th / 2)
                    n_gt += 1
                    box_gt = torch.cat([coord.unsqueeze(0) for coord in [tx1, ty1, tx2, ty2]]).view(1, -1)
                    sample_pred = output[sample_i]
                    if sample_pred is not None:
                        # Iterate through predictions where the class predicted is same as gt
                        for x1, y1, x2, y2, conf, obj_conf, obj_pred in sample_pred[sample_pred[:, 6] == obj_cls]:
                            box_pred = torch.cat([coord.unsqueeze(0) for coord in [x1, y1, x2, y2]]).view(1, -1)
                            iou = utils.bbox_iou(box_pred, box_gt)
                            if iou >= config.iou_thres:
                                correct += 1
                                break
        if n_gt:
            print('Batch [%d/%d] mAP: %.5f' % (step, len(dataloader), float(correct / n_gt)))

    print('Mean Average Precision: %.5f' % float(correct / n_gt))


if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, config.device_ids))
    eval()
