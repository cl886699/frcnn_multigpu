# tf2.0目标检测之csv 2 Tfrecord
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import cv2
import os
import numpy as np
import random
import colorsys
from tqdm import tqdm
from detection.models.detectors import faster_rcnn
from dota_data import ZipDotaDataset

random.seed(1234)


def draw_gt(image, bboxes, classes, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_classes = bboxes
    for i in range(len(out_boxes)):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        coor[0] = int(coor[0])
        coor[2] = int(coor[2])
        coor[1] = int(coor[1])
        coor[3] = int(coor[3])

        fontScale = 0.5
        class_ind = int(out_classes[i] - 1)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[1], coor[0]), (coor[3], coor[2])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s' % (classes[class_ind])
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


def draw_bbox(image, bboxes, classes, show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes = bboxes
    for i in range(len(out_classes)):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i] - 1)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image



def dispaly_cv(image, boxes, boxes_gt, label_gt, label_pre):
    image = image.numpy()
    image = image.astype(np.uint8)
    # print('lable_gt: ', label_gt)
    # print('lable_pre: ', label_pre)
    if image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
    image_gt = image.copy()
    n = boxes.shape[0]
    if not n:
        print("no instances to display ")
    for i in range(n):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if not np.any(boxes[i]):
            continue
        x1, y1, x2, y2, scores = boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, 8, 0)
        cv2.putText(image, str(label_pre[i]), (int(x1), int(y1+10)), cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)
        cv2.putText(image, str(round(scores, 2)), (int(x1) + 10, int(y1+10)),cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)

    ngt = boxes_gt.shape[0]
    # print("gt_box: ", boxes_gt)
    if not ngt:
        print("no instances to display ")
    for i in range(ngt):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if not np.any(boxes_gt[i]):
            continue
        x1, y1, x2, y2= boxes_gt[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        cv2.rectangle(image_gt, (x1, y1), (x2, y2), color, 1, 8, 0)
        cv2.putText(image_gt, str(label_gt[i]), (int(x1), int(y1+10)), cv2.FONT_HERSHEY_SIMPLEX,0.3,color,1)

    stack_image = np.hstack((image_gt, image))
    cv2.namedWindow('gt&pre',cv2.WINDOW_FREERATIO)
    cv2.imshow('gt&pre', stack_image)
    cv2.waitKey(0)


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cal_ap(gt_dict, pre__dict, categaries):
    result = {}
    for key in range(categaries):
        b1 = pre__dict[str(key+1)]
        if not b1:
            continue
        image_ids = [tt[0] for tt in b1]
        confidence = np.array([tt[5] for tt in b1])
        BB = np.array([tt[1:5] for tt in b1])
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = gt_dict[str(key + 1)][str(image_ids[d])]  # ann
            bb = BB[d, :].astype(float)
            ovmax = -np.inf  # 负数最大值
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih
                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)  # 最大重叠
                jmax = np.argmax(overlaps)  # 最大重合率对应的gt
            # 计算tp 和 fp个数
            if ovmax > 0.5:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1  # 标记为已检测
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
        # compute precision recall
        fp = np.cumsum(fp)  # np.cumsum() 按位累加
        tp = np.cumsum(tp)
        rec = tp / float(npos[str(key + 1)])
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        print('key+1: ', str(key + 1))
        print(ap)
        result[str(key + 1)] = ap
    return result


if __name__ == '__main__':
    tf_record_path = 'D:/datasets/dota/old_tfrecord/'
    total_categaries = 16
    _, val_datasets = ZipDotaDataset(tf_record_path, 1, crop_size=[512, 512, 3]).prepare(False, False)
    one_imgs, one_metas, one_bboxes, one_labels, _ = next(iter(val_datasets))
    one_imgs = tf.cast(one_imgs, tf.float32)
    one_metas = tf.cast(one_metas, tf.float32)
    model = faster_rcnn.FasterRCNN(
        num_classes=17)
    _ = model((one_imgs, one_metas), training=False)
    model.load_weights('./weights/epoch_6_loss_130.0.h5',
                       by_name=True)
    gt_bbox_dict = {str(key + 1): {} for key in range(total_categaries)}
    pre_bbox_dict = {str(key + 1): [] for key in range(total_categaries)}
    # number of true positive
    npos = {str(key + 1): 0 for key in range(total_categaries)}
    img_ids = 10000
    fd_cat = open(os.path.join(tf_record_path, 'dota.names'))
    label_str = fd_cat.read().split('\n')
    for val_imgs, val_metas, val_bboxes, val_labels, val_file in tqdm(val_datasets):
        val_bboxes = tf.squeeze(val_bboxes, axis=0)
        val_labels = tf.squeeze(val_labels, axis=0)
        label_mask = tf.squeeze(tf.where(tf.not_equal(val_labels, -1)), axis=1)
        boxes_gt = tf.gather(val_bboxes, label_mask)
        val_labels = tf.gather(val_labels, label_mask)
        boxes_gt = boxes_gt.numpy()
        val_labels = val_labels.numpy()
        boxes_gt = boxes_gt.astype(np.int)
        val_file = val_file.numpy()[0].decode('utf-8')
        val_imgs = tf.squeeze(val_imgs, axis=0)
        val_metas = tf.squeeze(val_metas, axis=0)
        val_predict_bboxes = []
        # print(val_bboxes)
        for key in range(total_categaries):
            # 现在数据返回的坐标顺序是y1,x1,y2,x2
            tmp_box = [[boxes_gt[indcc][1], boxes_gt[indcc][0], boxes_gt[indcc][3], boxes_gt[indcc][2]]
                       for indcc, cc in enumerate(val_labels) if cc == key + 1]
            # tmp_box = [boxes_gt[indcc] for indcc, cc in enumerate(val_labels) if cc == key + 1]
            det = [False] * len(tmp_box)
            gt_bbox_dict[str(key + 1)][str(img_ids)] = {'bbox': np.array(tmp_box), 'det': det}
            npos[str(key + 1)] += len(tmp_box)

        proposals = model.simple_test_rpn(val_imgs, val_metas)
        res = model.simple_test_bboxes(val_imgs, val_metas, proposals)
        for pos in range(res['class_ids'].shape[0]):
            label_id = int(res['class_ids'][pos])
            y1, x1, y2, x2 = [int(num) for num in list(res['rois'][pos])]
            tmp_list2 = [img_ids, x1, y1, x2, y2, float(res['scores'][pos])]
            val_predict_bboxes.append([x1, y1, x2, y2, float(res['scores'][pos])])
            pre_bbox_dict[str(label_id)].append(tmp_list2)
        img_ids += 1
        pre_image = val_imgs.numpy().copy()
        gt_image = draw_gt(image=val_imgs.numpy().astype(np.uint8), bboxes=[boxes_gt, val_labels], classes=label_str)
        pre_bbox = [np.array(val_predict_bboxes), res['scores'], res['class_ids']]
        pre_image = draw_bbox(image=pre_image.astype(np.uint8), bboxes=pre_bbox, classes=label_str)
        show_image = np.concatenate([gt_image, pre_image], axis=1)
        cv2.imshow('gt', show_image)
        cv2.waitKey(0)
    ap = cal_ap(gt_bbox_dict, pre_bbox_dict, total_categaries)
    print(ap)
    print('map: ', sum([ap['1'], ap['2']])/15)