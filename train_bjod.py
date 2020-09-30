# tf2.0目标检测之csv 2 Tfrecord
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import random
import cv2
from tqdm import tqdm
import datetime
import os
import time
from detection.models.detectors import faster_rcnn
from bjod_data import ZiptrainDataset, Zipvaluedata

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

random.seed(234)


def save_images(image, boxes, filen, label_pre, pth=''):
    image = image.numpy()
    image = image.astype(np.uint8)
    if image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR, image)
    n = boxes.shape[0]
    if not n:
        print("no instances to display ")
    for i in range(n):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        if not np.any(boxes[i]):
            continue
        x1, y1, x2, y2, _ = boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, 8, 0)
        cv2.putText(image, str(label_pre[i]), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 1)
    filen = filen[:-4] + '.jpg'
    cv2.imwrite(os.path.join(pth, filen), image)


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


def eval_modle(gt_dict, pre__dict, npos, categaries):
    result = {}
    for key in range(categaries):
        b1 = pre__dict[str(key + 1)]
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
        rec = tp / np.maximum(float(npos[str(key + 1)]), np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        print('key+1: ', str(key + 1))
        print(ap)
        result[str(key + 1)] = ap
    return result


class DistTrainer:
    def __init__(self, dis_strategy, ori_model, categaries, nu_devices, maxap=0.0, epoch=[0, 200], trian_dir=''):
        self.dist_strategy = dis_strategy
        self.model = ori_model
        self.num_devices = nu_devices
        self.trian_dir = trian_dir
        self.epochs = epoch
        self.maxap = maxap
        self.total_categaries = categaries
        self.optimizer = tf.keras.optimizers.SGD(1e-4, momentum=0.9, nesterov=True)

    # @tf.function
    def train_step(self, batch_imgs, batch_metas, batch_bboxes, batch_labels):
        with tf.GradientTape() as tape:
            rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
                self.model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
            loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

    def dist_train_step(self, batch_imgs, batch_metas, batch_bboxes, batch_labels):
        per_loss_value, per_rpn_class_loss, per_rpn_bbox_loss, per_rcnn_class_loss, per_rcnn_bbox_loss = self.dist_strategy.run(
            self.train_step,
            args=(batch_imgs, batch_metas, batch_bboxes, batch_labels))
        loss_value = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss_value, axis=None)
        rpn_class_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_class_loss, axis=None)
        rpn_bbox_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_bbox_loss, axis=None)
        rcnn_class_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rcnn_class_loss, axis=None)
        rcnn_bbox_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rcnn_bbox_loss, axis=None)
        return loss_value, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

    # @tf.function
    def test_step(self, batch_imgs, batch_metas, batch_bboxes, batch_labels):
        rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss = \
            self.model((batch_imgs, batch_metas, batch_bboxes, batch_labels), training=True)
        loss_value = rpn_class_loss + rpn_bbox_loss + rcnn_class_loss + rcnn_bbox_loss
        return loss_value, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

    def dist_test_step(self, batch_imgs, batch_metas, batch_bboxes, batch_labels):
        per_loss_value, per_rpn_class_loss, per_rpn_bbox_loss, per_rcnn_class_loss, per_rcnn_bbox_loss = self.dist_strategy.run(
            self.test_step,
            args=(batch_imgs, batch_metas, batch_bboxes, batch_labels))
        loss_value = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_loss_value, axis=None)
        rpn_class_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_class_loss, axis=None)
        rpn_bbox_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rpn_bbox_loss, axis=None)
        rcnn_class_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rcnn_class_loss, axis=None)
        rcnn_bbox_loss = self.dist_strategy.reduce(tf.distribute.ReduceOp.SUM, per_rcnn_bbox_loss, axis=None)
        return loss_value, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss

    def eval_step(self, val_dts):
        gt_bbox_dict = {str(key + 1): {} for key in range(self.total_categaries)}
        pre_bbox_dict = {str(key + 1): [] for key in range(self.total_categaries)}
        # number of true positive
        npos = {str(key + 1): 0 for key in range(self.total_categaries)}
        img_ids = 10000
        for val_imgs, val_metas, val_bboxes, val_labels, val_file in tqdm(val_dts):
            # if random.randint(1, 100) > 11:
            #     continue
            val_labels = tf.squeeze(val_labels, axis=0).numpy()
            val_bboxes = tf.squeeze(val_bboxes, 0).numpy().astype(np.int)
            val_imgs = tf.squeeze(tf.cast(val_imgs, tf.float32), axis=0)
            val_metas = tf.squeeze(tf.cast(val_metas, tf.float32), axis=0)
            val_predict_bboxes = []

            for key in range(self.total_categaries):
                tmp_box = [val_bboxes[indcc] for indcc, cc in enumerate(val_labels) if cc == key + 1]
                det = [False] * len(tmp_box)
                gt_bbox_dict[str(key + 1)][str(img_ids)] = {'bbox': np.array(tmp_box), 'det': det}
                npos[str(key + 1)] += len(tmp_box)

            proposals = self.model.simple_test_rpn(val_imgs, val_metas)
            res = self.model.simple_test_bboxes(val_imgs, val_metas, proposals)
            for pos in range(res['class_ids'].shape[0]):
                label_id = int(res['class_ids'][pos])
                y1, x1, y2, x2 = [int(num) for num in list(res['rois'][pos])]
                tmp_list2 = [img_ids, x1, y1, x2, y2, float(res['scores'][pos])]
                val_predict_bboxes.append([x1, y1, x2, y2, float(res['scores'][pos])])
                pre_bbox_dict[str(label_id)].append(tmp_list2)
            img_ids += 1
        return gt_bbox_dict, pre_bbox_dict, npos

    def rd_save_images(self, val_dts, img_save_path):
        for val_imgs, val_metas, _, _, val_file in tqdm(val_dts):
            if random.randint(1, 100) > 10:
                continue
            val_file = val_file.numpy()[0].decode('utf-8')
            val_imgs = tf.squeeze(tf.cast(val_imgs, tf.float32), axis=0)
            val_metas = tf.squeeze(tf.cast(val_metas, tf.float32), axis=0)
            val_predict_bboxes = []
            proposals = self.model.simple_test_rpn(val_imgs, val_metas)
            res = self.model.simple_test_bboxes(val_imgs, val_metas, proposals)
            for pos in range(res['class_ids'].shape[0]):
                y1, x1, y2, x2 = [int(num) for num in list(res['rois'][pos])]
                val_predict_bboxes.append([x1, y1, x2, y2, float(res['scores'][pos])])
            save_images(val_imgs, np.array(val_predict_bboxes), val_file, res['class_ids'], img_save_path)

    def train(self, train_ds, val_ds):
        # train model
        train_dts = self.dist_strategy.experimental_distribute_dataset(train_ds)
        val_dts = self.dist_strategy.experimental_distribute_dataset(val_ds)
        log_dir = self.trian_dir + 'log_dir/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(log_dir)
        index_step = 0
        time_start = time.time()
        input_flag = False
        for epoch in range(self.epochs[0], self.epochs[1]):
            loss_history = np.zeros(5)
            for (step, inputs) in enumerate(train_dts):
                batch_imgs, batch_metas, batch_bboxes, batch_labels, filen = inputs
                labels_tmp = tf.cast(tf.fill([1,1000], -1), tf.int32)
                if self.num_devices > 1:
                    for per_tensor in batch_labels.values:
                        if tf.equal(per_tensor, labels_tmp).numpy().all():
                            input_flag = True
                            print("skip this batch")
                            break
                        else:
                            pass
                    if input_flag:
                        input_flag = False
                        continue
                else:
                    if tf.equal(batch_labels, labels_tmp).numpy().all():
                        continue
                loss_value, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss \
                    = self.dist_train_step(batch_imgs, batch_metas, batch_bboxes, batch_labels)
                loss_history[0] += loss_value
                loss_history[1] += rpn_class_loss
                loss_history[2] += rpn_bbox_loss
                loss_history[3] += rcnn_class_loss
                loss_history[4] += rcnn_bbox_loss

                if step % 10 == 0:
                    if step:
                        loss_history = loss_history / 10
                        print('time:', round(time.time() - time_start, 2), 'epoch:', epoch, ', step:', step, ', loss:',
                              loss_history)
                        time_start = time.time()
                        with file_writer.as_default():
                            tf.summary.scalar('total_loss', loss_history[0], step=index_step)
                            tf.summary.scalar('rpn_class_loss', loss_history[1], step=index_step)
                            tf.summary.scalar('rpn_bbox_loss', loss_history[2], step=index_step)
                            tf.summary.scalar('rcnn_class_loss', loss_history[3], step=index_step)
                            tf.summary.scalar('rcnn_bbox_loss', loss_history[4], step=index_step)
                        file_writer.flush()
                        index_step += 1
                        loss_history = np.zeros(5)
                    else:
                        print('epoch:', epoch, ', step:', step, ', loss:', loss_history)
                if step % 2000 == 0:
                    weights_dir = self.trian_dir + 'weights/epoch_' + str(epoch) + '_loss_'
                    sum_loss = 0
                    for (val_step, inputs_val) in tqdm(enumerate(val_dts)):
                        batch_imgs, batch_metas, batch_bboxes, batch_labels, filen = inputs_val
                        labels_tmp = tf.cast(tf.fill([1, 1000], -1), tf.int32)
                        if self.num_devices > 1:
                            for per_tensor in batch_labels.values:
                                if tf.equal(per_tensor, labels_tmp).numpy().all():
                                    input_flag = True
                                    print("skip this batch")
                                    break
                                else:
                                    pass
                            if input_flag:
                                input_flag = False
                                continue
                        else:
                            if tf.equal(batch_labels, labels_tmp).numpy().all():
                                continue
                        loss_value, rpn_class_loss, rpn_bbox_loss, rcnn_class_loss, rcnn_bbox_loss \
                            = self.dist_test_step(batch_imgs, batch_metas, batch_bboxes, batch_labels)
                        sum_loss += loss_value
                    print('sum_loss: ', sum_loss)
                    if sum_loss > self.maxap:
                        self.maxap = sum_loss
                        self.model.save_weights(weights_dir + str(tf.round(sum_loss, 2).numpy()) + '.h5')


if __name__ == '__main__':
    PER_GPU_BATCHSIZE = 1
    dist_strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    num_devices = dist_strategy.num_replicas_in_sync
    print('Number of devices: {}'.format(num_devices))
    GLOBAL_BATCHSIZE = num_devices * PER_GPU_BATCHSIZE

    with dist_strategy.scope():
        if os.name == 'nt':
            tf_record_path = 'D:/datasets/bjod/'
            train_dir = './train_dir/'
        else:
            tf_record_path = '../../../../../datasets/bjod/'
            train_dir = './train_dir/'
        crop_size = [992, 992, 3]
        train_datasets = ZiptrainDataset(tf_record_path, 1, 96, crop_size=crop_size,
                                         roi_path='D:/datasets/bjod/roi_test/').prepare(True)

        val_train = Zipvaluedata(tf_record_path, crop_size=crop_size).prepare('train_api_97.record')
        val_test = Zipvaluedata(tf_record_path, crop_size=crop_size).prepare('val_api_19.record')

        one_imgs, one_metas, one_bboxes, one_labels, _ = next(iter(val_train))
        one_imgs = tf.expand_dims(tf.cast(one_imgs[0], tf.float32), axis=0)
        one_metas = tf.expand_dims(tf.cast(one_metas[0], tf.float32), axis=0)
        model = faster_rcnn.FasterRCNN(num_classes=2)
        _ = model((one_imgs, one_metas), training=False)

        model_ori = faster_rcnn.FasterRCNN(num_classes=81)
        _ = model_ori((one_imgs, one_metas), training=False)
        model_ori.load_weights('./weights/faster_rcnn_resnet101_fpn_coco2017_map35.h5',
                               by_name=True)
        model.backbone.set_weights(model_ori.backbone.get_weights())
        model.neck.set_weights(model_ori.neck.get_weights())
        model.rpn_head.set_weights(model_ori.rpn_head.get_weights())
        model.roi_align.set_weights(model_ori.roi_align.get_weights())
        # print(cc)
        model.summary()


        def __init__(self, dis_strategy, ori_model, categaries, nu_devices, maxap=0.0, epoch=[0, 200], trian_dir=''):
            self.dist_strategy = dis_strategy
            self.model = ori_model
            self.num_devices = nu_devices
            self.trian_dir = trian_dir
            self.epochs = epoch
            self.maxap = maxap
            self.total_categaries = categaries
            self.optimizer = tf.keras.optimizers.SGD(1e-4, momentum=0.9, nesterov=True)
        trainer = DistTrainer(dis_strategy=dist_strategy,
                              ori_model=model,
                              categaries=2,
                              nu_devices=1,
                              maxap=0.0,
                              epoch=[0, 200],
                              trian_dir=train_dir
                              )
        trainer.train(train_datasets, val_test)
