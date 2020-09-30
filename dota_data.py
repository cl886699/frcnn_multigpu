import os
import sys
import tensorlayer as tl
import tensorflow as tf
import numpy as np
import random
import cv2
from shapely.geometry import Polygon
from tqdm import tqdm
from skimage import transform

def show_images(image, boxes, filen, label_pre, pth=''):
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
        y1, x1, y2, x2 = boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, 8, 0)
        cv2.putText(image, str(label_pre[i]), (int((x1 + x2) / 2), int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color, 1)
    cv2.imshow('hello', image)
    cv2.waitKey(0)
    # filen = filen[:-4] + '.jpg'
    # cv2.imwrite(os.path.join(pth, filen), image)


class ZipDotaDataset:
    def __init__(self, dataset_dir, batch_size, crop_size=[512, 512, 3], thresh_minarea=0.2,
                 augment=True):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.min_area = thresh_minarea
        self.image_feature_description = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'encoded': tf.io.FixedLenFeature([], tf.string),
            'x_list': tf.io.VarLenFeature(tf.int64),
            'y_list': tf.io.VarLenFeature(tf.int64),
            'label_list': tf.io.VarLenFeature(tf.int64),
            'difficult': tf.io.VarLenFeature(tf.int64),
        }

    @staticmethod
    def flip_labels(bbx, coin, img_shape):
        if len(bbx) == 0:
            return bbx
        # bbox = np.squeeze(bbx, axis = 0)
        bbox = bbx.numpy()
        # print("bbox_labels: ", bbox)
        w = img_shape[0].numpy()
        h = img_shape[1].numpy()
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        if coin < 0.3:
            bbox[:, 0] = h - (bbox[:, 0] + bw)
            bbox[:, 2] = h - (bbox[:, 2] - bw)
            return bbox
        elif coin > 0.7:
            bbox[:, 1] = w - (bbox[:, 1] + bh)
            bbox[:, 3] = w - (bbox[:, 3] - bh)
            return bbox
        else:
            return bbx

    # 图片宽 高为w,h
    # bbox的bw,bh
    # 逆时针90度: 原点坐标变为了0，w
    # x+bw,y将为左上角坐标，变为 y,w-(x+bw)
    # 顺时针90度：原点坐标变为了h，0
    # x,y+bh将为左上角坐标，变为h-(y+bh),x
    # 180度：原点坐标变为了w,h
    # x+bw,y+bh将为左上角坐标，变为w-(x+bw) h-(y+bh)
    @staticmethod
    def rotate_labels(bbx, ik, img_shape, coin):
        if len(bbx) == 0:
            return bbx
        if coin < 0.5:
            # print("before: ", bbx.numpy())
            w = img_shape[0].numpy()
            h = img_shape[1].numpy()
            bbox = bbx.numpy()
            ik = ik.numpy()
            bw = bbx[:, 2] - bbx[:, 0]
            bh = bbx[:, 3] - bbx[:, 1]
            bw = bw.numpy()
            bh = bh.numpy()
            r_bbox = bbox.copy()

            if ik == 0:
                return r_bbox
            elif ik == 1:
                r_bbox[:, 0] = bbox[:, 1]
                r_bbox[:, 1] = w - (bbox[:, 0] + bw)
                r_bbox[:, 2] = bh + r_bbox[:, 0]
                r_bbox[:, 3] = bw + r_bbox[:, 1]
                # print("w,h,bw,bh: ", w, h, bw, bh)
            elif ik == 2:
                r_bbox[:, 0] = w - (bbox[:, 0] + bw)
                r_bbox[:, 1] = h - (bbox[:, 1] + bh)
                r_bbox[:, 2] = bw + r_bbox[:, 0]
                r_bbox[:, 3] = bh + r_bbox[:, 1]
            elif ik == 3:
                r_bbox[:, 0] = h - (bbox[:, 1] + bh)
                r_bbox[:, 1] = bbox[:, 0]
                r_bbox[:, 2] = bh + r_bbox[:, 0]
                r_bbox[:, 3] = bw + r_bbox[:, 1]
            return r_bbox
        else:
            return bbx

    def random_crop(self, img, x_list, y_list, labels):
        img = img.numpy()
        x_list = x_list.numpy()
        labels = labels.numpy()
        w_img, h_img, _ = img.shape
        cx_max = w_img - self.crop_size[0]
        cy_max = h_img - self.crop_size[1]
        re_index = []
        r_labels = []
        rr_labels = []
        ori_bbox = [[x_list[i], y_list[i]] for i in range(len(x_list))]
        ori_bbox = np.split(ori_bbox, list(range(4, len(ori_bbox), 4)))
        bboxes = []
        for index, _ in enumerate(range(20)):
            tl_x = random.randint(0, cx_max)
            tl_y = random.randint(0, cy_max)
            ori_contours = []
            # print("tl_x,tl_y: ", tl_x,tl_y)
            roi_img = Polygon([[tl_y, tl_x],
                               [self.crop_size[0] + tl_y, tl_x],
                               [self.crop_size[0] + tl_y, self.crop_size[1] + tl_x],
                               [tl_y, self.crop_size[1] + tl_x],
                               ])
            # print("roi_img: ", roi_img)
            for indexi, contours in enumerate(ori_bbox):
                p1 = Polygon(contours).buffer(0)
                pp = roi_img.intersection(p1)
                if pp.geom_type == 'Polygon':
                    if pp.area/p1.area > self.min_area and pp.is_valid:
                        r_labels.append(labels[indexi])
                        re_index.append(indexi)
                        ori_contours.append(pp)
                elif pp.geom_type == 'MultiPolygon':
                    mulpps = list(pp)
                    for mulpp in mulpps:
                        if mulpp.geom_type == 'Polygon':
                            if mulpp.area/p1.area > self.min_area and mulpp.is_valid:
                                r_labels.append(labels[indexi])
                                re_index.append(indexi)
                                ori_contours.append(mulpp)
                            else:
                                pass
                        else:
                            pass
                else:
                    continue
            # 有限次数内没有能达到crop要求,进行resize操作。原生的tl.prepro.obj_box_imresize操作有错误，
            # Tensorlayer
            # 中的imresize使用了scipy.misc.imresize方法，该方法已经被弃用了，需要更改image
            # resize的方法， 这里可以改为skimage.transform.resize(x, size, preserve_range=True, order=3)
            if re_index:
                img = img[tl_x:(self.crop_size[0] + tl_x), tl_y:(self.crop_size[1] + tl_y)]
                for inds, contours in enumerate(ori_contours):
                    coords = contours.bounds
                    xmin = int(coords[0] - tl_y)
                    ymin = int(coords[1] - tl_x)
                    xmax = int(coords[2] - tl_y)
                    ymax = int(coords[3] - tl_x)
                    if xmax > xmin and ymax > ymin:
                        bboxes.append([xmin, ymin, xmax, ymax])
                        rr_labels.append(r_labels[inds])
                return img, bboxes, rr_labels
            else:
                continue
        # tmp_bboxes = []
        # for inds, contours in enumerate(ori_bbox):
        #     contours = Polygon(contours).buffer(0)
        #     if contours.area < 1.0:
        #         continue
        #     coords = contours.bounds
        #     xmin = int(coords[0])
        #     ymin = int(coords[1])
        #     xmax = int(coords[2])
        #     ymax = int(coords[3])
        #     if xmax > xmin and ymax > ymin:
        #         tmp_bboxes.append([xmin, ymin, xmax, ymax])
        #         rr_labels.append(labels[inds])
        # if tmp_bboxes:
        #     tmp_bboxes = np.array(tmp_bboxes)
        #     xy_wh_bbox = tmp_bboxes.copy()
        #     xy_wh_bbox[:, 2] = tmp_bboxes[:, 2] - tmp_bboxes[:, 0]
        #     xy_wh_bbox[:, 3] = tmp_bboxes[:, 3] - tmp_bboxes[:, 1]
        #     img, xy_wh_bbox = tl.prepro.obj_box_imresize(img, xy_wh_bbox,
        #                                                  size=[self.crop_size[0], self.crop_size[1]])
        #     xy_wh_bbox = np.array(xy_wh_bbox)
        #     bboxes = xy_wh_bbox.copy()
        #     bboxes[:, 2] = xy_wh_bbox[:, 2] + xy_wh_bbox[:, 0]
        #     bboxes[:, 3] = xy_wh_bbox[:, 3] + xy_wh_bbox[:, 1]
        # else:
        #     img = transform.resize(img, self.crop_size[0:-1], preserve_range=True, order=3)
        img = transform.resize(img, self.crop_size[0:-1], preserve_range=True, order=3)
        return img, bboxes, rr_labels

    def bbox_convert(self, r_bbox, r_labels):
        #将bbox的形状统一化为1000*4
        if r_bbox.numpy().shape[0]:
            zeros_tmp = tf.zeros([1000, 4], tf.int64)
            r_bbox = tf.concat([r_bbox, zeros_tmp], axis=0)
            r_bbox = tf.slice(r_bbox, [0, 0], [1000, 4])
            r_bbox = tf.cast(r_bbox, tf.float32)
            labes_tmp = tf.cast(tf.fill([1000], -1), tf.int64)
            r_labels = tf.concat([r_labels, labes_tmp], axis=0)
            r_labels = tf.slice(r_labels, [0], [1000])
            r_labels = tf.cast(r_labels, tf.int32)
        else:
            r_bbox = tf.zeros([1000, 4], tf.float32)
            r_labels = tf.cast(tf.fill([1000], -1), tf.int32)
        r_bbox = r_bbox.numpy()
        # if r_bbox.shape[0]:
        cc = np.hsplit(r_bbox, 4)
        dd = [cc[1], cc[0], cc[3], cc[2]]
        r_bbox = np.hstack(dd)
        return r_bbox, r_labels

    def parse_image_function(self, example_proto):
        image_features = tf.io.parse_single_example(example_proto, self.image_feature_description)
        # print("type:", type(image_features['encoded']))
        x_image = tf.io.decode_png(image_features['encoded'], 3)
        file_name = image_features['filename']
        difficult = tf.sparse.to_dense(image_features['difficult'])
        x_list = tf.sparse.to_dense(image_features['x_list'])
        y_list = tf.sparse.to_dense(image_features['y_list'])
        label_list = tf.sparse.to_dense(image_features['label_list'])
        rimage_metas = tf.cast([512, 512, 3], tf.float32)
        parse_image, r_bbox, r_labels = tf.py_function(self.random_crop,
                                                       inp=[x_image, x_list, y_list, label_list],
                                                       Tout=[tf.uint8, tf.int64, tf.int64])
        if self.augment:
            # rotate
            coin = tf.random.uniform([], 0, 1.0)
            ik = tf.random.uniform([], minval=0, maxval=4, dtype="int32")
            parse_image = tf.cond(
                coin < 0.5,
                lambda: tf.image.rot90(parse_image, k=ik),
                lambda: parse_image)
            r_bbox = tf.py_function(self.rotate_labels, inp=[r_bbox, ik, self.crop_size, coin], Tout=[tf.int64])
            r_bbox = tf.squeeze(r_bbox, axis=0)
            # flip
            def f1(): return tf.image.flip_left_right(parse_image)
            def f2(): return tf.image.flip_up_down(parse_image)
            def f3(): return parse_image
            coin_flip = tf.random.uniform([], 0, 1.0)
            parse_image = tf.case([(tf.less(coin_flip, 0.3), f1),
                                   (tf.greater(coin_flip, 0.7), f2)],
                                  default=f3, exclusive=True)
            r_bbox = tf.py_function(self.flip_labels, inp=[r_bbox, coin_flip, self.crop_size], Tout=[tf.int64])
            r_bbox = tf.squeeze(r_bbox, axis=0)
        r_bbox, r_labels= tf.py_function(self.bbox_convert, inp=[r_bbox, r_labels], Tout=[tf.int64, tf.int32])
        # r_bbox = tf.squeeze(r_bbox, axis=0)
        # r_labels = tf.squeeze(r_labels, axis=0)
        parse_image = tf.cast(parse_image, tf.float32)
        r_bbox = tf.cast(r_bbox, tf.float32)
        return parse_image, rimage_metas, r_bbox, r_labels, file_name

    def prepare(self, train_aug=True, val_aug=False):
        parse_fn = lambda x: self.parse_image_function(x)
        self.augment = train_aug
        train_ds = tf.data.TFRecordDataset(os.path.join(self.dataset_dir, 'train21797.record')).map(parse_fn, num_parallel_calls=-1)
        train_ds = train_ds.shuffle(10).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # train_ds = train_ds.shuffle()

        self.augment = val_aug
        val_ds = tf.data.TFRecordDataset(os.path.join(self.dataset_dir, 'val162.record')).map(parse_fn, num_parallel_calls=-1)
        val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, val_ds


class ZipDotaDataset_notcrop:
    def __init__(self, dataset_dir, batch_size, crop_size=[512, 512, 3], thresh_minarea=0,
                 augment=True):
        self.dataset_dir = dataset_dir
        self.augment = augment
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.min_area = thresh_minarea
        self.image_feature_description = {
            'filename': tf.io.FixedLenFeature([], tf.string),
            'encoded': tf.io.FixedLenFeature([], tf.string),
            'x_list': tf.io.VarLenFeature(tf.int64),
            'y_list': tf.io.VarLenFeature(tf.int64),
            'label_list': tf.io.VarLenFeature(tf.int64),
            'difficult': tf.io.VarLenFeature(tf.int64),
        }

    @staticmethod
    def flip_labels(bbx, coin, img_shape):
        if len(bbx) == 0:
            return bbx
        # bbox = np.squeeze(bbx, axis = 0)
        bbox = bbx.numpy()
        # print("bbox_labels: ", bbox)
        w = img_shape[0].numpy()
        h = img_shape[1].numpy()
        bw = bbox[:, 2] - bbox[:, 0]
        bh = bbox[:, 3] - bbox[:, 1]
        if coin < 0.3:
            bbox[:, 0] = h - (bbox[:, 0] + bw)
            bbox[:, 2] = h - (bbox[:, 2] - bw)
            return bbox
        elif coin > 0.7:
            bbox[:, 1] = w - (bbox[:, 1] + bh)
            bbox[:, 3] = w - (bbox[:, 3] - bh)
            return bbox
        else:
            return bbx

    # 图片宽 高为w,h
    # bbox的bw,bh
    # 逆时针90度: 原点坐标变为了0，w
    # x+bw,y将为左上角坐标，变为 y,w-(x+bw)
    # 顺时针90度：原点坐标变为了h，0
    # x,y+bh将为左上角坐标，变为h-(y+bh),x
    # 180度：原点坐标变为了w,h
    # x+bw,y+bh将为左上角坐标，变为w-(x+bw) h-(y+bh)
    @staticmethod
    def rotate_labels(bbx, ik, img_shape, coin):
        if len(bbx) == 0:
            return bbx
        if coin < 0.5:
            # print("before: ", bbx.numpy())
            w = img_shape[0].numpy()
            h = img_shape[1].numpy()
            bbox = bbx.numpy()
            ik = ik.numpy()
            bw = bbx[:, 2] - bbx[:, 0]
            bh = bbx[:, 3] - bbx[:, 1]
            bw = bw.numpy()
            bh = bh.numpy()
            r_bbox = bbox.copy()

            if ik == 0:
                return r_bbox
            elif ik == 1:
                r_bbox[:, 0] = bbox[:, 1]
                r_bbox[:, 1] = w - (bbox[:, 0] + bw)
                r_bbox[:, 2] = bh + r_bbox[:, 0]
                r_bbox[:, 3] = bw + r_bbox[:, 1]
                # print("w,h,bw,bh: ", w, h, bw, bh)
            elif ik == 2:
                r_bbox[:, 0] = w - (bbox[:, 0] + bw)
                r_bbox[:, 1] = h - (bbox[:, 1] + bh)
                r_bbox[:, 2] = bw + r_bbox[:, 0]
                r_bbox[:, 3] = bh + r_bbox[:, 1]
            elif ik == 3:
                r_bbox[:, 0] = h - (bbox[:, 1] + bh)
                r_bbox[:, 1] = bbox[:, 0]
                r_bbox[:, 2] = bh + r_bbox[:, 0]
                r_bbox[:, 3] = bw + r_bbox[:, 1]
            return r_bbox
        else:
            return bbx

    def build_bbox(self, x_list, y_list, labels):
        x_list = x_list.numpy()
        labels = labels.numpy()
        r_labels = []
        ori_bbox = [[x_list[i], y_list[i]] for i in range(len(x_list))]
        ori_bbox = np.split(ori_bbox, list(range(4, len(ori_bbox), 4)))
        bboxes = []
        for indexi, contours in enumerate(ori_bbox):
            p1 = Polygon(contours)
            if p1.area > self.min_area:
                coords = p1.bounds
                xmin = int(coords[0])
                ymin = int(coords[1])
                xmax = int(coords[2])
                ymax = int(coords[3])
                if xmax > xmin and ymax > ymin:
                    bboxes.append([xmin, ymin, xmax, ymax])
                    r_labels.append(labels[indexi])
        return bboxes, r_labels

    def bbox_convert(self, r_bbox, r_labels):
        #将bbox的形状统一化为1000*4
        if r_bbox.numpy().shape[0]:
            zeros_tmp = tf.zeros([1000, 4], tf.int64)
            r_bbox = tf.concat([r_bbox, zeros_tmp], axis=0)
            r_bbox = tf.slice(r_bbox, [0, 0], [1000, 4])
            r_bbox = tf.cast(r_bbox, tf.float32)
            labes_tmp = tf.cast(tf.fill([1000], -1), tf.int64)
            r_labels = tf.concat([r_labels, labes_tmp], axis=0)
            r_labels = tf.slice(r_labels, [0], [1000])
            r_labels = tf.cast(r_labels, tf.int32)
        else:
            r_bbox = tf.zeros([1000, 4], tf.float32)
            r_labels = tf.cast(tf.fill([1000], -1), tf.int32)
        r_bbox = r_bbox.numpy()
        # if r_bbox.shape[0]:
        cc = np.hsplit(r_bbox, 4)
        dd = [cc[1], cc[0], cc[3], cc[2]]
        r_bbox = np.hstack(dd)
        return r_bbox, r_labels

    def parse_image_function(self, example_proto):
        image_features = tf.io.parse_single_example(example_proto, self.image_feature_description)
        # print("type:", type(image_features['encoded']))
        parse_image = tf.io.decode_png(image_features['encoded'], 3)
        file_name = image_features['filename']
        difficult = tf.sparse.to_dense(image_features['difficult'])
        x_list = tf.sparse.to_dense(image_features['x_list'])
        y_list = tf.sparse.to_dense(image_features['y_list'])
        label_list = tf.sparse.to_dense(image_features['label_list'])
        rimage_metas = tf.cast([1024, 1024, 3], tf.float32)
        r_bbox, r_labels = tf.py_function(self.build_bbox,
                                                       inp=[x_list, y_list, label_list],
                                                       Tout=[tf.int64, tf.int64])
        if self.augment:
            # rotate
            coin = tf.random.uniform([], 0, 1.0)
            ik = tf.random.uniform([], minval=0, maxval=4, dtype="int32")
            parse_image = tf.cond(
                coin < 0.5,
                lambda: tf.image.rot90(parse_image, k=ik),
                lambda: parse_image)
            r_bbox = tf.py_function(self.rotate_labels, inp=[r_bbox, ik, self.crop_size, coin], Tout=[tf.int64])
            r_bbox = tf.squeeze(r_bbox, axis=0)
            # flip
            def f1(): return tf.image.flip_left_right(parse_image)
            def f2(): return tf.image.flip_up_down(parse_image)
            def f3(): return parse_image
            coin_flip = tf.random.uniform([], 0, 1.0)
            parse_image = tf.case([(tf.less(coin_flip, 0.3), f1),
                                   (tf.greater(coin_flip, 0.7), f2)],
                                  default=f3, exclusive=True)
            r_bbox = tf.py_function(self.flip_labels, inp=[r_bbox, coin_flip, self.crop_size], Tout=[tf.int64])
            r_bbox = tf.squeeze(r_bbox, axis=0)
        r_bbox, r_labels= tf.py_function(self.bbox_convert, inp=[r_bbox, r_labels], Tout=[tf.int64, tf.int32])
        # r_bbox = tf.squeeze(r_bbox, axis=0)
        # r_labels = tf.squeeze(r_labels, axis=0)
        parse_image = tf.cast(parse_image, tf.float32)
        r_bbox = tf.cast(r_bbox, tf.float32)
        return parse_image, rimage_metas, r_bbox, r_labels, file_name

    def prepare(self, train_aug=True, val_aug=False):
        parse_fn = lambda x: self.parse_image_function(x)
        self.augment = train_aug
        train_ds = tf.data.TFRecordDataset(os.path.join(self.dataset_dir, 'train21797.record')).map(parse_fn, num_parallel_calls=-1)
        train_ds = train_ds.shuffle(10).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # train_ds = train_ds.shuffle()

        self.augment = val_aug
        val_ds = tf.data.TFRecordDataset(os.path.join(self.dataset_dir, 'val6952.record')).map(parse_fn, num_parallel_calls=-1)
        val_ds = val_ds.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, val_ds


if __name__ == '__main__':
    tf_record_path = 'D:/datasets/dota/'
    train_datasets, val_datasets = ZipDotaDataset_notcrop(tf_record_path, 1, crop_size=[1024, 1024, 3]).prepare(True, False)
    # print(len(train_datasets))
    a = 0
    for parse_image, rimage_metas, r_bbox, r_labels, file_name in tqdm(val_datasets):
        print(file_name)
        print(parse_image.shape)
        # parse_image = tf.squeeze(parse_image).numpy()
        bbox = tf.squeeze(r_bbox, 0).numpy()
        bbox = bbox.astype(np.int)
        r_labels = tf.squeeze(r_labels, 0).numpy()
        # print("after int: ", bbox)
        show_images(parse_image, bbox, 'fd', r_labels)