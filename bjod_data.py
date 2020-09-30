import os
import sys
import tensorflow as tf
import numpy as np
import random
import cv2
from tqdm import tqdm


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


class Zipvaluedata:
    def __init__(self, dataset_dir, crop_size=[512, 512, 3]):
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.image_feature_description = {
            'image/height': tf.io.VarLenFeature(tf.int64),
            'image/width': tf.io.VarLenFeature(tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/difficult': tf.io.VarLenFeature(tf.int64),
        }

    def random_crop(self, img, xmin, ymin, xmax, ymax, labels):
        img = img.numpy()
        labels = labels.numpy()
        w_img, h_img, _ = img.shape
        xmin = np.clip(xmin.numpy() * w_img, 0, w_img).astype(np.int)
        ymin = np.clip(ymin.numpy() * h_img, 0, h_img).astype(np.int)
        xmax = np.clip(xmax.numpy() * w_img, 0, w_img).astype(np.int)
        ymax = np.clip(ymax.numpy() * h_img, 0, h_img).astype(np.int)
        cx_max = w_img - self.crop_size[0]
        cy_max = h_img - self.crop_size[1]
        rr_labels = []
        bboxes = []
        rimage_metas = np.array(
            [self.crop_size[0], self.crop_size[1], 3, self.crop_size[0], self.crop_size[1], 3, self.crop_size[0],
             self.crop_size[1], 3, 1, 0])
        if cx_max or cy_max:
            for index, _ in enumerate(range(20)):
                tl_x = random.randint(0, cx_max)
                tl_y = random.randint(0, cy_max)
                r_img = img[tl_x:(self.crop_size[0] + tl_x), tl_y:(self.crop_size[1] + tl_y)]
                for inds in range(len(xmin)):
                    xmin_tmp = xmin[inds] - tl_y
                    ymin_tmp = ymin[inds] - tl_x
                    xmax_tmp = xmax[inds] - tl_y
                    ymax_tmp = ymax[inds] - tl_x
                    if 0 <= xmin_tmp < xmax_tmp < self.crop_size[0] and 0 <= ymin_tmp < ymax_tmp < self.crop_size[1]:
                        bboxes.append([xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp])
                        rr_labels.append(labels[inds])
                if rr_labels:
                    return r_img, bboxes, rr_labels, rimage_metas
                else:
                    continue
            return r_img, bboxes, rr_labels, rimage_metas
        else:
            for inds in range(len(xmin)):
                xmin_tmp = xmin[inds]
                ymin_tmp = ymin[inds]
                xmax_tmp = xmax[inds]
                ymax_tmp = ymax[inds]
                if 0 <= xmin_tmp < xmax_tmp < self.crop_size[0] and 0 <= ymin_tmp < ymax_tmp < self.crop_size[1]:
                    bboxes.append([xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp])
                    rr_labels.append(labels[inds])
            return img, bboxes, rr_labels, rimage_metas

    def bbox_convert(self, r_bbox, r_labels):
        # 将bbox的形状统一化为1000*4
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
        x_image = tf.io.decode_png(image_features['image/encoded'], 3)
        xmin_list = tf.sparse.to_dense(image_features['image/object/bbox/xmin'])
        xmax_list = tf.sparse.to_dense(image_features['image/object/bbox/xmax'])
        ymin_list = tf.sparse.to_dense(image_features['image/object/bbox/ymin'])
        ymax_list = tf.sparse.to_dense(image_features['image/object/bbox/ymax'])
        label_list = tf.sparse.to_dense(image_features['image/object/class/label'])
        filename = image_features['image/filename']
        # rimage_metas = tf.cast([512, 512, 3, 512, 512, 3, 512, 512, 3, 1, 0], tf.float32)
        parse_image, r_bbox, r_labels, rimage_metas = tf.py_function(self.random_crop,
                                                                     inp=[x_image, xmin_list, ymin_list, xmax_list,
                                                                          ymax_list, label_list],
                                                                     Tout=[tf.uint8, tf.int64, tf.int64, tf.float32])
        r_bbox, r_labels = tf.py_function(self.bbox_convert, inp=[r_bbox, r_labels], Tout=[tf.int64, tf.int32])
        parse_image = tf.cast(parse_image, tf.float32)
        r_bbox = tf.cast(r_bbox, tf.float32)
        r_labels = tf.cast(r_labels, tf.int32)
        parse_image = parse_image / 255.0
        return parse_image, rimage_metas, r_bbox, r_labels, filename

    def prepare(self, data):
        parse_fn = lambda x: self.parse_image_function(x)
        val_ds = tf.data.TFRecordDataset(os.path.join(self.dataset_dir, data)).map(parse_fn,
                                                                                   num_parallel_calls=-1)
        val_ds = val_ds.batch(1).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return val_ds


class ZiptrainDataset:
    def __init__(self, dataset_dir, batch_size, in_sampe_size, crop_size=[512, 512, 3], augment=True, roi_path=''):
        self.dataset_dir = dataset_dir
        self.sample_size = in_sampe_size
        self.augment = augment
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.roi_path = roi_path
        self.image_feature_description = {
            'image/height': tf.io.VarLenFeature(tf.int64),
            'image/width': tf.io.VarLenFeature(tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
            'image/object/difficult': tf.io.VarLenFeature(tf.int64),
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

    def random_crop(self, img, xmin, ymin, xmax, ymax, labels):
        img = img.numpy()
        labels = labels.numpy()
        w_img, h_img, _ = img.shape
        xmin = np.clip(xmin.numpy() * w_img, 0, w_img).astype(np.int)
        ymin = np.clip(ymin.numpy() * h_img, 0, h_img).astype(np.int)
        xmax = np.clip(xmax.numpy() * w_img, 0, w_img).astype(np.int)
        ymax = np.clip(ymax.numpy() * h_img, 0, h_img).astype(np.int)
        cx_max = w_img - self.crop_size[0]
        cy_max = h_img - self.crop_size[1]
        rr_labels = []
        bboxes = []
        rimage_metas = np.array(
            [self.crop_size[0], self.crop_size[1], 3, self.crop_size[0], self.crop_size[1], 3, self.crop_size[0],
             self.crop_size[1], 3, 1, 0])
        if cx_max or cy_max:
            for index, _ in enumerate(range(20)):
                tl_x = random.randint(0, cx_max)
                tl_y = random.randint(0, cy_max)
                r_img = img[tl_x:(self.crop_size[0] + tl_x), tl_y:(self.crop_size[1] + tl_y)]
                for inds in range(len(xmin)):
                    xmin_tmp = xmin[inds] - tl_y
                    ymin_tmp = ymin[inds] - tl_x
                    xmax_tmp = xmax[inds] - tl_y
                    ymax_tmp = ymax[inds] - tl_x
                    if 0 <= xmin_tmp < xmax_tmp < self.crop_size[0] and 0 <= ymin_tmp < ymax_tmp < self.crop_size[1]:
                        bboxes.append([xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp])
                        rr_labels.append(labels[inds])
                if rr_labels:
                    return r_img, bboxes, rr_labels, rimage_metas
                else:
                    continue
            return r_img, bboxes, rr_labels, rimage_metas
        else:
            for inds in range(len(xmin)):
                xmin_tmp = xmin[inds]
                ymin_tmp = ymin[inds]
                xmax_tmp = xmax[inds]
                ymax_tmp = ymax[inds]
                if 0 <= xmin_tmp < xmax_tmp < self.crop_size[0] and 0 <= ymin_tmp < ymax_tmp < self.crop_size[1]:
                    bboxes.append([xmin_tmp, ymin_tmp, xmax_tmp, ymax_tmp])
                    rr_labels.append(labels[inds])
            return img, bboxes, rr_labels, rimage_metas

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

    def bbox_fill_zeros(self, r_bbox, r_labels, num_labels=100):
        # 将bbox的形状统一化为1000*4
        if r_bbox.numpy().shape[0]:
            zeros_tmp = tf.zeros([num_labels, 4], tf.int64)
            r_bbox = tf.concat([r_bbox, zeros_tmp], axis=0)
            r_bbox = tf.slice(r_bbox, [0, 0], [num_labels, 4])
            r_bbox = tf.cast(r_bbox, tf.float32)
            labes_tmp = tf.cast(tf.fill([num_labels], -1), tf.int64)
            r_labels = tf.concat([r_labels, labes_tmp], axis=0)
            r_labels = tf.slice(r_labels, [0], [num_labels])
            r_labels = tf.cast(r_labels, tf.int32)
        else:
            r_bbox = tf.zeros([num_labels, 4], tf.float32)
            r_labels = tf.cast(tf.fill([num_labels], -1), tf.int32)
        r_bbox = r_bbox.numpy()
        # if r_bbox.shape[0]:
        cc = np.hsplit(r_bbox, 4)
        dd = [cc[1], cc[0], cc[3], cc[2]]
        r_bbox = np.hstack(dd)
        return r_bbox, r_labels

    def mixup_bbox(self, images, in_bbox, in_labels):
        if self.roi_path == '':
            return images, in_bbox, in_labels
        images = images.numpy()
        img_w = images.shape[1]
        img_h = images.shape[0]
        out_bbox = in_bbox.numpy()
        out_labels = in_labels.numpy()
        roi_files = os.listdir(self.roi_path)
        num_flags = random.randint(3, 8)
        file_list = random.sample(roi_files, num_flags)
        for indx in range(num_flags):
            tmp_bbox = []
            tmp_labels = []
            roi_img = cv2.imread(os.path.join(self.roi_path, file_list[indx]))
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            roi_w = roi_img.shape[1]
            roi_h = roi_img.shape[0]
            #确保没有目标之间得重叠
            while True:
                colaps_flags = False
                xmin = random.randint(0, img_w - roi_w)
                ymin = random.randint(0, img_h - roi_h)
                for orignal_bbox in out_bbox:
                    # print(orignal_bbox)
                    ixmin = max(xmin, orignal_bbox[0])
                    iymin = max(ymin, orignal_bbox[1])
                    ixmax = min(xmin+roi_w, orignal_bbox[2])
                    iymax = min(ymin+roi_h, orignal_bbox[3])
                    if ixmin < ixmax and iymin < iymax:
                        colaps_flags = True
                if colaps_flags:
                    continue
                else:
                    images[ymin:(ymin + roi_h), xmin:(xmin + roi_w)] = roi_img
                    tmp_bbox.append([xmin, ymin, xmin + roi_w, ymin + roi_h])
                    tmp_labels.append(1)
                    if out_labels.size == 0:
                        out_bbox = np.array(tmp_bbox)
                        out_labels = np.array(tmp_labels)
                    else:
                        out_bbox = np.concatenate([out_bbox, np.array(tmp_bbox)], axis=0)
                        out_labels = np.concatenate([out_labels, np.array(tmp_labels)], axis=0)
                    break
        return images, out_bbox, out_labels

    def parse_image_function(self, example_proto):
        image_features = tf.io.parse_single_example(example_proto, self.image_feature_description)
        x_image = tf.io.decode_png(image_features['image/encoded'], 3)
        xmin_list = tf.sparse.to_dense(image_features['image/object/bbox/xmin'])
        xmax_list = tf.sparse.to_dense(image_features['image/object/bbox/xmax'])
        ymin_list = tf.sparse.to_dense(image_features['image/object/bbox/ymin'])
        ymax_list = tf.sparse.to_dense(image_features['image/object/bbox/ymax'])
        label_list = tf.sparse.to_dense(image_features['image/object/class/label'])
        filename = image_features['image/filename']
        # rimage_metas = tf.cast([512, 512, 3, 512, 512, 3, 512, 512, 3, 1, 0], tf.float32)
        parse_image, r_bbox, r_labels, rimage_metas = tf.py_function(self.random_crop,
                                                                     inp=[x_image, xmin_list, ymin_list, xmax_list,
                                                                          ymax_list, label_list],
                                                                     Tout=[tf.uint8, tf.int64, tf.int64, tf.float32])
        parse_image, r_bbox, r_labels = tf.py_function(self.mixup_bbox,
                                                       inp=[parse_image, r_bbox, r_labels],
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
        r_bbox, r_labels = tf.py_function(self.bbox_convert, inp=[r_bbox, r_labels], Tout=[tf.int64, tf.int32])
        # r_bbox = tf.squeeze(r_bbox, axis=0)
        # r_labels = tf.squeeze(r_labels, axis=0)
        # r_bbox, r_labels= tf.py_function(self.bbox_fill_zeros, inp=[r_bbox, r_labels, 100], Tout=[tf.int64, tf.int32])
        # r_bbox = tf.squeeze(r_bbox, axis=0)
        # r_labels = tf.squeeze(r_labels, axis=0)
        parse_image = tf.cast(parse_image, tf.float32)
        parse_image = parse_image / 255.0
        r_bbox = tf.cast(r_bbox, tf.float32)
        return parse_image, rimage_metas, r_bbox, r_labels, filename

    def prepare(self, train_aug=True):
        parse_fn = lambda x: self.parse_image_function(x)
        self.augment = train_aug
        train_ds = tf.data.TFRecordDataset(os.path.join(self.dataset_dir, 'train_api_97.record')).map(parse_fn,
                                                                                                      num_parallel_calls=-1)
        train_ds = train_ds.skip(random.randint(0, 97 - self.sample_size)).take(self.sample_size)
        train_ds = train_ds.shuffle(100).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds


if __name__ == '__main__':
    tf_record_path = 'D:/datasets/bjod/'
    train_datasets = ZiptrainDataset(tf_record_path, 1, 96, crop_size=[992, 992, 3],
                                    roi_path='D:/datasets/bjod/roi_test/').prepare(True)

    val_datasets = Zipvaluedata(tf_record_path,[992,992,3]).prepare('val_api_19.record')
    a = 0
    for parse_image, rimage_metas, r_bbox, r_labels, file_name in tqdm(val_datasets):
        parse_image = parse_image.numpy()
        parse_image = np.squeeze(parse_image, axis=0)
        # file_name = file_name.numpy().decode('utf-8')
        r_bbox = r_bbox.numpy()
        r_bbox = np.squeeze(r_bbox, axis=0)
        parse_image = cv2.cvtColor(parse_image, cv2.COLOR_RGB2BGR)
        print(len(r_bbox))
        for i in range(len(r_bbox)):
            cv2.rectangle(parse_image, (r_bbox[i][1], r_bbox[i][0]), (r_bbox[i][3], r_bbox[i][2]), (255, 0, 255), 1)
        cv2.namedWindow('r', cv2.WINDOW_FREERATIO)
        cv2.imshow('r', parse_image)
        cv2.waitKey(0)
