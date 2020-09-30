import tensorflow as tf

from detection.utils.misc import *

class AnchorGenerator(object):
    def __init__(self, 
                 scales=(32, 64, 128, 256, 512), 
                 ratios=(0.5, 1, 2), 
                 feature_strides=(4, 8, 16, 32, 64)):
        '''Anchor Generator
        
        Attributes
        ---
            scales: 1D array of anchor sizes in pixels.
            ratios: 1D array of anchor ratios of width/height.
            feature_strides: Stride of the feature map relative to the image in pixels.
        '''
        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides

    @tf.function
    def generate_pyramid_anchors(self, img_shape):
        '''Generate the multi-level anchors for Region Proposal Network
        
        Args
        ---
            img_metas: [batch_size, 11]
        
        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        '''
        # generate anchors
        feature_shapes = [(img_shape[:, 0] // stride, img_shape[:, 1] // stride)
                          for stride in self.feature_strides]
        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ]
        anchors = tf.concat(anchors, axis=0)
        return anchors

        # generate valid flags

        # valid_flags = [
        #     self._generate_valid_flags(anchors, img_shape[i, 0:2])
        #     for i in range(img_shape.shape[0])
        # ]
        # valid_flags = tf.stack(valid_flags, axis=0)
        #
        # return anchors, valid_flags

    def _generate_valid_flags(self, anchors, img_shape):
        '''
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            img_shape: Tuple. (height, width, channels)
            
        Returns
        ---
            valid_flags: [num_anchors]
        '''
        # img_shape = tf.cast(img_shape, tf.float32)
        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 3] + anchors[:, 1]) / 2

        print(anchors.shape[0])

        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)
        
        valid_flags = tf.where(y_center <= img_shape[0], valid_flags, zeros)
        valid_flags = tf.where(x_center <= img_shape[1], valid_flags, zeros)
        
        return valid_flags
    
    def _generate_level_anchors(self, level, feature_shape):
        '''Generate the anchors given the spatial shape of feature map.
        
        Args
        ---
            feature_shape: (height, width)

        Returns
        ---
            numpy.ndarray [anchors_num, (y1, x1, y2, x2)]
        '''
        scale = self.scales[level]
        ratios = self.ratios
        feature_stride = self.feature_strides[level]
        # print(type(feature_shape[0]))
        # Get all combinations of scales and ratios
        scales, ratios = tf.meshgrid([float(scale)], ratios)
        scales = tf.reshape(scales, [-1])
        ratios = tf.reshape(ratios, [-1])
        
        # Enumerate heights and widths from scales and ratios
        heights = scales / tf.sqrt(ratios)
        widths = scales * tf.sqrt(ratios)

        a1 = tf.squeeze(feature_shape[0], axis=0)
        a2 = tf.squeeze(feature_shape[1], axis=0)
        print(a1)
        print(a2)
        shifts_y = tf.multiply(tf.range(a1), feature_stride)
        shifts_x = tf.multiply(tf.range(a2), feature_stride)

        # shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        # shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)
        
        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=2), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=2), (-1, 2))

        # Convert to corner coordinates (y1, x1, y2, x2)
        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)
        return boxes


if __name__=='__main__':
    generator = AnchorGenerator(
        scales=(32, 64, 128, 256, 512),
        ratios=(0.5, 1, 2),
        feature_strides=(4, 8, 16, 32, 64))
    img_metas = tf.constant([[512,5123,3]], tf.float32)
    anchors = generator.generate_pyramid_anchors(img_metas)
    print(anchors)
    # tf.Tensor([128.], shape=(1,), dtype=float32)
    # tf.Tensor([64.], shape=(1,), dtype=float32)
    # tf.Tensor([32.], shape=(1,), dtype=float32)
    # tf.Tensor([16.], shape=(1,), dtype=float32)
    # tf.Tensor([8.], shape=(1,), dtype=float32)

    # Tensor("floordiv:0", shape=(1,), dtype=float32)