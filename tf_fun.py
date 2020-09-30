import tensorflow as tf
@tf.function
def fun():
    a = tf.constant(10, tf.float32)
    print(a)
    b = tf.range(a)
    return b


if __name__ == '__main__':
    b = tf.constant(10, tf.float32)
    print(b.shape)