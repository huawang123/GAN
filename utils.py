import pandas as pd
import numpy as np
import tensorflow as tf

def bn(x, is_training, scope):
    return tf.contrib.layers.batch_norm(x,
                                        decay=0.9,
                                        updates_collections=None,
                                        epsilon=1e-5,
                                        scale=True,
                                        is_training=is_training,
                                        scope=scope)


def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)

def like_rgb_label(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape().as_list()
    y_shapes = y.get_shape().as_list()
    y = tf.reshape(y, [y_shapes[0],1, 1, y_shapes[-1]])#[batch_size, 1, 1, num_classes]
    y_ = tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[-1]])#[batch_size, width, height, num_classes]
    return y*y_

def conv2d(input_, output_dim, kernel=(3,3), stride=(2,2),padding='SAME', activation='',use_bn=False, is_training=False, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel[0], kernel[1], input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=0.02))
        conv = tf.nn.conv2d(input_, w, strides=[1, stride[0], stride[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        hidden = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if use_bn:
            hidden = bn(x=hidden, is_training=is_training, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
        elif activation == 'lrelu':
            hidden =  lrelu(hidden)
        elif activation == 'sigmoid':
            hidden =  tf.nn.sigmoid(hidden)
        return hidden

def deconv2d(input_, output_size, output_channel, kernel=(3,3), stride=(2,2),padding='SAME',
             activation='',use_bn=False, is_training=False,  name="deconv2d"):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        batch_size = input_.get_shape().as_list()[0]
        w = tf.get_variable('w', [kernel[0], kernel[1], output_channel, input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=0.02))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=[batch_size,output_size,output_size,output_channel],
                                        strides=[1, stride[0], stride[1], 1], padding=padding)
        biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.0))
        hidden = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if use_bn:
            hidden = bn(x=hidden, is_training=is_training, scope='bn')
        if activation == 'relu':
            hidden =  tf.nn.relu(hidden)
        elif activation == 'lrelu':
            hidden =  lrelu(hidden)
        elif activation == 'sigmoid':
            hidden =  tf.nn.sigmoid(hidden)
        return hidden

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, matrix) + bias

class data_iter(object):
    # filelist: 要生成batch的图像路径和标签的filelist
    # batch_size: 每个batch有多少张图片
    def __init__(self,filelist,batch_size):
        self.batch_size = batch_size
        self.filelist = filelist
        self.num_batches = 0
        self.pointer = 0
        self.x_batches = []
        self.y_batches = []
        self.creat_batches = self.get_csv_batch()
    # 生成相同大小的批次  CSV文件
    def get_csv_batch(self):
        with tf.variable_scope('input'):
            data = pd.read_csv(self.filelist, header=0, dtype=np.int)
            x, y = np.asarray(data.iloc[:, 1:]), data['label']
            self.num_batches = int(len(y) / self.batch_size)
            samples = self.num_batches * self.batch_size
            # for i in range(self.num_batches):
            #     self.x_batches.append(x[i])
            self.x_batches = np.split(x[:samples].reshape(self.batch_size, -1), self.num_batches, 1)
            self.y_batches = np.split(y[:samples].reshape(self.batch_size, -1), self.num_batches, 1)

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, np.squeeze(y)

    def reset_batch_pointer(self):
        self.pointer = 0

def load_model(sess, saver, restore_checkpoint):
    print('Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(restore_checkpoint)
    try:

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Sucessful loading checkpoing...%s' % ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            raise TypeError('no checkpoint in %s' % restore_checkpoint)
    except Exception as e:
        print(e)

def prams_summaries_all():
    # Add summaries for variables.
    prams0 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gen')
    prams1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'dis')
    prams = prams0 + prams1
    sum_list = []
    for p in prams:
        name = tf.summary.histogram(p.op.name, p)
        sum_list.append(name)
    return sum_list

def count_trainable_params():
    total_parameters = 0
    a = []
    for variable in tf.trainable_variables():
        a.append(variable)
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

def model_identifier(model_type):
    print ("training model is : %s " % model_type)

def save_model(saver, sess, logdir, global_stepe, gl, dl):
    save_path = logdir + 'model.ckpt'
    saver.save(sess, save_path, global_step=global_stepe)
    print('\nGen_loss {:.9f} and Dis_loss {:.9f} in step : {})'
          '\ncheckpoint has been saved in : {}'.format(gl, dl, global_stepe, logdir))

def view_samples(epoch, samples,shape, output_dir):
    """
    #用于可视化epoch后输出图片
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    import matplotlib
    # Force matplotlib to not use any Xwindows backend.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, cols = shape[0],shape[1]
    map = []
    for i in range(rows):
        tmp = []
        for j in range(cols):
            tmp.append(((samples[i*cols + j])))
        # tmp = np.hstack(tmp)
        map.append(np.hstack(tmp))
    map = np.asarray(np.vstack(map))
    plt.imshow(map)
    plt.title('Epoch: %s' % epoch)
    out_file = output_dir + '%s.png' % epoch
    plt.savefig(out_file, dpi=300)
    return out_file

def gen_gif(show_images_path, output_dir):
    from PIL import Image
    im = Image.open(show_images_path[0])
    images = []
    for i in range(len(show_images_path)):
        images.append(Image.open(show_images_path[i]))
    im.save(output_dir+'mnist.gif', save_all=True, append_images=images, loop=1, duration=1, comment=b"gen_images")

def gen_pair(file):
    import glob
    data_path = glob.glob(file+'*.jpg')
    label = np.ones(len(data_path),np.int32)
    return data_path,label

def get_image_label_pair(filelist_path):
    # 解析文本文件
    label_image = lambda x: x.strip().split('    ')
    with open(filelist_path) as f:
        label = [int(label_image(line)[0]) for line in f.readlines()]
    with open(filelist_path) as f:
        image_path_list = [label_image(line)[1] for line in f.readlines()]
    return image_path_list, label

def pre_processing(img, Isize, crop_size, method):
    # resize and crop
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize_images(img, [Isize, Isize], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # img = tf.random_crop(img, [crop_size, crop_size, 3])

    # preprocess
    if method == 'default':
        # img = tf.image.random_flip_left_right(img)
        # img = tf.image.random_brightness(img, max_delta=63)
        # img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # img = tf.image.per_image_standardization(img)
        # img = tf.cast(img, tf.float32) * (1. / 255)
        img = tf.reshape(img, [Isize, Isize, 3])

    # keras preprocess module
    return img

# 生成相同大小的批次
def get_batch(image, label, image_W=256, image_H=256, batch_size=32, capacity=256,min_after_dequeue=None, is_training=True):
    # image, label: 要生成batch的图像路径和标签list
    # image_W, image_H: 图片的宽高
    # batch_size: 每个batch有多少张图片
    # capacity: 队列容量
    # return: 图像和标签的batch

    # 将python.list类型转换成tf能够识别的格式
    with tf.variable_scope('input'):
        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.int64)
        # 生成队列
        input_queue = tf.train.slice_input_producer([image, label])
        image_contents = tf.read_file(input_queue[0])
        label = input_queue[1]
        image = tf.image.decode_jpeg(image_contents, channels=3)
        image = pre_processing(image, image_H, image_H, 'default')
        # 统一图片大小
        # image = tf.image.resize_images(image, [image_H, image_W], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.cast(image, tf.float32)
        # image = tf.image.per_image_standardization(image)   # 标准化数据
        if is_training:
            if not min_after_dequeue:
                image_batch, label_batch, filename = tf.train.batch([image, label, input_queue[0]],
                                                batch_size=batch_size,
                                                num_threads=64,   # 线程
                                                capacity=capacity)
            else:
                image_batch, label_batch, filename = tf.train.shuffle_batch([image, label, input_queue[0]],
                                                batch_size=batch_size,
                                                num_threads=64,  # 线程
                                                capacity=capacity + min_after_dequeue,
                                                min_after_dequeue=min_after_dequeue)

        else:
            image_batch, label_batch, filename = tf.train.batch([image, label, input_queue[0]],
                                                  batch_size=batch_size,
                                                  num_threads=64,  # 线程
                                                  capacity=capacity)
        return image_batch, label_batch, filename
