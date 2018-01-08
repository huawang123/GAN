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