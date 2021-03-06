import tensorflow as tf
from utils import conv2d, deconv2d, bn, linear, lrelu
def discriminator(x, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    batch_size = x.get_shape().as_list()[0]
    with tf.variable_scope("discriminator", reuse=reuse):
        net = conv2d(x, output_dim=64, kernel=(4, 4), stride=(2, 2), activation='lrelu', name='conv1')
        net = conv2d(net, output_dim=128, kernel=(4, 4), stride=(2, 2), activation='lrelu', use_bn=True, is_training=is_training, name='conv2')
        net = tf.reshape(net, [batch_size, -1])
        net = lrelu(bn(linear(net, 1024, scope='d_fc3'), is_training=is_training, scope='linear1'))
        out_logit = linear(net, 1, scope='linear2')
        out = tf.nn.sigmoid(out_logit)
        return out, out_logit, net

def generator(z, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("generator", reuse=reuse):
        net = tf.nn.relu(bn(linear(z, 1024, scope='g_fc1'), is_training=is_training, scope='linear1'))
        net = tf.nn.relu(bn(linear(net, 128 * 7 * 7, scope='g_fc2'), is_training=is_training, scope='linear2'))
        net = tf.reshape(net, [-1, 7, 7, 128])
        batch_size = net.get_shape().as_list()[0]
        net = deconv2d(net, output_size=14, output_channel=64, kernel=(4,4),stride=(2,2),activation='relu', use_bn=True,is_training=True,name='d_conv1')
        out = deconv2d(net, output_size=28, output_channel=1, kernel=(4, 4), stride=(2, 2), activation='sigmoid', name='gen_images')
        return out