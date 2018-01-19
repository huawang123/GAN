from utils import *
def discriminator(x, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("discriminator", reuse=reuse):
        batch_size = x.get_shape().as_list()[0]
        net = conv2d(x, output_dim=64, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True,is_training=is_training, name='conv1')
        net = conv2d(net, output_dim=64, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=is_training, name='conv2')
        flat_net = conv2d(net, output_dim=64, kernel=(7, 7), stride=(1, 1), padding='VALID', activation='relu', use_bn=True, is_training=is_training, name='conv3')
        net = conv2d(flat_net, output_dim=64*7*7, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True,is_training=is_training, name='d_conv2')
        net = tf.reshape(net, [batch_size, 7, 7, -1])
        net = deconv2d(net, output_size=14, output_channel=64, kernel=(4, 4), stride=(2, 2), activation='relu', use_bn=True, is_training=True, name='d_conv3')
        out = deconv2d(net, output_size=28, output_channel=1, kernel=(4, 4), activation='sigmoid', stride=(2, 2),name='out')
        # recon loss
        recon_error = tf.sqrt(2 * tf.nn.l2_loss(out - x)) / batch_size
        return out, recon_error, flat_net
#
def generator(z, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("generator", reuse=reuse):
        batch_size = z.get_shape().as_list()[0]
        z_label = tf.reshape(z, [batch_size, 1, 1, -1])
        net = conv2d(z_label, output_dim=1024, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True, is_training=is_training, name='conv0')
        net = conv2d(net, output_dim=128 * 7 * 7, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True,is_training=is_training, name='conv1')
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = deconv2d(net, output_size=14, output_channel=128, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=True, name='d_conv1')
        out = deconv2d(net, output_size=28, output_channel=1, kernel=(3, 3), stride=(2, 2), activation='sigmoid', name='gen_images')
        return out