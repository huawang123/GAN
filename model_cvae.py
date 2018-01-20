from utils import *
def encode(x, y, z_dim, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("discriminator", reuse=reuse):
        rgb_label = like_rgb_label(x, y)
        image = concat([x, rgb_label], 3)
        net = conv2d(image, output_dim=64, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True,is_training=is_training, name='conv1')
        net = conv2d(net, output_dim=128, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=is_training, name='conv2')
        net = conv2d(net, output_dim=256, kernel=(7, 7), stride=(1, 1), padding='VALID', activation='relu', use_bn=True, is_training=is_training, name='conv3')
        gaussian_params = tf.squeeze(conv2d(net, output_dim=2 * z_dim, kernel=(1, 1), stride=(1, 1), name='liner'))
        # The mean parameter is unconstrained
        mean = gaussian_params[:, :z_dim]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, z_dim:])
        return mean, stddev
# Bernoulli decoder
def decode(z, y, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("generator", reuse=reuse):
        z_label = concat([z, y], 1)
        batch_size = z_label.get_shape().as_list()[0]
        z_label = tf.reshape(z_label, [batch_size, 1, 1, -1])
        net = conv2d(z_label, output_dim=1024, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True, is_training=is_training, name='conv0')
        net = conv2d(net, output_dim=128 * 7 * 7, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True,is_training=is_training, name='conv1')
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = deconv2d(net, output_size=14, output_channel=128, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=True, name='d_conv1')
        out = deconv2d(net, output_size=28, output_channel=1, kernel=(3, 3), stride=(2, 2), activation='sigmoid', name='gen_images')
        return out