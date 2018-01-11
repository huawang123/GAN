from ut import *

def classify(x,y_dim, is_training=True, reuse=False):
    with tf.variable_scope('classify', reuse=reuse):
        net = conv2d(x, 128, kernel=(1, 1), stride=(1, 1), padding='VALID', use_bn=True, is_training=is_training, name='fc')
        logits = conv2d(net, y_dim, kernel=(1,1),stride=(1,1),padding='VALID',use_bn=True, is_training=is_training,name='logits')
        spares_logits = tf.squeeze(logits, [1,2], name='sapres_logits')
        predict = tf.nn.softmax(spares_logits)
        return predict, spares_logits

def discriminator(x, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("discriminator", reuse=reuse):
        batch_size = x.get_shape().as_list()[0]
        net = conv2d(x, output_dim=64, kernel=(3, 3), stride=(2, 2), activation='lrelu', use_bn=True, is_training=is_training,  name='conv1')
        net = conv2d(net, output_dim=128, kernel=(3, 3), stride=(2, 2), activation='lrelu', use_bn=True, is_training=is_training, name='conv2')
        net = conv2d(net, output_dim=256, kernel=(3, 3), stride=(2, 2), activation='lrelu', use_bn=True, is_training=is_training, name='conv4')
        net = conv2d(net, output_dim=512, kernel=(4, 4), stride=(1, 1), activation='lrelu', padding='VALID', use_bn=True, is_training=is_training, name='conv5')
        out_logit = conv2d(net, output_dim=1, kernel=(1, 1), stride=(1, 1),  name='conv6')
        out = tf.nn.sigmoid(out_logit)
        return out, out_logit, net

def generator(z, y, is_training=True, reuse=False):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    with tf.variable_scope("generator", reuse=reuse):
        z_label = concat([z, y], 1)
        batch_size = z_label.get_shape().as_list()[0]
        z_label = tf.reshape(z_label,[batch_size, 1, 1, -1])
        net = conv2d(z_label, output_dim=1024, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True, is_training=is_training, name='conv0')
        net = conv2d(net, output_dim=128 * 4 * 4, kernel=(1, 1), stride=(1, 1), activation='relu', use_bn=True, is_training=is_training, name='conv1')
        net = tf.reshape(net, [-1, 4, 4, 128])
        net = deconv2d(net, output_size=7, output_channel=128, kernel=(3,3),stride=(2,2),activation='relu', use_bn=True,is_training=True,name='d_conv1')
        net = deconv2d(net, output_size=14, output_channel=128, kernel=(3, 3), stride=(2, 2), activation='relu',use_bn=True, is_training=True, name='d_conv2')
        out = deconv2d(net, output_size=28, output_channel=1, kernel=(3, 3), stride=(2, 2), activation='sigmoid', name='gen_images')
        return out