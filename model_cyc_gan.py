from ut import *

def build_resnet_block(x,dim,is_training,name):
    with tf.variable_scope(name):
        net = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        net = conv2d(net,output_dim=dim,kernel=(3,3),stride=(1,1),padding='VALID', activation='relu', use_bn=True,is_training=is_training, name='conv1')
        net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out = conv2d(net, output_dim=dim, kernel=(3, 3), stride=(1, 1), padding='VALID', use_bn=True,is_training=is_training, name='conv2')
        return out + x

def build_generator_resnet_6blocks(x, is_training=True, reuse=False, name='gen'):
    with tf.variable_scope(name,reuse=reuse):
        net = conv2d(x, output_dim=32, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=is_training, name='conv1')
        net = conv2d(net, output_dim=64, kernel=(3, 3), stride=(2, 2),  activation='relu', use_bn=True, is_training=is_training, name='conv2')
        encode = conv2d(net, output_dim=128, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=is_training, name='conv3')

        o_r1 = build_resnet_block(encode, 128,is_training=is_training, name="r1")
        o_r2 = build_resnet_block(o_r1, 128,is_training=is_training, name="r2")
        o_r3 = build_resnet_block(o_r2, 128,is_training=is_training, name="r3")
        o_r4 = build_resnet_block(o_r3, 128,is_training=is_training, name="r4")
        o_r5 = build_resnet_block(o_r4, 128,is_training=is_training, name="r5")
        o_r6 = build_resnet_block(o_r5, 128,is_training=is_training, name="r6")

        net = deconv2d(o_r6, output_size=64, output_channel=128, kernel=(3, 3), stride=(2, 2), activation='relu',use_bn=True, is_training=True, name='d_conv1')
        net = deconv2d(net, output_size=128, output_channel=128, kernel=(3, 3), stride=(2, 2), activation='relu',use_bn=True, is_training=True, name='d_conv2')
        out = deconv2d(net, output_size=256, output_channel=3, kernel=(3, 3), stride=(2, 2), name='gen_images')

        # Adding the tanh layer
        out_gen = tf.nn.tanh(out, "t1")
        return out_gen

def build_gen_discriminator(x, is_training=True, reuse=False, name='dis'):
    with tf.variable_scope(name, reuse=reuse):
        net = conv2d(x, output_dim=64, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True,is_training=is_training, name='conv1')
        net = conv2d(net, output_dim=128, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True, is_training=is_training, name='conv2')
        net = conv2d(net, output_dim=128, kernel=(3, 3), stride=(2, 2), activation='relu', use_bn=True,is_training=is_training, name='conv3')
        net = conv2d(net, output_dim=64, kernel=(4, 4), stride=(1, 1), activation='relu', use_bn=True, is_training=is_training, name='fc')
        logit = tf.squeeze(conv2d(net, output_dim=1, kernel=(1, 1), stride=(1, 1), name='liner'))
        return logit
