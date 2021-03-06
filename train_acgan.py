import tensorflow as tf
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
from ut import *
from model_acgan import *

gpu = 1
learning_rate = 0.0004
img_width = 28
img_height = 28
depth = 1
z_dim = 100
batch_size = 32
num_classes = 10
max_epoch = 25
weiht_decay = 0.00002
total_samples = 42000
train_flag = False
train_data_path = '/home/wh/working/train.csv'
log_path = '/storage/wanghua/kaggle/log/gan_mnist_ac/'
restore_checkpoint = '/storage/wanghua/kaggle/filelist/'
output_dir = log_path + 'gengrate_images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def tr():
    global_step = tf.Variable(0, name="global_step", trainable=False)

    # input

    image_dims = [img_width, img_height, depth]
    input = tf.placeholder(tf.float32, [batch_size] + [img_width * img_height * depth], 'real_data')
    inputss = tf.reshape(input, [batch_size] + image_dims, 'rgb')
    inputs = tf.cast(inputss, tf.float32) * (1. / 255)
    spares_label = tf.placeholder(tf.int32, [batch_size, ], name='spares_label')
    label = tf.one_hot(spares_label, depth=num_classes, name='label')
    z_prior = tf.placeholder(tf.float32, [batch_size, z_dim], name="z_prior")

    d_real, d_real_logits, real_fc = discriminator(inputs, is_training=True, reuse=False)
    faka_data = generator(z_prior, label, is_training=True, reuse=False)
    d_fake, d_fake_logits, dake_fc = discriminator(faka_data, is_training=True, reuse=True)
    predict, real_spares_logits = classify(real_fc, is_training=True, reuse=False)
    predict, fake_spares_logits = classify(dake_fc, is_training=True, reuse=True)

    # divide trainable variables into a group for D and a group for G
    t_vars = tf.trainable_variables()
    d_params = [var for var in t_vars if 'dis' in var.name]
    g_params = [var for var in t_vars if 'gen' in var.name]
    c_params = t_vars

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake)))
    d_l2_loss = tf.add_n([weiht_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    dis_loss = d_loss_real + d_loss_fake + d_l2_loss

    g_l2_loss = tf.add_n([weiht_decay * tf.nn.l2_loss(var) for var in g_params])
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake))) + g_l2_loss

    c_real_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=real_spares_logits))
    c_fake_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=fake_spares_logits))
    c_l2_loss = tf.add_n([0.1*weiht_decay * tf.nn.l2_loss(var) for var in c_params])
    c_loss = c_fake_loss + c_real_loss + c_l2_loss

    # optimizer
    optimizer_g = tf.train.AdamOptimizer(learning_rate,beta1=0.5)
    optimizer_d = tf.train.AdamOptimizer(learning_rate,beta1=0.5)
    optimizer_c = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    # trainer
    d_trainer = optimizer_d.minimize(dis_loss, var_list=d_params)
    g_trainer = optimizer_g.minimize(gen_loss, var_list=g_params)
    c_trainer = optimizer_c.minimize(c_loss, var_list=c_params)

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", dis_loss)
    g_loss_sum = tf.summary.scalar("g_loss", gen_loss)

    c_real_loss_sum = tf.summary.scalar("c_real_loss", c_real_loss)
    c_fake_loss_sum = tf.summary.scalar("c_fake_loss", c_fake_loss)
    c_loss_sum = tf.summary.scalar("c_loss", c_loss)
    g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    c_sum = tf.summary.merge([c_real_loss_sum, c_fake_loss_sum, c_loss_sum])

    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        count_trainable_params()
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver()
        load_model(sess=sess, saver=saver, restore_checkpoint=restore_checkpoint)
        with tf.device('/gpu:%d' % gpu):
            show_images_path = []
            for epoch in range(max_epoch):
                data_load = data_iter('train.csv', batch_size)
                setps = int(total_samples / batch_size)
                for step in range(setps):
                    x, y = data_load.next_batch()
                    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_dim)).astype(np.float32)
                    dl,summary_str_d, _  = sess.run([dis_loss,d_sum, d_trainer], feed_dict={input:x, spares_label:y, z_prior:z_sample_val})
                    train_writer.add_summary(summary_str_d, epoch*setps+step)
                    print('[Epoch: %s]  Step: %s  Dis_loss: %s' % (epoch, step, dl))

                    if step % 10 == 0:
                        for j in range(10):
                            z_sample_val = np.random.normal(0, 1, size=(batch_size, z_dim)).astype(np.float32)
                            gl,summary_str_g, _, cl,summary_str_c, _ = sess.run([gen_loss,g_sum, g_trainer, c_loss, c_sum, c_trainer],
                                                         feed_dict={input:x, spares_label:y, z_prior:z_sample_val})
                            train_writer.add_summary(summary_str_g, epoch * setps + step)
                            train_writer.add_summary(summary_str_c, epoch * setps + step)
                            print('[Epoch: %s]  Step: %s  -------------Gen_loss:  %s-------------C_loss: %s' % (epoch, step, gl, cl))
                            # tmp = view_samples(-3, np.squeeze(pl), (4, 8), output_dir)
                z_sample_val = np.random.normal(0, 1, size=(batch_size, z_dim)).astype(np.float32)
                [im] = sess.run([faka_data], feed_dict={input:x, spares_label:y, z_prior:z_sample_val})
                tmp = view_samples(epoch,np.squeeze(im),(4,8), output_dir)
                show_images_path.append(tmp)
                save_model(saver, sess, log_path, epoch, gl, dl)
            gen_gif(show_images_path, output_dir)

def te():
    p = 100#多少张图
    spares_label = tf.placeholder(tf.int32, [p, ], name='spares_label')
    label = tf.one_hot(spares_label, depth=num_classes, name='label')
    z_prior = tf.placeholder(tf.float32, [p, z_dim], name="z_prior")
    faka_data = generator(z_prior, label, is_training=False, reuse=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        count_trainable_params()
        tf.summary.FileWriter(log_path, sess.graph)
        variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'gen')
        saver = tf.train.Saver(variables_to_restore)
        load_model(sess=sess, saver=saver, restore_checkpoint=log_path)
        y = np.tile(np.arange(10), [10])
        with tf.device('/gpu:%d' % gpu):
            z_sample_val = np.random.normal(0, 1, size=(p, z_dim)).astype(np.float32)
            [im] = sess.run([faka_data], feed_dict={spares_label:y, z_prior:z_sample_val})
            _ = view_samples(-1, np.squeeze(im),(10,10), output_dir)

if train_flag:
    tr()
else:
    te()
