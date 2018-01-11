import tensorflow as tf
import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
from ut import *
from model_cgan import *

learning_rate = 0.0004
img_width = 28
img_height = 28
depth = 1
z_dim = 100
batch_size = 32
num_classes = 10
other_character = 2
y_dim = num_classes + other_character
len_discrete_code = num_classes
max_epoch = 25
weiht_decay = 0.00002
total_samples = 42000
gpu = 0
train_flag = False
SUPERVISED = True
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
    label = tf.placeholder(tf.float32, [batch_size, y_dim], name='label')
    # label = tf.one_hot(spares_label, depth=num_classes, name='label')
    z_prior = tf.placeholder(tf.float32, [batch_size, z_dim], name="z_prior")

    d_real, d_real_logits, real_fc = discriminator(inputs, is_training=True, reuse=False)
    faka_data = generator(z_prior, label, is_training=True, reuse=False)
    d_fake, d_fake_logits, dake_fc = discriminator(faka_data, is_training=True, reuse=True)
    # predict, real_spares_logits = classify(real_fc, is_training=True, reuse=False)
    predict, fake_spares_logits = classify(dake_fc, y_dim, is_training=True, reuse=False)

    # divide trainable variables into a group for D and a group for G
    t_vars = tf.trainable_variables()
    d_params = [var for var in t_vars if 'dis' in var.name]
    g_params = [var for var in t_vars if 'gen' in var.name]
    q_params = t_vars

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake)))
    d_l2_loss = tf.add_n([weiht_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    dis_loss = d_loss_real + d_loss_fake + d_l2_loss

    g_l2_loss = tf.add_n([weiht_decay * tf.nn.l2_loss(var) for var in g_params])
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.ones_like(d_fake))) + g_l2_loss

    # discrete code : categorical
    # 记录前十个信息量为mnist数字类别
    disc_code_est = fake_spares_logits[:, : len_discrete_code]
    # 记录标签的类别,分为有监督和无监督两类，由SUPERVISED决定
    disc_code_tg = label[:, :len_discrete_code]
    # q_disc_loss=disc_code_tg*(-log(sigmoid(disc_code_est)))+(1-disc_code_tg)*(-log(1-sigmoid(disc_code_est)))
    q_disc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_code_est, labels=disc_code_tg))

    # continuous code : gaussian
    # 计算除交叉商的高斯函数部分损失，用于完善q_loss
    cont_code_est = fake_spares_logits[:, len_discrete_code:]
    cont_code_tg = label[:, len_discrete_code:]
    q_cont_loss = tf.reduce_mean(tf.reduce_sum(tf.square(cont_code_tg - cont_code_est), axis=1))
    # get information loss
    # 类别损失与特征损失求和
    q_loss = q_disc_loss + q_cont_loss

    # optimizer
    optimizer_g = tf.train.AdamOptimizer(learning_rate,beta1=0.5)
    optimizer_d = tf.train.AdamOptimizer(learning_rate,beta1=0.5)
    optimizer_c = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    # trainer
    d_trainer = optimizer_d.minimize(dis_loss, var_list=d_params)
    g_trainer = optimizer_g.minimize(gen_loss, var_list=g_params)
    q_trainer = optimizer_c.minimize(q_loss, var_list=q_params)

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", dis_loss)
    g_loss_sum = tf.summary.scalar("g_loss", gen_loss)

    q_real_loss_sum = tf.summary.scalar("q_disc_loss", q_disc_loss)
    q_fake_loss_sum = tf.summary.scalar("q_cont_loss", q_cont_loss)
    q_loss_sum = tf.summary.scalar("q_loss", q_loss)
    g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    c_sum = tf.summary.merge([q_real_loss_sum, q_fake_loss_sum, q_loss_sum])

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
                    y = np.asarray(np.eye(num_classes)[y],dtype=float)
                    if not SUPERVISED:
                        y = np.random.multinomial(1, len_discrete_code * [float(1.0 / len_discrete_code)], size=[batch_size])
                    batch_codes = np.concatenate((y, np.random.uniform(-1, 1, size=(batch_size, other_character))), axis=1)
                    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_dim)).astype(np.float32)
                    dl,summary_str_d, _  = sess.run([dis_loss,d_sum, d_trainer], feed_dict={input:x, label:batch_codes, z_prior:z_sample_val})
                    train_writer.add_summary(summary_str_d, epoch*setps+step)
                    print('[Epoch: %s]  Step: %s  Dis_loss: %s' % (epoch, step, dl))

                    if step % 10 == 0:
                        for j in range(10):
                            z_sample_val = np.random.normal(0, 1, size=(batch_size, z_dim)).astype(np.float32)
                            gl,summary_str_g, _, cl,summary_str_c, _ = sess.run([gen_loss,g_sum, g_trainer, q_loss, c_sum, q_trainer],
                                                         feed_dict={input:x, label:batch_codes, z_prior:z_sample_val})
                            train_writer.add_summary(summary_str_g, epoch * setps + step)
                            train_writer.add_summary(summary_str_c, epoch * setps + step)
                            print('[Epoch: %s]  Step: %s  -------------Gen_loss:  %s-------------Q_loss: %s' % (epoch, step, gl, cl))
                            # tmp = view_samples(-3, np.squeeze(pl), (4, 8), output_dir)
                z_sample_val = np.random.normal(0, 1, size=(batch_size, z_dim)).astype(np.float32)
                [im] = sess.run([faka_data], feed_dict={input:x, label:batch_codes, z_prior:z_sample_val})
                tmp = view_samples(epoch,np.squeeze(im),(4,8), output_dir)
                show_images_path.append(tmp)
                save_model(saver, sess, log_path, epoch, gl, dl)
            gen_gif(show_images_path, output_dir)

def te():
    p = 100#多少张图
    label = tf.placeholder(tf.float32, [p, y_dim], name='spares_label')
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
        y = np.tile(np.arange(num_classes), [10])
        y =  np.asarray(np.eye(num_classes)[y],dtype=float)
        other_char = np.ones((p, other_character))
        other_char[:,0]=0
        batch_codes = np.concatenate((y, other_char), axis=1)
        with tf.device('/gpu:%d' % gpu):
            z_sample_val = np.random.normal(0, 1, size=(p, z_dim)).astype(np.float32)
            [im] = sess.run([faka_data], feed_dict={label:batch_codes, z_prior:z_sample_val})
            _ = view_samples(101, np.squeeze(im),(10,10), output_dir)

if train_flag:
    tr()
else:
    te()