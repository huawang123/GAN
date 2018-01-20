import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
from model_vae import *

learning_rate = 0.0004
img_width = 28
img_height = 28
depth = 1
z_dim = 100
batch_size = 32
max_epoch = 100
weiht_decay = 0.00002
total_samples = 42000
gpu = 3
train_flag = False

# EBGAN Parameter
margin = max(1,batch_size/64.)        # margin for loss function
pt_loss_weight = 0.01

train_data_path = '/home/wh/working/train.csv'
log_path = '/storage/wanghua/kaggle/log/gan_mnist_eb/'
restore_checkpoint = '/storage/wanghua/kaggle/filelist/'
output_dir = log_path + 'gengrate_images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# borrowed from https://github.com/shekkizh/EBGAN.tensorflow/blob/master/EBGAN/Faces_EBGAN.py
def pullaway_loss(embeddings):
    """
    Pull Away loss calculation
    :param embeddings: The embeddings to be orthogonalized for varied faces. Shape [batch_size, embeddings_dim]
    :return: pull away term loss
    """
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    similarity = tf.matmul(
        normalized_embeddings, normalized_embeddings, transpose_b=True)
    bs = tf.cast(tf.shape(embeddings)[0], tf.float32)
    pt_loss = (tf.reduce_sum(similarity) - bs) / (bs * (bs - 1))
    return pt_loss

def tr():
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # input
    image_dims = [img_width, img_height, depth]
    input = tf.placeholder(tf.float32, [batch_size] + [img_width * img_height * depth], 'real_data')
    inputss = tf.reshape(input, [batch_size] + image_dims, 'rgb')
    inputs = tf.cast(inputss, tf.float32) * (1. / 255)

    mu, sigma = encode(inputs, z_dim = 100, is_training=True, reuse=False)
    # sampling by re-parameterization technique
    eplision = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z_re = mu + sigma * eplision
    faka_data = decode(z_re, is_training=True, reuse=False)
    faka_data = tf.clip_by_value(faka_data, 1e-8, 1 - 1e-8)

    # loss
    Reconstruction_loss = tf.reduce_sum(-inputs * tf.log(faka_data) - (1 - inputs) * tf.log(1 - faka_data), [1, 2])
    #重构误差也可以用mse
    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1,[1])
    neg_loglikelihood = tf.reduce_mean(Reconstruction_loss)
    KL_divergence = tf.reduce_mean(KL_divergence)
    loss = neg_loglikelihood + KL_divergence

    t_vars = tf.trainable_variables()
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate,beta1=0.5)
    # trainer
    trainer = optimizer.minimize(loss, var_list=t_vars)

    nll_sum = tf.summary.scalar("neg_loglikelihood", neg_loglikelihood)
    kl_sum = tf.summary.scalar("KL_divergence", KL_divergence)
    loss_sum = tf.summary.scalar("loss", loss)
    sum_list = prams_summaries_all()
    sum = tf.summary.merge([loss_sum,kl_sum,nll_sum])

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
                    ls,summary_str, _, nll, kl  = sess.run([loss, sum, trainer, neg_loglikelihood, KL_divergence], feed_dict={input:x})
                    train_writer.add_summary(summary_str, epoch * setps + step)
                    print('[Epoch: %s]  Step: %s  Loss: %s---------Neg_loglikelihood: %s-----------KL_divergence: %s' % (epoch, step, ls, nll, kl))
                     # tmp = view_samples(-3, np.squeeze(pl), (4, 8), output_dir)
                [im] = sess.run([faka_data], feed_dict={input:x})
                tmp = view_samples(epoch,np.squeeze(im),(4,8), output_dir)
                show_images_path.append(tmp)
                save_model(saver, sess, log_path, epoch, dl=ls, gl=0)
            gen_gif(show_images_path, output_dir)

def te():
    p = 100#多少张图
    z_prior = tf.placeholder(tf.float32, [p, z_dim], name="z_prior")
    faka_data = decode(z_prior, is_training=False, reuse=False)

    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        count_trainable_params()
        tf.summary.FileWriter(log_path, sess.graph)
        variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'gen')
        saver = tf.train.Saver(variables_to_restore)
        load_model(sess=sess, saver=saver, restore_checkpoint=log_path)
        with tf.device('/gpu:%d' % gpu):
            z_sample_val = gaussian(p, z_dim)
            [im] = sess.run([faka_data], feed_dict={z_prior:z_sample_val})
            _ = view_samples(-1, np.squeeze(im),(10,10), output_dir)

if train_flag:
    tr()
else:
    te()
