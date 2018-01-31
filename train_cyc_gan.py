import os
from scipy.misc import imsave
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
from model_began import *

learning_rate = 0.0002
img_width = 256
img_height = 256
img_layer = 3
batch_size = 1
max_epoch = 100

gpu = 3
pool_size = 25
max_images=100
train_flag = False

A_train_data_path = '/home/wh/cy/input/trainA/'
B_train_data_path = '/home/wh/cy/input/trainB/'
log_path = '/storage/wanghua/kaggle/log/gan_cycle/'
restore_checkpoint = '/storage/wanghua/kaggle/log/gan_cycle/'
output_dir = log_path + 'gengrate_images/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def input_setup():
    '''
    This function basically setup variables for taking image input.
    filenames_A/filenames_B -> takes the list of all training images
    self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
    '''
    import glob
    filenames_A = glob.glob(A_train_data_path + "*.jpg")
    queue_length_A = tf.size(filenames_A)
    filenames_B = glob.glob(B_train_data_path + "*.jpg")
    queue_length_B = tf.size(filenames_B)

    filename_queue_A = tf.train.string_input_producer(filenames_A)
    filename_queue_B = tf.train.string_input_producer(filenames_B)

    image_reader = tf.WholeFileReader()
    _, image_file_A = image_reader.read(filename_queue_A)
    _, image_file_B = image_reader.read(filename_queue_B)

    image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [256, 256]), 127.5), 1)
    image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [256, 256]), 127.5), 1)
    return image_A,image_B

def input_read(sess, image_A, image_B):
    '''
    It reads the input into from the image folder.
    self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
    self.A_input/self.B_input -> Stores all the training images in python list
    '''
    # Loading images into the tensors
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    # num_files_A = sess.run(self.queue_length_A)
    # num_files_B = sess.run(self.queue_length_B)
    A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
    B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
    for i in range(max_images):
        image_tensor = sess.run(image_A)
        if (image_tensor.size == img_height * img_width * batch_size * img_layer):
            A_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))
    for i in range(max_images):
        image_tensor = sess.run(image_B)
        if (image_tensor.size == img_height * img_width * batch_size * img_layer):
            B_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))
    coord.request_stop()
    coord.join(threads)
    fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layer))
    fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layer))
    return A_input,B_input,fake_images_A,fake_images_B

def save_training_images(epoch, i ,fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp,A_input,B_input):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    imsave(output_dir + str(epoch) + "_" + str(i) + ".jpg", ((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
    imsave(output_dir + str(epoch) + "_" + str(i) + ".jpg",((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
    imsave(output_dir + str(epoch) + "_" + str(i) + ".jpg",((cyc_A_temp[0] + 1) * 127.5).astype(np.uint8))
    imsave(output_dir + str(epoch) + "_" + str(i) + ".jpg", ((cyc_B_temp[0] + 1) * 127.5).astype(np.uint8))
    imsave(output_dir + str(epoch) + "_" + str(i) + ".jpg", ((A_input[i][0] + 1) * 127.5).astype(np.uint8))
    imsave(output_dir + str(epoch) + "_" + str(i) + ".jpg",((B_input[i][0] + 1) * 127.5).astype(np.uint8))

def fake_image_pool(num_fakes, fake, fake_pool):
    ''' This function saves the generated image to corresponding pool of images.
    In starting. It keeps on feeling the pool till it is full and then randomly selects an
    already stored image and replace it with new one.'''
    import random
    if (num_fakes < pool_size):
        fake_pool[num_fakes] = fake
        return fake
    else:
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0, pool_size - 1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = fake
            return temp
        else:
            return fake
def tr():
    global_step = tf.Variable(0, name="global_step", trainable=False)
    # input
    image_A, image_B = input_setup()

    input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

    fake_pool_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="fake_pool_A")
    fake_pool_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="fake_pool_B")

    fake_B = build_generator_resnet_6blocks(input_A, is_training=True, reuse=False, name='A2B')
    fake_A = build_generator_resnet_6blocks(input_B, is_training=True, reuse=False, name='B2A')
    logitA = build_gen_discriminator(input_A, is_training=True, reuse=False, name='DA')
    logitB = build_gen_discriminator(input_B, is_training=True, reuse=False, name='DB')

    logitA_fake = build_gen_discriminator(fake_A, is_training=True, reuse=True, name='DA')
    logitB_fake = build_gen_discriminator(fake_B, is_training=True, reuse=True, name='DB')
    cyc_A = build_generator_resnet_6blocks(fake_B, is_training=True, reuse=True, name='B2A')
    cyc_B = build_generator_resnet_6blocks(fake_A, is_training=True, reuse=True, name='A2B')

    logitA_fake_pool = build_gen_discriminator(fake_pool_A, is_training=True, reuse=True, name='DA')
    logitB_fake_pool = build_gen_discriminator(fake_pool_B, is_training=True, reuse=True, name='DB')

    cyc_loss = tf.reduce_mean(tf.abs(input_A - cyc_A)) + tf.reduce_mean(tf.abs(input_B - cyc_B))
    gd_lossA = tf.reduce_mean(tf.squared_difference(logitA_fake, 1))
    gd_lossB = tf.reduce_mean(tf.squared_difference(logitB_fake, 1))
    g_lossB2A = gd_lossA + cyc_loss
    g_lossA2B = gd_lossB + cyc_loss
    d_lossA = (tf.reduce_mean(tf.square(logitA_fake_pool)) + tf.reduce_mean(tf.squared_difference(logitA,1)))/2.0
    d_lossB = (tf.reduce_mean(tf.square(logitB_fake_pool)) + tf.reduce_mean(tf.squared_difference(logitB, 1))) / 2.0

    t_vars = tf.trainable_variables()
    d_A_vars = [var for var in t_vars if 'DA' in var.name]
    g_B2A_vars = [var for var in t_vars if 'B2A' in var.name]
    d_B_vars = [var for var in t_vars if 'DB' in var.name]
    g_A2B_vars = [var for var in t_vars if 'A2B' in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    d_A_trainer = optimizer.minimize(d_lossA, var_list=d_A_vars)
    d_B_trainer = optimizer.minimize(d_lossB, var_list=d_B_vars)
    g_A2B_trainer = optimizer.minimize(g_lossA2B, var_list=g_A2B_vars)
    g_B2A_trainer = optimizer.minimize(g_lossB2A, var_list=g_B2A_vars)

    for var in t_vars: print(var.name)
    # Summary variables for tensorboard
    g_lossA2B_summ = tf.summary.scalar("g_lossA2B", g_lossA2B)
    g_lossB2A_summ = tf.summary.scalar("g_B_loss", g_lossB2A)
    d_A_loss_summ = tf.summary.scalar("d_A_loss", d_lossA)
    d_B_loss_summ = tf.summary.scalar("d_B_loss", d_lossB)

    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        count_trainable_params()
        train_writer = tf.summary.FileWriter(log_path, sess.graph)
        saver = tf.train.Saver()
        load_model(sess=sess, saver=saver, restore_checkpoint=restore_checkpoint)

        num_fake_inputs = 0
        A_input, B_input, fake_images_A, fake_images_B = input_read(sess, image_A, image_B)
        with tf.device('/gpu:%d' % gpu):
            for epoch in range(max_epoch):
                saver.save(sess, log_path, global_step=epoch)
                for i in range(0, 10):
                    fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([fake_A, fake_B, cyc_A, cyc_B],
                                                                                feed_dict={input_A: A_input[i],input_B: B_input[i]})

                    save_training_images(epoch,i,fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp,A_input,B_input)
                for ptr in range(max_images):
                    _, fake_A2B_tmp, g_l_A2B, summary_str_A2B  = sess.run([g_A2B_trainer, fake_B, g_lossA2B, g_lossA2B_summ],
                                                                 feed_dict={input_A: A_input[ptr],input_B: B_input[ptr]})
                    train_writer.add_summary(summary_str_A2B, epoch * ptr + ptr)
                    fake_A2B_temp1 = fake_image_pool(num_fake_inputs, fake_A2B_tmp, fake_images_B)

                    _, d_l_B, summary_str_DB = sess.run([d_B_trainer, d_lossB, d_B_loss_summ],
                                                 feed_dict={input_A: A_input[ptr],input_B: B_input[ptr],fake_pool_B: np.asarray(fake_A2B_temp1)})
                    train_writer.add_summary(summary_str_DB, epoch * ptr + ptr)

                    _, fake_B2A_tmp, g_l_B2A, summary_str_B2A = sess.run([g_B2A_trainer, fake_B, g_lossB2A, g_lossB2A_summ],
                                                                feed_dict={input_A: A_input[ptr],input_B: B_input[ptr]})
                    train_writer.add_summary(summary_str_B2A, epoch * ptr + ptr)
                    fake_B2A_temp1 = fake_image_pool(num_fake_inputs, fake_B2A_tmp, fake_images_A)

                    _, d_l_A, summary_str_DA = sess.run([d_A_trainer, d_lossA, d_A_loss_summ],
                                                 feed_dict={input_A: A_input[ptr], input_B: B_input[ptr],fake_pool_A: np.asarray(fake_B2A_temp1)})
                    train_writer.add_summary(summary_str_DA, epoch * ptr + ptr)
                    print("[Epoch:%s  Ptr:%s]Gen A2B Loss: %s Dis B Loss:%s Gen A2B Loss:%s Dis A Loss:%s" %
                          (epoch, ptr, g_l_A2B, d_l_B, g_l_B2A, d_l_A))
                    num_fake_inputs += 1


def te():
    image_A, image_B = input_setup()

    input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
    input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")

    fake_B = build_generator_resnet_6blocks(input_A, is_training=True, reuse=False, name='A2B')
    fake_A = build_generator_resnet_6blocks(input_B, is_training=True, reuse=False, name='B2A')

    os.environ["CUDA_VISIBLE_DEVICES"] = '%d' % gpu
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=sess_config) as sess:
        count_trainable_params()
        tf.summary.FileWriter(log_path, sess.graph)
        variables_to_restoreA2B = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,'A2B')
        variables_to_restoreB2A = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'B2A')
        saver = tf.train.Saver(variables_to_restoreA2B + variables_to_restoreB2A)
        load_model(sess=sess, saver=saver, restore_checkpoint=log_path)

        A_input, B_input, _, _ = input_read(sess, image_A, image_B)
        with tf.device('/gpu:%d' % gpu):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            for i in range(0, 1):
                fake_A_temp, fake_B_temp = sess.run([fake_A, fake_B], feed_dict={input_A: A_input[i],input_B: B_input[i]})
                imsave(output_dir + str(i) + ".jpg",((fake_A_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave(output_dir + str(i) + ".jpg",((fake_B_temp[0] + 1) * 127.5).astype(np.uint8))
                imsave(output_dir + str(i) + ".jpg",((A_input[i][0] + 1) * 127.5).astype(np.uint8))
                imsave(output_dir + str(i) + ".jpg",((B_input[i][0] + 1) * 127.5).astype(np.uint8))

if train_flag:
    tr()
else:
    te()