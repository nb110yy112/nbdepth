#coding=utf-8

import numpy as np
import tensorflow as tf
import net
import time
import os
import cv2

""" param """
epoch = 50
iteration = 1000
batch_size = 64
lr = 0.0002
n_critic = 5
gpu_id = 0
img_width = 1242
img_height = 360
data_pool = []
path = '/home/gan/img/SfMLearner/2011_09_26/2011_09_26_drive_0001_sync/image_00/data/'
start_time = time.time()
end_time = start_time

""" data """
for file in os.listdir(path):
    im1 = cv2.imread(path + file)
    im1 = cv2.resize(im1, (img_width, img_height))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im1 = np.array(im1)/255.
    data_pool.append(im1)
data_pool = np.array(data_pool)

""" graphs """
with tf.device('/gpu:%d' % gpu_id):
    ''' models '''
    generator = net.flow_net
    discriminator = net.img_dis

    ''' graph '''
    # inputs
    real = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 1])
    z = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 1])

    # dicriminate real
    r_logit = discriminator(real, reuse=False)

    # generate and discrinate fake
    fake = generator(z)
    f_logit = discriminator(fake)

    # losses
    def gradient_penalty(real, fake, f):
        def interpolate(a, b):
            shape = tf.concat((tf.shape(a)[0:1], tf.tile([1], [a.shape.ndims - 1])), axis=0)
            alpha = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.get_shape().as_list())
            return inter

        x = interpolate(real, fake)
        pred = f(x)
        gradients = tf.gradients(pred, x)[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
        gp = tf.reduce_mean((slopes - 1.)**2)
        return gp

    wd = tf.reduce_mean(r_logit) - tf.reduce_mean(f_logit)
    gp = gradient_penalty(real, fake, discriminator)
    d_loss = -wd + gp * 10.0
    g_loss = -tf.reduce_mean(f_logit)

    # otpims
    d_var = utils.trainable_variables('img_diff')
    g_var = utils.trainable_variables('img_cmb' or 'img_pro')
    d_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(d_loss, var_list=d_var)
    g_step = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(g_loss, var_list=g_var)

    # summaries
    d_summary = utils.summary({wd: 'wd', gp: 'gp'})
    g_summary = utils.summary({g_loss: 'g_loss'})

""" train """
''' init '''
# session
sess = utils.session()
# iteration counter
it_cnt, update_cnt = utils.counter()
# saver
saver = tf.train.Saver(max_to_keep=10)
# summary writer
summary_writer = tf.summary.FileWriter('./summaries/flownet', sess.graph)

''' initialization '''
ckpt_dir = './checkpoints/flownet'
utils.mkdir(ckpt_dir + '/')
if not utils.load_checkpoint(ckpt_dir, sess):
    sess.run(tf.global_variables_initializer())

''' train '''
try:
    max_it = epoch * iteration
    for it in range(sess.run(it_cnt), max_it):
        sess.run(update_cnt)

        # which epoch
        epoch = it // iteration
        it_epoch = it % iteration + 1
        real_ipt = []
        z_ipt = []

        # train D
        for i in range(n_critic):
            # batch data
            num = np.random.randint(0, len(data_pool), batch_size)
            for i in range(batch_size):
                z_ipt.append(data_pool[num[i]])
                real_ipt.append(data_pool[[num[i]]])
            real_ipt = np.array(real_ipt)
            z_ipt = np.array(z_ipt)
            d_summary_opt, _ = sess.run([d_summary, d_step], feed_dict={real: real_ipt, z: z_ipt})
        summary_writer.add_summary(d_summary_opt, it)

        # train G
        num = np.random.randint(0, len(data_pool), batch_size)
        for i in range(batch_size):
            z_ipt.append(data_pool[num[i]])
        z_ipt = np.array(z_ipt)
        g_summary_opt, _ = sess.run([g_summary, g_step], feed_dict={z: z_ipt})
        summary_writer.add_summary(g_summary_opt, it)

        # display
        if it % 1 == 0:
            print("Epoch: (%3d) (%5d/%5d)" % (epoch, it_epoch, batch_epoch))

        # save
        if (it + 1) % 1000 == 0:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch))
            print('Model saved in file: % s' % save_path)

        # sample
        if (it + 1) % 100 == 0:
            f_sample_opt = sess.run(fake, feed_dict={z: z_ipt})

            save_dir = './sample_images_while_training/flownet'
            utils.mkdir(save_dir + '/')
            utils.imwrite(utils.immerge(f_sample_opt, 8, 8), '%s/Epoch_(%d)_(%dof%d).jpg' % (save_dir, epoch, it_epoch, batch_epoch))

except Exception as e:
    traceback.print_exc()
finally:
    print(" [*] Close main session!")
    sess.close()

