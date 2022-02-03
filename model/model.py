from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from six.moves import xrange

from model.utils.interpolate import *
from model.utils.custom_ops import *
from model.utils.data_io import DataSetLoader, saveSampleResults
from model.utils.parzen_ll import ParsenDensityEsimator
from model.utils.inception_model import *

from model.modules import *

import scipy.io as sio
import model.utils.fid_util2 as fid_util

class CoopVAEBM(object):
    def __init__(self, num_epochs=200, image_size=64, batch_size=100, num_channel=3, nTileRow=12, nTileCol=12,
                 descriptor_type='SA', generator_type='SA', encoder_type='SA',
                 d_lr=0.001, vae_lr=0.0001, beta1_vae=0.5, beta1_des=0.5,
                 des_step_size=0.002, des_sample_steps=10, des_refsig=0.016,
                 gen_step_size=0.1, gen_sample_steps=0, gen_refsig=0.3, gen_latent_size=100, weight_latent_loss =2.2,
                 data_path='./data/', log_step=10, category='volcano',
                 sample_dir='./synthesis', model_dir='./checkpoints', log_dir='./log', test_dir='./test',
                 prefetch=True, read_len=500, output_dir='', calculate_inception=False, calculate_FID=False):
        self.descriptor_type = descriptor_type
        self.generator_type = generator_type
        self.encoder_type = encoder_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.nTileRow = nTileRow
        self.nTileCol = nTileCol
        self.num_chain = nTileRow * nTileCol
        self.beta1_des = beta1_des
        self.beta1_vae = beta1_vae
        self.prefetch = prefetch
        self.read_len = read_len
        self.category = category
        self.num_channel = num_channel
        self.calculate_inception = calculate_inception
        self.calculate_FID = calculate_FID
        self.output_dir = output_dir

        self.weight_latent_loss = weight_latent_loss

        self.d_lr = d_lr
        #self.g_lr = g_lr
        self.vae_lr = vae_lr
        self.delta1 = des_step_size
        self.sigma1 = des_refsig
        self.delta2 = gen_step_size
        self.sigma2 = gen_refsig
        self.t1 = des_sample_steps
        self.t2 = gen_sample_steps

        self.data_path = data_path
        self.log_step = log_step

        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.test_dir = test_dir
        self.z_size = gen_latent_size

        self.syn = tf.placeholder(shape=[None, self.image_size, self.image_size, self.num_channel], dtype=tf.float32, name='syn')
        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, self.num_channel], dtype=tf.float32, name='obs')
        self.z = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z')

        self.debug = False

        self.descriptor = descriptor
        self.generator = generator
        self.encoder = encoder

    def build_model(self):

        self.global_step = tf.Variable(-1, trainable=False)
        self.decayed_learning_rate_d = tf.train.exponential_decay(self.d_lr, self.global_step, 20, 0.98)
        self.decayed_learning_rate_vae = tf.train.exponential_decay(self.vae_lr, self.global_step, 20, 0.98)
        self.update_lr = tf.assign_add(self.global_step, 1)

        self.gen_res = self.generator(self.z, self.generator_type, self.image_size, reuse=False)
        obs_res = self.descriptor(self.obs, self.descriptor_type, reuse=False)
        syn_res = self.descriptor(self.syn, self.descriptor_type, reuse=True)

        self.code, self.mn, self.std = self.encoder(self.syn, self.encoder_type, self.z_size, reuse=False)
        self.AE_res = self.generator(self.code, self.generator_type, self.image_size, reuse=True)

        # VAE loss
        diff = tf.reshape(tf.square(self.AE_res-self.obs), [-1, self.num_channel * self.image_size * self.image_size])
        vae_recon_loss = tf.reduce_sum(diff, 1)

        vae_latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * self.std - tf.square(self.mn) - tf.exp(2.0 * self.std), 1)
        self.vae_loss = tf.reduce_mean(vae_recon_loss + self.weight_latent_loss * vae_latent_loss)

        self.vae_loss_mean, self.vae_loss_update = tf.contrib.metrics.streaming_mean(self.vae_loss)
        vae_vars = [var for var in tf.trainable_variables() if var.name.startswith('VAE')]
        vae_optim = tf.train.AdamOptimizer(self.decayed_learning_rate_vae, beta1=self.beta1_vae)

        vae_grads_vars = vae_optim.compute_gradients(self.vae_loss, var_list=vae_vars)
        vae_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in vae_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_vae_grads = vae_optim.apply_gradients(vae_grads_vars)


        self.recon_err = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0)), 2))
        self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean(self.recon_err)

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

        self.des_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))
        self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(self.des_loss)

        des_optim = tf.train.AdamOptimizer(self.decayed_learning_rate_d, beta1=self.beta1_des)
        des_grads_vars = des_optim.compute_gradients(self.des_loss, var_list=des_vars)
        des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients(des_grads_vars)


        # symbolic langevins
        self.langevin_descriptor = self.langevin_dynamics_descriptor(self.syn)

        tf.summary.scalar('des_loss', self.des_loss_mean)
        tf.summary.scalar('vae_loss', self.vae_loss_mean)


        self.summary_op = tf.summary.merge_all()

    def langevin_dynamics_descriptor(self, syn_arg):
        def cond(i, syn):
            return tf.less(i, self.t1)

        def body(i, syn):
            noise = tf.random_normal(shape=[self.num_chain, self.image_size, self.image_size, self.num_channel], name='noise')
            syn_res = self.descriptor(syn, self.descriptor_type, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            # syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad) + self.delta1 * noise

            syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad)
            #syn = syn + 0.5 * self.delta1 * self.delta1 * grad
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics_descriptor"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])
            return syn

    def langevin_dynamics_generator(self, z_arg):
        def cond(i, z):
            return tf.less(i, self.t2)

        def body(i, z):
            noise = tf.random_normal(shape=[self.num_chain, self.z_size], name='noise')

            gen_res = self.generator(z, self.generator_type, self.image_size, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - gen_res),
                                       axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
            z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) + self.delta2 * noise
            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z

    def langevin_dynamics_generator_recovery(self, z_arg, visible_map):
        def cond(i, z):
            return tf.less(i, self.t2)

        def body(i, z):
            noise = tf.random_normal(shape=[self.num_chain, self.z_size], name='noise')

            gen_res = self.generator(z, self.generator_type, self.image_size, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * visible_map *(tf.square(self.obs - gen_res)), axis=0)
            #gen_loss = tf.reduce_mean( visible_map * (tf.square(self.obs - gen_res)), axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
            z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) + self.delta2 * noise
            #z = z - 50 *  grad
            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z

    def train(self, sess):

        self.build_model()

        # Prepare training data

        dataset = DataSetLoader(self)

        # initialize training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sample_results_des = np.random.randn(self.num_chain * dataset.num_batch, self.image_size, self.image_size, self.num_channel)
        sample_results_gen = np.random.randn(self.num_chain * dataset.num_batch, self.image_size, self.image_size, self.num_channel)

        saver = tf.train.Saver(max_to_keep=6)

        writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # measure 1: Parzon-window based likelihood
        if self.calculate_FID:
            fid_util.init_fid()
            FID_log_file = os.path.join(self.output_dir, 'fid.txt')
        # measure 2: inception score
        if self.calculate_inception:
            inception_log_file = os.path.join(self.output_dir, 'inception.txt')

        # make graph immutable
        tf.get_default_graph().finalize()

        # store graph in protobuf
        with open(self.model_dir + '/graph.proto', 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))


        # train
        minibatch = -1

        for epoch in xrange(self.num_epochs):

            start_time = time.time()
            sess.run(self.update_lr)

            for i in xrange(dataset.num_batch):
                minibatch = minibatch + 1
                obs_data = dataset.get_batch()

                # Step G1: generate X ~ N(0, 1)
                z_vec = np.random.randn(self.num_chain, self.z_size)
                g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
                # Step D1: obtain synthesized images Y
                if self.t1 > 0:
                    syn = sess.run(self.langevin_descriptor, feed_dict={self.syn: g_res})


                # Step D2: update D net
                d_loss = sess.run([self.des_loss, self.des_loss_update, self.apply_d_grads],
                                  feed_dict={self.obs: obs_data, self.syn: syn})[0]

                # Step G2: update VAE net
                vae_loss = sess.run([self.vae_loss, self.vae_loss_update, self.apply_vae_grads],
                                  feed_dict={self.obs: syn, self.syn: syn})[0]


                # Compute MSE
                mse = sess.run([self.recon_err, self.recon_err_update],
                               feed_dict={self.obs: obs_data, self.syn: syn})[0]

                sample_results_gen[i * self.num_chain:(i + 1) * self.num_chain] = g_res
                sample_results_des[i * self.num_chain:(i + 1) * self.num_chain] = syn


            end_time = time.time()
            #[des_loss_avg, gen_loss_avg, mse_avg, summary] = sess.run([self.des_loss_mean, self.gen_loss_mean,
            #                                                           self.recon_err_mean, self.summary_op])

            [des_loss_avg, vae_loss_avg, mse_avg, summary] = sess.run([self.des_loss_mean, self.vae_loss_mean,
                                                                       self.recon_err_mean, self.summary_op])

            [decayed_lr_d, decayed_lr_vae] = sess.run([self.decayed_learning_rate_d, self.decayed_learning_rate_vae])

            writer.add_summary(summary, minibatch)
            writer.flush()
            print('Epoch #{:d}, avg.des loss: {:.4f}, avg.vae loss: {:.4f}, '
                  'avg.L2 dist: {:4.4f}, time: {:.2f}s, learning rate: EBM {:.6f}, VAE {:.6f}'.format(epoch, des_loss_avg, vae_loss_avg,
                  mse_avg, end_time - start_time, decayed_lr_d, decayed_lr_vae))


            if epoch % self.log_step == 0:
                # save check points
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

            if (self.calculate_inception or self.calculate_FID) and epoch % 25 == 0:

                num_example_fid = 10000
                num_examples = max((len(dataset)* self.calculate_inception),  (num_example_fid * self.calculate_FID))

                sample_results, abc = self.val_generation(sess, epoch, num_examples)

                if self.calculate_inception:

                    #sample_results_partial = sample_results_des[:len(dataset)]
                    sample_results = np.minimum(1, np.maximum(-1, sample_results))
                    sample_results = (sample_results + 1) / 2 * 255
                    print("Computing Inception score")
                    m, s = get_inception_score(sample_results)
                    print("Inception score: mean {}, sd {}".format(m, s))
                    fo = open(inception_log_file, 'a')
                    fo.write("Epoch {}: mean {}, sd {} \n".format(epoch, m, s))
                    fo.close()

                if self.calculate_FID:

                    sample_results = sample_results[:num_example_fid]

                    obs_partial = dataset.images[:len(sample_results)]

                    print("Computing FID score")
                    fid_ebm = fid_util.get_fid(sess, sample_results, obs_partial)
                    print("FID : {}".format(fid_ebm))
                    fo = open(FID_log_file, 'a')
                    fo.write("Epoch {}: FID {} \n".format(epoch, fid_ebm))
                    fo.close()

            elif epoch % 25 == 0:

                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleResults(syn, "%s/des_%06d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)
                saveSampleResults(g_res, "%s/gen_%06d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)


    def val_generation(self, sess, epoch, num_examples):


        num_batches = int(math.ceil(num_examples / self.num_chain))

        sample_results_des = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, self.num_channel)
        sample_results_gen = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, self.num_channel)


        for i in xrange(num_batches):

            # Step G0: generate X ~ N(0, 1)
            z_vec = np.random.randn(self.num_chain, self.z_size)
            g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
            # Step D1: obtain synthesized images Y
            if self.t1 > 0:
                syn = sess.run(self.langevin_descriptor, feed_dict={self.syn: g_res})

            sample_results_des[i * self.num_chain:(i + 1) * self.num_chain] = syn
            sample_results_gen[i * self.num_chain:(i + 1) * self.num_chain] = g_res

            if i==0:
                if not os.path.exists(self.sample_dir):
                    os.makedirs(self.sample_dir)
                saveSampleResults(syn, "%s/des_%06d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)
                saveSampleResults(g_res, "%s/gen_%06d.png" % (self.sample_dir, epoch), col_num=self.nTileCol)

        sample_results_des = sample_results_des[:num_examples]
        sample_results_gen = sample_results_gen[:num_examples]

        return sample_results_des, sample_results_gen


    def train_finetune(self, sess, ckpt, starting_epoch=0):

        assert ckpt is not None, 'no checkpoint provided.'

        self.build_model()

        saver = tf.train.Saver(max_to_keep=4)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))


        # Prepare training data
        dataset = DataSetLoader(self)

        # initialize training

        sample_results_des = np.random.randn(self.num_chain * dataset.num_batch, self.image_size, self.image_size, self.num_channel)
        sample_results_gen = np.random.randn(self.num_chain * dataset.num_batch, self.image_size, self.image_size, self.num_channel)



        writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # measure 1: Parzon-window based likelihood
        if self.calculate_FID:
            fid_util.init_fid()
            FID_log_file = os.path.join(self.output_dir, 'fid.txt')

        # measure 2: inception score
        if self.calculate_inception:
            inception_log_file = os.path.join(self.output_dir, 'inception.txt')
            #inception_write_file = os.path.join(self.output_dir, 'inception.mat')

        # make graph immutable
        #tf.get_default_graph().finalize()

        # store graph in protobuf
        #with open(self.model_dir + '/graph.proto', 'w') as f:
        #    f.write(str(tf.get_default_graph().as_graph_def()))

        inception_mean, inception_sd = [], []


        # train
        minibatch = -1

        #sess.run(self.init_global_step)

        starting_epoch = sess.run(self.global_step) + 1

        for epoch in xrange(starting_epoch, self.num_epochs):

            start_time = time.time()
            sess.run(self.update_lr)

            for i in xrange(dataset.num_batch):
                minibatch = minibatch + 1
                obs_data = dataset.get_batch()

                # Step G1: generate X ~ N(0, 1)
                z_vec = np.random.randn(self.num_chain, self.z_size)
                g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
                # Step D1: obtain synthesized images Y
                if self.t1 > 0:
                    syn = sess.run(self.langevin_descriptor, feed_dict={self.syn: g_res})


                # variational inference

                # Step D2: update D net
                d_loss = sess.run([self.des_loss, self.des_loss_update, self.apply_d_grads],
                                  feed_dict={self.obs: obs_data, self.syn: syn})[0]

                # Step G2: update VAE net
                vae_loss = sess.run([self.vae_loss, self.vae_loss_update, self.apply_vae_grads],
                                  feed_dict={self.obs: syn, self.syn: syn})[0]


                # Compute MSE
                mse = sess.run([self.recon_err, self.recon_err_update],
                               feed_dict={self.obs: obs_data, self.syn: syn})[0]

                sample_results_gen[i * self.num_chain:(i + 1) * self.num_chain] = g_res
                sample_results_des[i * self.num_chain:(i + 1) * self.num_chain] = syn


            end_time = time.time()


            [des_loss_avg, vae_loss_avg, mse_avg, summary] = sess.run([self.des_loss_mean, self.vae_loss_mean,
                                                                       self.recon_err_mean, self.summary_op])

            [decayed_lr_d, decayed_lr_vae] = sess.run([self.decayed_learning_rate_d, self.decayed_learning_rate_vae])

            writer.add_summary(summary, minibatch)
            writer.flush()
            print('Epoch #{:d}, avg.des loss: {:.4f}, avg.vae loss: {:.4f}, '
                  'avg.L2 dist: {:4.4f}, time: {:.2f}s, learning rate: EBM {:.6f}, VAE {:.6f}'.format(epoch, des_loss_avg, vae_loss_avg,
                  mse_avg, end_time - start_time, decayed_lr_d, decayed_lr_vae))


            if epoch % self.log_step == 0:
                # save check points
                if not os.path.exists(self.model_dir):
                    os.makedirs(self.model_dir)
                saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=epoch)

            if (self.calculate_inception or self.calculate_FID) and epoch % 20 == 0:

                num_example_fid = 10000
                num_examples = max((len(dataset)* self.calculate_inception),  (num_example_fid * self.calculate_FID))

                sample_results, abc = self.val_generation(sess, epoch, num_examples)


                if self.calculate_inception:

                    #sample_results_partial = sample_results_des[:len(dataset)]
                    sample_results = np.minimum(1, np.maximum(-1, sample_results))
                    sample_results = (sample_results + 1) / 2 * 255
                    print("Computing Inception score")
                    m, s = get_inception_score(sample_results)
                    print("Inception score: mean {}, sd {}".format(m, s))
                    fo = open(inception_log_file, 'a')
                    fo.write("Epoch {}: mean {}, sd {} \n".format(epoch, m, s))
                    fo.close()
                    #inception_mean.append(m)
                    #inception_sd.append(s)
                    #sio.savemat(inception_write_file, {'mean': np.asarray(inception_mean), 'sd': np.asarray(inception_sd)})

                if self.calculate_FID:

                    sample_results = sample_results[:num_example_fid]

                    #sample_results_partial = sample_results_des[:100]
                    obs_partial = dataset.images[:len(sample_results)]

                    print("Computing FID score")
                    fid_ebm = fid_util.get_fid(sess, sample_results, obs_partial)
                    print("FID : {}".format(fid_ebm))
                    fo = open(FID_log_file, 'a')
                    fo.write("Epoch {}: FID {} \n".format(epoch, fid_ebm))
                    fo.close()



    def interpolation(self, sess, ckpt, sample_size, useTraining=False):
        assert ckpt is not None, 'no checkpoint provided.'

        if useTraining:
            dataset = DataSetLoader(self)

        obs_res = self.descriptor(self.obs, self.descriptor_type, reuse=False)
        gen_res = self.generator(self.z, self.generator_type, self.image_size, reuse=False)
        get_codes = self.encoder(self.obs, self.encoder_type, self.z_size, reuse=False)[0]
        langevin_descriptor = self.langevin_dynamics_descriptor(self.obs)
        num_batches = int(math.ceil(sample_size / self.num_chain))

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        for i in xrange(num_batches):

            if useTraining:
                obs_data = dataset.get_batch()
                num_get = np.min([self.num_chain, np.shape(obs_data)[0]])
                obs_data = obs_data[:num_get, :, :, :]
                z_vec = sess.run(get_codes, feed_dict={self.obs: obs_data})
            else:
                z_vec = np.random.randn(self.num_chain, self.z_size)
            # g_res = sess.run(gen_res, feed_dict={self.z: z_vec})
            # saveSampleResults(g_res, "%s/gen%03d.png" % (self.test_dir, i), col_num=self.nTileCol)

            # output interpolation results
            interp_z = linear_interpolator(z_vec, npairs=self.nTileRow, ninterp=self.nTileCol)
            interp = sess.run(gen_res, feed_dict={self.z: interp_z})

            interp = sess.run(langevin_descriptor, feed_dict={self.obs: interp})
            saveSampleResults(interp, "%s/interp%03d.png" % (self.test_dir, i), col_num=self.nTileCol)

        print("The results are saved in a folder: {}".format(self.test_dir))

    def sampling(self, sess, ckpt, sample_size, sample_step, calculate_inception=False, calculate_FID=False):
        assert ckpt is not None, 'no checkpoint provided.'

        self.t1 = sample_step

        gen_res = self.generator(self.z, self.generator_type, self.image_size, reuse=False)
        obs_res = self.descriptor(self.obs, self.descriptor_type, reuse=False)

        self.langevin_descriptor = self.langevin_dynamics_descriptor(gen_res)
        num_batches = int(math.ceil(sample_size / self.num_chain))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        sample_results_des = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, self.num_channel)
        for i in xrange(num_batches):
            z_vec = np.random.randn(self.num_chain, self.z_size)

            # synthesis by generator
            g_res = sess.run(gen_res, feed_dict={self.z: z_vec})
            saveSampleResults(g_res, "%s/gen%03d_test.png" % (self.test_dir, i), col_num=self.nTileCol)

            # synthesis by descriptor and generator
            syn = sess.run(self.langevin_descriptor, feed_dict={self.z: z_vec})
            saveSampleResults(syn, "%s/des%03d_test.png" % (self.test_dir, i), col_num=self.nTileCol)

            sample_results_des[i * self.num_chain:(i + 1) * self.num_chain] = syn

            if i % 10 == 0:
                print("Sampling batches: {}, from {} to {}".format(i, i * self.num_chain,
                                                                   min((i+1) * self.num_chain, sample_size)))
        sample_results_des = sample_results_des[:sample_size]
        sample_results_des = np.minimum(1, np.maximum(-1, sample_results_des))


        if calculate_inception:
            print("Computing inception score")
            sample_results_des = (sample_results_des + 1) / 2 * 255
            m, s = get_inception_score(sample_results_des)
            print("Inception score: mean {}, sd {}".format(m, s))

        if calculate_FID:
            dataset = DataSetLoader(self)

            obs_partial = dataset.images[:len(sample_results_des)]

            print("Computing FID score")
            fid_util.init_fid()
            fid_ebm = fid_util.get_fid(sess, sample_results_des, obs_partial)
            print("FID : {}".format(fid_ebm))


        sampling_output_file = os.path.join(self.output_dir, 'samples_des.npy')
        np.save(sampling_output_file, sample_results_des)
        print("The results are saved in folder: {}".format(self.output_dir))


    def visualize_refinement(self, sess, ckpt, num_batches=1):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.z, self.generator_type, self.image_size, reuse=False)
        obs_res = self.descriptor(self.obs, self.descriptor_type, reuse=False)

       # num_batches = int(math.ceil(sample_size / self.num_chain))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        #sample_results_des = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, self.num_channel)
        total_images = np.random.randn(self.t1, self.num_chain, self.image_size, self.image_size,  self.num_channel)
        for i in xrange(num_batches):

            z_vec = np.random.randn(self.num_chain, self.z_size)

            # synthesis by generator
            g_res = sess.run(gen_res, feed_dict={self.z: z_vec})
            saveSampleResults(g_res, "%s/%03d_batch_0_step.png" % (self.test_dir, i), col_num=self.nTileCol)

            total_steps = self.t1
            for j in xrange(total_steps):
                self.t1 = j+1
                langevin_descriptor = self.langevin_dynamics_descriptor(self.obs)
                syn = sess.run(langevin_descriptor, feed_dict={self.obs: g_res})
                total_images[j,:,:,:,:] = syn
                saveSampleResults(syn, "%s/%03d_batch_%03d_step.png" % ( self.test_dir, i, j+1), col_num=self.nTileCol)

            refinement_visualization = np.random.randn( self.nTileRow * self.nTileCol, self.image_size, self.image_size, self.num_channel)

            for j in xrange(self.nTileRow):
                refinement_visualization[j  *self.nTileCol ,:,:,:] = g_res[j, :, :, :]

                for k in xrange(total_steps):

                    refinement_visualization[j *self.nTileCol + 1 + k, : , :, : ] = total_images[k, j, :, :, :]

                #for k in xrange(int(total_steps/2)):
                    #refinement_visualization[j *self.nTileCol + 1 + k, : , :, : ] = total_images[k*2, j, :, :, :]


            saveSampleResults(refinement_visualization, "%s/%03d_batch_final.png" % (self.test_dir, i), col_num=self.nTileRow)

        print("The results are saved in folder: {}".format(self.test_dir))


    def inpaint(self, sess, ckpt, num_batches=1):
        assert ckpt is not None, 'no checkpoint provided.'

        dataset = DataSetLoader(self)

        obs_res = self.descriptor(self.obs, self.descriptor_type, reuse=False)
        inference  = self.encoder(self.obs, self.encoder_type, self.z_size, reuse=False)[0]
        gen = self.generator(self.z, self.generator_type, self.image_size, reuse=False)

        langevin_descriptor = self.langevin_dynamics_descriptor(self.obs)

        visible_map = tf.placeholder(shape=[None, self.image_size, self.image_size, self.num_channel], dtype=tf.float32, name='visible')


        #num_batches = int(math.ceil(sample_size / self.batch_size))

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))
        inference_Langevin = self.langevin_dynamics_generator_recovery(self.z, visible_map)

        images = np.random.randn(self.num_chain, self.image_size, self.image_size, self.num_channel)


        for i in xrange(num_batches):

            obs_data = dataset.get_batch()
            num_get = np.min([self.nTileRow, np.shape(obs_data)[0]])
            obs_data = obs_data[:num_get,:,:,:]
            mask_data = obs_data
            mask_data[:, int(self.image_size/2):,:,:]=0

            mask_data = np.tile(mask_data, (self.nTileCol, 1, 1, 1))
            saveSampleResults(mask_data, "%s/masked_data_%03d.png" % (self.test_dir, i), col_num=self.nTileRow)

            # vae + Langevin
            code = sess.run(inference, feed_dict={self.obs: mask_data})
            decode = sess.run(gen, feed_dict={self.z: code})
            recover_data1 = sess.run(langevin_descriptor, feed_dict={self.obs: decode})
            recover_data1[:, :int(self.image_size / 2), :, :] = mask_data[:, :int(self.image_size / 2), :, :]
            saveSampleResults(recover_data1, "%s/inpaint_VAE_EBM_%03d.png" % (self.test_dir, i), col_num=self.nTileRow)

            # Langevin only
            recover_data2 = sess.run(langevin_descriptor, feed_dict={self.obs: mask_data})
            recover_data2[:, :int(self.image_size / 2), :, :] = mask_data[:, :int(self.image_size / 2), :, :]
            saveSampleResults(recover_data2, "%s/inpaint_EBM_only_%03d.png" % (self.test_dir, i), col_num=self.nTileRow)

            #recover_data2[:, :int(self.image_size / 2), :, :] = mask_data[:, :int(self.image_size / 2), :, :]

            # Langevin inference
            initialize_z = np.random.randn(self.num_chain, self.z_size)
            visible_map_input = np.zeros(shape=mask_data.shape)
            visible_map_input[:, :int(self.image_size/2),:,:] = 1
            code_Langevin = sess.run(inference_Langevin, feed_dict={self.z: initialize_z, self.obs: mask_data, visible_map: visible_map_input})

            decode_Langevin = sess.run(gen, feed_dict={self.z: code_Langevin})
            decode_Langevin[:, :int(self.image_size / 2), :, :] = mask_data[:, :int(self.image_size / 2), :, :]
            recover_data3 = sess.run(langevin_descriptor, feed_dict={self.obs: decode_Langevin})
            recover_data3[:, :int(self.image_size / 2), :, :] = mask_data[:, :int(self.image_size / 2), :, :]

            #

            #saveSampleResults(decode_Langevin, "%s/inpaint_decode_Langevin%03d.png" % (self.test_dir, i), col_num=self.nTileRow)

            saveSampleResults(recover_data3, "%s/inpaint_Lengevin_inference_EBM%03d.png" % (self.test_dir, i), col_num=self.nTileRow)

        print("The results are saved in a folder: {}".format(self.test_dir))

    def reconstruction(self, sess, ckpt, num_batches=3):
        assert ckpt is not None, 'no checkpoint provided.'

        dataset = DataSetLoader(self)

        obs_res = self.descriptor(self.obs, self.descriptor_type, reuse=False)
        inference  = self.encoder(self.obs, self.encoder_type, self.z_size, reuse=False)[0]
        gen = self.generator(self.z, self.generator_type, self.image_size, reuse=False)

        langevin_descriptor = self.langevin_dynamics_descriptor(self.obs)


        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))
        #inference_Langevin = self.langevin_dynamics_generator_recovery(self.z, visible_map)

        #images = np.random.randn(self.num_chain, self.image_size, self.image_size, self.num_channel)

        for i in xrange(num_batches):

            obs_data = dataset.get_batch()
            #num_get = np.min([self.nTileRow, np.shape(obs_data)[0]])

            saveSampleResults(obs_data, "%s/reconstruction_obs%03d.png" % (self.test_dir, i),
                              col_num=self.nTileRow)

            # vae + Langevin
            encode = sess.run(inference, feed_dict={self.obs: obs_data})
            decode = sess.run(gen, feed_dict={self.z: encode})

            saveSampleResults(decode, "%s/reconstruction_gen%03d.png" % (self.test_dir, i),
                              col_num=self.nTileRow)

            refined = sess.run(langevin_descriptor, feed_dict={self.obs: decode})

            saveSampleResults(refined, "%s/reconstruction_refined%03d.png" % (self.test_dir, i),
                              col_num=self.nTileRow)


        print("The results are saved in a folder: {}".format(self.test_dir))
