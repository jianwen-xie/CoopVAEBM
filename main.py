from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from model.model import CoopVAEBM

FLAGS = tf.app.flags.FLAGS

# learning parameters
tf.flags.DEFINE_string('descriptor_type', 'cifar', 'descriptor / EBM network type: [scene/mnist/cifar]') # ebm
tf.flags.DEFINE_string('generator_type', 'cifar', 'generator network type: [scene/mnist/cifar]')   # generator model
tf.flags.DEFINE_string('encoder_type', 'cifar', 'encoder network type: [scene/mnist/cifar]')       # inference model

#tf.flags.DEFINE_string('descriptor_type', 'mnist', 'descriptor network type: [scene/mnist/cifar]')
#tf.flags.DEFINE_string('generator_type', 'mnist2', 'generator network type: [scene/mnist/cifar]')
#tf.flags.DEFINE_string('encoder_type', 'mnist_web', 'encoder network type: [scene/mnist/cifar]')


tf.flags.DEFINE_integer('image_size', 32, 'Image size to rescale images') # 28 for mnist, 32 for cifar10
tf.flags.DEFINE_integer('num_channel', 3, 'number of channel') # 1 for mnist, 3 for cifar10
tf.flags.DEFINE_integer('batch_size', 250, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 5000, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 30, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 30, 'Column number of synthesized images') # total number of synthesis is nTileRow x nTileCol
tf.flags.DEFINE_float('beta1_des', 0.5, 'Momentum term of adam for descriptor / EBM')
tf.flags.DEFINE_float('beta1_gen', 0.5, 'Momentum term of adam for generator')

tf.flags.DEFINE_float('beta1_vae', 0.5, 'Momentum term of adam for VAE')

# parameters for descriptorNet / EBM
tf.flags.DEFINE_float('d_lr', 0.009, 'Initial learning rate for descriptor') # 0.009
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 15, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics')


# parameters for VAENet
tf.flags.DEFINE_float('vae_lr', 0.0002, 'Initial learning rate for VAE')  # 0.0002
tf.flags.DEFINE_float('weight_latent_loss', 2, 'weight of latent loss in the total VAE loss')

# parameters for generatorNet
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 0, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')
tf.flags.DEFINE_integer('gen_latent_size', 100, 'Number of dimensions of latent variables')# 100

# utils
tf.flags.DEFINE_string('data_dir', './data', 'The data directory')
tf.flags.DEFINE_string('category', 'cifar', 'The name of dataset: [cifar/mnist/mnist-fashion]')
tf.flags.DEFINE_boolean('prefetch', True, 'True if reading all images at once')
tf.flags.DEFINE_boolean('calculate_inception', False, 'True if inception score is calculated')
tf.flags.DEFINE_boolean('calculate_FID', False, 'True if FID score is calculated')
tf.flags.DEFINE_integer('read_len', 500, 'Number of batches per reading')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 50, 'Number of minibatches to save output results')


# training or testing
tf.flags.DEFINE_boolean('test', True, 'True if in testing mode')
tf.flags.DEFINE_string('test_type', 'syn', 'testing type: [inter/syn/recon/visual/finetune]: inpaint: inpainting | inter: interpolation | syn: synthesis |visual: visualization | finetune: finetune the model')
tf.flags.DEFINE_string('ckpt', 'pretrained/checkpoints/cifar/model.ckpt-3000', 'Checkpoint path to load: e.g., pretrained/checkpoints/cifar/model.ckpt-3000')
tf.flags.DEFINE_integer('sample_size', 100, 'Number of images to generate during test.')


def main(_):

    import datetime
    extra_info = 'vae_lr=' + str(FLAGS.vae_lr) + ' ' + ' d_lr=' + str(FLAGS.d_lr) + ' weight_latent=' + str(FLAGS.weight_latent_loss)
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    output_dir = ''.join([output_dir] + [extra_info])

    if len(sys.argv) > 1:
        output_dir = ''.join([output_dir] + sys.argv[1:])

    sample_dir = os.path.join(output_dir, 'synthesis')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'checkpoints')
    if FLAGS.test_type == 'inter':
        test_dir = os.path.join(output_dir, 'test/interpolation')
    elif FLAGS.test_type == 'syn':
        test_dir = os.path.join(output_dir, 'test/synthesis')
    elif FLAGS.test_type == 'inpaint':
        test_dir = os.path.join(output_dir, 'test/inpaint')
    elif FLAGS.test_type == 'recon':
        test_dir = os.path.join(output_dir, 'test/reconstruction')
    elif FLAGS.test_type == 'visual':
        test_dir = os.path.join(output_dir, 'test/visualization')
    elif FLAGS.test_type == 'finetune':
        test_dir = os.path.join(output_dir, 'test/finetune')
    else:
        return NotImplementedError

    model = CoopVAEBM(

        descriptor_type=FLAGS.descriptor_type,
        generator_type=FLAGS.generator_type,
        encoder_type=FLAGS.encoder_type,
        num_epochs=FLAGS.num_epochs,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        num_channel=FLAGS.num_channel,
        beta1_des=FLAGS.beta1_des,
        beta1_vae=FLAGS.beta1_vae,
        nTileRow=FLAGS.nTileRow, nTileCol=FLAGS.nTileCol,
        d_lr=FLAGS.d_lr, vae_lr=FLAGS.vae_lr,
        des_refsig=FLAGS.des_refsig, gen_refsig=FLAGS.gen_refsig,
        des_step_size=FLAGS.des_step_size, gen_step_size=FLAGS.gen_step_size,
        des_sample_steps=FLAGS.des_sample_steps, gen_sample_steps=FLAGS.gen_sample_steps,
        gen_latent_size=FLAGS.gen_latent_size, weight_latent_loss=FLAGS.weight_latent_loss,
        log_step=FLAGS.log_step, data_path=FLAGS.data_dir, category=FLAGS.category,
        sample_dir=sample_dir, log_dir=log_dir, model_dir=model_dir, test_dir=test_dir,
        prefetch=FLAGS.prefetch, read_len=FLAGS.read_len, output_dir=output_dir,
        calculate_inception=FLAGS.calculate_inception, calculate_FID=FLAGS.calculate_FID
    )

    with tf.Session() as sess:
        if FLAGS.test:

            if tf.gfile.Exists(test_dir):
                tf.gfile.DeleteRecursively(test_dir)
            tf.gfile.MakeDirs(test_dir)

            if FLAGS.test_type == 'syn':
                model.sampling(sess, FLAGS.ckpt, sample_size=50000, sample_step=30, calculate_inception=True)
            elif FLAGS.test_type == 'inter':
                model.interpolation(sess, FLAGS.ckpt, 55000, useTraining=False)
            elif FLAGS.test_type == 'inpaint':
                model.inpaint(sess, FLAGS.ckpt, num_batches=3)
            elif FLAGS.test_type == 'recon':
                model.reconstruction(sess, FLAGS.ckpt, num_batches=3)
            elif FLAGS.test_type == 'visual':
                model.visualize_refinement(sess, FLAGS.ckpt, num_batches=6)
            elif FLAGS.test_type == 'finetune':
                model.train_finetune(sess, FLAGS.ckpt)
            else:
                return NotImplementedError

        else:


            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)

            if tf.gfile.Exists(sample_dir):
                tf.gfile.DeleteRecursively(sample_dir)
            tf.gfile.MakeDirs(sample_dir)

            if tf.gfile.Exists(model_dir):
                tf.gfile.DeleteRecursively(model_dir)
            tf.gfile.MakeDirs(model_dir)

            model.train(sess)

if __name__ == '__main__':
    tf.app.run()
