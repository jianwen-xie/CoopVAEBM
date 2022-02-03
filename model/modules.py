import tensorflow as tf
from utils.custom_ops import *


def descriptor(inputs, descriptor_type, reuse=False):
    with tf.variable_scope('des', reuse=reuse):

        if descriptor_type == 'scene':

            out = tf.layers.conv2d(inputs, 64, kernel_size=(5, 5), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

            out = tf.layers.conv2d(out, 256, kernel_size=(3, 3), strides=(1, 1), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

            out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out


        elif descriptor_type == 'cifar':

            out = tf.layers.conv2d(inputs, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

            out = tf.layers.conv2d(out, 128, kernel_size=(4, 4), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

            out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

            out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv4')

            out = tf.layers.conv2d(out, 1, list(out.shape[1:3]), strides=(1, 1), padding='valid',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out

        elif descriptor_type == 'cifar_2_song':

            out = tf.layers.conv2d(inputs, 96, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv1')

            out = tf.layers.conv2d(out, 96, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv2')

            out = tf.layers.conv2d(out, 96, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv3')

            out = tf.layers.conv2d(out, 192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv4')

            out = tf.layers.conv2d(out, 192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv5')

            out = tf.layers.conv2d(out, 192, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv6')

            out = tf.layers.conv2d(out, 192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv7')

            out = tf.layers.conv2d(out, 192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv8')

            out = tf.layers.conv2d(out, 192, kernel_size=(1, 1), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv9')


            out = tf.layers.conv2d(out, 1, list(out.shape[1:3]), strides=(1, 1), padding='valid',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')



            # out = conv(inputs, 96, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv1')
            # out = tf.nn.elu(out)
            # out = conv(out, 96, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv2')
            # out = tf.nn.elu(out)
            # out = conv(out, 96, kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv3')
            # out = tf.nn.elu(out)
            # out = conv(out, 192, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv4')
            # out = tf.nn.elu(out)
            # out = conv(out, 192, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv5')
            # out = tf.nn.elu(out)
            # out = conv(out, 192, kernel=3, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv6')
            # out = tf.nn.elu(out)
            # out = conv(out, 192, kernel=3, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv7')
            # out = tf.nn.elu(out)
            # out = conv(out, 192, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv8')
            # out = tf.nn.elu(out)
            # out = conv(out, 192, kernel=1, stride=1, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv9')
            # out = tf.nn.elu(out)
            # out = conv2d(out, 1, list(out.shape[1:3]), strides=(1, 1), padding='VALID', activate_fn=None, name="fc")
            return out

        elif descriptor_type == 'cifar_2_elu':

            out = tf.layers.conv2d(inputs, 64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.leaky_relu, name='conv1')

            out = tf.layers.conv2d(out, 128, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.leaky_relu, name='conv2')

            out = tf.layers.conv2d(out, 256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.leaky_relu, name='conv3')

            out = tf.layers.conv2d(out, 256, kernel_size=(5, 5), strides=(2, 2), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.leaky_relu, name='conv4')

            out = tf.layers.conv2d(out, 1, list(out.shape[1:3]), strides=(1, 1), padding='valid',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out

        elif descriptor_type == 'cifar_2_residual':


            out = residual_block_ebm(inputs, 64, "convt1")

            out1 = residual_block_ebm(out, 128, "convt2")

            out2 = residual_block_ebm(out1, 256, "convt3")

            out = tf.layers.conv2d(out2, 1, list(out2.shape[1:3]), strides=(1, 1), padding='valid',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out


        elif descriptor_type == 'cifar_2_elu_pooling':

            out = tf.layers.conv2d(inputs, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv1')


            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv2')

            out = tf.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(out)

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv3')

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv4')

            out = tf.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(out)

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(1, 1), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv5')

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(2, 2), padding='same',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005),
                                   activation=tf.nn.elu, name='conv6')

            out = tf.layers.conv2d(out, 1, list(out.shape[1:3]), strides=(1, 1), padding='valid',
                                   kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out

        elif descriptor_type == 'SA':

            is_training = True
            #out = dilated_conv2d(input_, 64, kernel=(3, 3), rate=2, padding='SAME', activate_fn=None, name='dil_conv1')
            out = conv2d(inputs, 64, kernal=(3, 3), strides=(1, 1), padding='SAME', activate_fn=None, name="conv1")
            out = tf.contrib.layers.batch_norm(out, is_training=is_training)
            out = leaky_relu(out)

            #out = dilated_conv2d(out, 128, kernel=(2, 2), rate=2, padding='SAME', activate_fn=None, name='dil_conv2')
            #out = conv2d(out, 128, kernal=(1, 1), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2")
            out = conv2d(out, 128, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2")
            out = tf.contrib.layers.batch_norm(out, is_training=is_training)
            out = leaky_relu(out)

            #out = non_local_block_sim(out, name="SA_1")

            #out = dilated_conv2d(out, 256, kernel=(2, 2), rate=2, padding='SAME', activate_fn=None, name='dil_conv3')
            #out = conv2d(out, 256, kernal=(1, 1), strides=(2, 2), padding='SAME', activate_fn=None, name="conv3")
            out = conv2d(out, 256, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv3")

            out = tf.contrib.layers.batch_norm(out, is_training=is_training)
            out = leaky_relu(out)

            #out = non_local_block_sim(out, name="SA_2")

            #out = dilated_conv2d(out, 256, kernel=(2, 2), rate=2, padding='SAME', activate_fn=None, name='dil_conv4')
            out = conv2d(out, 256, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv4")
            out = tf.contrib.layers.batch_norm(out, is_training=is_training)
            out = leaky_relu(out)

            out = conv2d(out, 1, list(out.shape[1:3]), strides=(1, 1), padding='VALID', activate_fn=None, name="fc")
            out = tf.contrib.layers.batch_norm(out, is_training=is_training)
            out = leaky_relu(out)





            #out = non_local_block_sim(out, name="SA_2")
            #convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            #out = leaky_relu(out)

            return out

        elif descriptor_type == 'cifar1':

            out = tf.layers.conv2d(inputs, 64, kernel_size=(5, 5), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

            out = tf.layers.conv2d(out, 256, kernel_size=(3, 3), strides=(1, 1), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

            out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out

        elif descriptor_type == 'mnist':

            out = tf.layers.conv2d(inputs, 64, kernel_size=(5, 5), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

            out = tf.layers.conv2d(out, 128, kernel_size=(3, 3), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

            out = tf.layers.conv2d(out, 256, kernel_size=(3, 3), strides=(1, 1), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

            out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out

        elif descriptor_type == 'mnist2':

            out = tf.layers.conv2d(inputs, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

            out = tf.layers.conv2d(out, 128, kernel_size=(4, 4), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

            out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

            out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
            kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

            return out

        else:
            return NotImplementedError


def encoder(inputs, encoder_type, z_size, reuse=False):
    with tf.variable_scope('VAE_encoder', reuse=reuse):


        if encoder_type == 'cifar':

             out = tf.layers.conv2d(inputs, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

             out = tf.layers.conv2d(out, 128, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

             out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

             out = tf.layers.conv2d(out, 512, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv4')

             #out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
             #kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))

        elif encoder_type == 'mnist':

             out = tf.layers.conv2d(inputs, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

             out = tf.layers.conv2d(out, 128, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

             out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

             out = tf.layers.conv2d(out, 512, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv4')

             #out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
             #kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))

        elif encoder_type == 'mnist_web':

             out = tf.layers.conv2d(inputs, 64, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')
             out = tf.nn.dropout(out, 0.8)

             out = tf.layers.conv2d(out, 64, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')
             out = tf.nn.dropout(out, 0.8)

             out = tf.layers.conv2d(out, 64, kernel_size=(4, 4), strides=(1, 1), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')
             out = tf.nn.dropout(out, 0.8)

             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)

             # sd = 0.5 * (1e-6 + tf.nn.softplus(tf.layers.dense(out, units=z_size)))

             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))

        elif encoder_type == 'mnist_web_no_drop':

             out = tf.layers.conv2d(inputs, 64, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')
             #out = tf.nn.dropout(out, 0.8)

             out = tf.layers.conv2d(out, 64, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')
             #out = tf.nn.dropout(out, 0.8)

             out = tf.layers.conv2d(out, 64, kernel_size=(4, 4), strides=(1, 1), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')
             #out = tf.nn.dropout(out, 0.8)

             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))


        elif encoder_type == 'mnist_web_norm':

             is_training =True
             out = conv2d(inputs, 64, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv1")
             out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)

             out = conv2d(out, 64, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2")
             out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)

             out = conv2d(out, 64, kernal=(4, 4), strides=(1, 1), padding='SAME', activate_fn=None, name="conv3")
             out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)


             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))


        elif encoder_type == 'mnist_erik2':

             is_training =True
             out = conv2d(inputs, 128, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv1")
             #out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)

             out = conv2d(out, 256, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2")
             #out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)

             out = conv2d(out, 512, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv3")
             #out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)


             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             log_sd = tf.layers.dense(out, units=z_size)

             log_sd = 1e-6 + tf.nn.softplus(log_sd)

             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(0.5*log_sd))



             sd = log_sd

        elif encoder_type == 'mnist_erik':

             is_training =True
             out = conv2d(inputs, 64, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv1")
             #out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)

             out = conv2d(out, 64, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv2")
             #out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)

             out = conv2d(out, 64, kernal=(4, 4), strides=(2, 2), padding='SAME', activate_fn=None, name="conv3")
             #out = tf.contrib.layers.batch_norm(out, is_training=is_training)
             out = leaky_relu(out)


             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             log_sd = tf.layers.dense(out, units=z_size)

             log_sd = 1e-6 + tf.nn.softplus(log_sd)


             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(0.5*log_sd))

             sd = log_sd

        elif encoder_type == 'SA':

             out = tf.layers.conv2d(inputs, 64, kernel_size=(3, 3), strides=(1, 1), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')
             #    kernel_initializer=tf.contrib.layers.xavier_initializer()
             out = tf.layers.conv2d(out, 128, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

             out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

             out = tf.layers.conv2d(out, 512, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv4')

             #out = tf.layers.conv2d(out, 100, list(out.shape[1:3]), strides=(1, 1), padding='valid',
             #kernel_initializer=tf.initializers.random_normal(stddev=0.005), name='fc')

             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))

        else:

             out = tf.layers.conv2d(inputs, 64, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv1')

             out = tf.layers.conv2d(out, 128, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv2')

             out = tf.layers.conv2d(out, 256, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005), activation=tf.nn.leaky_relu, name='conv3')

             out = tf.layers.conv2d(out, 512, kernel_size=(4, 4), strides=(2, 2), padding='same',
             kernel_initializer=tf.initializers.random_normal(stddev=0.005),  activation=tf.nn.leaky_relu, name='conv4')

             out = tf.contrib.layers.flatten(out)
             mn = tf.layers.dense(out, units=z_size)
             sd = 0.5 * tf.layers.dense(out, units=z_size)
             epsilon = tf.random_normal(tf.stack([tf.shape(out)[0], z_size]))
             z = mn + tf.multiply(epsilon, tf.exp(sd))

        return z, mn, sd


def generator(inputs, generator_type, image_size, reuse=False, is_training=True):
    with tf.variable_scope('VAE_generator', reuse=reuse):
        if generator_type == 'scene':

            inputs = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])

            convt1 = convt2d(inputs, (None, image_size // 16, image_size // 16, 512), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt1")
            convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
            convt1 = leaky_relu(convt1)

            convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(5, 5)
                             , strides=(2, 2), padding="SAME", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            convt3 = convt2d(convt2, (None, image_size // 4, image_size // 4, 128), kernal=(5, 5)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = leaky_relu(convt3)

            convt4 = convt2d(convt3, (None, image_size // 2, image_size // 2, 64), kernal=(5, 5)
                             , strides=(2, 2), padding="SAME", name="convt4")
            convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
            convt4 = leaky_relu(convt4)

            convt5 = convt2d(convt4, (None, image_size, image_size, 3), kernal=(5, 5)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt5 = tf.nn.tanh(convt5)

            return convt5


        elif generator_type == 'cifar':

            convt1 = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])

            convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            convt3 = convt2d(convt2, (None, image_size // 4, image_size // 4, 128), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = leaky_relu(convt3)

            convt4 = convt2d(convt3, (None, image_size // 2, image_size // 2, 64), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt4")
            convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
            convt4 = leaky_relu(convt4)

            convt5 = convt2d(convt4, (None, image_size, image_size, 64), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt5 = leaky_relu(convt5)

            convt6 = convt2d(convt5, (None, image_size, image_size, 3), kernal=(3, 3)
                             , strides=(1, 1), padding="SAME", name="convt6")
            convt6 = tf.nn.tanh(convt6)

            return convt6

        elif generator_type == 'cifar_2_convt2d':

            convt1 = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])

            convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            convt3 = convt2d(convt2, (None, image_size // 4, image_size // 4, 128), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = leaky_relu(convt3)

            convt4 = convt2d(convt3, (None, image_size // 2, image_size // 2, 64), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt4")
            convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
            convt4 = leaky_relu(convt4)

            convt5 = convt2d(convt4, (None, image_size, image_size, 64), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt5 = tf.contrib.layers.batch_norm(convt5, is_training=is_training)
            convt5 = leaky_relu(convt5)

            convt6 = convt2d(convt5, (None, image_size, image_size, 3), kernal=(3, 3)
                             , strides=(1, 1), padding="SAME", name="convt6")
            convt6 = tf.nn.tanh(convt6)

            return convt6


        elif generator_type == 'cifar_2_conv2d_up':

            convt1 = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])

            convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            conv3 = conv2d(convt2, 128, kernal=(3, 3), strides = (1, 1), name='conv3')
            conv3 = usample(conv3)
            conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training)
            conv3 = leaky_relu(conv3)

            conv4 = conv2d(conv3, 64, kernal=(3, 3), strides=(1, 1), name='conv4')
            conv4 = usample(conv4)
            conv4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training)
            conv4 = leaky_relu(conv4)

            conv5 = conv2d(conv4, 64, kernal=(3, 3), strides=(1, 1), name='conv5')
            conv5 = usample(conv5)
            conv5 = tf.contrib.layers.batch_norm(conv5, is_training=is_training)
            conv5 = leaky_relu(conv5)

            conv6 = conv2d(conv5, 3, kernal=(3, 3), strides=(1, 1), name='conv6')
            conv6 = tf.nn.tanh(conv6)

            return conv6

        elif generator_type == 'cifar_2_dil_conv2d_up':

            convt1 = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])


            convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            conv3 = dilated_conv2d(convt2, 128, kernel=(3, 3), rate=2, padding='SAME', activate_fn=None, name='dil_conv1')
            #conv3 = conv2d(convt2, 128, kernal=(3, 3), strides = (1, 1), name='conv3')
            conv3 = usample(conv3)
            conv3 = tf.contrib.layers.batch_norm(conv3, is_training=is_training)
            conv3 = leaky_relu(conv3)

            conv4 = dilated_conv2d(conv3, 64, kernel=(3, 3), rate=2, padding='SAME', activate_fn=None, name='dil_conv2')
            #conv4 = conv2d(conv3, 64, kernal=(3, 3), strides=(1, 1), name='conv4')
            conv4 = usample(conv4)
            conv4 = tf.contrib.layers.batch_norm(conv4, is_training=is_training)
            conv4 = leaky_relu(conv4)

            conv5 = dilated_conv2d(conv4, 64, kernel=(3, 3), rate=2, padding='SAME', activate_fn=None, name='dil_conv3')
            #conv5 = conv2d(conv4, 64, kernal=(3, 3), strides=(1, 1), name='conv5')
            conv5 = usample(conv5)
            conv5 = tf.contrib.layers.batch_norm(conv5, is_training=is_training)
            conv5 = leaky_relu(conv5)

            conv6 = dilated_conv2d(conv5, 3, kernel=(3, 3), rate=2, padding='SAME', activate_fn=None, name='dil_conv4')
            #conv6 = conv2d(conv5, 3, kernal=(3, 3), strides=(1, 1), name='conv6')
            conv6 = tf.nn.tanh(conv6)

            return conv6

        elif generator_type == 'SA':

            convt1 = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])

            convt2 = convt2d(convt1, (None, image_size // 8, image_size // 8, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            convt2 = non_local_block_sim(convt2, name="SA_4")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            convt3 = convt2d(convt2, (None, image_size // 4, image_size // 4, 128), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = leaky_relu(convt3)

            #convt3 = non_local_block_sim(convt3, name="SA_4")
            #convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            #convt3 = leaky_relu(convt3)


            convt4 = convt2d(convt3, (None, image_size // 2, image_size // 2, 64), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt4")
            convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
            convt4 = leaky_relu(convt4)

            convt5 = convt2d(convt4, (None, image_size, image_size, 64), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt5 = leaky_relu(convt5)

            convt6 = convt2d(convt5, (None, image_size, image_size, 3), kernal=(3, 3)
                             , strides=(1, 1), padding="SAME", name="convt6")
            convt6 = tf.nn.tanh(convt6)

            return convt6

        elif generator_type == 'residual':

            inputs = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])

            convt1 = convt2d(inputs, (None, 4, 4, 512), kernal=(4, 4), strides=(1, 1), padding="VALID", name="convt1")

            convt2 = residual_block(convt1, 256, "convt2")

            convt3 = residual_block(convt2, 128, "convt3")

            convt4 = residual_block(convt3, 64, "convt4")

            convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
            convt4 = leaky_relu(convt4)

            convt5 = convt2d(convt4, (None, image_size, image_size, 64), kernal=(3, 3)
                             , strides=(1, 1), padding="SAME", name="convt5")
            convt5 = tf.contrib.layers.batch_norm(convt5, is_training=is_training)
            convt5 = leaky_relu(convt5)

            convt6 = convt2d(convt5, (None, image_size, image_size, 3), kernal=(3, 3)
                             , strides=(1, 1), padding="SAME", name="convt6")

            #convt5 = conv2d(convt4, 3, kernal=(3, 3), strides=(1, 1), name='conv5')

            convt6 = tf.nn.tanh(convt6)

            return convt6


        elif generator_type == 'mnist':

            inputs = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])
            convt1 = convt2d(inputs, (None, 4, 4, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt1")
            convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
            convt1 = leaky_relu(convt1)

            convt2 = convt2d(convt1, (None, 7, 7, 128), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = leaky_relu(convt2)

            convt3 = convt2d(convt2, (None, 14, 14, 64), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = leaky_relu(convt3)

            convt4 = convt2d(convt3, (None, image_size, image_size, 1), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt4 = tf.nn.tanh(convt4)

            return convt4

        elif generator_type == 'mnist2':

            inputs = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])
            convt1 = convt2d(inputs, (None, 4, 4, 256), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt1")
            convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
            convt1 = tf.nn.relu(convt1)

            convt2 = convt2d(convt1, (None, 7, 7, 128), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = tf.nn.relu(convt2)

            convt3 = convt2d(convt2, (None, 14, 14, 64), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = tf.nn.relu(convt3)

            convt4 = convt2d(convt3, (None, image_size, image_size, 1), kernal=(3, 3)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt4 = tf.nn.tanh(convt4)

            return convt4

        elif generator_type == 'mnist_erik':

            inputs = tf.reshape(inputs, [-1, 1, 1, inputs.get_shape()[1]])
            convt1 = convt2d(inputs, (None, 4, 4, 1024), kernal=(4, 4)
                             , strides=(1, 1), padding="VALID", name="convt1")
            convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
            convt1 = tf.nn.relu(convt1)

            convt2 = convt2d(convt1, (None, 7, 7, 512), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt2")
            convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
            convt2 = tf.nn.relu(convt2)

            convt3 = convt2d(convt2, (None, 14, 14, 256), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt3")
            convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
            convt3 = tf.nn.relu(convt3)

            convt4 = convt2d(convt3, (None, image_size, image_size, 1), kernal=(4, 4)
                             , strides=(2, 2), padding="SAME", name="convt5")
            convt4 = tf.nn.tanh(convt4)

            return convt4

        else:
            return NotImplementedError
