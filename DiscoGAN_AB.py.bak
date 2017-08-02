import tensorflow as tf
import tensorflow.contrib as tf_contrib
import celebA
import os

tf.contrib.layers
layers = tf_contrib.layers

batch_size = 32
max_step = 10000
save_steps = 5000
min_queue_examples = 256

image_size = 64

train_dir = "./train/"

# Hyper parameter
learning_rate = 0.0002
adam_beta1 = 0.9
adam_beta2 = 0.999
adam_epsilon = 1e-08

# Choose style
style_A = 'Blond_Hair'
style_B = 'Black_Hair'
constraint = 'Male'
constraint_type = '1'
is_test = False
n_test = 200


arg_scope = tf_contrib.framework.arg_scope
def lrelu(x, leak=0.2, name='lrelu'):
    # https://github.com/tensorflow/tensorflow/issues/4079
    with tf.variable_scope(name) :
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def relu(x, name='relu') :
    with tf.variable_scope(name) :
        return tf.nn.relu(x)

def generator(scope_name, x, is_training=True, reuse=False) :
    with tf.variable_scope(scope_name + '/generator') as scope :
        if reuse :
            scope.reuse_variables()
        batch_norm_params = {'decay' : 0.999,
                             'epsilon' : 0.001,
                             'is_training' : is_training,
                             'scope' : 'batch_norm'}

        with arg_scope([layers.conv2d, layers.conv2d_transpose],
                       kernel_size = [4,4],
                       stride=[2,2],
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params,
                       weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')) :
            with arg_scope([layers.conv2d], activation_fn=lrelu) :
                conv1 = layers.conv2d(inputs=x, num_outputs=64, normalizer_fn=None, biases_initializer=None, scope='ge_conv1')
                conv2 = layers.conv2d(inputs=conv1, num_outputs=64*2, scope='ge_conv2')
                conv3 = layers.conv2d(inputs=conv2, num_outputs=64*4, scope='ge_conv3')
                conv4 = layers.conv2d(inputs=conv3, num_outputs=64*8, scope='ge_conv4')

            with arg_scope([layers.conv2d_transpose], activation_fn=tf.nn.relu) :
                d_conv1 = layers.conv2d_transpose(inputs=conv4, num_outputs=64*4, scope='ge_dconv1')
                d_conv2 = layers.conv2d_transpose(inputs=d_conv1, num_outputs=64*2, scope='ge_dconv2')
                d_conv3 = layers.conv2d_transpose(inputs=d_conv2, num_outputs=64, scope='ge_dconv3')
                # original code use sigmoid
                generated_image = layers.conv2d_transpose(inputs=d_conv3, num_outputs=3,activation_fn=tf.nn.tanh, normalizer_fn=None, biases_initializer=None, scope='generated_image')

                return generated_image
def discriminator(scope_name, x, reuse=False) :
    with tf.variable_scope(scope_name + '/discriminator') as scope :
        if reuse :
            scope.reuse_variables()
            # tf.get_variable_scope().reuse_variables()
        batch_norm_params = {'decay' : 0.999,
                             'epsilon' : 0.001,
                             'scope' : 'batch_norm'}

        with arg_scope([layers.conv2d],
                       kernel_size=[4,4],
                       stride=[2,2],
                       activation_fn=lrelu,
                       normalizer_fn=layers.batch_norm,
                       normalizer_params=batch_norm_params,
                       weights_regularizer=layers.l2_regularizer(0.0001, scope='l2_decay')) :

            conv1 = layers.conv2d(inputs=x, num_outputs=64, normalizer_fn=None, biases_initializer=None, scope='dis_conv1')
            conv2 = layers.conv2d(inputs=conv1, num_outputs=64*2, scope='dis_conv2')
            conv3 = layers.conv2d(inputs=conv2, num_outputs=64*4, scope='dis_conv3')
            conv4 = layers.conv2d(inputs=conv3, num_outputs=64*8, scope='dis_conv4')

            # padding option same? valid?
            discriminator_value = layers.conv2d(inputs=conv4, num_outputs=1, stride=[1,1], padding='VALID', activation_fn=None, normalizer_fn=None, biases_initializer=None, scope='discriminator_value')

            return discriminator_value
def GAN_loss(logits, is_real=True) :
    if is_real :
        labels = tf.ones_like(logits)
    else :
        labels = tf.zeros_like(logits)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))

    return loss

def Reconstruction_loss(x, y, method='MSE' ) :
    if method == 'MSE' :
        return tf.losses.mean_squared_error(labels=x, predictions=y)
    elif method == 'cosine' :
        return tf.losses.cosine_distance(labels=x, predictions=y)
    elif method == 'hinge' :
        return tf.losses.hinge_loss(labels=x, logits=y)

def DiscoGAN_model(mode) :
    with tf.Graph().as_default() :
        domain_A_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./cat_dog/trainA/*.jpg"), capacity=200)
        domain_B_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./cat_dog/trainB/*.jpg"), capacity=200)
        image_reader = tf.WholeFileReader()

        _, A_file = image_reader.read(domain_A_queue)
        _, B_file = image_reader.read(domain_B_queue)

        domain_A = tf.image.decode_jpeg(A_file)
        domain_B = tf.image.decode_jpeg(B_file)

        domain_A = tf.cast(tf.reshape(domain_A, shape=[64,64,3]), dtype=tf.float32) / 255.0
        domain_B = tf.cast(tf.reshape(domain_B, shape=[64,64,3]), dtype=tf.float32) / 255.0

        domain_A_batch = tf.train.shuffle_batch([domain_A],
                                                batch_size=batch_size,
                                                num_threads=1,
                                                capacity=min_queue_examples +3 * batch_size,
                                                min_after_dequeue=min_queue_examples,
                                                name="domain A")
        domain_B_batch = tf.train.shuffle_batch([domain_B],
                                                batch_size=batch_size,
                                                num_threads=1,
                                                capacity=min_queue_examples + 3 * batch_size,
                                                min_after_dequeue=min_queue_examples,
                                                name="domain B")

        if mode == 'translate' :
            G_AB = generator('G_AB', domain_A_batch, is_training=False)
            G_BA = generator('G_BA', domain_B_batch, is_training=False)

            G_ABA = generator('G_BA', G_AB, is_training=False, reuse=True)
            G_BAB = generator('G_AB', G_BA, is_training=False, reuse=True)

        if mode == 'train' :
            # generate image A->B or B->A
            G_AB = generator('G_AB', domain_A)
            G_BA = generator('G_BA', domain_B)

            # reconstruct image A->B->A or B->A->B
            G_ABA = generator('G_BA', G_AB, reuse=True)
            G_BAB = generator('G_AB', G_BA, reuse=True)

            # discriminate real image A or B
            D_real_A = discriminator('A', domain_A)
            D_real_B = discriminator('B', domain_B)

            # discriminate generated image A or B
            D_generated_A = discriminator('A', G_BA, reuse=True)
            D_generated_B = discriminator('B', G_AB, reuse=True) # logits_fake_A2B

            # GAN loss of real A or generated A
            loss_real_A = GAN_loss(logits=D_real_A, is_real=True)
            loss_generated_A = GAN_loss(logits=D_generated_A, is_real=False)

            # GAN loss of real B or generated B
            loss_real_B = GAN_loss(logits=D_real_B, is_real=True)
            loss_generated_B = GAN_loss(logits=D_generated_B, is_real=False)

            # loss of discriminator
            loss_discriminator_A = loss_real_A + loss_generated_A
            loss_discriminator_B = loss_real_B + loss_generated_B

            # loss of GAN
            # is_real = True... cuz, paper formula
            # How well it belongs to domain A or B
            loss_GAN_A = GAN_loss(logits=D_generated_A, is_real=True)
            loss_GAN_B = GAN_loss(logits=D_generated_B, is_real=True)

            # Reconstruction loss
            loss_reconst_A = Reconstruction_loss(domain_A, G_ABA)
            loss_reconst_B = Reconstruction_loss(domain_B, G_BAB)

            # Generator loss
            loss_G_AB = loss_GAN_B + loss_reconst_A
            loss_G_BA = loss_GAN_A + loss_reconst_B

            # Final loss
            loss_Generator = loss_G_AB + loss_G_BA
            loss_Discriminator = loss_discriminator_A + loss_discriminator_B
            # Separate variables for each function
            t_vars = tf.trainable_variables()

            D_vars = [var for var in t_vars if 'discriminator' in var.name]
            G_vars = [var for var in t_vars if 'generator' in var.name]

            for var in G_vars:
                print(var.name)
            for var in D_vars:
                print(var.name)

            # Add summaries.
            # Add loss summaries
            tf.summary.scalar("losses/loss_Discriminator", loss_Discriminator)
            tf.summary.scalar("losses/loss_Discriminator_A", loss_discriminator_A)
            tf.summary.scalar("losses/loss_Discriminator_B", loss_discriminator_B)
            tf.summary.scalar("losses/loss_Generator", loss_Generator)
            tf.summary.scalar("losses/loss_Generator_AB", loss_G_AB)
            tf.summary.scalar("losses/loss_Generator_BA", loss_G_BA)

            # Add histogram summaries
            for var in D_vars:
                tf.summary.histogram(var.op.name, var)
            for var in G_vars:
                tf.summary.histogram(var.op.name, var)

            # Add image summaries
            tf.summary.image('domain_A', domain_A, max_outputs=4)
            tf.summary.image('domain_B', domain_B, max_outputs=4)
            tf.summary.image('generated_images_AB', G_AB, max_outputs=4)
            tf.summary.image('generated_images_BA', G_BA, max_outputs=4)
            tf.summary.image('generated_images_ABA', G_ABA, max_outputs=4)
            tf.summary.image('generated_images_BAB', G_BAB, max_outputs=4)

        print('complete DiscoGAN_model')

        tf.summary.scalar('learning_rate', learning_rate)



        train_D = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon).minimize(loss_Discriminator, var_list=D_vars)
        train_G = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=adam_beta1, beta2=adam_beta2, epsilon=adam_epsilon).minimize(loss_Generator, var_list=G_vars)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        train_dir = "./train/"
        with tf.Session() as sess :
            sess.run(tf.global_variables_initializer())

            try :
                saver.restore(sess=sess, save_path=train_dir+"model.ckpt")
                print("model resotred")
            except :
                print("model not restored")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            summary_op = tf.summary.merge_all()


            for step in range(max_step) :

                _, _, loss_D, loss_G = sess.run([train_D, train_G, loss_Discriminator, loss_Generator])


                print("step: %d loss_D: %f, loss_G: %f"
                      % (step, loss_D, loss_G))


                if step % 200 == 0 :
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)

                if step % save_steps == 0 :
                    checkpoint_path = os.path.join(train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)


            coord.request_stop()
            coord.join(threads)
            print('finish training')




def main() :
    DiscoGAN_model(mode='train')


main()

