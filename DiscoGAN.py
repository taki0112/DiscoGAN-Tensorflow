from ops import *
from utils import *
from glob import glob
import time

class DiscoGAN(object):
    def __init__(self, sess, epoch, dataset, batch_size, learning_rate, beta1, beta2, weight_decay, checkpoint_dir, result_dir, log_dir, sample_dir):
        self.model_name = 'DiscoGAN'
        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.dataset_name = dataset

        self.print_freq = 100
        self.epoch = epoch
        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        self.height = 256
        self.width = 256
        self.channel = 3

        self.trainA, self.trainB = prepare_data(dataset_name=self.dataset_name)
        self.num_batches = max(len(self.trainA) // self.batch_size, len(self.trainB) // self.batch_size)
        # may be i will use deque

    def generator(self, x, is_training=True, reuse=False, scope="generator"):
        with tf.variable_scope(scope, reuse=reuse):
            conv_dim = [128, 256, 512]
            deconv_dim = [256,128,64]

            """ CONV """
            x = conv_layer(x, filter_size=64, kernel=[4,4], stride=2, padding=1, layer_name='conv1')
            x = lrelu(x, 0.2)

            index = 2
            for dim in conv_dim :
                x = conv_layer(x, filter_size=dim, kernel=[4,4], stride=2, padding=1, layer_name='conv'+str(index))
                x = Batch_Normalization(x, training=is_training, scope='batch'+str(index-1))
                x = lrelu(x, 0.2)
                index += 1

            """ DECONV """

            index = 1
            for dim in deconv_dim :
                x = deconv_layer(x, filter_size=dim, kernel=[4,4], stride=2, padding=1, layer_name='deconv'+str(index))
                x = Batch_Normalization(x, training=is_training, scope='batch'+str(index+3))
                x = relu(x)
                index += 1

            x = deconv_layer(x, filter_size=3, kernel=[4,4], stride=2, padding=1, layer_name='deconv4')
            x = tanh(x)

            return x

    def discriminator(self, x, is_training=True, reuse=False, scope="discriminator"):
        with tf.variable_scope(scope, reuse=reuse):
            conv_dim = [128, 256, 512]

            x = conv_layer(x, filter_size=64, kernel=[4,4], stride=2, padding=1, layer_name='conv1')
            x = lrelu(x, 0.2)

            index = 2
            for dim in conv_dim :
                x = conv_layer(x, filter_size=dim, kernel=[4,4], stride=2, padding=1, layer_name='conv'+str(index))
                x = Batch_Normalization(x, training=is_training, scope='batch'+str(index-1))
                x = lrelu(x, 0.2)
                index += 1

            x = conv_layer(x, filter_size=1, kernel=[4,4], stride=1, padding='VALID', layer_name='conv5')
            # x = sigmoid(x)

            return x
    def build_model(self):
        self.domain_A = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.channel], name='domain_A') # real A
        self.domain_B = tf.placeholder(tf.float32, [self.batch_size, self.width, self.height, self.channel], name='domain_B') # real B

        """ Define Generator, Discriminator """
        # Generator
        self.fake_B = self.generator(self.domain_A, is_training=True, scope='generator_B') # B'
        self.fake_A = self.generator(self.domain_B, is_training=True, scope='generator_A') # A'

        self.recon_A = self.generator(self.fake_B, is_training=True, reuse=True, scope='generator_A') # A -> B' -> A
        self.recon_B = self.generator(self.fake_A, is_training=True, reuse=True, scope='generator_B') # B -> A -> B

        # Discriminator
        self.dis_real_A = self.discriminator(self.domain_A, is_training=True, scope='discriminator_A')
        self.dis_real_B = self.discriminator(self.domain_B, is_training=True, scope='discriminator_B')

        self.dis_fake_A = self.discriminator(self.fake_A, is_training=True, reuse=True, scope='discriminator_A')
        self.dis_fake_B = self.discriminator(self.fake_B, is_training=True, reuse=True, scope='discriminator_B')



        """ Loss Function """
        # Discriminator
        self.loss_real_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_real_A, labels=tf.ones_like(self.dis_real_A)))
        self.loss_fake_A = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_fake_A, labels=tf.zeros_like(self.dis_fake_A)))

        self.loss_real_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_real_B, labels=tf.ones_like(self.dis_real_B)))
        self.loss_fake_B = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_fake_B, labels=tf.zeros_like(self.dis_fake_B)))

        self.D_A_loss = self.loss_real_A + self.loss_fake_A
        self.D_B_loss = self.loss_real_B + self.loss_fake_B

        # Generator
        self.recon_A_loss = tf.losses.mean_squared_error(labels=self.domain_A, predictions=self.recon_A)
        self.recon_B_loss = tf.losses.mean_squared_error(labels=self.domain_B, predictions=self.recon_B)

        self.G_A_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_fake_A, labels=tf.ones_like(self.dis_fake_A))) + self.recon_A_loss
        self.G_B_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dis_fake_B, labels=tf.ones_like(self.dis_fake_B))) + self.recon_B_loss

        self.Discriminator_loss = self.D_A_loss + self.D_B_loss
        self.Generator_loss = self.G_A_loss + self.G_B_loss

        """ Training """
        t_vars = tf.trainable_variables()
        G_vars = [var for var in t_vars if 'generator' in var.name]
        D_vars = [var for var in t_vars if 'discriminator' in var.name]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            Adam = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2)
            self.G_optim = Adam.minimize(self.Generator_loss, var_list=G_vars)
            self.D_optim = Adam.minimize(self.Discriminator_loss, var_list=D_vars)


        """" Summary """
        self.g_loss = tf.summary.scalar('G_loss', self.Generator_loss)
        self.d_loss = tf.summary.scalar('D_loss', self.Discriminator_loss)

    def train(self):
        # initialize all variables
        tf.global_variables_initializer().run()

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                random_index_A = np.random.choice(len(self.trainA), size=self.batch_size, replace=False)
                random_index_B = np.random.choice(len(self.trainB), size=self.batch_size, replace=False)
                batch_A_images = self.trainA[random_index_A]
                batch_B_images = self.trainB[random_index_B]

                # Update D
                _, summary_str = self.sess.run(
                    [self.D_optim, self.d_loss],
                    feed_dict = {self.domain_A : batch_A_images, self.domain_B : batch_B_images})
                self.writer.add_summary(summary_str, counter)

                # Update G
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.G_optim, self.g_loss],
                    feed_dict = {self.domain_A : batch_A_images, self.domain_B : batch_B_images})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time))

                if np.mod(counter, 100) == 0 :
                    save_images(batch_A_images, [self.batch_size, 1],
                                './{}/real_A_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))
                    save_images(batch_B_images, [self.batch_size, 1],
                                './{}/real_B_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))

                    save_images(fake_A, [self.batch_size, 1],
                                './{}/fake_A_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))
                    save_images(fake_B, [self.batch_size, 1],
                                './{}/fake_B_{:02d}_{:04d}.jpg'.format(self.sample_dir, epoch, idx+2))

                # After an epoch, start_batch_id is set to zero
                # non-zero value is only for the first epoch after loading pre-trained model
                start_batch_id = 0

                # save model
                self.save(self.checkpoint_dir, counter)

            # save model for final step
            self.save(self.checkpoint_dir, counter)


    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def test(self):
        tf.global_variables_initializer().run()
        test_A_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testA'))
        test_B_files = glob('./dataset/{}/*.*'.format(self.dataset_name + '/testB'))

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(self.result_dir, 'index.html')
        index = open(index_path, 'w')
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        for sample_file  in test_A_files : # A -> B
            print('Processing A image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.fake_B, feed_dict = {self.domain_A :sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")

        for sample_file  in test_B_files : # B -> A
            print('Processing B image: ' + sample_file)
            sample_image = np.asarray(load_test_data(sample_file))
            image_path = os.path.join(self.result_dir,'{0}'.format(os.path.basename(sample_file)))

            fake_img = self.sess.run(self.fake_A, feed_dict = {self.domain_B : sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()