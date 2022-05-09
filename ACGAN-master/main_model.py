# [Controllable PV Scenario Generation via Mixup-based Deep Generative Networks]

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from load import load_solar_data
from interpret import interpret_for_pattern


class Discriminator:

    def __init__(self, input_shape, w_dim):
        super(Discriminator, self).__init__()
        self.input_shape = input_shape
        self.conv_1 = layers.Conv2D(64, (5, 5), 2, padding='same')
        self.conv_2 = layers.Conv2D(128, (5, 5), 2, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        self.fc_1 = layers.Dense(w_dim)
        self.fc_2 = layers.Dense(1)
        self.relu1 = layers.LeakyReLU(alpha=0.2)
        self.relu2 = layers.LeakyReLU(alpha=0.2)
        self.relu3 = layers.LeakyReLU(alpha=0.2)

    def model(self):
        inputs = layers.Input(self.input_shape)
        x = self.conv_1(inputs)
        x = self.relu1(x)
        x = self.conv_2(x)
        x = self.bn_1(x)
        x = self.relu2(x)
        x = layers.Flatten()(x)
        x = self.fc_1(x)
        x = self.bn_2(x)
        x = self.relu1(x)
        out_logits = self.fc_2(x)
        out = keras.activations.sigmoid(out_logits)

        model = keras.models.Model(inputs=inputs, outputs=[x, out], name="Discriminator")
        return model


class Generator:
    def __init__(self, z_shape, w_dim):
        super(Generator, self).__init__()
        self.z_shape = z_shape

        self.fc_1 = layers.Dense(w_dim)
        self.fc_2 = layers.Dense(128 * 6 * 6)
        self.bn_1 = layers.BatchNormalization()
        self.bn_2 = layers.BatchNormalization()
        self.bn_3 = layers.BatchNormalization()
        self.up_conv_1 = layers.Conv2DTranspose(64, 4, 2, padding='same')
        self.up_conv_2 = layers.Conv2DTranspose(1, 4, 2, padding='same')

    def model(self):
        inputs = layers.Input(self.z_shape)
        x = self.fc_1(inputs)
        x = self.bn_1(x)
        x = layers.ReLU()(x)
        x = self.fc_2(x)
        x = self.bn_2(x)
        x = layers.ReLU()(x)
        x = layers.Reshape((6, 6, 128))(x)
        x = self.up_conv_1(x)
        x = self.bn_3(x)
        x = layers.ReLU()(x)
        o = self.up_conv_2(x)

        model = keras.models.Model(inputs=inputs, outputs=o, name="Generator")
        return model


class Classifier:
    def __init__(self, y_dim, inputs_shape, w_dim):
        super(Classifier, self).__init__()
        self.y_dim = y_dim
        self.inputs_shape = inputs_shape
        self.fc_1 = layers.Dense(w_dim)
        self.fc_2 = layers.Dense(self.y_dim)
        self.bn_1 = layers.BatchNormalization()

    def model(self):
        inputs = layers.Input(self.inputs_shape)
        x = self.fc_1(inputs)
        x = self.bn_1(x)
        x = layers.LeakyReLU(alpha=0.2)(x)
        out_logits = self.fc_2(x)
        out = keras.layers.Softmax()(out_logits)

        model = keras.models.Model(inputs=inputs, outputs=[out, out_logits], name="Classifier")
        return model


class Mixup_ACGAN():
    def __init__(
            self,
            epochs=6000,
            batch_size=64,
            image_shape=[24, 24, 1],
            dim_z=144,
            dim_y=12,  # The parameters for controlling the number of events
            dim_W=1024,
            dim_channel=1,
            learning_rate=5e-4,
            mixup=0.0
    ):
        self.mixup = mixup
        self.epochs= epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W = dim_W
        self.dim_channel = dim_channel

        self.g_optimizer = optimizers.Adam(lr=5 * self.lr, beta_1=0.5)
        self.d_optimizer = optimizers.Adam(lr=1 * self.lr, beta_1=0.5)
        self.c_optimizer = optimizers.Adam(lr=5 * self.lr, beta_1=0.5)
        self.g = self.generator_()
        self.d = self.discriminator_()
        self.c = self.classifier_()

        self.g.summary()
        self.d.summary()
        self.c.summary()

    def generator_(self):
        return Generator(self.dim_z + self.dim_y, self.dim_W).model()

    def discriminator_(self):
        return Discriminator(self.image_shape, self.dim_W).model()

    def classifier_(self):
        return Classifier(self.dim_y, self.dim_W, 64).model()

    def data_prepare_(self):
        datasets = load_solar_data()
        Y = interpret_for_pattern()
        trX, teX = datasets[:int(datasets.shape[0] * 0.8)], datasets[int(datasets.shape[0] * 0.8):]
        trY, teY = Y[:int(datasets.shape[0] * 0.8)], Y[int(datasets.shape[0] * 0.8):]
        import matplotlib.pyplot as plt
        plt.plot(trX[0])
        plt.show()
        return trX, trY, teX, teY

    def d_loss_fun(self, d_fake_logits, d_real_logits):
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_logits), logits=d_real_logits))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_logits), logits=d_fake_logits))
        total_loss = d_loss_fake+d_loss_real
        return total_loss

    def g_loss_fun(self, logits):
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(logits), logits=logits))
        return g_loss

    def q_loss_fun(self, code_logit_real, code_logit_fake,batch_labels):
        q_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=code_logit_real))
        q_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=code_logit_fake))
        q_loss = q_real_loss+q_fake_loss
        return q_loss

    @tf.function
    def train_step(self, X_batch, Y_batch):
        z = np.random.uniform(-1, 1, [self.batch_size, self.dim_z]).astype(np.float32)
        latent_code = tf.concat([z, Y_batch], 1)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape, tf.GradientTape() as q_tape:
            fake_imgs = self.g(latent_code, training=True)

            d_fake_out, d_fake_logits = self.d(fake_imgs, training=True)
            d_real_out, d_real_logits = self.d(X_batch, training=True)
            d_loss = self.d_loss_fun(d_fake_logits, d_real_logits)
            g_loss = self.g_loss_fun(d_fake_logits)
            code_fake, code_logit_fake = self.c(d_fake_out, training=True)
            code_real, code_logit_real = self.c(d_real_out, training=True)
            c_loss = self.q_loss_fun(code_real, code_fake, tf.cast(Y_batch, tf.float32))

        gradients_of_d = d_tape.gradient(d_loss, self.d.trainable_variables)
        gradients_of_g = g_tape.gradient(g_loss, self.g.trainable_variables)
        # c loss backprop to all the trainable-variables
        trainable_variables_q = self.c.trainable_variables + self.d.trainable_variables + self.g.trainable_variables
        gradients_q = q_tape.gradient(c_loss, trainable_variables_q)

        self.d_optimizer.apply_gradients(zip(gradients_of_d, self.d.trainable_variables))
        self.g_optimizer.apply_gradients(zip(gradients_of_g, self.g.trainable_variables))
        self.c_optimizer.apply_gradients(zip(gradients_q, trainable_variables_q))

        return d_loss, g_loss, c_loss, d_fake_out, d_real_out

    def train(self):
        trX, trY, _, _ = self.data_prepare_()

        trY = tf.one_hot(trY, depth=12)
        print("trY_shape", trY.shape)

        gen_loss_all = []
        P_real = []
        P_fake = []
        P_distri = []
        discrim_loss = []
        # begin training
        for epoch in range(self.epochs):
            print("====epoch" + str(epoch) + "====")
            trD = np.concatenate([trX, trY], axis=-1)
            Xs = trD[:, :trX.shape[-1]]
            Ys = trD[:, trX.shape[-1]:]
            Xs = Xs.reshape([-1, 24, 24, 1])

            for start, end in zip(
                    range(0, len(trY), self.batch_size),
                    range(self.batch_size, len(trY), self.batch_size)
            ):
                X_batch = Xs[start:end].reshape([-1, 24, 24, 1])
                Y_batch = Ys[start:end]

                if self.mixup > 0:
                    alpha = np.random.uniform(0, self.mixup)
                    n = np.random.randint(0, self.batch_size)
                    X_batch = alpha * X_batch[n, np.newaxis] + (1 - alpha) * X_batch
                    Y_batch = alpha * Y_batch[n, np.newaxis] + (1 - alpha) * Y_batch

                d_loss, g_loss, c_loss, d_fake_out, d_real_out = self.train_step(X_batch, Y_batch)
            print("d_real", np.mean(d_real_out))
            print("d_fake", np.mean(d_fake_out))
            print("c_loss", np.mean(c_loss))
            print("")
            self.saveWeights()

    def saveWeights(self):
        self.g.save_weights("record/generator.h5")
        self.d.save_weights("record/discriminator.h5")
        self.c.save_weights("record/classifier.h5")

    def inference(self):
        gen = self.g
        gen.load_weights("record/generator.h5")
        z = np.random.uniform(-1, 1, [1, self.dim_z]).astype(np.float32)
        z = np.tile(z, [12, 1])
        y = np.arange(0, 12)
        print(y)
        y = tf.one_hot(y, depth=12)
        # y = np.random.uniform(0, 1, [12, self.dim_y]).astype(np.float32)
        x = gen.call(tf.concat([z, y], 1))
        x = np.array(x)
        x = x.reshape((-1, 576))
        import matplotlib.pyplot as plt
        for i in range(12):
            plt.plot(x[i],linewidth =1.0)
        plt.show()


if __name__ == '__main__':
    Mixup_ACGAN().inference()
