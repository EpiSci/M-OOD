# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Apr. 2019 by wontak ryu.
ryu071511@gmail.com.
https://github.com/RRoundTable/EEN-with-Keras.

Building keras model.
'''
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU,Conv2DTranspose, Dense, ZeroPadding2D, Lambda, Input
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K
import argparse
import tensorflow as tf
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-task', type=str, default='poke', help='breakout | seaquest | flappy | poke | driving')
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-model', type=str, default='latent-3layer')
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-nfeature', type=int, default=64, help='number of feature maps')
parser.add_argument('-n_latent', type=int, default=4, help='dimensionality of z')
parser.add_argument('-lrt', type=float, default=0.0005, help='learning rate')
parser.add_argument('-epoch_size', type=int, default=500)
parser.add_argument('-loss', type=str, default='l2', help='l1 | l2')
parser.add_argument('-gpu', type=int, default=1)
parser.add_argument('-datapath', type=str, default='.', help='data folder')
parser.add_argument('-save_dir', type=str, default='./results/', help='where to save the models')
parser.add_argument('-height', type=int, default=50)
parser.add_argument('-width', type=int, default=50)
parser.add_argument('-nc', type=int, default=3)
parser.add_argument('-npred', type=int, default=1)
parser.add_argument('-n_out', type=int, default=3)
opt = parser.parse_args()

# setting
K.set_image_data_format("channels_first")
print("imgae data format : ",K.image_data_format())

"""
Convolution : (W-F+2P)/S+1
DeConvolution : S*(W-1)+F-P
"""
def g_network_encoder(opt):
    """Deterministic encoder
    :param opt: parser
    :return: keras Model
    """
    model = Sequential(name="g_encoder")
    # layer 1
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature,(3,3),(2,2),"valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (3, 3), (2, 2),"valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (2, 2), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def g_network_decoder(opt):
    """Deterministic decoder
    :param opt: parser
    :return: keras Model
    """
    k = 4 # poke
    model=Sequential(name="g_decoder")
    # layer 4
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (3,3), (1,1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (3, 3), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (4, 4), (2, 2), "valid"))
    return model

def phi_network_conv(opt):
    """Encoder for residual images(error)
    :param opt: parser
    :return: keras Model
    """
    model = Sequential(name="phi_conv")
    # layer 1
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (3, 3), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (3, 3), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (2, 2), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def phi_network_fc(opt):
    """FC layer for encoded error
    :param opt: parser
    :return: keras Model
    """
    model = Sequential(name="phi_fc")
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(opt.n_latent, activation='tanh'))
    return model

# conditional network
def f_network_encoder(opt):
    """Conditional encoder
    :param opt: parser
    :return: keras Model
    """
    model = Sequential(name="f_encoder")
    # layer 1
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (3, 3), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 2
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (3, 3), (2, 2), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 3
    model.add(ZeroPadding2D(1))
    model.add(Conv2D(opt.nfeature, (2, 2), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    return model

def f_network_decoder(opt:'parser'):
    """Conditional decoder
    :param opt: parser
    :return: keras Model
    """
    model = Sequential(name="f_decoder")
    # layer 4
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (3, 3), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 5
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.nfeature, (3, 3), (1, 1), "valid"))
    model.add(BatchNormalization())
    model.add(ReLU())
    # layer 6
    model.add(ZeroPadding2D(1))
    model.add(Conv2DTranspose(opt.n_out, (4, 4), (2, 2), "valid"))
    return model

def encoder_latent(opt):
    """Latent variable layer
    :param opt: parser
    :return: keras Model
    """
    model=Sequential(name="encoder_latent")
    model.add(Dense(opt.nfeature))
    return model

# Deterministic Model
class DeterministicModel:
    def __init__(self, opt):
        self.opt = opt
        self.g_network_encoder = g_network_encoder(self.opt)
        self.g_network_decoder = g_network_decoder(self.opt)

    def build(self):
        inputs = Input((self.opt.nc, self.opt.height, self.opt.width))
        outputs = self.g_network_decoder(self.g_network_encoder(inputs))
        model = Model(inputs, outputs)
        return model

    def get_layer(self):
        return [self.g_network_encoder, self.g_network_decoder]

class MultiInputLayer(layers.Layer):
    """
    Custom layer
    """
    def __init__(self, output_dim, opt):
        self.output_dim = output_dim
        self.opt = opt
        super(MultiInputLayer, self).__init__()

    def build(self, input_shape):
        """Create a trainable weight
        :param input_shape: [cond, target]
        """
        self.g_network_encoder = g_network_encoder(self.opt)
        self.g_network_decoder = g_network_decoder(self.opt)
        self.f_network_encoder = f_network_encoder(self.opt)
        self.phi_network_conv = phi_network_conv(self.opt)
        self.phi_network_fc = phi_network_fc(self.opt)
        self.encoder_latent = encoder_latent(self.opt)
        super(MultiInputLayer, self).build(input_shape)

    def call(self, x):
        """Feedforward
        :param x:  [inputs, targets]
        :return: hidden tensor
        """
        inputs = x[0]
        targets = x[1]
        pred_g = self.g_network_decoder(self.g_network_encoder(inputs))

        # residual
        r = Lambda((lambda x: x[1] - x[0]))([pred_g, targets])
        out_dim = K.int_shape(self.phi_network_conv(r))  # shape=(?, 64, 7, 7)
        z = self.phi_network_fc(K.reshape(self.phi_network_conv(r),
                                          (self.opt.batch_size, out_dim[1] * out_dim[2] * out_dim[3])))
        z = K.reshape(z, (self.opt.batch_size, self.opt.n_latent))
        z_emb = self.encoder_latent(z)
        z_emb = K.reshape(z_emb, (self.opt.batch_size, self.opt.nfeature, 1, 1))
        s = self.f_network_encoder(inputs)
        return Lambda((lambda x: tf.math.add(x[0], x[1])))([s, z_emb]) # tf.math.add : broadcast

    def get_layers(self):
        layers = [self.g_network_encoder, self.g_network_decoder, self.f_network_encoder,
                 self.phi_network_conv, self.phi_network_fc, self.encoder_latent]
        return layers

# Latent Variable Model
class LatentResidualModel3Layer:
    """
    Our Model : Error-Encoding-Network
    """
    def __init__(self, opt) -> None:
        """
        :param opt: parser
        """
        self.opt = opt
        self.g_network_encoder = g_network_encoder(self.opt)
        self.g_network_decoder = g_network_decoder(self.opt)
        self.phi_network_conv = phi_network_conv(self.opt)
        self.phi_network_fc = phi_network_fc(self.opt)
        self.f_network_encoder = f_network_encoder(self.opt)
        self.f_network_decoder = f_network_decoder(self.opt)
        self.encoder_latent = encoder_latent(self.opt)
        self.hidden = MultiInputLayer([64, 7, 7], self.opt)

    def build(self):
        """Error Encoding Network
        :return: keras Model
        """
        inputs_ = Input((self.opt.nc, self.opt.height, self.opt.width))
        targets_ = Input((self.opt.nc, self.opt.height, self.opt.width))
        h = self.hidden([inputs_, targets_])
        pred_f = self.f_network_decoder(h)
        model_f = Model([inputs_,targets_], pred_f)
        return model_f

    def get_model_z(self):
        """
        return latent variable model
        :return: keras Model
        """
        inputs = Input((self.opt.nc, self.opt.height, self.opt.width))
        targets =Input((self.opt.nc, self.opt.height, self.opt.width))
        z_emb = Lambda(self.get_latent)([inputs, targets])
        return Model([inputs, targets], z_emb)

    def get_latent(self, x):
        """
        :param x: [inputs, targets]
         - inputs : numpy array
         - targets : numpy array
        :return: latent variable
        """
        inputs = x[0]
        targets = x[1]
        pred_g = self.g_network_decoder(self.g_network_encoder(inputs))
        # residual
        r = Lambda((lambda x: x[0] - x[1]))([targets, pred_g])
        out_dim = K.int_shape(self.phi_network_conv(r))  # shape=(?, 64, 7, 7)
        z = self.phi_network_fc(K.reshape(self.phi_network_conv(r),
                                          (self.opt.batch_size, out_dim[1] * out_dim[2] * out_dim[3])))
        z = K.reshape(z, (self.opt.batch_size, self.opt.n_latent))
        return z

    def decode(self, inputs, z):
        """
        :param inputs: numpy_array(images)
        :param z: latent variable
        :return: prediction with latent variable
        """
        inputs = K.reshape(inputs, (self.opt.batch_size,
                                    self.opt.ncond * self.opt.nc,
                                    self.opt.height,
                                    self.opt.width))
        z = tf.convert_to_tensor(z)
        z_emb = self.encoder_latent(z)
        z_emb =  K.reshape(z_emb, (self.opt.batch_size, self.opt.nfeature,1, 1))
        s = self.f_network_encoder(inputs)
        h = Lambda((lambda x: tf.math.add(x[0], x[1])))([s, z_emb])
        pred = self.f_network_decoder(h)
        return K.eval(pred)

    def get_layers(self):
        layers = [self.g_network_encoder, self.g_network_decoder, self.f_network_encoder,
                  self.phi_network_conv, self.phi_network_fc, self.encoder_latent, self.f_network_decoder]
        return layers

    def load_weights(self, model):
        """Update layers
        :param model: trained model
        """
        transfer_layer = model.layers[2].get_layers()
        transfer_layer.append(model.layers[3])
        self.g_network_encoder = transfer_layer[0]
        self.g_network_decoder = transfer_layer[1]
        self.f_network_encoder = transfer_layer[2]
        self.phi_network_conv = transfer_layer[3]
        self.phi_network_fc = transfer_layer[4]
        self.encoder_latent = transfer_layer[5]
        self.f_network_decoder = transfer_layer[6]

class BaselineModel3Layer:
    def __init__(self, opt):
        self.opt = opt
        self.f_network_encoder = f_network_encoder(self.opt)
        self.f_network_decoder = f_network_decoder(self.opt)

    def build(self):
        inputs = Input((self.opt.nc, self.opt.height, self.opt.width))
        h = self.f_network_encoder(inputs)
        pred = self.f_network_decoder(h)
        return Model(inputs, pred)

if __name__ == '__main__':
    EEN = LatentResidualModel3Layer(opt)
    model = EEN.build()
    EEN.get_model_z()
    model.compile(optimizer = "Adam", loss = "mse")