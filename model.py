# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 11:56:19 2018

@author: mbilkhu
"""

import tensorflow as tf

from keras.layers import Conv2D, LeakyReLU, BatchNormalization, UpSampling2D
from keras.layers import Input, Concatenate, Flatten, Dense, Activation
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob

def conv2d(input_layer, filters, kernel):
    w = Conv2D(filters=filters, kernel_size = kernel, strides=2, padding='same')(input_layer)
    x = LeakyReLU(alpha = 0.2)(w)
    y = BatchNormalization()(x)
    return y

def deconv2d(input_layer, concat_layer, filters, kernel):
    w = UpSampling2D(size = (2,2))(input_layer)
    x = Conv2D(filters=filters, kernel_size = kernel, padding='same')(w)
    y = LeakyReLU(alpha = 0.2)(x)
    z = Concatenate()([y, concat_layer])
    return z


class GAN():
    def __init__(self):
        self.df = 16
        self.batch_size = 8
        self.epochs = 1
        self.input_image_shape = (256, 256, 3)
        
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        
        self.d_A.compile(optimizer = Adam(lr = 0.0002),
                         loss = 'mse', metrics = ["accuracy"])
        self.d_B.compile(optimizer = Adam(lr = 0.0002),
                         loss = 'mse', metrics = ["accuracy"])
        
        A = Input(shape = self.input_image_shape)
        B = Input(shape = self.input_image_shape)
        
        self.gA2B = self.build_generator()
        self.gB2A = self.build_generator()
        
        # Model not compiled
        fake_B = self.gA2B(A)
        fake_A = self.gB2A(B)
        
        rec_A = self.gB2A(fake_B)
        rec_B = self.gA2B(fake_A)
        
        iden_A = self.gB2A(A)
        iden_B = self.gA2B(B)
        
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)
        
        self.d_A.trainable = False
        self.d_B.trainable = False
        
        self.combined = Model(inputs = [A, B],
                                     outputs = [valid_A, valid_B,
                                                fake_A, fake_B,
                                                      rec_A, rec_B,
                                                      iden_A, iden_B])
    
        self.combined.compile(loss=['mse','mse','mse','mse','mse','mse','mse','mse'],
                              optimizer = Adam(lr=0.0002), 
                              metrics = ["accuracy"])
        

    def build_generator(self):
        inp = Input(shape = (256, 256, 3))
        x1 = conv2d(inp, self.df, (3, 3))
        x2 = conv2d(x1, self.df*2, (3,3))
        x3 = conv2d(x2, self.df*4, (3,3))
        x4 = conv2d(x3, self.df*8, (3,3))
        
        y1 = deconv2d(x4, x3, self.df*4, (3,3))
        y2 = deconv2d(y1, x2, self.df*2, (3,3))
        y3 = deconv2d(y2, x1, self.df, (3,3))
        y4 = UpSampling2D(size=(2,2))(y3)
        
        out = Conv2D(3, (1,1), padding='same')(y4)
        
        return Model(inputs = inp, outputs = out)
    
    def build_discriminator(self):
        inp = Input(shape = (256, 256, 3))
        x1 = conv2d(inp, self.df, (3, 3))
        x2 = conv2d(x1, self.df*2, (3,3))
        x3 = conv2d(x2, self.df*4, (3,3))
        x4 = conv2d(x3, self.df*8, (3,3))
        x5 = Flatten()(x4)
        x5 = Dense(512, activation='relu')(x5)
        x6 = Dense(96, activation = 'relu')(x5)
        x7 = Dense(1, activation='sigmoid')(x6)
        
        return Model(inputs = inp, outputs = x7)
    
    def load_image(self, path):
        img = load_img(path, target_size = self.input_image_shape[:-1])
        img = img_to_array(img)
        img = img/127.5 - 1
        return img
    
    def data_generator(self):
        train_a = glob.glob('datasets/cityscapes/trainA/*')
        train_b = glob.glob('datasets/cityscapes/trainB/*')
        
        while True:
            x, y = [], []
            idx = np.random.choice(995, self.batch_size)
            for i in idx:
                x.append(self.load_image(train_a[i]))
                y.append(self.load_image(train_b[i]))
            x = np.array(x)
            y = np.array(y)
            
            yield x, y
            
    def train(self, epochs = 15, batch_size = 8):
        data = self.data_generator()
        fakes = np.zeros((batch_size, 1))
        valid = np.ones((batch_size, 1))
        
        for ep in range(epochs):
            img_A, img_B = next(data)
            
            print('-- Training DA --')
            self.d_A.fit(img_A, valid, epochs = 1, batch_size = batch_size)
            fake_A = self.gB2A.predict(img_B)
            self.d_A.fit(fake_A, fakes, epochs = 1, batch_size = batch_size)
            
            print('-- Training DB --')
            self.d_B.fit(img_B, valid, epochs = 1, batch_size = batch_size)
            fake_B = self.gA2B.predict(img_A)
            self.d_B.fit(fake_B, fakes, epochs = 1, batch_size = batch_size)
            
            print('-- Training Combined Model --')
            hist = self.combined.fit([img_A, img_B], 
                                            [valid, valid,
                                             img_B, img_A,
                                             img_A, img_B,
                                             img_A, img_B],
                                             epochs = 1,
                                             batch_size = batch_size)

        
gan = GAN()
gan.train()
