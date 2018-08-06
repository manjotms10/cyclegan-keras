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
import cv2

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
        self.df = 64
        self.batch_size = 16
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
                                                      rec_A, rec_B,
                                                      iden_A, iden_B])
    
        self.combined.compile(loss=['mse','mse','mse','mse','mse','mse'], 
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
        
        out = Conv2D(3, (3,3), padding='same')(y4)
        
        return Model(inputs = inp, outputs = out)
    
    def build_discriminator(self):
        inp = Input(shape = (256, 256, 3))
        x1 = conv2d(inp, self.df, (3, 3))
        x2 = conv2d(x1, self.df*2, (3,3))
        x3 = conv2d(x2, self.df*4, (3,3))
        x4 = conv2d(x3, self.df*8, (3,3))
        x5 = Flatten()(x4)
        x6 = Dense(96, activation = 'relu')(x5)
        x7 = Dense(1, activation='relu')(x6)
        
        return Model(inputs = inp, outputs = x7)
    
    def data_generator(self):
        train_a = glob.glob('datasets/apple2orange/trainA/*')
        train_b = glob.glob('datasets/apple2orange/trainB/*')
        x = []
        y = []
        while True:
            idx = np.random.choice(995, self.batch_size)
            for i in idx:
                x.append(img_to_array(load_img(train_a[i])))
                y.append(img_to_array(load_img(train_b[i])))
            x = np.array(x)
            y = np.array(y)
            
            yield x, y
            
    def train(self):
        data = self.data_generator()
        epochs = self.epochs
        fakes = np.zeros((self.batch_size, 1))
        valid = np.ones((self.batch_size, 1))
        
        for ep in range(epochs):
            img_A, img_B = next(data)
            
            l1 = self.d_A.train_on_batch(img_A, valid)
            l2 = self.d_B.train_on_batch(img_B, valid)
            
            fake_B = self.gA2B.predict(img_A)
            fake_A = self.gB2A.predict(img_B)
            
            l3 = self.d_A.train_on_batch(fake_B, fakes)
            l4 = self.d_A.train_on_batch(fake_A, fakes)
            
            hist = self.combined.train_on_batch([img_A, img_B], 
                                                [valid, valid,
                                                 img_A, img_B,
                                                 img_A, img_B])
        
gan = GAN()
gan.train()
