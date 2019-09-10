import tensorflow as tf
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np
import os

class Models:
    def __init__(self, input_size, model, modelPath, preModelPath='', pretrained=False):
        self.inW = input_size[0]
        self.inH = input_size[1]
        self.inC = input_size[2]
        self.pretrained = pretrained
        self.preModelPath = preModelPath
        self.modelPath = modelPath
        if model == 'unet':self.get_unet_keras()


    def get_unet_keras(self):
        lr = tf.keras.layers
        inputs = tf.keras.Input(shape=(self.inW, self.inH, self.inC))
        conv1 = lr.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv1 = lr.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        pool1 = lr.MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = lr.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = lr.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        pool2 = lr.MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = lr.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = lr.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        pool3 = lr.MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = lr.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = lr.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = lr.Dropout(0.5)(conv4)
        pool4 = lr.MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = lr.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = lr.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = lr.Dropout(0.5)(conv5)

        up6 = lr.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            lr.UpSampling2D(size=(2, 2))(drop5))
        merge6 = lr.concatenate([drop4, up6], axis=3)
        conv6 = lr.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = lr.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

        up7 = lr.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            lr.UpSampling2D(size=(2, 2))(conv6))
        merge7 = lr.concatenate([conv3, up7], axis=3)
        conv7 = lr.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = lr.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

        up8 = lr.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            lr.UpSampling2D(size=(2, 2))(conv7))
        merge8 = lr.concatenate([conv2, up8], axis=3)
        conv8 = lr.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = lr.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = lr.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            lr.UpSampling2D(size=(2, 2))(conv8))
        merge9 = lr.concatenate([conv1, up9], axis=3)
        conv9 =lr. Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = lr.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = lr.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv10 = lr.Conv2D(1, 1, activation='sigmoid')(conv9)

        self.model = tf.keras.models.Model(inputs=inputs, outputs=conv10)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

        if self.pretrained and self.preModelPath != '':
            self.model.load_weights(self.preModelPath)


    def train(self, generator, steps, epochs):
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(self.modelPath, monitor='loss',
                                                              verbose=1, save_best_only=True)
        self.model.fit_generator(generator,steps_per_epoch=steps,epochs=epochs,callbacks=[model_checkpoint])

    def predict_images(self, test_path, postfix = 'pred'):
        for file in os.listdir(test_path):
            name, ext = file.split('.')
            im = io.imread(test_path + file)
            im = self.preprocess(im)
            pred = self.model.predict(im)
            pred_im = pred[0, ..., 0]
            outFile = test_path + name + '_pred.' + ext
            io.imsave(outFile, pred_im)

    def preprocess(self, img):
        im = resize(img, (256, 256))
        if len(im.shape) > self.inC:
            im = rgb2gray(im)
        if np.max(im) > 1.0:
            im = im / 255.0
        im = np.reshape(im, [1, 256, 256, self.inC])
        return im