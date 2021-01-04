import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,)
parser.add_argument('--Epochs',type=int, default=10)
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--LR', type=float, default=0.03)

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

if__name__ == '__main__' :
    args = parser.parse_args()

    model = args.model
    Lr = args.LR
    Epochs = args.Epochs
    Batch_size = args.BATCH_SIZE
    
    train_ds, test_ds, validation_ds = utils.load_datasets(Batch_size)
    
    model = utils.load_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=Lr), metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_ds,
          EPOCHS=Epochs,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])
    
    model.save('./models')
    utils.graph(history)
    
