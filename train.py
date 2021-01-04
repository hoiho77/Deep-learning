import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--model')
parser.add_argument('--Epochs',type=int, default=10)
parser.add_argument('--BATCH_SIZE', type=int, default=32)
parser.add_argument('--LR', type=float, default=0.03)

def process_images(image, label):
    image = tf.image.per_image_standardization(image)
    image = tf.image.resize(image, (227,227))
    return image, label

def load_datasets(Batch_size) :
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    #Clarify class_names
    CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    #validation data is the first 4000 images within the 20,00 training data (cannot use all datasets due to limits of memory)
    validation_images, validation_labels = train_images[:4000], train_labels[:4000]
    train_images, train_labels = train_images[4000:20000], train_labels[4000:20000]
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))
    
    train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
    test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
    validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
    
    train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=Batch_size, drop_remainder=True))
    test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=Batch_size, drop_remainder=True))
    validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=Batch_size, drop_remainder=True))

def graph(history) :
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], 'g-', label='accuracy')
    plt.plot(history.history['val_accuracy'], 'k--', label='val_accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='best')
    
    plt.save('./training_graph.jpg')
    
if__name__ == '__main__' :
    args = parser.parse_args()

    #model = args.model
    Lr = args.LR
    Epochs = args.Epochs
    Batch_size = args.BATCH_SIZE
    
    train_ds, test_ds, validation_ds = load_datasets(Batch_size)
    
    model = load_model()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.optimizers.SGD(lr=Lr), metrics=['accuracy'])
    model.summary()
    
    history = model.fit(train_ds,
          EPOCHS=Epochs,
          validation_data=validation_ds,
          validation_freq=1,
          callbacks=[tensorboard_cb])
    
    model.save('./models')
    graph(history)
    
