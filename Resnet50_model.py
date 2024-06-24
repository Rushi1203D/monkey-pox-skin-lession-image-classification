print(classification_report(y_true, y_pred))

**ResNet50**

import tensorflow as tf
from tensorflow import keras
from keras.applications import resnet
from keras.layers import Flatten , Dense , Activation
from keras import optimizers , Sequential
from keras.optimizers import Adam
from keras import models , layers
from keras.layers import Input
from keras.models import Model
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import numpy as np
import itertools
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount("/content/gdrive")

train_path='/content/gdrive/MyDrive/Dataset/train'
val_path='/content/gdrive/MyDrive/Dataset/val'
test_data='/content/gdrive/MyDrive/Dataset/test'

train_data = tf.keras.utils.image_dataset_from_directory(
    directory = train_path,
    image_size =(224,224),
    batch_size = 32,
    label_mode = "categorical",
    seed = 43
)

val_data = tf.keras.utils.image_dataset_from_directory(
    directory = val_path,
    image_size =(224,224),
    batch_size = 32,
    label_mode = "categorical",
    seed = 43
)

test_data = tf.keras.utils.image_dataset_from_directory(
    directory = test_data,
    image_size =(224,224),
    batch_size = 32,
    label_mode = "categorical",
    seed = 43
)

resnet= tf.keras.applications.ResNet50(
                   include_top=False,
                   input_shape=(224,224,3),
                   pooling='avg',
                   classes=4,
                   weights='imagenet')

for each_layer in resnet.layers:
        each_layer.trainable=False

model = Sequential()
model.add(resnet)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(optimizer=Adam(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_data,
                    validation_data=val_data,
                    epochs=100)

model.evaluate(test_data)

classes = ["cp", "me", "mp", "nl"]
name = "ResNet50 Confusion matrix"

def plot_confusion_matrix(cm,
                          classes,
                          name,
                          cmap,
                          normalize=False,
                          title='ResNet50 Confusion matrix'):

    #plt.figure(figsize=(6,3))
    plt.subplot(1,3,3)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, weight = 'bold')
    plt.yticks(tick_marks, classes, weight = 'bold')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Labels', fontweight='bold')
    plt.xlabel('Predicted Labels', fontweight='bold')


def create_confusion_matrix(cm_plot_labels, name, y_true,y_pred, map_col):
    #Get the true and predicted labels
    y_predict_classes, y_true_classes = y_pred,y_true

    #Compute the confusion matrix
    confusion_matrix_computed = confusion_matrix(y_true_classes, y_predict_classes)

    #Plot the confusion matrix
    plot_confusion_matrix(confusion_matrix_computed, cm_plot_labels, name, map_col)
    # plot_confusion_matrix(conf_mat=confusion_matrix_computed, class_names=cm_plot_labels)


predictions = np.array([])
labels =  np.array([])
for x, y in test_data:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])



import keras
from matplotlib import pyplot as plt

plt.subplots(figsize=(11.5,3))
plt.subplot(1,3,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ResNet50 Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['Train_Acc', 'Val_Acc'], loc='lower right')

plt.subplot(1,3,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('ResNet50 Model Loss')
plt.ylabel('Loss')
plt.xlabel('Number of Epochs')
plt.legend(['Train_Loss', 'Val_Loss'], loc='upper right')

create_confusion_matrix(classes, name, labels, predictions, 'Greens')
