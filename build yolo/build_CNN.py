import cv2
import os
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import losses
from keras import optimizers
import random
import h5py
def build_data(path_dir,label):
    data_path = os.path.join(path_dir,'*g')
    files = glob.glob(data_path)
    data = []
    labels = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.resize(img,(200,200))
        data.append(img)
        labels.append(label)
    return data,labels
if __name__ == '__main__':
    data_label = os.listdir('Data')
    data_label.sort(reverse = True)
    label_to_index = {label:i for i,label in enumerate(data_label)}
    print(label_to_index)
    data  = []
    labels = []
    for label in data_label:
        y = [0 for i in range(len(data_label))]
        y[label_to_index[label]] = 1

        dt,lb = build_data(os.path.join('Data',label),y)
        data = data +dt
        labels = labels +lb
    indexes = list(range(len(data)))
    random.shuffle(indexes)
    x_train = []
    y_train = []
    for i in indexes:
        x_train.append(data[i])
        y_train.append(labels[i])
    x_train = np.array(x_train)
    y_train = np.array(y_train)

# 5. Định nghĩa model
    model = Sequential()

    model.add(Conv2D(8, (3, 3), activation='relu', input_shape=(200, 200, 3),strides=2))

    model.add(Conv2D(16, (3, 3), activation='relu',strides=2))

    model.add(Conv2D(32, (3, 3), activation='relu',strides=2))
    model.add(Conv2D(64, (3, 3), activation='relu',strides=2))
    model.add(Conv2D(128, (3, 3), activation='relu',strides=2))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dense(len(data_label),activation='softmax'))

    model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(lr=0.0001))
    model.fit(x_train, y_train, batch_size=128, epochs = 100)

    out = model.predict(x_train[0:1])

    print(out,y_train[0])
    for result in out:
        max = 0
        id_max = 0
        for id in range(len(result)):
            if result[id] >= max:
                max = result[id]
                id_max = id
        if max > 0.8:
            print('This is',data_label[id_max])
        else:
            print('What is this? I dont know')



