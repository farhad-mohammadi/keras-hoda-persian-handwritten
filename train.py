from HodaDatasetReader import  read_hoda_dataset
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

#load dataset
print('Reading train dataset (Train 60000.cdb)...')
x_train, y_train = read_hoda_dataset(dataset_path='dataset\Train 60000.cdb',
    images_height=32,
    images_width=32,
    one_hot=False,
    reshape=False)

print('Reading test dataset (Test 20000.cdb)...')
x_test, y_test = read_hoda_dataset(dataset_path='dataset\Test 20000.cdb',
    images_height=32,
    images_width=32,
    one_hot=False,
    reshape=False)

print('Reading remaining samples dataset (RemainingSamples.cdb)...')
x_remaining, y_remaining = read_hoda_dataset('dataset\RemainingSamples.cdb',
    images_height=32,
    images_width=32,
    one_hot=False,
    reshape=False)

#convert numerical label to categorical label
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test,10)
y_remaining= keras.utils.to_categorical(y_remaining,10)

# the shape of the input images
in_shape = x_train.shape[1:]
#the number of classes
n_classes = 10

# define model
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=in_shape))
model.add(MaxPool2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

# define loss and optimizer
model.compile(loss = 'categorical_crossentropy',optimizer='adam', metrics=['accuracy'])

#print(model.summary())

# fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=1, batch_size=64, verbose=1)

#save the model
model.save('model.h5')

