from HodaDatasetReader import  read_hoda_dataset
from tensorflow import keras
from keras.models import load_model

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

# load model
model = load_model('model\\model.h5')

# evaluate the model
loss, acc = model.evaluate(x_train, y_train, verbose=0)
print('Train Accuracy: {} %'.format(round(acc*100,2)))

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print('Test Accuracy: {} %'.format(round(acc*100,2)))

loss, acc = model.evaluate(x_remaining, y_remaining, verbose=0)
print('Remaining Accuracy: {} %'.format(round(acc*100,2)))