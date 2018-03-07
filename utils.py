# add original author

import os
import pickle as pickle

import numpy as np
import wget
import tarfile

URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

def download_CIFAR10(_dir):
  """ download CIFAR10 dataset """
  os.makedirs(_dir)
  original_dir = os.getcwd()
  os.chdir(_dir)
  filename = wget.download(URL)
  _file = tarfile.open(filename)
  _file.extractall()
  os.chdir(original_dir)

def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f, encoding = 'latin1')
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y

def load_CIFAR10(_dir):
  """ load all of cifar """
  if not os.path.isdir(_dir):
    download_CIFAR10(_dir)
  data_dir = _dir + '/' + 'cifar-10-batches-py'
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(data_dir, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(data_dir, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def prepare_splits(X_train, y_train, X_test, y_test):
  # Split the data into train, val, and test sets. In addition we will
  # create a small development set as a subset of the training data;
  # we can use this for development so our code runs faster.
  num_training = 49000
  num_validation = 1000
  num_test = 1000
  num_dev = 500

  # Our validation set will be num_validation points from the original
  # training set.
  mask = range(num_training, num_training + num_validation)
  X_val = X_train[mask]
  y_val = y_train[mask]

  # Our training set will be the first num_train points from the original
  # training set.
  mask = range(num_training)
  X_train = X_train[mask]
  y_train = y_train[mask]

  # We will also make a development set, which is a small subset of
  # the training set.
  mask = np.random.choice(num_training, num_dev, replace=False)
  X_dev = X_train[mask]
  y_dev = y_train[mask]

  # We use the first num_test points of the original test set as our
  # test set.
  mask = range(num_test)
  X_test = X_test[mask]
  y_test = y_test[mask]

  X_train, X_val, X_test, X_dev = preprocess_data(X_train, X_val,
                                                  X_dev, X_test)

  return X_train, y_train, X_val, y_val, X_dev, y_dev, X_test, y_test

def preprocess_data(X_train, X_val, X_dev, X_test):
  mean_image = np.mean(X_train, axis = 0, keepdims = True)
  X_train -= mean_image
  X_val -= mean_image
  X_test -= mean_image
  X_dev -= mean_image

  return X_train, X_val, X_test, X_dev

  # Whitenning
  # Standardization

def get_minibatch(X, y, bs):
  sz = len(y)
  indices = np.random.choice(sz, bs)
  return X[indices], y[indices]

def get_minibatch_onehot(X, y, bs):
  sz = len(y)
  indices = np.random.choice(sz, bs)
  X_batch = X[indices]
  y_batch = y[indices]
  y_one_hot = np.zeros((bs, 10))
  y_one_hot[y_batch] = 1
  return X_batch, y_one_hot


# Xtr, Ytr, Xte, Yte = load_CIFAR10('data')
# data = prepare_splits(Xtr, Ytr, Xte, Yte)
# X_train, y_train, X_val, y_val, X_dev, y_dev, X_test, y_test = data

# print(X_train.shape, X_val.shape, X_dev.shape, X_test.shape)
