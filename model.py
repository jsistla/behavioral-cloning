import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.utils.visualize_util import plot
import matplotlib.pyplot as plt
import utils

tf.python.control_flow_ops = tf

number_of_epochs = 12 
number_of_samples_per_epoch = 20032
number_of_validation_samples = 6400
learning_rate = 1e-4
activation_relu = 'elu'

# try to load a previously saved model
model_path = 'model'

# This model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
model = Sequential()

model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(64, 64, 3)))

# starts with five convolutional and maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation(activation_relu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164))
model.add(Activation(activation_relu))

# Testing to see if model converges faster.
#model.add(Dense(512))
#model.add(Activation(activation_relu))

model.add(Dense(100))
model.add(Activation(activation_relu))

model.add(Dense(50))
model.add(Activation(activation_relu))

model.add(Dense(10))
model.add(Activation(activation_relu))

model.add(Dense(1))

model.summary()

model.compile(optimizer=Adam(learning_rate), loss="mse", )

plot(model, to_file='model.png', show_shapes=True)
try:
    model.load_weights(model_path+'.h5')

	
except IOError:
    print ('no previous model found....\n')

# a callback to save a list of the losses over each batch during training
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))


# a callback to save a list of the accuracies over each batch during training
class AccHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_acc = []
        
    def on_batch_end(self, batch, logs={}):
        self.train_acc.append(logs.get('acc'))

loss_hist = LossHistory()
acc_hist = AccHistory()
early_stop = EarlyStopping(monitor='val_loss', patience=3, 
						   verbose=0, mode='min')
checkpoint = ModelCheckpoint('checkpoints/'+model_path+'-{epoch:02d}-{val_loss:.4f}', 
							 monitor='val_loss',verbose=0, save_best_only=True, 
							 save_weights_only=False, mode='auto')

# creating generators for training and validationi data
train_gen = utils.generate_next_batch()
validation_gen = utils.generate_next_batch()

history = model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1,callbacks=[early_stop, checkpoint])

# finally save our model and weights
utils.save_model(model)

"""
# plot the data for training and test
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('history')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
"""
