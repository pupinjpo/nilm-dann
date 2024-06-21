import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout
from keras import optimizers
from keras import regularizers
import pickle

adam = optimizers.Adam(lr=0.00001,beta_1=0.9, beta_2=0.999 ,amsgrad=False)
f_input = Input(shape=(599,1))
H = Conv1D(30, 10, kernel_regularizer=regularizers.l2(0.001), activation='relu', use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', padding='valid')(f_input)
H = Dropout(0.5)(H)
H = Conv1D(30, 8, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu', padding='valid')(H)
H = Conv1D(40, 6, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu', padding='valid')(H)
H = Dropout(0.5)(H)
H = Conv1D(50, 5, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu', padding='valid')(H)
H = Conv1D(50, 5, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu', padding='valid')(H)
H = Dropout(0.5)(H)
H = Conv1D(1024, 570, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', activation='relu', padding='valid')(H)
H = Dropout(0.5)(H)
f_V = Dense(100, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', kernel_initializer='he_normal', activation='linear')(H)
F = Model(f_input,f_V)
F.compile(loss='mse', optimizer=adam)
c_input = Input(shape=(1,100))
H=Dense(100, kernel_regularizer=regularizers.l2(0.001), activation='linear')(c_input)
c_V=Dense(2, kernel_regularizer=regularizers.l2(0.001), use_bias='True', bias_initializer='he_normal', activation='sigmoid')(H)
C=Model(c_input, c_V)
C.compile(loss='mse', optimizer=adam)
net_input = Input(shape=(599,1))
H = F(net_input)
H = C(H)
net = Model(net_input, H)
net.compile(loss='mse', metrics=['accuracy'], optimizer=adam)
net.load_weights('in/microwave_net.h5')

with open('in/testing_dataset_microwave.pkl', 'rb') as f:
    data = pickle.load(f)
labels = data['lab']
input = data['data'][0, :, 0, :]
input = np.swapaxes(input, axis1=0, axis2=1)
input = np.expand_dims(input,axis=2)
output = net.predict(input)
output_length = np.shape(output)[0]
output_labels = np.zeros(output_length)
for i in range(0, output_length):
    if output[i, 0, 0] < output[i, 0, 1]:
        output_labels[i] = 1

print('Model accuracy [%]')
print(np.sum(np.abs(output_labels-labels[0,:]))/output_length)
