import sys
import socket
import numpy as np
from time import time, sleep
import mne
from mne import io

# EEGNet-specific imports
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm

### Functionality:
## This file must be able to do the following, in this order
#  1. establish a TCP connection with the Acquisition Client in OpenViBE
#  2. define a functional tensorflow model
#  3. load in pre-trained weights for that model
#  4. send a TCP stimulation to signal the loading is finished
#  5. recieve EEG data over the TCP connection
#  6. check the shape (crop/resize the data if necessary for the model)
#  7. make a predict() call on the data to estimate its class
#  8. send a TCP stimulation to indicate the predicted class
#  9. repeat steps 5-8 until the program is finished
# 10. FUTURE: trigger external device depending on classification (shock, wheelchair, etc.)


#-------------------------------------------------------------------------
###  1. establish a TCP connection with the Acquisition Client in OpenViBE
## The tag format is the same as with Shared Memory Tagging. It comprises three blocks of 8 bytes:
#
# ----------------------------------------------------------------------
# |  padding  (8 bytes) |  event id (8 bytes)  |  timestamp (8 bytes)  |
# ----------------------------------------------------------------------
#  
# The padding is only for consistency with Shared Memory Tagging and has no utility.
# The event id informs about the type of event happening.
# The timestamp is the posix time (ms since Epoch) at the moment of the event.
# It the latter is set to 0, the acquisition server issues its own timestamp upon reception of the stimulation.

# host and port of tcp tagging server
HOST = '127.0.0.1'
PORT = 15361  # defaults

# transform a value into an array of byte values in little-endian order.
def to_byte(value, length):
    for x in range(length):
        yield value%256
        value//=256


def sendOVstim(ID, sock, delay=0):  # Artificial delay (ms). It may need to be increased if the time to send the tag is too long and causes tag loss.
    # create the three pieces of the tag, padding, event_id and timestamp
    padding = [0] * 8
    event_id = list(to_byte(ID, 8))
    timestamp = list(to_byte(int(time() * 1000) + delay, 8))  # can be either posix time in ms, or 0 to have acquisition server do it
    sock.sendall(bytearray(padding + event_id + timestamp))  # send stimulation marker


# connect to (s)end port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
print("Sending socket connected!")

# connect to (r)eceive port
"""
OpenViBE TCP writer
1) if the outputs of the box are raw numeric values, the box first
sends every connecting client eight variables of unit32 (32 bytes in total)

2) After the possible global header, the data itself is sent.
The data is a stream of float64 for signal and streammatrix. The data orientation is [nsamples x nchannels]
i.e. all channels for one sample are sent in a sequence, then all channels of the next sample, and so on.

For stimulations, the clsdata is unit64 if user chooses raw, or char strings otherwise
"""
r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # open TCP socket
r.connect(('127.0.0.1', 5678))  # connect to OV TCP writer port
print("Receiving socket connected!")

# read the global header before receiving any streams
# all header values are uint32 - do not read 32 bytes at once, just 4 bytes at a time
# variable names sourced from documentation: http://openvibe.inria.fr//documentation/3.1.0/Doc_BoxAlgorithm_TCPWriter.html
Version = np.frombuffer(r.recv(4), np.uint32)[0]
Endianness = np.frombuffer(r.recv(4), np.uint32)[0]
_ = np.frombuffer(r.recv(4), np.uint32)[0]  # Frequency bytes always returning as 0... so hard code it for now
Frequency = 512
Channels = np.frombuffer(r.recv(4), np.uint32)[0]
Samples_per_chunk = np.frombuffer(r.recv(4), np.uint32)[0]
Reserved0 = np.frombuffer(r.recv(4), np.uint32)[0]
Reserved1 = np.frombuffer(r.recv(4), np.uint32)[0]
Reserved2 = np.frombuffer(r.recv(4), np.uint32)[0]


#----------------------------------------------
###  2. define a functional tensorflow model
def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))
    
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)


#-------------------------------------------------
###  3. load in pre-trained weights for that model
classes = 4
chans = 9
samples = 513  # hard coded from network training #
# Check the channel count, ideally use the channel selector in OpenViBE to align the script input with the expected channel count
print("Channel count not the same as what the model is expecting!" if Channels != chans else "Channel count matches!")

# Configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
model = EEGNet(nb_classes = classes, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

model.load_weights('C:/Users/Owner/Documents/OpenViBE Scenarios/TCP/4class_80trial_wet.h5')


#---------------------------------------------------------------
###  4. send a TCP stimulation to signal the loading is finished
EVENT_ID = 5+0x8207  # currently set to OVTK_StimulationId_TrainCompleted to signal player continue
sendOVstim(EVENT_ID, s)

#---------------------------------------------------------------
###  4.5. Prepare for prediction loop

PREDICTION_EPOCHS = 100 # Number of predictions to make
SAMPLE_SIZE = 8 # Size of float64 data type in bytes
DESIRED_SHAPE = (Channels, samples) # Desired shape for eeg_epoch
MODEL_SHAPE = (1, Channels, samples, 1) # Desired shape for model prediction

# Accounts for any leftover samples at the end of buffer not used for prediction sample
leftovers = None

# Keeps track of the number of predictions made
Num_Preds = 0

while (Num_Preds < PREDICTION_EPOCHS):
    #-----------------------------------------------
    ###  5. recieve EEG data over the TCP connection
    # begin reading stream from OV
    # first define the number of samples needed for classification
    gathered = 0
    needed = samples
    eeg_epoch = None

    if leftovers:  # if we have any leftovers, use them
            print("Leftover =", leftovers)
            eeg_epoch = leftovers

    while gathered < needed:  # gather that many samples
        chunk = np.zeros((Channels, Samples_per_chunk)).astype(np.float64)
        for i in range(Channels):
            data = r.recv(SAMPLE_SIZE*Samples_per_chunk)  # float64 (8bytes) * (number of channels)
            chunk[i] = np.frombuffer(data, np.float64)  # should be (32,)
        
        if (eeg_epoch is None):
            eeg_epoch = chunk
        else:
            eeg_epoch = np.concatenate((eeg_epoch, chunk), axis=1)  # v-stack into a single "epoch"
        gathered += Samples_per_chunk
    
    if (eeg_epoch.shape != DESIRED_SHAPE):
        leftovers = eeg_epoch[DESIRED_SHAPE[0]:, DESIRED_SHAPE[1]:]
        eeg_epoch = eeg_epoch[:DESIRED_SHAPE[0], :DESIRED_SHAPE[1]]

    #------------------------------------------------------------------------
    ###  6. check the shape (crop/resize the data if necessary for the model)
    print("Initial shape =", eeg_epoch.shape)
    data_sample = eeg_epoch.reshape(MODEL_SHAPE)
    print("New shape =", data_sample.shape)


    #--------------------------------------------------------------
    ###  7. make a predict() call on the data to estimate its class
    probs = model.predict(data_sample)
    preds = probs.argmax(axis = -1)
    print(probs)
    print(preds)


    #-------------------------------------------------------------
    ###  8. send a TCP stimulation to indicate the predicted class
    # TODO: replace prints with proper OV stim codes
    if preds[0] == 0: print("left")
    elif preds[0] == 1: print("right")
    elif preds[0] == 2: print("up")
    elif preds[0] == 3: print("down")

    Num_Preds += 1
