import pickle
import numpy as np
import pyriemann
import sklearn
import scipy
import matplotlib as mpl
mpl.use('Qt5Agg') # for using pyplot (pip install pyqt5)
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt, sosfiltfilt

# Pyriemann with OV Python scripting plugin --------------------------------------------------- written by Kyungho Won
# In the future, constant values will be changed to variables belong to Python scripitng plugin
#
# Step
# 1. Loads covariance matrices estimated using calibration EEG at the beginning and fits MDM (__init__)
# 2. During test scenario, python scripting module receives the segmented EEG from OpenViBE every epoch (input: signal)
# 3. In Python scripting plugin, the segmented EEG is band-pass filtered and transformed to a covariance matrix
# 4. The Fitted MDM predicts the current label with the covariance matrix
# 5. Python scripting plugin sends stimulution (predicted labels) as an output (output: stimulation)

# 6. Ohter external modules could be added

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
	nyq = fs/2
	low = lowcut/nyq
	high = highcut/nyq
	sos = butter(order, [low, high], btype='band', output='sos')
	# demean before filtering
	meandat = np.mean(data, axis=1)
	data = data - meandat[:, np.newaxis]
	y = sosfiltfilt(sos, data) # zero-phase filter # data: [ch x time]
	# specify pandlen to make the result the same as Matlab filtfilt()
	return y

def draw_feedback(nth, nClass):
	labels_arr = ['LEFT','RIGHT','UP','DOWN']
	mpl.rcParams['toolbar'] = 'None' # Remove tool bar (upper bar)

	plt.clf()
	plt.plot(0,0)
	ax = plt.gca()
	ax.set_facecolor('black')
	plt.xlim([-10, 10])
	plt.ylim([-10, 10])
	plt.axis('off')
	plt.title('%02d Predicted: %s' %(nth, labels_arr[int(nClass)-1]))

	if nClass == 1: # left
		plt.arrow(0,0, -4, 0, width=1)
	elif nClass == 2: # right
		plt.arrow(0,0, 4, 0, width=1)
	elif nClass == 3: # up
		plt.arrow(0,0, 0, 4, width=1)
	elif nClass == 4: # down
		plt.arrow(0,0, 0, -4, width=1)


class MyOVBox(OVBox):
	def __init__(self):
		OVBox.__init__(self)
		self.signalHeader = None
		self.nth_trial = 0

	def initialize(self):
		# Append to the box output a stimulation header. 
		self.output[0].append(OVStimulationHeader(0., 0.))

		# Load covariance matrices estimated from the calibrated EEG
		load_file = open(self.setting['Trained model path'], 'rb')
		trained = pickle.load(load_file)
		self.mdm = pyriemann.classification.MDM()
		self.mdm.metric = 'Riemann'
		self.mdm.fit(trained['COV'], trained['Labels'])	
		print('Training accuracy is', np.sum(self.mdm.predict(trained['COV'])==trained['Labels'])/len(trained['Labels']))
		print('== Trained COV:', trained['COV'].shape)
		print('==', self.mdm)
		print('\n\n')

		# User defined parameters
		self.lowbp = int(self.setting['low bp'])
		self.highbp = int(self.setting['high bp'])
		self.filterorder = int(self.setting['filter order'])
		self.sampling = int(self.setting['sampling rate'])
		self.isfeedback = self.setting['Feedback']
		self.ans_mi = [769, 770, 780, 774] # left right up down

		plt.ion()

	def process(self):
		for chunkIdx in range( len(self.input[0]) ):
			# borrowed from python-signal-average.py
			if(type(self.input[0][chunkIdx]) == OVSignalHeader): # called only once
				self.signalHeader = self.input[0].pop()

			elif(type(self.input[0][chunkIdx]) == OVSignalBuffer): # called every epoch
				chunk = self.input[0].pop()
				numpyBuffer = np.array(chunk, dtype=np.float64).reshape(tuple(self.signalHeader.dimensionSizes))
				# numpyBuffer has [ch x time]
				numpyBuffer = butter_bandpass_filter(numpyBuffer, self.lowbp, self.highbp, self.sampling, self.filterorder)

				# Pyriemann only accpets 3D inputs with [nMatrices, nCh, nTime]
				cur_input = np.expand_dims(numpyBuffer, axis=0) # now (1, nCh, nTime)
				COV_cur = pyriemann.estimation.Covariances().fit_transform(cur_input)
				predict_class = self.mdm.predict(COV_cur) # among [1, 2, 3, 4]
				print(predict_class)

				# send stimulation (classified results)
				stimSet = OVStimulationSet(self.getCurrentTime(), self.getCurrentTime()+1./self.getClock())
				stimSet.append(OVStimulation(self.ans_mi[int(predict_class)-1], self.getCurrentTime(), 0.))
				self.output[0].append(stimSet)
				self.nth_trial = self.nth_trial + 1

				if self.isfeedback == 'True':
					draw_feedback(self.nth_trial, predict_class)
								
	def uninitialize(self):
		end = self.getCurrentTime()
		self.output[0].append(OVStimulationEnd(end,end))
		print('uninitialize')
		plt.ioff()
		plt.close()

box = MyOVBox()	# When it ends (the last call)