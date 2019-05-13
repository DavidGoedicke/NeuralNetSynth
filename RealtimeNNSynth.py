#!/usr/bin/env python
# coding: utf-8
import math
import numpy as np

import mido
import librosa

import IPython.display as ipd
import matplotlib.pyplot as plt
from torch import optim
import torch.utils.data as utils

import torch.nn.functional as F
from torch import nn
import torch

from torch.autograd import Variable
from threading import Thread
from time import sleep
import time
import pyaudio

from  numpy_ringbuffer import RingBuffer


#model  = models.resnet18(pretrained=False)
#model.load_state_dict(torch.load("../models//UrbanSoundResnet18Fft2048Hop128.pth", map_location=lambda storage, loc: storage));
#model.cpu()
#model.eval()


sr=44100
n_fft=2048
hopLength=1024
SequenceLength = 25
BatchSize=128

pa = pyaudio.PyAudio()
running = True
MidiInputArray=np.zeros(128)
AUdioOutArray=np.zeros(500)
inport = mido.open_input()
globalBuffer = np.zeros(1024)

############
class LSTMNetwork(nn.Module):
    def __init__(self):
        super(LSTMNetwork,self).__init__()
        # Defining the layers, 128, 64, 10 units each

        #self.num_layers=1
        self.hiddenLayerSize=1024
        #self.word_lstm_init_h = nn.Parameter(torch.zeros(self.num_layers, BatchSize, self.hiddenLayerSize).type(torch.FloatTensor), requires_grad=True)
        #self.word_lstm_init_c = nn.Parameter(torch.zeros(self.num_layers, BatchSize, self.hiddenLayerSize).type(torch.FloatTensor), requires_grad=True)
        #self.word_lstm_init_h.cuda()
        #self.word_lstm_init_c.cuda()
        #self.fc0 = nn.LSTM(128, self.hiddenLayerSize,num_layers=self.num_layers,batch_first=True)

        self.fc0 = nn.Linear(128, self.hiddenLayerSize)
        self.fc1 = nn.Linear(self.hiddenLayerSize, self.hiddenLayerSize*2)
        self.fc2 = nn.Linear(self.hiddenLayerSize*2, n_fft+2)
        #self.h0 = torch.zeros(self.num_layers, 1, (n_fft+2*2)).cuda() # 2 for bidirection
        #self.c0 = torch.zeros(self.num_layers, 1, (n_fft+2*2)).cuda()


    def forward(self, x, ActualBatchSize):
        ''' Forward pass through the network, returns the output logits '''
        #self.h0=h0
        #self.c0=c0
        #x=x.view(SequenceLength,ActualBatchSize,128)
        #x,hidden = self.lstm(x,(self.word_lstm_init_h,self.word_lstm_init_c)) #(self.h0,self.c0)

        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return x,0,0

model = LSTMNetwork()
model.load_state_dict(torch.load('./Demo.model'))
model.eval()

def callback(in_data, frame_count, time_info, status):
    global globalBuffer
    data = globalBuffer
    print(np.sum(globalBuffer))
    return (data, pyaudio.paContinue)
stream = pa.open(format = pyaudio.paFloat32,
                    channels=1,
                    rate=44100,
                    output=True,
                    stream_callback=callback)


def startProgram(targetLength=20):
    global globalBuffer
    stream.start_stream()
    t0 = time.time()
    sharedBuffer =[]
    while running:
        for msg in inport.iter_pending():
            if msg.type=='note_on':
                MidiInputArray[msg.note]+=1.0;
            elif msg.type=='note_off':
                MidiInputArray[msg.note]-=1.0;
        with torch.no_grad():
            output ,_,_= model.forward(torch.Tensor(MidiInputArray),0)
            outputArray=output.numpy()
            #print(outputArray.shape)
            a=np.array(np.int((n_fft/2+1)) *[1+1j])
            a.real=outputArray[:np.int(n_fft/2+1)]
            a.imag=outputArray[np.int(n_fft/2+1):]

            sharedBuffer.append(a)
            if(len(sharedBuffer)>=2):
                sharedBuffer=sharedBuffer[-2:]
                transformedArray=np.array(sharedBuffer).T
                #print(transformedArray.shape)
                Y_infered2 = librosa.istft(transformedArray,hop_length=hopLength)
                globalBuffer = Y_infered2
                #print(np.sum(globalBuffer))
        if ( targetLength>0 )and ( (time.time()-t0)>=targetLength):
            break;


if __name__ == '__main__':
    print("Starting Running");
    startProgram(0)
    print("Stopping!");
    time.sleep(2)
    stream.close()
    pa.terminate()




# In[ ]:
