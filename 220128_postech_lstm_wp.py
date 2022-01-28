
import numpy as np
import urllib.request
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data
from tqdm import tqdm

urllib.request.urlretrieve("https://www.gutenberg.org/files/2600/2600-0.txt",
filename="war_and_peace.txt")
f = open('war_and_peace.txt', 'rb')

sentences = []
for sentence in f: 
    sentence = sentence.strip() # remove \r, \n with strip()
    # sentence = sentence.lower() # put down to lower cases
    # sentence = sentence.decode('ascii', 'ignore') # delete bytes like \xe2\x80\x99 
    # sentence = sentence.decode()
    sentence = sentence.decode('ascii', 'ignore') 
    if len(sentence) > 0:
        sentences.append(sentence)
f.close() 

total_data = ' '.join(sentences)
print('Total length of the data: %d' % len(total_data))
sentences= []

char_vocab = sorted(list(set(total_data)))
vocab_size = len(char_vocab)
print ('Size of vocabulary: {}'.format(vocab_size))

# %%
char_to_index = dict((char, index) for index, char in enumerate(char_vocab)) # indexing the vocabs

# %%
index_to_char = {}
for key, value in char_to_index.items():
    index_to_char[value] = key

# %%
seq_length = 35
n_samples = len(total_data) - seq_length + 1
print ('Number of data samples with seq_length: {}'.format(n_samples))

# encode the whole data
encoded = [char_to_index[c] for c in total_data]

# %%
## split the encoded data
train_X = []
train_y = []

test_X = []
test_y = []

sample_batch = int(n_samples/10)
for i in range(10):
    for j in range(sample_batch-1):
        if j < int(np.floor(0.9*sample_batch)):
            # train input
            X_encoded = encoded[(i* sample_batch) + j : (i* sample_batch) + j + seq_length]
            train_X.append(X_encoded)

            # train label
            y_encoded = encoded[(i* sample_batch) + j + 1 : (i* sample_batch) + j + seq_length + 1]
            train_y.append(y_encoded)
        else :
            X_encoded = encoded[(i* sample_batch) + j : (i* sample_batch) + j + seq_length]
            test_X.append(X_encoded)

            y_encoded = encoded[(i* sample_batch) + j + 1 : (i* sample_batch) + j + seq_length + 1]
            test_y.append(y_encoded)
    

# %%
print("Data pre-processing starts...")
x_one_hot = [np.eye(vocab_size)[x] for x in tqdm(train_X,desc="train set")]
x_one_hot_test = [np.eye(vocab_size)[x] for x in tqdm(test_X,desc="test set")]
print("Data pre-processing Done!")

# %%
print('Size of training data : {}'.format(len(x_one_hot)))
print('Size of training label : {}'.format(len(train_y)))

print('Size of test data : {}'.format(len(x_one_hot_test)))
print('Size of test label : {}'.format(len(test_y)))

# %%

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
import torch
from torch import nn
from torch import optim
from torch import functional as F 

# Imports from aihwkit.
# from aihwkit.nn import AnalogLSTM
# from aihwkit.optim import AnalogSGD
# from aihwkit.simulator.configs import SingleRPUConfig, FloatingPointRPUConfig
# from aihwkit.simulator.configs import InferenceRPUConfig
# from aihwkit.simulator.configs.devices import (ConstantStepDevice, LinearStepDevice, SoftBoundsDevice, 
# TransferCompound, ReferenceUnitCell, IdealDevice, FloatingPointDevice)
# from aihwkit.simulator.configs.utils import (
#     WeightNoiseType, WeightClipType, WeightModifierType)
# from aihwkit.simulator.presets import GokmenVlasovPreset
# from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
# from aihwkit.nn import AnalogLinear, AnalogSequential

from time import time
device_torch = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device_torch = torch.device("cpu")

NUM_LAYERS = 2
INPUT_SIZE = vocab_size # number of chracters
EMBED_SIZE = 20
HIDDEN_SIZE = 64
OUTPUT_SIZE = vocab_size
DROPOUT_RATIO = 0

SEQ_LEN = seq_length 


class LSTMNetwork_noEmbedding(nn.Sequential):

    def __init__(self,n_layers=2):
        super().__init__()
        self.n_layers = n_layers
        self.dropout = nn.Dropout(DROPOUT_RATIO)
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=n_layers,
                               dropout=DROPOUT_RATIO, bias=True)
        self.decoder =nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE, bias=True)

        
    def forward(self, x_in, in_states): 
        """ Forward pass """
        out, _ = self.lstm(x_in, in_states)
        out = out.view(out.size()[0]*out.size()[1], HIDDEN_SIZE)
        out = self.decoder(out)
        
        return out

    def init_hidden(self, seq_length):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, seq_length, HIDDEN_SIZE).zero_().to(device_torch),
                  weight.new(self.n_layers, seq_length, HIDDEN_SIZE).zero_().to(device_torch))

        return hidden

# %%
class CustomDataset(data.Dataset):
    def __init__(self, data_x, data_y):
        super(CustomDataset, self).__init__()
        self.x = data_x
        self.y = data_y
        
    def __getitem__(self, index):
        
        self.x_data = torch.from_numpy(self.x[index]).float()
        self.y_data = torch.from_numpy(np.asarray(self.y[index])).long()
        return self.x_data, self.y_data

    def __len__(self):
        return len(self.x)

# %%

## Training 

BATCH_SIZE = 1
train_data = CustomDataset(x_one_hot, train_y)
test_data = CustomDataset(x_one_hot_test, test_y)

train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)
test_loader = data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, num_workers=8, shuffle=False)

# %%
def validation_test(model,dataset):
    val_states = model.init_hidden(seq_length)
    val_losses= []
    model.eval()
    for idx, (inputs, target) in tqdm(enumerate(dataset) , desc="Validation"):
        t_i = inputs.to(device_torch)
        t_o= target.view(target.shape[0]*target.shape[1]).to(device_torch)

        val_pred = model(t_i,val_states)
        val_loss = criterion(val_pred, t_o)

        val_losses.append(val_loss.detach().cpu())
    val_losses = np.array(val_losses)
    loss_out = np.mean(val_losses)

    return loss_out
                

# %%

LEARNING_RATE = 0.005
EPOCHS = 50

model = LSTMNetwork_noEmbedding().to(device_torch)
print(model)
# optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
# optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
# optimizer.regroup_param_groups(model)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# train
val_losses =[]
losses = []
total_time = time()
sampling = 50000

for i in range(EPOCHS):
    model.train()
    temp =[]
    time_init = time()
    states = model.init_hidden(seq_length)
    total_loss = 0

    for idx, (inputs, target) in tqdm(enumerate(train_loader), desc="Training in Epoch {}".format(i)):
        d_i = inputs.to(device_torch)
        d_o = target.view(target.shape[0]*target.shape[1]).to(device_torch)
        optimizer.zero_grad()
        pred = model(d_i, states)

        loss = criterion(pred, d_o)
        loss.backward()

        #nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        # print loss in every sampling time
        if (idx+1) % sampling == 0:
            print("Epoch: {}/{}...".format(i+1, EPOCHS),
                    "Step: {}...".format(idx+1),
                    "Loss: {:.4f}...".format(loss.item()))

    losses.append(loss.item())

    val_loss_epoch = validation_test(model,test_loader)
    val_losses.append(val_loss_epoch)

    print('Epoch = %d: Train Loss= %f / Validation Loss= %f' % (i+1, loss.item(),val_loss_epoch))
    print('Time : ', time()-time_init)

print("Total time taken : %f" % (time()-total_time))
model_name = '2201278_lstm_fp_batchsize1.net'

checkpoint = {'n_hidden': HIDDEN_SIZE,
              'n_layers': model.n_layers,
              'state_dict': model.state_dict(),
              'tokens': char_vocab,
              'val_loss': val_losses,
              'train_loss': losses}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

# %%
