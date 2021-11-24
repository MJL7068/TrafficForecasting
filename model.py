import torch
from torch import nn, optim

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import sklearn.metrics

from lstm import LSTM
from trainer import Trainer


#trainer = Trainer(model = model, epochs = 500, train_set = train_data, valid_data=valid_data, lr=0.001, print_every=100)

#_ = trainer.train()

# validation_cut is the number of rows in valid_data. The number of rows in one days data is 96
# y_pred: an array of indeces in var that are to be predicted 
def prep_data(data, var, year, month, validation_cut, y_pred):
    data[var] = data[var].astype(np.float32)

    #data = data[(data.YEAR == 2019) & (data.MONTH == 2)]
    data.set_index(pd.Series(list(range(data.shape[0]))), inplace = True)

    train_data = data[var][:-validation_cut].to_numpy().reshape(-1, len(var))
    valid_data = data[var][-validation_cut:].to_numpy().reshape(-1, len(var))

    t_scaler = MinMaxScaler(feature_range=(-1, len(var)))
    v_scaler = MinMaxScaler(feature_range=(-1, len(var)))
    train_data = t_scaler.fit_transform(train_data)
    valid_data = v_scaler.fit_transform(valid_data)

    # convert training data to tensor
    train_data = torch.tensor(train_data, dtype = torch.float32)
    valid_data = torch.tensor(valid_data, dtype = torch.float32)

    valid_x = valid_data[:-1,:]
    valid_y = valid_data[1:,y_pred]
    valid_data = (valid_x, valid_y)

    input_size = len(var)
    hidden_size = 100
    num_layers = 1
    output_size = len(y_pred)

    model = LSTM(input_size, hidden_size, num_layers, output_size)

    return train_data, valid_data, t_scaler, v_scaler, model

def plot_results(model, train_data, valid_x, t_scaler, v_scaler, data, y_pred):
    hs = None
    train_preds, hs = model(train_data.unsqueeze(0), hs)
    for i in range(train_data.shape[1] - len(y_pred)):
        train_preds = torch.cat((train_preds, train_preds[:,0].reshape(-1,1)), 1)
    train_preds = t_scaler.inverse_transform(train_preds.detach())
    train_preds = train_preds[:,0].reshape(-1, 1)

    valid_preds, hs = model(valid_x.unsqueeze(0), hs)
    for i in range(valid_x.shape[1] - len(y_pred)):
        valid_preds = torch.cat((valid_preds, valid_preds[:,0].reshape(-1,1)), 1)
    valid_preds = v_scaler.inverse_transform(valid_preds.detach())
    valid_preds = valid_preds[:,[0]].reshape(-1, 1)

    train_time = data.index[:-validation_cut]
    valid_time = data.index[-(validation_cut - 1):]

    # Show all predictions
    plt.plot(train_time, train_preds[:,].squeeze(), 'r--', label = 'Training Predictions')
    plt.plot(valid_time, valid_preds.squeeze(), 'g--', label = 'Validation Predictions')
    plt.plot(data.index, data[np.array(var)[y_pred]].to_numpy(), label = 'Actual')
    plt.show()

    # Show validation predictions
    plt.title('Predicting the number of vehicles')
    plt.plot(range(0, validation_cut - 1), valid_preds.squeeze(), 'g--', label = 'Validation Predictions')
    plt.plot(range(0, validation_cut), data[np.array(var)[y_pred]][-(validation_cut):].to_numpy(), label = 'Actual')
    plt.xticks(range(0, validation_cut, 16) + [96], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'])
    plt.xlabel('Time (EST)')
    plt.ylabel('Number of vehicles')
    plt.legend()
    plt.show()


def get_batches(data, window, y_pred):
    """
    Takes data with shape (n_samples, n_features) and creates mini-batches
    with shape (1, window). 
    """

    L = len(data)
    for i in range(L - window):
        x_sequence = data[i:i + window]
        #y_sequence = data[i+1: i + window + 1]
        y_sequence = data[i+1: i + window + 1, y_pred].reshape(-1, len(y_pred))
        yield x_sequence, y_sequence

def train(model, epochs, train_set, valid_data = None, lr = 0.001, print_every = 10, y_pred = [0, 1]):

    criterion = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    train_loss = []
    valid_loss = []
    
    for e in range(epochs):
        
        hs = None
        t_loss = 0

        # originally window_size was 12 for a monthly flight dataset
        window_size = 4
        for x, y in get_batches(train_set, window_size, y_pred):

            opt.zero_grad()
            
            # Create batch_size dimension
            x = x.unsqueeze(0)
            out, hs = model(x, hs)
            hs = tuple([h.data for h in hs])
            
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            t_loss += loss.item()
            
        if valid_data is not None:
                model.eval()
                val_x, val_y = valid_data
                val_x = val_x.unsqueeze(0)
                preds, _ = model(val_x, hs)
                v_loss = criterion(preds, val_y)
                valid_loss.append(v_loss.item())
                
                model.train()
            
        train_loss.append(np.mean(t_loss))
            
            
        if e % print_every == 0:
            print('Epoch {}:\nTraining Loss: {}'.format(e, train_loss[-1]))
            if valid_data is not None:
                print('Validation Loss: {}'.format(valid_loss[-1]))
    
    plt.figure(figsize=[8., 6.])
    plt.plot(train_loss, label='Training Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.show()

data = pd.read_csv('/home/matleino/Desktop/lstm_test/combined_data_traffic_est.csv')
var = ['ha', 'pa', 'deg', 'sade', '0', '1', '2', '3', '4', 'IS_CLOSE']
# We need to get rid of nans in deg column
data.loc[621:623,'deg'] = [0.7, 0.8, 0.9]
#validation_cut = (96*28)
validation_cut = 96
y_pred = [0, 1]
data = data[(data.YEAR == 2019) & (data.MONTH == 2)]
#data = data[(data.YEAR == 2019) & ((data.MONTH == 2) | (data.MONTH == 3))]
train_data, valid_data, t_scaler, v_scaler, model = prep_data(data, var, year = 2019, month = 2, validation_cut = validation_cut, y_pred = y_pred)
valid_x, valid_y = valid_data
#model = model.load_state_dict(torch.load('/home/matleino/Desktop/lstm_test/experiments/models/exp_17.pt'))

print("start training")
train(model, 50, train_data, lr = 0.0005, valid_data = valid_data, y_pred = y_pred)

#torch.save(model.state_dict(), '/home/matleino/Desktop/lstm_test/experiments/models/exp_18.pt')

plot_results(model, train_data, valid_x, t_scaler, v_scaler, data, y_pred)