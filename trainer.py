import torch
from torch import nn, optim

import numpy as np

class Trainer:

    def __init__(self, model, epochs, train_set, valid_data=None, lr=0.001, print_every=100):
        self.model = model
        self.epochs = epochs
        self.train_set = train_set
        self.valid_data = valid_data
        self.lr = lr
        self.print_every = print_every

    def get_batches(data, window):
        L = len(data)
        for i in range(L - window):
            x_sequence = data[i:i + window]
            y_sequence = data[i+1: i + window + 1] 
            yield x_sequence, y_sequence

    def train(self):
        criterion = nn.MSELoss()
        opt = optim.Adam(self.model.parameters(), lr = self.lr)
    
        train_loss = []
        valid_loss = []
    
        for e in range(self.epochs):
        
            hs = None
            t_loss = 0

            print(self.train_set)
            print(self.train_set.shape)
            for x, y in self.get_batches(self.train_set, 42):

                opt.zero_grad()
            
                # Create batch_size dimension
                x = x.unsqueeze(0)
                out, hs = self.model(x, hs)
                hs = tuple([h.data for h in hs])
            
                loss = criterion(out, y)
                loss.backward()
                opt.step()
                t_loss += loss.item()
            
            if self.valid_data is not None:
                self.model.eval()
                val_x, val_y = self.valid_data
                val_x = val_x.unsqueeze(0)
                preds, _ = self.model(val_x, hs)
                v_loss = criterion(preds, val_y)
                valid_loss.append(v_loss.item())
                
                self.model.train()
            
        train_loss.append(np.mean(t_loss))
            
            
        if e % self.print_every == 0:
            print('Epoch {}:\nTraining Loss: {}'.format(e, train_loss[-1]))
            if self.valid_data is not None:
                print('Validation Loss: {}'.format(valid_loss[-1]))
    
        #plt.figure(figsize=[8., 6.])
        #plt.plot(train_loss, label='Training Loss')
        #plt.plot(valid_loss, label='Validation Loss')
        #plt.title('Loss vs Epochs')
        #plt.xlabel('Epochs')
        #plt.savefig('/wrk/users/matleino/testLSTM/losses.png')
        #plt.show()
