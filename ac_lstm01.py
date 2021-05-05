import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


class LSTM11(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=30, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=15, output_size=3):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        aa=self.hidden_layer_size 
        self.lstm1 = nn.LSTM(input_size, hidden_layer_size)
        self.lstm2 = nn.LSTM(hidden_layer_size, hidden_layer_size*2)

        self.linear1 = nn.Linear(hidden_layer_size*2, output_size*2)
        self.linear2 = nn.Linear(output_size*2, output_size)

        self.hidden_cell1 = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        bb=self.hidden_cell1
        print(bb)
        self.hidden_cell2 = (torch.zeros(1,1,self.hidden_layer_size*2),
                            torch.zeros(1,1,self.hidden_layer_size*2))

    def forward(self, input_seq):
        lstm_out1, self.hidden_cell1 = self.lstm1(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell1)
        cc=lstm_out1
#        print(cc)
        lstm_out2, self.hidden_cell2 = self.lstm2(lstm_out1.view(len(input_seq) ,1, -1), self.hidden_cell2)
        predictions1 = self.linear1(lstm_out2.view(len(input_seq), -1))
        predictions2 = self.linear2(predictions1.view(len(input_seq), -1))
        return predictions2[-1]    

data = pd.read_csv('c:\\neustar\\data\\ondeq.csv', delimiter=',', 
                   error_bad_lines=False, float_precision='round_trip')

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 5
plt.rcParams["figure.figsize"] = fig_size
plt.title('forme d onda')
plt.ylabel('valori')
plt.autoscale(axis='x', tight=True)
plt.grid(True)
plt.plot(data['20dc'])
plt.plot(data['50dc'])
plt.plot(data['triangle'])
plt.show()
aa=0
bb=0
cc=0

test_data_size = int(data.shape[0]/10)
train_data = data[:-test_data_size]
test_data = data[-test_data_size:]
scaler = MinMaxScaler(feature_range=(0, 1))
train_data_norm = scaler.fit_transform(train_data)
train_data_normalized = torch.FloatTensor(train_data_norm)
train_window = int(test_data_size)
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
    
model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 15
for i in range(epochs):
    for seq, labels in train_inout_seq:
#        print(seq)
#        print(labels)
        optimizer.zero_grad()
        model.hidden_cell1 = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        model.hidden_cell2 = (torch.zeros(1, 1, model.hidden_layer_size*2),
                            torch.zeros(1, 1, model.hidden_layer_size*2))

        y_pred = model(seq)
        y_pred = y_pred.view(1,-1)
        
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%2 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = test_data_size*2
test_inputs = train_data_normalized[-test_data_size:].tolist()
print(test_inputs)

import joblib
PATH = 'data/aaaaa.pkl'
joblib.dump(model, PATH)
model11=joblib.load(PATH)




model11.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        model11.hidden = (torch.zeros(1, 1, model11.hidden_layer_size),
                        torch.zeros(1, 1, model11.hidden_layer_size))
        test_inputs.append(model11(seq).tolist())

actual_predictions = scaler.inverse_transform(np.array(test_inputs[test_data_size:] ))
print(actual_predictions)

x = np.arange(data.shape[0]-test_data_size*2, data.shape[0], 1)
print(x)

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data['20dc'])
plt.plot(x,actual_predictions[:,:1])
plt.show()

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data['50dc'])
plt.plot(x,actual_predictions[:,1:2])
plt.show()

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data['triangle'])
plt.plot(x,actual_predictions[:,2:3])
plt.show()



print('FINE')


