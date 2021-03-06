import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pdb
import numpy as np

torch.manual_seed(10)
np.random.seed(10)

# Hyper Parameters 
input_size = 6
hidden_size = 200
num_classes = 1
num_epochs = 25
batch_size = 1
learning_rate = 0.001


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
  
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    
    # Create Neural Network
    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  

    sensor_data = np.loadtxt('./sensor_data/sensor_data.txt')
    
    # Mark how many steps beforehand the collision needs to be predicted
    sensor_data_row_0 = sensor_data[:,-1]
    sensor_data_row_1 = np.roll(sensor_data[:,-1], -1) 
    sensor_data_row_2 = np.roll(sensor_data[:,-1], -2) 
    sensor_data_row_3 = np.roll(sensor_data[:,-1], -3)
    sensor_data_row_4 = np.roll(sensor_data[:,-1], -4)
    sensor_data[:,-1] = sensor_data_row_0 + sensor_data_row_1 + sensor_data_row_2 + sensor_data_row_3 + sensor_data_row_4
    np.savetxt('./sensor_data/labeled_sensor_data.txt',sensor_data)
    
    
    
    collision_full_data = sensor_data[ sensor_data[:,-1] > 0 ]
    
    # Duplicating collision Data for faster learning    
    for i in range(5):
        sensor_data = np.append(sensor_data,collision_full_data,axis=0)
    
    
    # Shuffle the sensor data
    np.random.shuffle(sensor_data)
    
    sensor_nn_data = torch.Tensor(sensor_data[:,:-1])
    collision_data = torch.Tensor(collision_full_data[:,:-1])

    sensor_nn_labels = torch.Tensor(sensor_data[:,-1]).view(-1,1)
    collision_sensor_nn_labels = torch.Tensor(collision_full_data[:,-1]).view(-1,1)
    
    total = sensor_nn_data.shape[0]
    train_size = int(0.70*total)
    test_size = total - train_size
    train_sensor_nn_data = sensor_nn_data[:train_size]
    train_sensor_nn_labels = sensor_nn_labels[:train_size]
    
    
    for j in range(num_epochs):

        losses = 0
        for i in range(train_size):  
            input_values = Variable(sensor_nn_data[i])
            labels = Variable(sensor_nn_labels[i])
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = net(input_values)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses += loss.item()
            
        print ('Epoch %d, Loss: %.4f' %(j+1, losses/sensor_nn_data.shape[0]))       
        torch.save(net.state_dict(), './saved_nets/nn_bot.pkl')
           
        


