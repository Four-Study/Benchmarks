#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision
import torch.optim as optim


# In[3]:


train = torchvision.datasets.FashionMNIST('./datasets', train=True, download=True,transform=transforms.Compose([transforms.ToTensor()]))
test = torchvision.datasets.FashionMNIST('./datasets', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()]))
trainset = torch.utils.data.DataLoader(train, batch_size=256, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)


# In[3]:


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__() 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.pool  = nn.AvgPool2d(kernel_size=2)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        y = self.conv1(x)
        y = F.relu(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = F.relu(y)
        y = self.pool(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = F.relu(y)
        y = self.fc2(y)
        y = F.relu(y)
        y = self.fc3(y)
        return y


# In[4]:


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
net = LeNet5().to(device)
print(net)
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


# ### Train the model first

# In[5]:


for epoch in range(10): 
    for data in trainset:  
        X, y = data
        X = X.to(device)
        y = y.to(device) 
        net.zero_grad()  
        output = net(X)  
        loss = loss_criterion(output, y)  

        # Backpropergation 
        loss.backward()  
        optimizer.step()  
    print("epoch:", epoch + 1, "loss:", loss.item())


# In[6]:


correct = 0
total = 0
p_vals = []
for i in range(10):
    p_vals.append([])
with torch.no_grad():
    for data in testset:
        X, y = data
        X = X.to(device)
        y = y.to(device)
        output = net(X)
        probs = F.softmax(output, 1)
        
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
        for idx, i in enumerate(probs):
            p_vals[y[idx]].append(i[y[idx]].item())

print("Accuracy: ", round(correct/total, 5))


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 5, sharex = True, figsize=(22, 8))
plt.rcParams['font.size'] = '16'
for i in range(10):
    
    _ = axs[i//5, i % 5].hist(np.array(p_vals[i]), 20)
    axs[i//5, i % 5].set_title(trainset.dataset.classes[i], fontsize=20)

#     _ = ax2.hist(np.array(p_vals[3]), 20)
#     ax2.set_title(trainset.dataset.classes[3])

#     _ = ax3.hist(np.array(p_vals[7]), 20)
#     ax3.set_title(trainset.dataset.classes[7])

#     _ = ax4.hist(np.array(p_vals[9]), 20)
#     ax4.set_title(trainset.dataset.classes[9])

plt.savefig('output_score.pdf', dpi = 150)


# In[15]:


fig, axs = plt.subplots(2, 1, sharex = False, figsize=(4, 8))
plt.rcParams['font.size'] = '16'
for i in [8, 9]:
    
    _ = axs[i-8].hist(np.array(p_vals[i]))
    axs[i-8].set_title(trainset.dataset.classes[i], fontsize=20)

#     _ = ax2.hist(np.array(p_vals[3]), 20)
#     ax2.set_title(trainset.dataset.classes[3])

#     _ = ax3.hist(np.array(p_vals[7]), 20)
#     ax3.set_title(trainset.dataset.classes[7])

#     _ = ax4.hist(np.array(p_vals[9]), 20)
#     ax4.set_title(trainset.dataset.classes[9])
plt.tight_layout()
plt.savefig('output_score.pdf', dpi = 150)


# In[ ]:




