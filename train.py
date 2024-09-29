import json
import numpy as np
from nltk_utlis import tokenise, stem, bag

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNetwork

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenise(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', ',','!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
Y_train = []

for (w, tag) in xy:
    b = bag(w, all_words)
    X_train.append(b)

    label = tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

class ChatDataSet(Dataset):
    
    def __init__(self) -> None:
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index) -> tuple:
        return self.x_data[index], self.y_data[index]
    
    def __len__(self) -> int:
        return self.n_samples
    
# Hyperparameters
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# Loss and Optimiser
CE = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(words)
        loss = CE(outputs, labels)

        # backprop and optimiser
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}: loss = {loss.item():.4f}')
    
print(f'Final loss = {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training Complete. File saved to {FILE}")
