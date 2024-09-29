import random
import json
import torch
from model import NeuralNetwork
from nltk_utlis import tokenise, bag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE, weights_only=False)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
words = data['words']
tags = data['tags']
model_state = data['model_state']

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
