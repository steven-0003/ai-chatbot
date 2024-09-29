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

bot_name = "Bob"
print("Let's chat! Type 'quit' to exit")

def get_response(msg):
    sentence = tokenise(msg)
    X = bag(sentence, words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]
    
    
    if prob.item()>0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        return f"{bot_name}: Sorry, I do not understand"