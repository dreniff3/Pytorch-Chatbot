import random
import json
import torch
from model import NeuralNet
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

# get data to create model
FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # load model state
model.eval()  # set model to evaluation mode

# begin chat
bot_name = "Hal"
print("Let's chat! Type 'quit' to exit")
while True:
    sentence = input("You: ")
    if sentence == 'quit':
        break

    # tokenize user input
    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    # model expects this shape:
    # 1 row for 1 sample, with x[0] columns
    x = x.reshape(1, x.shape[0])
    # conver to torch tensor (matrix)
    x = torch.from_numpy(x)  # bag_of_words returns numpy arr

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # check if probability for this tag is high enough
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
