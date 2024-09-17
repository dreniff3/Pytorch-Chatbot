## Pytorch Chatbot

This project is a simple chatbot built using a **feedforward neural network** with **PyTorch**. The model is trained on predefined intents and can respond to basic questions based on those intents.

### Features

- Uses **feedforward neural networks** for intent classification.
- Includes **Natural Language Toolkit** (**NLTK**) utilities for tokenization and stemming.
- Trained on a custom dataset of intents defined in a JSON file.
- Familiar-looking GUI interface for chatting with the bot made with **Tkinter**.

### Usage

1. Install the necessary dependencies:
   ```
   pip i torch nltk
2. Train the model:
   - Define your chatbot intents in the `intents.json` file.
   - Run the `train.py` script to train the neural network:
     ```
     python train.py
     ```
    NOTE: This trains the model and saves it as `data.pth` for later use.
3. After training, you can interact with the chatbot by running:
   ```
   python app.py
   ```

### Structure

- **train.py:** Script for training the neural network.
- **chat.py:** Script for interacting with the chatbot.
- **model.py:** Contains the neural network architecture.
- **nltk_utils.py:** Functions for text preprocessing (e.g., tokenization, stemming).
- **intents.json:** Defines the chatbot's intents and responses.

### Credits

This project follows a tutorial created by **Patrick Loeber**. You can find the full tutorial series [here](https://www.youtube.com/playlist?list=PLqnslRFeH2UrFW4AUgn-eY37qOAWQpJyg).
