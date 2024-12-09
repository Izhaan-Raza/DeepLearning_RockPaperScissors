import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import csv


DATA_FILE = 'rps_data.csv'


def load_data():
    if os.path.exists(DATA_FILE):
        data = []
        with open(DATA_FILE, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                data.append([int(row[0]), int(row[1])])
        return np.array(data)
    else:
        return np.array([], dtype=int).reshape(0, 2)

def save_data(data):
    with open(DATA_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

data = load_data()


if len(data) == 0:
    data = np.random.choice([0, 1, 2], size=(1000, 2))  
    save_data(data)  

X = data[:, 0] 
y = data[:, 1]  


X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)


train_size = int(0.8 * len(X_tensor))  
val_size = len(X_tensor) - train_size  


X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]


class RPSModel(nn.Module):
    def __init__(self):
        super(RPSModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)  
        self.fc2 = nn.Linear(10, 3)  

    def forward(self, x):
        x = torch.relu(self.fc1(x.float()))  
        x = self.fc2(x)
        return x

model = RPSModel()
criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    outputs = model(X_train.unsqueeze(1))  
    loss = criterion(outputs, y_train)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


model.eval()
with torch.no_grad():
    val_outputs = model(X_val.unsqueeze(1))
    _, predicted = torch.max(val_outputs, 1)
    accuracy = (predicted == y_val).sum().item() / y_val.size(0)
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')


def predict_move(previous_move):
    model.eval()
    with torch.no_grad():
        input_move = torch.tensor([previous_move], dtype=torch.long)
        output = model(input_move.unsqueeze(0))
        _, predicted_move = torch.max(output, 1)
        return predicted_move.item()


def play_game():
    print("Model Ready........")
    moves = ["Rock", "Paper", "Scissors"]
    
    previous_move = random.choice([0, 1, 2])  #
    print(f"modelstarts with: {moves[previous_move]}")

    while True:
        
        user_move = input("\nEnter your move (Rock, Paper, Scissors): ").strip().lower()
        
        if user_move == "rock":
            user_move = 0
        elif user_move == "paper":
            user_move = 1
        elif user_move == "scissors":
            user_move = 2
        else:
            print("Invalid input, please enter 'Rock', 'Paper', or 'Scissors'.")
            continue
        
       
        model_move = predict_move(previous_move)
        print(f"modelpredicts: {moves[model_move]}")

        
        if model_move == user_move:
            print("It's a tie!")
        elif (model_move == 0 and user_move == 1) or (model_move == 1 and user_move == 2) or (model_move == 2 and user_move == 0):
            print("You win!")
        else:
            print("modelwins!")

       
        previous_move = user_move
        
        
        save_data([[user_move, model_move]])

        play_again = input("\nDo you want to play again? (yes/no): ").strip().lower()
        if play_again != 'yes':
            print("Thanks for playing!")
            break


play_game()
