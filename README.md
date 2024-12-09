# Ai generated readme file beacuse im lazyyyyyyyyyyyyy

Here's a `README.md` file that you can use for your GitHub repository. This file explains the project, how to set it up, and how to run it.

### `README.md`

```markdown
# Rock, Paper, Scissors AI with PyTorch

This project implements a simple Rock, Paper, Scissors game where the AI (trained using a neural network built with PyTorch) learns from the moves played by the user. The AI predicts the next move based on its training data, which is updated after each round. 

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [How it Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The AI learns to play Rock, Paper, Scissors using the neural network trained on the previous rounds of the game. Every time the player and the model play a round, the moves are saved to a CSV file and used to further train the model in future runs.

### Features:
- **Neural Network Training**: The model is trained using PyTorch's neural network framework.
- **CSV Storage**: All moves are saved in a CSV file (`rps_data.csv`) to persist the data and help the AI learn over time.
- **Prediction**: The AI predicts the player's next move based on its own previous moves and the training it has received.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rps-ai-pytorch.git
   cd rps-ai-pytorch
   ```

2. Install the required dependencies:
   ```bash
   pip install torch numpy
   ```

## Usage

To run the game, simply execute the Python script:

```bash
python rps_game.py
```

The AI will start the game by making a random move, and you will be prompted to enter your move. The game continues until you decide to quit.

### Game Flow:
1. The model will predict its move based on previous data.
2. You will input your move (Rock, Paper, or Scissors).
3. The winner is determined based on standard Rock, Paper, Scissors rules.
4. The moves (player and model) are saved into a CSV file for future training.

## How it Works

1. **Training the Model**:
   - The model is a simple neural network that takes the player's previous move (0 = Rock, 1 = Paper, 2 = Scissors) as input and predicts the model's next move.
   - The dataset (`rps_data.csv`) stores the previous moves and corresponding model moves, which the model uses to learn.
   
2. **Model Architecture**:
   - A feedforward neural network with one hidden layer is used.
   - The input is the player's previous move (an integer between 0 and 2), and the output is the predicted next move (Rock, Paper, or Scissors).

3. **Training**:
   - The model is trained using the cross-entropy loss and the Adam optimizer.
   - After each round of the game, the moves are saved to the CSV file, and the model is further trained during the next run.

4. **Prediction**:
   - The model predicts the AI's next move based on the player's last move using the trained neural network.

## Contributing

Contributions are welcome! If you find any bugs or would like to suggest improvements, feel free to open an issue or submit a pull request.

## License

This project is open source and available under the [MIT License](LICENSE).
```

### Instructions to Add to GitHub:
1. **Create a New Repository** on GitHub.
2. **Clone the Repository** locally.
   ```bash
   git clone https://github.com/yourusername/rps-ai-pytorch.git
   ```
3. **Add Your Code and README**:
   - Copy your Python code (e.g., `rps_game.py`) and this `README.md` into your project folder.
4. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Initial commit with Rock, Paper, Scissors AI and README"
   git push origin main
   ```

This `README.md` file provides a detailed overview of your project, making it easy for others to understand and run the code.
