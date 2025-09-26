# Deep Q-Learning Snake Game 🐍🧠

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-red.svg)](https://pytorch.org)
[![Pygame](https://img.shields.io/badge/Pygame-2.5.2-green.svg)](https://pygame.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A sophisticated Deep Q-Learning implementation that trains an AI agent to master the classic Snake game through reinforcement learning. This project demonstrates advanced neural network design using PyTorch, featuring real-time visualization, performance tracking, and continuous learning capabilities.

## 🎯 Project Overview

This project implements a complete Deep Q-Network (DQN) system that learns optimal gameplay strategies for Snake through trial and error, using advanced reinforcement learning techniques including experience replay, target networks, and epsilon-greedy exploration.

### Key Achievements
- **Intelligent Agent**: AI learns complex gameplay strategies autonomously
- **Real-time Visualization**: Interactive GUI showing training progress and gameplay
- **Performance Analytics**: Integrated plotting system for monitoring learning curves
- **Model Persistence**: Save/load functionality for continuous training sessions
- **Robust Architecture**: Clean, modular codebase following software engineering best practices

## 🚀 Features

### 🎮 Game Environment
- **Classic Snake Gameplay**: Traditional snake mechanics with modern implementation
- **Responsive Controls**: Smooth keyboard input handling for human play
- **Dynamic Food Placement**: Intelligent food positioning algorithm
- **Collision Detection**: Robust boundary and self-collision systems
- **Score Tracking**: Real-time score display and record keeping

### 🧠 AI Agent & Deep Learning
- **Deep Q-Network Architecture**: 3-layer neural network (11→256→3)
- **Experience Replay**: Efficient memory buffer (100K experiences) for stable learning
- **Epsilon-Greedy Strategy**: Balanced exploration vs exploitation with adaptive decay
- **Reward System**: Sophisticated reward engineering (+10 food, -10 death)
- **State Representation**: 11-dimensional feature vector capturing game dynamics

### 📊 Training & Analytics
- **Real-time Performance Plots**: Live graphs showing score progression and learning curves
- **Training Metrics**: Game count, current score, and record tracking
- **Model Checkpointing**: Automatic saving of best-performing models
- **Hyperparameter Tuning**: Configurable learning rates, batch sizes, and network architecture

### 🔧 Technical Implementation
- **PyTorch Integration**: Full neural network implementation with GPU support
- **Modular Design**: Clean separation of concerns (game, agent, model, visualization)
- **Memory Management**: Efficient deque-based experience replay buffer
- **State Management**: Comprehensive game state encoding for optimal learning

## 🏗️ Architecture

### Project Structure
```
SnakeAI/
├── snake_game.py          # Human-playable Snake game implementation
├── Deep_Q_Learning/
│   ├── agent.py           # DQN Agent with training logic
│   ├── model.py           # Neural network architecture & trainer
│   ├── snakeAI.py         # AI-optimized game environment
│   ├── helper.py          # Visualization and plotting utilities
│   ├── pretrained_agent.py# Pre-trained model loading and inference
│   └── model/             # Saved model checkpoints
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

### Neural Network Architecture
```python
Linear_QNet(
  Input Layer:  11 features (state representation)
  Hidden Layer: 256 neurons (ReLU activation)
  Output Layer: 3 actions [straight, right, left]
)
```

### State Representation (11 Features)
1. **Danger Detection** (3 features): Immediate collision risks in all directions
2. **Movement Direction** (4 features): Current snake direction encoding
3. **Food Location** (4 features): Relative food position (up/down/left/right)

## 🔬 Deep Q-Learning Implementation

### Core Algorithms

**Bellman Equation Implementation:**
```
Q(s, a) = (1 - α) * Q(s, a) + α * [r + γ * max(Q(s', a'))]
```

**Key Components:**
- **Experience Replay**: Breaks correlation between consecutive experiences
- **Target Network**: Stabilizes training by using separate target Q-values
- **Epsilon Decay**: `ε = 80 - n_games` for adaptive exploration
- **Reward Engineering**: Strategic reward structure for optimal learning

### Training Process
1. **State Observation**: Extract 11-dimensional feature vector
2. **Action Selection**: Epsilon-greedy policy with neural network prediction
3. **Environment Interaction**: Execute action and observe reward/next state
4. **Experience Storage**: Store transition in replay memory
5. **Batch Training**: Sample random batch for neural network updates
6. **Model Persistence**: Save best-performing models automatically

## 📦 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Abhishek-1804/SnakeAI.git
   cd SnakeAI
   ```

2. **Create Virtual Environment** (Recommended)
   ```bash
   # Windows/Linux
   python -m venv snake_env
   
   # macOS
   python3 -m venv snake_env
   ```

3. **Activate Virtual Environment**
   ```bash
   # Windows
   snake_env\Scripts\activate
   
   # macOS/Linux
   source snake_env/bin/activate
   ```

4. **Install Dependencies**
   ```bash
   # Windows/Linux
   pip install -r requirements.txt
   
   # macOS
   pip3 install -r requirements.txt
   ```

## 🎮 Usage Guide

### Human Gameplay
Experience the classic Snake game with responsive controls:
```bash
python snake_game.py
```
**Controls:** Arrow keys for movement

### AI Training
Train the Deep Q-Learning agent from scratch:
```bash
cd Deep_Q_Learning
python agent.py
```
**Features:**
- Real-time performance visualization
- Automatic model saving for best scores
- Training progress metrics and statistics

### Pre-trained Model Inference
Run a saved model to see trained performance:
```bash
cd Deep_Q_Learning
python pretrained_agent.py
```
**Configuration:** Update `model_path` in the script to point to your saved model

## 📊 Training Results & Performance

### Learning Progression
- **Initial Phase**: Random exploration with high epsilon values
- **Learning Phase**: Gradual improvement as agent discovers effective strategies
- **Optimization Phase**: Fine-tuning for maximum score achievement
- **Convergence**: Stable performance with consistent high scores

### Performance Metrics
- **Score Tracking**: Real-time game score monitoring
- **Mean Score Analysis**: Running average for trend identification
- **Record Keeping**: Best performance tracking and model checkpointing
- **Loss Visualization**: Neural network training loss progression

## 🔧 Hyperparameter Configuration

### Key Parameters
```python
# Training Configuration
MAX_MEMORY = 100_000      # Experience replay buffer size
BATCH_SIZE = 1000         # Training batch size
LEARNING_RATE = 0.001     # Neural network learning rate
GAMMA = 0.9               # Discount factor for future rewards
EPSILON_DECAY = 80        # Exploration decay rate

# Network Architecture
INPUT_SIZE = 11           # State feature dimensions
HIDDEN_SIZE = 256         # Hidden layer neurons
OUTPUT_SIZE = 3           # Action space size

# Game Environment
BLOCK_SIZE = 20           # Game grid resolution
SPEED = 150               # Game/training speed
```

## 🛠️ Technical Deep Dive

### Reinforcement Learning Components

**Environment**: Modified Snake game optimized for AI training
- State space: 11-dimensional continuous features
- Action space: 3 discrete actions (straight, left, right)
- Reward structure: +10 (food), -10 (death), 0 (movement)
- Terminal conditions: Wall collision, self-collision, timeout

**Agent**: DQN implementation with advanced features
- Neural network: Multi-layer perceptron with ReLU activation
- Memory: Circular buffer for experience replay
- Policy: Epsilon-greedy with linear decay
- Training: Adam optimizer with MSE loss

### Code Architecture Highlights

**Modular Design**:
- `SnakeAI`: Game environment optimized for agent interaction
- `Agent`: DQN agent with training and inference capabilities
- `Linear_QNet`: PyTorch neural network implementation
- `QTrainer`: Training loop with Bellman equation updates

**Performance Optimizations**:
- Batch processing for efficient GPU utilization
- Memory-efficient experience replay buffer
- Vectorized state representation for fast computation
- Real-time visualization without blocking training

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Commit Changes**: `git commit -m 'Add amazing feature'`
4. **Push to Branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Areas for Contribution
- **Algorithm Improvements**: Double DQN, Dueling DQN, Rainbow DQN
- **Hyperparameter Optimization**: Automated tuning systems
- **Visualization Enhancements**: Advanced plotting and analysis tools
- **Performance Optimization**: GPU acceleration, distributed training
- **Documentation**: Tutorials, code comments, examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **Pygame Community**: For the game development library
- **OpenAI**: For pioneering research in Deep Q-Learning
- **Reinforcement Learning Community**: For advancing the field

## 📧 Contact

**Abhishek Deshpande**
- GitHub: [@Abhishek-1804](https://github.com/Abhishek-1804)
- Website: [abhishekdp.com](https://abhishekdp.com)
- LinkedIn: [Connect with me](https://linkedin.com/in/abhishek-deshpande)

---

⭐ **Star this repository if you found it helpful!** ⭐

*Built with passion for AI and game development* 🚀