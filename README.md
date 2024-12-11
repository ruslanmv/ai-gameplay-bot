# AI-Powered Gameplay Bot for MMORPGs

## Introduction

This project is a cutting-edge initiative to develop an AI bot capable of playing MMORPGs (Massively Multiplayer Online Role-Playing Games) by learning directly from gameplay videos available on platforms like YouTube and Twitch. The bot watches these videos, analyzes the actions of human players, and maps them to corresponding keyboard, mouse, or gamepad inputs. By training on this data, the bot can replicate human-like gameplay and improve its performance over time using reinforcement learning (RL).

This project the the continuation of the previos project [BOT-MMORPG-AI](https://github.com/ruslanmv/BOT-MMORPG-AI) with now is enhanced with GenAI and RL.

The core idea is to mimic how humans learn to play games:
1. **Observation**: Watching expert players on platforms like YouTube or Twitch to understand strategies and gameplay mechanics.
2. **Action Mapping**: Deducing the inputs (e.g., key presses, mouse movements) required to achieve the observed in-game actions.
3. **Training**: Using this mapped data to train advanced machine learning models (Neural Networks and Transformers) to perform these actions.
4. **Self-Improvement**: Enhancing gameplay through RL by experimenting with strategies and refining actions based on feedback.

This repository provides all necessary scripts and tools to process gameplay videos, build datasets, train models, and deploy an AI gameplay bot.

---

## Repository Structure

```plaintext
project/
├── data/
│   ├── raw/
│   │   ├── gameplay_videos/                # Folder for raw gameplay videos
│   │   ├── annotations/                    # Folder for annotated data (manual/auto-generated)
│   └── processed/
│       ├── nn_dataset.csv                  # Dataset for Neural Network model
│       ├── transformer_dataset.csv         # Dataset for Transformer model
│       ├── enriched_dataset.csv            # Dataset enriched with generative AI
│
├── models/
│   ├── neural_network/
│   │   ├── nn_model.py                     # Neural Network model definition
│   │   ├── nn_training.py                  # Training script for Neural Network
│   │   ├── nn_finetune.py                  # Fine-tuning script for NN real-time performance
│   │   └── nn_rl_integration.py            # Reinforcement Learning integration for NN
│   ├── transformer/
│   │   ├── transformer_model.py            # Transformer model definition
│   │   ├── transformer_training.py         # Training script for Transformer
│   │   ├── transformer_finetune.py         # Fine-tuning script for Transformer real-time performance
│   │   └── transformer_rl_integration.py   # Reinforcement Learning integration for Transformer
│
├── scripts/
│   ├── video_processing.py                 # Script to extract character actions and motions
│   ├── input_mapping.py                    # Script to map actions to keyboard/mouse inputs
│   ├── dataset_builder.py                  # Script to build datasets from video-input mappings
│   ├── generative_ai_enrichment.py         # Script to enrich datasets with generative AI
│
├── evaluation/
│   ├── model_comparison.py                 # Script to compare NN and Transformer performance
│   ├── real_time_tests.py                  # Real-time performance testing scripts
│   └── feedback_iteration.py               # Script for iterating based on feedback
│
├── deployment/
│   ├── deploy_nn.py                        # Deployment script for Neural Network model
│   ├── deploy_transformer.py               # Deployment script for Transformer model
│   └── real_time_controller.py             # Unified controller for real-time commands
│
├── tests/
│   ├── test_video_processing.py            # Unit tests for video processing
│   ├── test_input_mapping.py               # Unit tests for input mapping
│   ├── test_nn_model.py                    # Unit tests for Neural Network
│   ├── test_transformer_model.py           # Unit tests for Transformer
│   ├── test_rl_integration.py              # Unit tests for Reinforcement Learning integration
│   └── test_real_time_performance.py       # Performance testing framework
│
├── notebooks/
│   ├── data_analysis.ipynb                 # Jupyter notebook for dataset analysis
│   ├── nn_training_logs.ipynb              # Logs and visualizations for NN training
│   ├── transformer_training_logs.ipynb     # Logs and visualizations for Transformer training
│   ├── evaluation_results.ipynb            # Evaluation and comparison of models
│
├── README.md                               # Project overview and setup instructions
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore file
└── setup.py                                # Project setup script
```

---

## Workflow

```mermaid
graph TD
    A[Start: Analyze Gameplay Videos] --> B[Extract Character Actions and Motions]
    B --> C[Map Actions to Inputs (Keyboard/Mouse/Gamepad)]
    C --> D[Build Dataset from Video-Input Mappings]
    D --> E1[Train Initial Neural Network Model]
    D --> E2[Train Initial Transformer Model]
    E1 --> F1[Enhance NN Dataset with Goals and Game Context]
    E2 --> F2[Enhance Transformer Dataset with Goals and Game Context]
    F1 --> G1[Incorporate Generative AI for NN Dataset Enrichment]
    F2 --> G2[Incorporate Generative AI for Transformer Dataset Enrichment]
    G1 --> H1[Fine-Tune Neural Network for Real-Time Performance]
    G2 --> H2[Fine-Tune Transformer for Real-Time Performance]
    H1 --> I1[Integrate RL with Neural Network for Self-Improvement]
    H2 --> I2[Integrate RL with Transformer for Self-Improvement]
    I1 --> J1[Deploy Optimized NN Gameplay AI Model]
    I2 --> J2[Deploy Optimized Transformer Gameplay AI Model]
    J1 --> K[Iterate and Refine Based on Feedback]
    J2 --> K[Iterate and Refine Based on Feedback]
```

---

## Setup Instructions

### Prerequisites
1. Install Python (>= 3.8).
2. Install `pip` and `virtualenv` for dependency management.
3. Clone this repository:
   ```bash
   git clone https://github.com/ruslnmv/ai-gameplay-bot.git
   cd ai-gameplay-bot
   ```

### Install Dependencies
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   venv\Scripts\activate     # For Windows
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
1. Place raw gameplay videos from YouTube or Twitch in the `data/raw/gameplay_videos/` directory.
2. Run the video processing and dataset builder scripts:
   ```bash
   python scripts/video_processing.py
   python scripts/dataset_builder.py
   ```

### Training Models
- **Neural Network**:
  ```bash
  python models/neural_network/nn_training.py
  ```
- **Transformer**:
  ```bash
  python models/transformer/transformer_training.py
  ```

### Evaluation
1. Compare model performance:
   ```bash
   python evaluation/model_comparison.py
   ```
2. Test real-time performance:
   ```bash
   python evaluation/real_time_tests.py
   ```

### Deployment
- Deploy the preferred model (Neural Network or Transformer):
  ```bash
  python deployment/deploy_nn.py       # For Neural Network
  python deployment/deploy_transformer.py  # For Transformer
  ```

---

## How to Run
1. Gather gameplay videos from platforms like YouTube or Twitch.
2. Process the videos and build the dataset.
3. Train both Neural Network and Transformer models.
4. Evaluate the models and deploy the one with the best performance.
5. Use the real-time controller to integrate the bot into gameplay.

---

## Contributing
Feel free to fork the repository, submit issues, or contribute through pull requests. Let’s build the future of AI gaming together!

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
