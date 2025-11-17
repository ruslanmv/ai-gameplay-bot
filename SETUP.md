# AI Gameplay Bot - Setup Guide

Complete setup instructions for the AI Gameplay Bot project.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Generating Sample Data](#generating-sample-data)
4. [Training Models](#training-models)
5. [Running the Application](#running-the-application)
6. [Using the Web Interface](#using-the-web-interface)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python:** 3.8 or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Disk Space:** 2GB minimum
- **OS:** Linux, macOS, or Windows

### Recommended Setup

- **Python:** 3.9 or 3.10
- **RAM:** 8GB or more
- **GPU:** NVIDIA GPU with CUDA support (optional, for faster training)
- **OS:** Ubuntu 20.04+ or macOS 11+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-gameplay-bot.git
cd ai-gameplay-bot
```

### 2. Create a Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** If you have a CUDA-enabled GPU, install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
```

## Generating Sample Data

Before training models, you need data. You can either use your own gameplay data or generate sample data for testing.

### Option A: Generate Sample Data (Quick Start)

```bash
python scripts/generate_sample_data.py
```

This creates:
- 30 synthetic game frames
- Neural network dataset (30 samples)
- Transformer dataset (100 samples)
- Sample feedback data

### Option B: Use Your Own Gameplay Data

#### Step 1: Record Gameplay

Record your gameplay and save the video to `data/raw/gameplay_videos/sample_video.mp4`

#### Step 2: Extract Frames

```bash
python scripts/video_processing.py
```

This extracts frames from the video at 1 FPS and saves them to `data/processed/frames/`

#### Step 3: Create Annotations

Create `data/raw/annotations/actions.txt` with one action per line (matching each frame):

```
move_forward
move_forward
turn_right
attack
jump
...
```

**Supported Actions:**
- move_forward
- move_backward
- turn_left
- turn_right
- attack
- jump
- interact
- use_item
- open_inventory
- cast_spell

#### Step 4: Build Datasets

```bash
python scripts/dataset_builder.py
```

#### Step 5: (Optional) Enrich Dataset

```bash
python scripts/generative_ai_enrichment.py
```

This augments your dataset by adding synthetic samples.

## Training Models

### Training Neural Network Model

```bash
python models/neural_network/nn_training.py
```

**Configuration** (edit `nn_training.py`):
- `BATCH_SIZE`: 32 (reduce if out of memory)
- `NUM_EPOCHS`: 50
- `LEARNING_RATE`: 0.001

**Output:**
- Trained model: `models/neural_network/nn_model.pth`
- Training history: `models/neural_network/training_history.json`

### Training Transformer Model

```bash
python models/transformer/transformer_training.py
```

**Configuration** (edit `transformer_training.py`):
- `BATCH_SIZE`: 16
- `NUM_EPOCHS`: 30
- `LEARNING_RATE`: 0.0001
- `SEQUENCE_LENGTH`: 10

**Output:**
- Trained model: `models/transformer/transformer_model.pth`
- Training history: `models/transformer/training_history.json`

## Running the Application

### Method 1: Using the Web Interface (Recommended)

#### Step 1: Start the Control Backend

```bash
python deployment/control_backend.py
```

This starts the control server on `http://localhost:8000`

#### Step 2: Open the Web Interface

Open your browser and navigate to:
```
file:///path/to/ai-gameplay-bot/frontend/index.html
```

Or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Then open: `http://localhost:3000`

#### Step 3: Use the Control Panel

1. Click "Start NN" or "Start Transformer" to launch services
2. Choose which model to use
3. Test predictions with the "Run test prediction" button

### Method 2: Manual Service Start

#### Start Neural Network Service

```bash
python deployment/deploy_nn.py
```

Service runs on `http://localhost:5000`

#### Start Transformer Service

```bash
python deployment/deploy_transformer.py
```

Service runs on `http://localhost:5001`

#### Test Predictions

```bash
python deployment/real_time_controller.py
```

## Using the Web Interface

### Service Management

- **Start NN**: Launches the Neural Network prediction service
- **Stop NN**: Stops the Neural Network service
- **Start Transformer**: Launches the Transformer prediction service
- **Stop Transformer**: Stops the Transformer service
- **Start both services**: Launches both services simultaneously

### Model Selection

Use the dropdown to choose which model the game controller will use:
- **Neural Network**: Faster, lower latency
- **Transformer**: More context-aware, potentially more accurate

### Testing

Click "Run test prediction" to send a random state to the selected model and see the predicted action. This helps verify that services are running correctly.

### System Messages

The log box at the bottom shows all system activities, errors, and status updates.

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'torch'

**Solution:**
```bash
pip install torch torchvision
```

#### 2. Model file not found

**Error:** `FileNotFoundError: models/neural_network/nn_model.pth`

**Solution:** Train the model first:
```bash
python models/neural_network/nn_training.py
```

#### 3. Port already in use

**Error:** `OSError: [Errno 48] Address already in use`

**Solution:** Kill the process using the port:

```bash
# Find process
lsof -i :5000

# Kill process
kill -9 <PID>
```

Or change the port in the deployment script.

#### 4. CUDA out of memory

**Solution:** Reduce batch size in training scripts or use CPU:

```python
device = torch.device('cpu')
```

#### 5. Frontend can't connect to backend

**Solution:**
1. Ensure control backend is running: `python deployment/control_backend.py`
2. Check that backend is on port 8000
3. Look for CORS errors in browser console
4. Try disabling browser ad-blockers

### Performance Optimization

#### For Faster Training

1. Use a GPU with CUDA support
2. Increase batch size (if you have enough RAM)
3. Use mixed precision training
4. Reduce model size

#### For Faster Inference

1. Use the Neural Network model (faster than Transformer)
2. Reduce input feature dimensions
3. Use batch processing
4. Enable model quantization

### Getting Help

- **Issues:** Report bugs at https://github.com/your-username/ai-gameplay-bot/issues
- **Documentation:** Check `data/README.md` for data format details
- **API Reference:** See `API.md` for endpoint documentation

## Next Steps

After setup:

1. **Customize Actions:** Edit action mappings in `scripts/input_mapping.py`
2. **Improve Models:** Collect more training data and retrain
3. **Fine-tune:** Use `nn_finetune.py` and `transformer_finetune.py`
4. **Deploy:** Set up on a production server with gunicorn
5. **Monitor:** Use the feedback system to track model performance

## Production Deployment

For production deployment:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn (example for NN service)
gunicorn -w 4 -b 0.0.0.0:5000 deployment.deploy_nn:app

# Use supervisor or systemd for process management
```

See `docs/PRODUCTION.md` for detailed production deployment instructions.
