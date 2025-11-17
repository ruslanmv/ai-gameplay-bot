# Data Directory Structure

This directory contains all data used by the AI Gameplay Bot project.

## Directory Organization

```
data/
├── raw/                          # Raw, unprocessed data
│   ├── gameplay_videos/          # Original gameplay video files
│   │   └── sample_video.mp4      # Example: gameplay recording
│   └── annotations/              # Manual annotations and labels
│       └── actions.txt           # Frame-by-frame action labels
│
└── processed/                    # Processed, ready-to-use data
    ├── frames/                   # Extracted video frames
    │   ├── frame_0000.jpg
    │   ├── frame_0001.jpg
    │   └── ...
    ├── nn_dataset.csv            # Dataset for neural network training
    ├── transformer_dataset.csv   # Dataset for transformer training
    ├── enriched_dataset.csv      # Augmented dataset
    └── updated_nn_dataset.csv    # Dataset updated with feedback
```

## Data Formats

### 1. Actions Annotation File (`actions.txt`)

Format: One action per line, corresponding to each extracted frame

```
move_forward
turn_right
attack
jump
...
```

**Supported Actions:**
- `move_forward` - Move character forward
- `move_backward` - Move character backward
- `turn_left` - Turn character left
- `turn_right` - Turn character right
- `attack` - Perform attack action
- `jump` - Make character jump
- `interact` - Interact with object/NPC
- `use_item` - Use item from inventory
- `open_inventory` - Open inventory menu
- `cast_spell` - Cast spell/ability

### 2. Neural Network Dataset (`nn_dataset.csv`)

CSV format with feature columns and action labels:

```csv
feature_0,feature_1,...,feature_127,action
0.234,0.567,...,0.891,0
0.345,0.678,...,0.912,1
...
```

**Columns:**
- `feature_0` to `feature_127`: Extracted image features (128 dimensions)
- `action`: Action index (0-9 corresponding to action types)

**Action Indices:**
```
0: move_forward
1: move_backward
2: turn_left
3: turn_right
4: attack
5: jump
6: interact
7: use_item
8: open_inventory
9: cast_spell
```

### 3. Transformer Dataset (`transformer_dataset.csv`)

Same format as neural network dataset but typically larger and includes temporal sequences.

### 4. Enriched Dataset (`enriched_dataset.csv`)

Augmented version of the neural network dataset with synthetic samples added using data augmentation techniques (noise injection, feature perturbation).

### 5. User Feedback Data (`feedback/user_feedback.csv`)

CSV format tracking model predictions and corrections:

```csv
predicted_action,correct_action,confidence,timestamp,is_correct
move_forward,move_forward,0.95,2024-01-01 10:00:00,1
attack,jump,0.67,2024-01-01 10:05:00,0
...
```

**Columns:**
- `predicted_action`: Action predicted by the model
- `correct_action`: Actual correct action (ground truth)
- `confidence`: Model confidence score (0-1)
- `timestamp`: When prediction was made
- `is_correct`: 1 if prediction matches ground truth, 0 otherwise

## Generating Sample Data

To quickly generate sample data for testing:

```bash
python scripts/generate_sample_data.py
```

This will create:
- 30 synthetic game frames
- Neural network dataset (30 samples)
- Transformer dataset (100 samples)
- Sample feedback data (20 samples)

## Using Your Own Data

### Recording Gameplay

1. Record gameplay video and save to `data/raw/gameplay_videos/`
2. Extract frames:
   ```bash
   python scripts/video_processing.py
   ```

### Creating Annotations

1. Review extracted frames in `data/processed/frames/`
2. Create `data/raw/annotations/actions.txt`
3. Write one action per line (same order as frames)
4. Ensure number of actions matches number of frames

### Building Datasets

```bash
python scripts/dataset_builder.py
```

This will:
- Load frames and annotations
- Extract image features
- Create training-ready CSV files

### Data Augmentation

To increase dataset size:

```bash
python scripts/generative_ai_enrichment.py
```

This applies data augmentation techniques to expand your dataset.

## Best Practices

1. **Frame Rate**: Extract 1-2 frames per second for smooth action transitions
2. **Annotation Quality**: Ensure actions match frame content accurately
3. **Dataset Size**: Minimum 500-1000 samples recommended for good performance
4. **Balance**: Try to balance different action types in your dataset
5. **Validation**: Reserve 20% of data for validation/testing

## Troubleshooting

**Mismatch between frames and actions:**
- Check that actions.txt has exactly one line per frame
- Verify frames are sorted correctly (alphabetically)

**Low model accuracy:**
- Increase dataset size
- Use data augmentation
- Check annotation quality
- Balance action distribution

**Memory issues with large datasets:**
- Process in batches
- Reduce image resolution
- Use data generators instead of loading all at once
