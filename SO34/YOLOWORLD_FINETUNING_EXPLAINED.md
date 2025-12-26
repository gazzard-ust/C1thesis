# Fine-Tuning YOLO-World XL for Crack Detection

## Project Overview

This document describes the complete process of fine-tuning a YOLO-World XL model for crack detection on pipes, intended for deployment on a TurtleBot3 robot with CO2 gas monitoring capabilities.

### Objective
Detect three types of cracks on pipes:
- **Dummy crack**
- **Paper crack**
- **PVC pipe crack**

---

## Understanding the Evaluation Metrics

Before diving into the results, let's understand what each metric means:

### What is mAP (Mean Average Precision)?

**mAP** is the primary metric used to evaluate object detection models. Think of it as a score that measures how well your model can:
1. **Find objects** (Did it detect all the cracks?)
2. **Locate them accurately** (Are the bounding boxes in the right place?)
3. **Be confident correctly** (When it says "I'm 90% sure this is a crack," is it right?)

#### mAP@50 vs mAP@50-95

| Metric | What it measures | Strictness |
|--------|------------------|------------|
| **mAP@50** | Accuracy when bounding box overlaps ground truth by at least 50% | Lenient |
| **mAP@50-95** | Average accuracy across overlaps from 50% to 95% | Strict |

**Analogy:** Imagine you're playing darts.
- **mAP@50** = Did you hit somewhere on the dartboard? (easier)
- **mAP@50-95** = How close to the bullseye on average? (harder)

**Why mAP@50-95 is preferred:** It's the standard COCO metric and gives a better picture of how precisely your model locates objects. A high mAP@50-95 means your bounding boxes tightly fit the actual objects.

### What is IoU (Intersection over Union)?

IoU measures how much your predicted bounding box overlaps with the actual (ground truth) bounding box.

```
                 Area of Overlap
    IoU = ────────────────────────────
           Area of Union (both boxes)
```

**Visual Example:**
```
┌─────────────┐
│  Ground     │
│  Truth ┌────┼────┐
│        │████│    │  ← Overlap area
└────────┼────┘    │
         │ Prediction│
         └──────────┘

IoU = (████ shaded area) / (total area covered by both boxes)
```

| IoU Value | Meaning |
|-----------|---------|
| 1.0 (100%) | Perfect match - boxes are identical |
| 0.5 (50%) | Decent overlap - acceptable detection |
| 0.0 (0%) | No overlap - completely wrong |

### What is Precision?

**Precision** answers: *"When the model says it found a crack, how often is it actually correct?"*

```
                    True Positives (correct detections)
    Precision = ─────────────────────────────────────────────
                True Positives + False Positives (all detections)
```

**Example:**
- Model detects 100 "cracks"
- 98 are actually cracks (True Positives)
- 2 are false alarms (False Positives - maybe shadows or scratches)
- Precision = 98/100 = **0.98 or 98%**

**High precision means:** Few false alarms. When the model says "crack," you can trust it.

### What is Recall?

**Recall** answers: *"Of all the actual cracks in the images, how many did the model find?"*

```
                  True Positives (correct detections)
    Recall = ──────────────────────────────────────────────────
             True Positives + False Negatives (missed detections)
```

**Example:**
- There are 100 actual cracks in the test images
- Model correctly finds 97 of them
- Model misses 3 cracks
- Recall = 97/100 = **0.97 or 97%**

**High recall means:** The model rarely misses cracks. It finds almost everything.

### What is F1 Score?

**F1 Score** is the harmonic mean of Precision and Recall - it balances both metrics.

```
                2 × Precision × Recall
    F1 Score = ────────────────────────
                Precision + Recall
```

**Why use F1?** 
- A model could have 100% precision by only detecting obvious cracks (but missing many)
- A model could have 100% recall by marking everything as a crack (but many false alarms)
- F1 Score penalizes extreme trade-offs and rewards balanced performance

| F1 Score | Interpretation |
|----------|----------------|
| 0.90+ | Excellent - production ready |
| 0.80-0.90 | Good - minor improvements needed |
| 0.70-0.80 | Acceptable - needs more training data |
| Below 0.70 | Poor - significant issues |

### What is AP (Average Precision) per Class?

While mAP averages across ALL classes, **AP** shows performance for EACH class individually.

This helps identify:
- Which crack types are easy to detect
- Which crack types need more training data
- Class imbalance issues

---

## Final Results

### Overall Performance

| Metric | Test Set | Validation Set | What This Means |
|--------|----------|----------------|-----------------|
| **mAP@50-95** | **0.8692** | 0.8583 | Model locates cracks very precisely (86.9% accuracy at strict thresholds) |
| **mAP@50** | 0.9841 | 0.9768 | Model finds cracks in roughly the right location 98.4% of the time |
| **Precision** | 0.9799 | 0.9883 | Only 2% false alarms - very reliable detections |
| **Recall** | 0.9752 | 0.9633 | Finds 97.5% of all cracks - misses very few |
| **F1 Score** | 0.9775 | 0.9756 | Excellent balance between precision and recall |

### Interpreting These Results

✅ **mAP@50-95 = 0.8692 (86.92%)**
- This is considered **excellent** for object detection
- For reference, state-of-the-art models on COCO dataset achieve ~50-60% mAP@50-95
- Our specialized model on a focused task achieves much higher

✅ **Precision = 0.9799 (97.99%)**
- Out of 100 detections, ~98 are real cracks
- Very few false positives - the robot won't chase shadows

✅ **Recall = 0.9752 (97.52%)**
- Out of 100 actual cracks, the model finds ~98
- Very few missed detections - critical for safety applications

### Per-Class Performance (Test Set)

| Class | AP@50-95 | Interpretation |
|-------|----------|----------------|
| Dummy crack | 0.9560 (95.6%) | **Excellent** - Model learned this class very well |
| PVC pipe crack | 0.9116 (91.2%) | **Very Good** - Strong performance despite few training images |
| Paper crack | 0.7400 (74.0%) | **Good** - Lower than others, needs more diverse examples |

### Why is Paper Crack Lower?

Paper crack has the lowest AP despite having the most training images (399). This could be because:

1. **High variation:** Paper cracks may look very different from each other
2. **Subtle features:** Paper cracks might be harder to distinguish from background
3. **Annotation quality:** Some annotations might be inconsistent

**Solution:** Collect more diverse Paper crack images and ensure consistent annotation.

---

## 1. Dataset Preparation (Roboflow)

### 1.1 Data Collection

Images were collected for three crack types:

| Class | Images | Notes |
|-------|--------|-------|
| Paper crack | 399 | Most samples |
| Dummy crack | ~100 | Moderate samples |
| PVC pipe crack | 8 | **Very few samples** ⚠️ |

**⚠️ Class Imbalance Warning:** 
PVC pipe crack has only 8 images compared to 399 for Paper crack (50x difference). This can cause the model to be biased toward detecting Paper cracks. Augmentation helps but collecting more PVC images is recommended.

### 1.2 Annotation Process

**What is Annotation?**
Annotation means drawing bounding boxes around objects and labeling them. This teaches the model what to look for.

**Steps:**
1. Upload images to [Roboflow](https://roboflow.com)
2. Draw a tight bounding box around each crack
3. Assign the correct label (Dummy crack, Paper crack, or PVC pipe crack)
4. Approve the annotation

**Good vs Bad Annotation:**
```
GOOD ✅                          BAD ❌
┌──────────┐                    ┌────────────────┐
│ ████████ │ ← Tight fit        │                │
│ ████████ │                    │    ████████    │ ← Too much padding
└──────────┘                    │                │
                                └────────────────┘
```

### 1.3 Preprocessing Configuration

**What is Preprocessing?**
Preprocessing prepares images before training by standardizing their format.

| Setting | Value | Why? |
|---------|-------|------|
| Auto-Orient | Enabled | Fixes rotated images from camera |
| Resize | 640×640 | YOLO-World expects square images; 640 is standard size |
| Auto-Adjust Contrast | Enabled | Makes cracks more visible in low-contrast images |

### 1.4 Augmentation Configuration

**What is Augmentation?**
Augmentation creates new training images by modifying existing ones. This helps the model learn to recognize cracks in different conditions without collecting more real images.

| Augmentation | Value | What It Does | Why It Helps |
|--------------|-------|--------------|--------------|
| Flip | Horizontal + Vertical | Mirrors images | Cracks can appear in any orientation |
| Rotation | -15° to +15° | Tilts images slightly | Camera won't always be perfectly level |
| Shear | ±10° | Skews images | Compensates for perspective distortion |
| Brightness | -25% to +25% | Darkens/lightens | Works in different lighting conditions |
| Exposure | -15% to +15% | Adjusts light sensitivity | Handles over/underexposed images |
| Blur | Up to 1.5px | Slightly blurs | Works even if camera is slightly out of focus |
| Noise | Up to 1% | Adds grain | Handles noisy camera sensors |

**Output Multiplier: 3x**
Each original image generates 3 augmented versions, tripling the dataset size.

**Augmentations NOT Applied:**

| Augmentation | Why Skipped |
|--------------|-------------|
| Grayscale | Cracks may have color differences we want to preserve |
| Hue | Changing colors could make cracks unrecognizable |
| Crop | Might accidentally cut off the crack |

### 1.5 Dataset Split

**Why Split the Data?**

| Split | Percentage | Images | Purpose |
|-------|------------|--------|---------|
| **Train** | 88% | 2,292 | Model learns from these images |
| **Validation** | 8% | 217 | Model checks progress during training (prevents overfitting) |
| **Test** | 4% | 108 | Final evaluation - model never sees these until the end |

**Why separate Test from Validation?**
- Validation data influences training (model adjusts based on validation performance)
- Test data is completely "unseen" - gives true performance estimate

**Total after augmentation: 2,617 images**

### 1.6 Export Format

Export as **YOLOv8** format, which creates:
- `data.yaml` - Configuration file with paths and class names
- `images/` - Image files
- `labels/` - Text files with bounding box coordinates

---

## 2. Environment Setup (NVIDIA DGX)

### What is DGX?
NVIDIA DGX is a powerful computer designed for AI training with high-end GPUs (A100). It's much faster than a regular laptop for training deep learning models.

### 2.1 Create Conda Environment

**What is Conda?**
Conda is a package manager that creates isolated Python environments. This prevents conflicts between different projects.

```bash
# Create new environment named "yoloworld" with Python 3.10
conda create -n yoloworld python=3.10 -y

# Activate the environment
conda activate yoloworld
```

### 2.2 Install Dependencies

```bash
# PyTorch - The deep learning framework
# cu121 means CUDA 12.1 (for NVIDIA GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# YOLO and utilities
pip install ultralytics    # YOLO-World model
pip install wandb          # Experiment tracking
pip install roboflow       # Dataset download
pip install opencv-python-headless  # Image processing
pip install tensorboard    # Training visualization
```

### 2.3 Download Dataset from Roboflow

```python
from roboflow import Roboflow

# Connect to Roboflow with your API key
rf = Roboflow(api_key="YOUR_API_KEY")

# Access your project
project = rf.workspace("your-workspace").project("pipe-crack-detection")

# Download specific version in YOLOv8 format
version = project.version(1)
dataset = version.download("yolov8")
```

**Dataset structure after download:**
```
pipe-crack-detection-1/
├── data.yaml           ← Configuration file
├── train/
│   ├── images/         ← Training images
│   └── labels/         ← Training labels (bounding boxes)
├── valid/
│   ├── images/         ← Validation images
│   └── labels/         ← Validation labels
└── test/
    ├── images/         ← Test images
    └── labels/         ← Test labels
```

---

## 3. Training Configuration

### 3.1 Model Selection

**Model:** YOLO-World XL (`yolov8x-worldv2.pt`)

**What is YOLO-World?**
YOLO (You Only Look Once) is a fast object detection model. YOLO-World is a special version that can detect objects using text descriptions, making it excellent for custom classes.

**Why YOLO-World XL?**
| Variant | Size | Speed | Accuracy |
|---------|------|-------|----------|
| YOLO-World S | Small | Fastest | Lower |
| YOLO-World M | Medium | Fast | Medium |
| YOLO-World L | Large | Slower | Higher |
| **YOLO-World XL** | **Extra Large** | **Slowest** | **Highest** ✓ |

We chose XL for maximum accuracy. Speed is less important since we're running on a powerful laptop, not the robot itself.

### 3.2 Training Parameters Explained

| Parameter | Value | What It Means |
|-----------|-------|---------------|
| **Epochs** | 100 | Train for 100 complete passes through the dataset |
| **Batch Size** | 16 | Process 16 images at once (limited by GPU memory) |
| **Image Size** | 640×640 | Input resolution (matches preprocessing) |
| **Learning Rate** | 0.0002 | How fast the model adjusts weights (small = stable, large = fast but unstable) |
| **Optimizer** | AdamW | Algorithm for updating model weights (AdamW works well for vision models) |
| **Weight Decay** | 0.05 | Regularization to prevent overfitting (penalizes large weights) |
| **Patience** | 20 | Stop early if no improvement for 20 epochs |
| **Device** | GPU 0 | Use the first GPU |

**What is an Epoch?**
One epoch = the model sees every training image once. More epochs = more learning, but too many can cause overfitting.

**What is Batch Size?**
Instead of learning from one image at a time, the model processes multiple images together. Larger batches are faster but need more GPU memory.

**What is Overfitting?**
When a model memorizes the training data instead of learning general patterns. It performs great on training data but poorly on new images.

### 3.3 Training Script

```python
from ultralytics import YOLOWorld
from datetime import datetime
import wandb

# Initialize Weights & Biases for experiment tracking
wandb.init(project="yoloworld-crack-detection")

# Load pre-trained YOLO-World XL model
model = YOLOWorld("yolov8x-worldv2.pt")

# Set our custom classes
model.set_classes(["Dummy crack", "Paper crack", "PVC pipe crack"])

# Start training
results = model.train(
    data="pipe-crack-detection-1/data.yaml",  # Dataset config
    epochs=100,              # Number of training cycles
    batch=16,                # Images per batch
    imgsz=640,               # Image size
    lr0=0.0002,              # Initial learning rate
    optimizer="AdamW",       # Optimizer algorithm
    weight_decay=0.05,       # Regularization strength
    patience=20,             # Early stopping patience
    device=0,                # GPU to use
    project="crack_detection_runs",  # Output folder
    name=f"yoloworld_xl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    save_period=10,          # Save checkpoint every 10 epochs
    plots=True,              # Generate training plots
)

# Log final metrics
wandb.log({
    "final/mAP50-95": results.results_dict["metrics/mAP50-95(B)"],
    "final/mAP50": results.results_dict["metrics/mAP50(B)"],
    "final/precision": results.results_dict["metrics/precision(B)"],
    "final/recall": results.results_dict["metrics/recall(B)"],
})

wandb.finish()
```

### 3.4 Running Training on DGX

```bash
# CUDA_VISIBLE_DEVICES=3 selects GPU #3
# This is important when multiple users share the DGX
CUDA_VISIBLE_DEVICES=3 python train_yoloworld_wandb.py
```

**Training Time:** ~1.08 hours (100 epochs on A100-40GB)

This is fast because:
- A100 GPU is very powerful (40GB memory, thousands of cores)
- Dataset is relatively small (2,617 images)
- YOLO is optimized for speed

---

## 4. Evaluation

### 4.1 Evaluation Script

```python
from ultralytics import YOLOWorld

# Load the best model (saved during training)
model = YOLOWorld("crack_detection_runs/yoloworld_xl_20251226_171626/weights/best.pt")

# Evaluate on test set
results = model.val(
    data="pipe-crack-detection-1/data.yaml",
    split="test",        # Use test split
    imgsz=640,
    batch=16,
    conf=0.25,           # Confidence threshold
    iou=0.45,            # IoU threshold for NMS
)

# Print metrics
print(f"mAP@50-95: {results.box.map:.4f}")
print(f"mAP@50: {results.box.map50:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall: {results.box.mr:.4f}")
```

**What is Confidence Threshold (conf=0.25)?**
The model only reports detections it's at least 25% confident about. Lower threshold = more detections (but more false positives).

**What is NMS (Non-Maximum Suppression)?**
When the model detects the same object multiple times, NMS keeps only the best detection and removes duplicates.

### 4.2 Results Interpretation

**Test Set Performance:**
```
mAP@50-95:  0.8692  ← 86.92% - Excellent!
mAP@50:     0.9841  ← 98.41% - Near perfect at lenient threshold
Precision:  0.9799  ← 97.99% - Very few false alarms
Recall:     0.9752  ← 97.52% - Finds almost all cracks
F1 Score:   0.9775  ← 97.75% - Excellent balance
```

**What Do These Numbers Mean in Practice?**

| Scenario | With Our Model |
|----------|----------------|
| 100 actual cracks in images | Model finds ~98 of them |
| Model makes 100 detections | ~98 are real cracks, ~2 are false |
| Bounding box accuracy | 87% of boxes tightly fit the cracks |

**Inference Speed (DGX A100):**
```
Preprocess:  2.2ms   ← Resizing, normalizing
Inference:   17.9ms  ← Model prediction
Postprocess: 8.6ms   ← NMS, formatting results
─────────────────────
Total:       ~28.7ms per image (~35 FPS)
```

**Note:** This speed is on a powerful A100 GPU. On your laptop, expect 15-30 FPS depending on your GPU.

---

## 5. Model Export

### 5.1 Best Model Location

After training, two model files are saved:

```
crack_detection_runs/yoloworld_xl_20251226_171626/weights/
├── best.pt    ← Best performance during training (USE THIS)
└── last.pt    ← Final epoch (may be slightly worse)
```

**Always use `best.pt`** - it's the checkpoint with the highest validation mAP.

### 5.2 Copying the Model

Copy the best model to your deployment folder:

```bash
cp crack_detection_runs/yoloworld_xl_20251226_171626/weights/best.pt ./best.pt
```

The `best.pt` file (~150MB) contains:
- Model architecture
- Trained weights
- Class names
- Training configuration

This single file is all you need for deployment on your laptop.

---

## 6. Deployment

### 6.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        LAPTOP                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              gazzard_gui_v3.py                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │ │
│  │  │  YOLO-World  │  │    Depth     │  │    Robot     │ │ │
│  │  │  (best.pt)   │  │  Estimation  │  │   Control    │ │ │
│  │  │              │  │              │  │  (cmd_vel)   │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘ │ │
│  │         │                 │                 │          │ │
│  │         └─────────────────┴─────────────────┘          │ │
│  │                    ROS2 Node                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                  │
│              Web Interface (http://localhost:5000)           │
└───────────────────────────│──────────────────────────────────┘
                            │ ROS2 Topics
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                      TurtleBot3 (RPi)                         │
│   Camera → /image/compressed                                  │
│   CO2 Sensor → /co2_concentration                             │
│   Motors ← /cmd_vel                                           │
└───────────────────────────────────────────────────────────────┘
```

### 6.2 Files Required

```
deployment/
├── best.pt                 # Trained model weights (~150MB)
├── gazzard_gui_v3.py      # Main application
└── requirements.txt        # Python dependencies
```

### 6.3 Configuration

At the top of `gazzard_gui_v3.py`:

```python
# ============================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================

MODEL_PATH = "best.pt"  # Path to your fine-tuned model

CRACK_CLASSES = [
    "Dummy crack",
    "Paper crack",
    "PVC pipe crack",
]

DETECTION_CONFIDENCE = 0.25  # Minimum confidence to report detection
IOU_THRESHOLD = 0.45         # IoU threshold for NMS

STOPPING_DISTANCE_CM = 30.0  # Robot stops this far from crack
EMERGENCY_STOP_CM = 20.0     # Emergency stop distance
```

### 6.4 Running the System

```bash
# Start the application
python gazzard_gui_v3.py

# Open web interface in browser
# http://localhost:5000
```

---

## 7. Weights & Biases Tracking

### What is Weights & Biases (wandb)?

A tool for tracking machine learning experiments. It automatically logs:
- Training metrics (loss, mAP) over time
- System info (GPU usage, memory)
- Model checkpoints
- Hyperparameters

**Why Use It?**
- Compare different training runs
- Share results with teammates
- Reproduce experiments later

### Project Dashboard

- **Project:** `yoloworld-crack-detection`
- **Training Run:** `yoloworld_xl_20251226_171626`
- **Evaluation Run:** `eval_20251226_184800`
- **URL:** https://wandb.ai/ai231/yoloworld-crack-detection

### Metrics Tracked

| Metric | When Logged |
|--------|-------------|
| Training loss | Every batch |
| Validation mAP | Every epoch |
| Learning rate | Every epoch |
| Final test metrics | End of training |
| Per-class AP | End of training |

---

## 8. Lessons Learned

### What Worked Well ✅

1. **YOLO-World XL** - Excellent transfer learning; high accuracy with limited data
2. **Roboflow** - Made data preparation and augmentation easy
3. **AdamW optimizer** - Stable training with good convergence
4. **Augmentation** - Helped compensate for limited real images
5. **wandb** - Made tracking experiments effortless

### Areas for Improvement ⚠️

1. **Class Imbalance**
   - PVC pipe crack: Only 8 images
   - Paper crack: 399 images
   - This 50x difference can bias the model

2. **Paper Crack Performance**
   - AP@50-95: 0.74 (lower than other classes)
   - May need more diverse examples

3. **Dataset Size**
   - 2,617 images is relatively small
   - More data would improve generalization

### Recommendations for Future Work

| Issue | Solution |
|-------|----------|
| Class imbalance | Collect more PVC pipe crack images (target: 100+) |
| Paper crack AP | Add more varied Paper crack examples |
| Generalization | Collect images in different lighting, angles |
| Real-time speed | Test on actual deployment hardware |

---

## 9. Glossary

| Term | Definition |
|------|------------|
| **Annotation** | Drawing bounding boxes and labels on images |
| **AP** | Average Precision - performance metric for one class |
| **Augmentation** | Creating new training images by modifying existing ones |
| **Batch Size** | Number of images processed together |
| **Bounding Box** | Rectangle that surrounds a detected object |
| **Confidence** | Model's certainty about a detection (0-1) |
| **Epoch** | One complete pass through all training data |
| **F1 Score** | Harmonic mean of precision and recall |
| **Fine-tuning** | Adapting a pre-trained model to a new task |
| **Ground Truth** | The correct answer (human annotations) |
| **IoU** | Intersection over Union - measures box overlap |
| **mAP** | Mean Average Precision - overall detection accuracy |
| **NMS** | Non-Maximum Suppression - removes duplicate detections |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Precision** | Fraction of detections that are correct |
| **Recall** | Fraction of actual objects that were detected |
| **Validation Set** | Data used to tune hyperparameters during training |
| **Test Set** | Data used for final evaluation (never seen during training) |

---

## 10. References

- [Ultralytics YOLO-World Documentation](https://docs.ultralytics.com/models/yolo-world/)
- [Roboflow Documentation](https://docs.roboflow.com/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [TurtleBot3 e-Manual](https://emanual.robotis.com/docs/en/platform/turtlebot3/overview/)
- [COCO Evaluation Metrics](https://cocodataset.org/#detection-eval)

---

## Appendix: Training Output

```
100 epochs completed in 1.079 hours.
Optimizer stripped from crack_detection_runs/yoloworld_xl_20251226_171626/weights/last.pt
Optimizer stripped from crack_detection_runs/yoloworld_xl_20251226_171626/weights/best.pt

Validating crack_detection_runs/yoloworld_xl_20251226_171626/weights/best.pt...
YOLOv8x-worldv2 summary: 127 layers, 72,856,217 parameters, 273.3 GFLOPs

        Class     Images  Instances    Box(P        R    mAP50  mAP50-95)
          all        217        247    0.987    0.963    0.977     0.846
  Dummy crack         58         84    0.987    0.988    0.989     0.938
PVC pipe crack         78         81    0.989    0.938    0.975     0.841
   Paper crack         80         82    0.986    0.963    0.967     0.758

Speed: 0.1ms preprocess, 3.9ms inference, 0.1ms loss, 1.9ms postprocess per image
```

**Reading the Output:**
- **127 layers** - Depth of the neural network
- **72,856,217 parameters** - Number of trainable weights (~73 million)
- **273.3 GFLOPs** - Computational complexity (billion floating point operations)
- **Box(P)** - Precision
- **R** - Recall
- **mAP50** - mAP at 50% IoU
- **mAP50-95** - mAP averaged from 50% to 95% IoU

---

*Document created: December 26, 2025*
*Model version: yoloworld_xl_20251226_171626*
*Author: Gazzard Project Team*
