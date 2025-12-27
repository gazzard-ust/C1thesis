# Expected Outputs Measurement Guide

## Overview

This guide explains how to measure and validate each Expected Output (E.O.) for the Gazzard gas leak detection system. **Both software measurements and physical validation are required** to ensure accuracy.

---

## Equipment Needed

### Measurement Tools
| Item | Purpose |
|------|---------|
| Tape measure (metric) | Measure actual distances |
| Stopwatch / Timer | Measure approach duration |
| Protractor or angle finder | Measure alignment angles |
| CO2 gas canister or source | Create controlled leak |
| CO2 meter (handheld) | Validate CO2 readings |
| Tripod or fixed mount | Consistent camera positioning |
| Markers / Tape | Mark positions on floor |

### Test Setup
| Item | Purpose |
|------|---------|
| Pipe with visible crack | Detection target |
| Controlled environment | Reduce variables |
| Grid markings on floor | Measure robot position |
| Reference ruler in frame | Calibrate depth estimation |

---

## Test Environment Setup

### 1. Prepare the Test Area
```
┌────────────────────────────────────────────────────────────┐
│                                                            │
│    [CO2 Source]                                            │
│         ↓                                                  │
│    ┌─────────┐                                             │
│    │  Pipe   │ ← Crack location (marked)                   │
│    │  with   │                                             │
│    │  Crack  │                                             │
│    └─────────┘                                             │
│         │                                                  │
│         │ ← Measure this distance                          │
│         │                                                  │
│    ┌─────────┐                                             │
│    │ Robot   │ ← Starting position (marked)                │
│    │ Start   │                                             │
│    └─────────┘                                             │
│                                                            │
│    [Grid markings every 10cm on floor]                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 2. Mark Reference Points
- Mark crack location on floor with tape
- Mark robot starting position
- Create grid lines every 10cm for distance reference
- Place CO2 source near the crack

### 3. Controlled Variables
- Consistent lighting
- No wind/air currents (affects CO2 dispersion)
- Same crack sample for repeated tests
- Same starting distance

---

## E.O. 3.1: mAP and Inference Speed

### What to Measure
| Metric | Type | How to Get |
|--------|------|------------|
| mAP@50-95 | Software | From training results |
| mAP@50 | Software | From training results |
| Precision | Software | From training results |
| Recall | Software | From training results |
| Inference Speed | Software | From `/benchmark` endpoint |

### mAP (Already Measured During Training)

These values come from your training evaluation:

```
mAP@50-95:  0.8692
mAP@50:     0.9841
Precision:  0.9799
Recall:     0.9752
F1 Score:   0.9775
```

**No physical measurement needed** - this is from the test dataset evaluation.

### Inference Speed Measurement

#### Software Measurement:
```bash
# 1. Start the system
python gazzard_gui_v4.py

# 2. Wait 60 seconds for data collection

# 3. Get benchmark results
curl http://localhost:5000/benchmark
```

#### Physical Validation (Optional):
To verify the FPS is real-time capable:

1. Count frames visually for 10 seconds
2. Compare to reported FPS
3. Check for lag or dropped frames in video feed

#### Recording Results:

| Metric | Value | Unit |
|--------|-------|------|
| YOLO-World XL | ___ ± ___ | ms |
| Depth-Anything-V2 | ___ ± ___ | ms |
| Total Pipeline | ___ ± ___ | ms |
| Throughput | ___ ± ___ | FPS |
| Samples Collected | ___ | frames |

---

## E.O. 4.1: Accuracy of Identifying Gas Leak Source

### What to Measure
| Metric | Type | Description |
|--------|------|-------------|
| Detection Accuracy | Software + Physical | Did robot correctly identify leak location? |
| True Positives | Physical validation | Robot stopped at actual leak |
| False Positives | Physical validation | Robot stopped but no leak there |
| False Negatives | Physical validation | Leak exists but robot missed it |

### Test Procedure

#### Setup:
1. Place pipe with crack at known location
2. Place CO2 source at the crack (simulating leak)
3. Mark the "ground truth" leak position
4. Position robot at starting point (e.g., 150cm away)

#### Run Test:
```bash
# 1. Start system
python gazzard_gui_v4.py

# 2. Open web interface
http://localhost:5000

# 3. Click "Start" to begin search

# 4. Wait for TARGET_CONFIRMED state

# 5. Record results
curl http://localhost:5000/expected_outputs
```

#### Physical Validation:

| Step | Action | Record |
|------|--------|--------|
| 1 | Measure actual distance from robot to crack | ___ cm |
| 2 | Verify CO2 level at robot position with handheld meter | ___ ppm |
| 3 | Check if robot is facing the crack | Yes / No |
| 4 | Verify crack is visible in camera frame | Yes / No |

#### Accuracy Calculation:

Run **N trials** (recommended: N = 10 minimum):

```
                    Successful Confirmations
Accuracy (%) = ─────────────────────────────── × 100
                    Total Trials
```

#### Recording Results:

| Trial | Leak Present | Robot Confirmed | Result | Distance Error |
|-------|--------------|-----------------|--------|----------------|
| 1 | Yes | Yes | TP | +2.5 cm |
| 2 | Yes | Yes | TP | -1.2 cm |
| 3 | Yes | No | FN | N/A |
| 4 | No | No | TN | N/A |
| 5 | No | Yes | FP | N/A |
| ... | ... | ... | ... | ... |

**Confusion Matrix:**

|  | Predicted: Leak | Predicted: No Leak |
|--|-----------------|-------------------|
| **Actual: Leak** | True Positive (TP) | False Negative (FN) |
| **Actual: No Leak** | False Positive (FP) | True Negative (TN) |

**Metrics to Calculate:**
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## E.O. 4.2: Distance Estimation Accuracy

### What to Measure
| Metric | Type | Description |
|--------|------|-------------|
| Estimated Distance | Software | From Depth-Anything-V2 |
| Actual Distance | Physical | Tape measure |
| Estimation Error | Calculated | Actual - Estimated |

### Test Procedure

#### Setup:
1. Place crack at known distances: 30cm, 50cm, 80cm, 100cm, 150cm
2. Mark each position clearly

#### For Each Distance:

```bash
# 1. Position robot at marked distance from crack
# 2. Ensure crack is visible and detected
# 3. Record software estimate
curl http://localhost:5000/robot_status | jq '.crack.distance'

# 4. Measure actual distance with tape measure
```

#### Physical Measurement Method:

```
            ┌─────────────┐
            │    Crack    │
            └──────┬──────┘
                   │
                   │ ← Measure from crack center
                   │    to robot camera lens
                   │
            ┌──────┴──────┐
            │   Robot     │
            │   Camera    │
            └─────────────┘
```

**Measure from:**
- Crack center point
- To robot's camera lens (not robot body)

#### Recording Results:

| Actual (cm) | Estimated (cm) | Error (cm) | Error (%) |
|-------------|----------------|------------|-----------|
| 30 | ___ | ___ | ___ |
| 50 | ___ | ___ | ___ |
| 80 | ___ | ___ | ___ |
| 100 | ___ | ___ | ___ |
| 150 | ___ | ___ | ___ |

**Calculate:**
```
Error (cm) = Estimated - Actual
Error (%) = |Error| / Actual × 100

Mean Absolute Error (MAE) = Σ|Error| / N
Root Mean Square Error (RMSE) = √(Σ(Error²) / N)
```

#### Final Stopping Distance:

| Trial | Target (cm) | Actual Stop (cm) | Error (cm) |
|-------|-------------|------------------|------------|
| 1 | 30 | ___ | ___ |
| 2 | 30 | ___ | ___ |
| 3 | 30 | ___ | ___ |
| ... | ... | ... | ... |

---

## E.O. 4.3: Optimal Linear and Angular Speed

### What to Measure
| Metric | Type | Description |
|--------|------|-------------|
| Linear Speed | Software | From `/expected_outputs` |
| Angular Speed | Software | From `/expected_outputs` |
| Approach Time | Software + Physical | Time from start to stop |
| Distance Traveled | Physical | Actual path length |

### Test Procedure

#### Software Measurement:
```bash
# After a complete approach sequence:
curl http://localhost:5000/expected_outputs | jq '.EO_4_3_optimal_speed'
```

#### Physical Validation:

**Method 1: Stopwatch**
1. Start timer when robot begins moving toward crack
2. Stop timer when robot reaches TARGET_CONFIRMED
3. Compare to software `approach_duration_sec`

**Method 2: Video Analysis**
1. Record video of robot approach
2. Count frames and calculate actual speed
3. Compare to commanded speed

**Method 3: Distance/Time**
1. Mark starting position
2. Mark ending position
3. Measure distance traveled
4. Calculate: Speed = Distance / Time

#### Recording Results:

| Trial | Start Dist (cm) | End Dist (cm) | Time (s) | Calc Speed (cm/s) | Software Speed (m/s) |
|-------|-----------------|---------------|----------|-------------------|---------------------|
| 1 | 150 | 28 | ___ | ___ | ___ |
| 2 | 150 | 30 | ___ | ___ | ___ |
| 3 | 150 | 29 | ___ | ___ | ___ |

**Configured vs Actual:**

| Parameter | Configured | Actual Mean | Difference |
|-----------|------------|-------------|------------|
| Linear Speed (m/s) | 0.05 | ___ | ___ |
| Angular Speed (rad/s) | 0.15 | ___ | ___ |

---

## E.O. 4.4: Alignment Deviation

### What to Measure
| Metric | Type | Description |
|--------|------|-------------|
| Centering Error | Software | Pixels from image center |
| Angular Offset | Physical | Degrees off from crack |
| Lateral Offset | Physical | cm left/right of crack centerline |

### Test Procedure

#### Software Measurement:
```bash
curl http://localhost:5000/expected_outputs | jq '.EO_4_4_alignment_deviation'
```

#### Physical Measurement:

**Method 1: Angular Offset**

```
                    Crack
                      │
                      │
            θ ←───────┼─────── Robot heading
                      │
                      │
                    Robot
```

1. When robot stops, measure angle between:
   - Robot's forward direction
   - Line to crack center
2. Use protractor or calculate from lateral offset

**Method 2: Lateral Offset**

```
        ← d →
    ────┬─────────── Crack centerline
        │
        │
      Robot
```

1. Extend a line from robot camera perpendicular to heading
2. Measure distance `d` from this line to crack center
3. Positive = crack is to the right, Negative = to the left

#### Recording Results:

| Trial | Pixel Error (px) | Lateral Offset (cm) | Angular Offset (°) | Aligned? |
|-------|------------------|---------------------|-------------------|----------|
| 1 | ___ | ___ | ___ | Yes/No |
| 2 | ___ | ___ | ___ | Yes/No |
| 3 | ___ | ___ | ___ | Yes/No |

**Alignment Criteria:**
- Pixel Error ≤ 30 pixels = Aligned
- Lateral Offset ≤ 5 cm = Aligned  
- Angular Offset ≤ 10° = Aligned

**Calculate:**
```
Alignment Accuracy (%) = (Aligned Trials / Total Trials) × 100
Mean Pixel Error = Σ|Pixel Error| / N
Mean Lateral Offset = Σ|Lateral Offset| / N
```

---

## Complete Test Protocol

### Before Testing
- [ ] Charge robot battery fully
- [ ] Calibrate CO2 sensor
- [ ] Check camera focus
- [ ] Mark test positions on floor
- [ ] Prepare data recording sheets

### Test Sequence

```
┌─────────────────────────────────────────────────────────────┐
│                    COMPLETE TEST SEQUENCE                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. SETUP PHASE                                             │
│     □ Position pipe with crack                              │
│     □ Place CO2 source                                      │
│     □ Position robot at starting mark                       │
│     □ Record starting distance: ___ cm                      │
│                                                             │
│  2. CALIBRATION PHASE                                       │
│     □ Start gazzard_gui_v4.py                               │
│     □ Wait for CO2 baseline calibration (10 sec)            │
│     □ Record baseline CO2: ___ ppm                          │
│                                                             │
│  3. DETECTION PHASE                                         │
│     □ Click "Start" on web interface                        │
│     □ Start stopwatch                                       │
│     □ Observe robot behavior                                │
│                                                             │
│  4. CONFIRMATION PHASE                                      │
│     □ Wait for TARGET_CONFIRMED                             │
│     □ Stop stopwatch                                        │
│     □ Record approach time: ___ sec                         │
│                                                             │
│  5. MEASUREMENT PHASE                                       │
│     □ Measure actual distance to crack: ___ cm              │
│     □ Measure lateral offset: ___ cm                        │
│     □ Measure angular offset: ___ °                         │
│     □ Verify CO2 with handheld meter: ___ ppm               │
│                                                             │
│  6. DATA COLLECTION PHASE                                   │
│     □ curl http://localhost:5000/expected_outputs/save      │
│     □ Save video recording                                  │
│     □ Fill in data sheet                                    │
│                                                             │
│  7. RESET PHASE                                             │
│     □ Click "Reset" on web interface                        │
│     □ Return robot to starting position                     │
│     □ Repeat from step 1 for next trial                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Number of Trials

| E.O. | Minimum Trials | Recommended |
|------|----------------|-------------|
| 3.1 (Inference) | 500 frames | 1000+ frames |
| 4.1 (Accuracy) | 10 trials | 20+ trials |
| 4.2 (Distance) | 5 per distance × 5 distances | 10 per distance |
| 4.3 (Speed) | 10 trials | 20+ trials |
| 4.4 (Alignment) | 10 trials | 20+ trials |

---

## Data Recording Template

### Trial Record Sheet

```
═══════════════════════════════════════════════════════════════
TRIAL #: ___    DATE: ___________    TIME: ___________
═══════════════════════════════════════════════════════════════

SETUP:
  Starting Distance:     ___________ cm
  CO2 Source Position:   ___________ 
  Crack Type:            □ Dummy  □ Paper  □ PVC

SOFTWARE READINGS (from /expected_outputs):
  Estimated Distance:    ___________ cm
  Final CO2 Level:       ___________ ppm
  Centering Error:       ___________ pixels
  Approach Duration:     ___________ sec
  Linear Speed Used:     ___________ m/s
  Angular Speed Used:    ___________ rad/s

PHYSICAL MEASUREMENTS:
  Actual Distance:       ___________ cm
  Lateral Offset:        ___________ cm
  Angular Offset:        ___________ °
  Handheld CO2 Reading:  ___________ ppm

CALCULATED ERRORS:
  Distance Error:        ___________ cm  (_____%)
  
RESULT:
  □ True Positive (Correct detection)
  □ False Positive (Detected but wrong location)
  □ False Negative (Missed detection)
  □ True Negative (Correctly no detection)

NOTES:
_______________________________________________________________
_______________________________________________________________

═══════════════════════════════════════════════════════════════
```

---

## Final Report Table Templates

### Table 1: Model Performance (E.O. 3.1)

| Metric | Value |
|--------|-------|
| mAP@50-95 | 0.8692 |
| mAP@50 | 0.9841 |
| Precision | 0.9799 |
| Recall | 0.9752 |
| F1 Score | 0.9775 |
| YOLO-World XL Inference | ___ ± ___ ms |
| Depth-Anything-V2 Inference | ___ ± ___ ms |
| Total Pipeline | ___ ± ___ ms |
| Throughput | ___ ± ___ FPS |

### Table 2: Leak Detection Accuracy (E.O. 4.1)

| Metric | Value |
|--------|-------|
| Total Trials | ___ |
| True Positives | ___ |
| False Positives | ___ |
| False Negatives | ___ |
| True Negatives | ___ |
| **Accuracy** | ___% |
| **Precision** | ___% |
| **Recall** | ___% |

### Table 3: Distance Estimation (E.O. 4.2)

| Actual (cm) | Estimated (cm) | Error (cm) | Error (%) |
|-------------|----------------|------------|-----------|
| 30 | ___ ± ___ | ___ | ___ |
| 50 | ___ ± ___ | ___ | ___ |
| 80 | ___ ± ___ | ___ | ___ |
| 100 | ___ ± ___ | ___ | ___ |
| 150 | ___ ± ___ | ___ | ___ |
| **MAE** | | ___ cm | |
| **RMSE** | | ___ cm | |

### Table 4: Approach Speed (E.O. 4.3)

| Parameter | Configured | Actual | Unit |
|-----------|------------|--------|------|
| Linear Speed | 0.05 | ___ ± ___ | m/s |
| Angular Speed | 0.15 | ___ ± ___ | rad/s |
| Approach Time | - | ___ ± ___ | sec |

### Table 5: Alignment Accuracy (E.O. 4.4)

| Metric | Value | Unit |
|--------|-------|------|
| Mean Centering Error | ___ ± ___ | pixels |
| Mean Lateral Offset | ___ ± ___ | cm |
| Mean Angular Offset | ___ ± ___ | degrees |
| Alignment Success Rate | ___% | |

---

## Summary

| E.O. | Software Measurement | Physical Measurement Required |
|------|---------------------|------------------------------|
| 3.1 mAP | From training | ❌ No |
| 3.1 Inference | `/benchmark` endpoint | ❌ No (optional validation) |
| 4.1 Accuracy | `/expected_outputs` | ✅ Yes - verify correct location |
| 4.2 Distance | `/expected_outputs` | ✅ Yes - tape measure |
| 4.3 Speed | `/expected_outputs` | ✅ Yes - stopwatch |
| 4.4 Alignment | `/expected_outputs` | ✅ Yes - measure offset |

**Physical measurements are essential for E.O. 4.1-4.4 to validate software estimates!**

---

*Document Version: 1.0*
*Created: December 2025*
*For: Gazzard Gas Leak Detection System*
