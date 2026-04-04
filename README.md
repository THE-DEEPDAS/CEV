# Hackzion Hackathon -- Segmentation Documentation

## Duality AI's Offroad Semantic Scene Segmentation

------------------------------------------------------------------------

## 1. Overview

Duality AI presents the Offroad Autonomy Segmentation Challenge, where
participants train a semantic segmentation model using synthetic desert
environment data and test it on unseen environments.

The dataset is generated using Falcon, Duality AI's digital twin
simulation platform.

### Objectives

-   Train a robust semantic segmentation model\
-   Evaluate performance on unseen environments\
-   Optimize for accuracy, generalization, and efficiency

------------------------------------------------------------------------

## Importance of Digital Twins

-   Enables scalable and cost-effective data generation\
-   Allows controlled variations (weather, time, terrain)\
-   Improves robustness of models for real-world deployment

------------------------------------------------------------------------

## Hackathon Structure

### Dataset Details

-   Generated using FalconEditor\
-   Contains desert environment images

### Classes

  ID      Class Name
  ------- ----------------
  100     Trees
  200     Lush Bushes
  300     Dry Grass
  500     Dry Bushes
  550     Ground Clutter
  600     Flowers
  700     Logs
  800     Rocks
  7100    Landscape
  10000   Sky

------------------------------------------------------------------------

## i. Hackathon Tasks

### 1. AI Engineering

-   Train and fine-tune model\
-   Evaluate performance\
-   Optimize accuracy and inference time

### 2. Documentation & Presentation

-   Document workflow\
-   Show loss graphs and metrics\
-   Prepare final report and presentation

------------------------------------------------------------------------

## ii. Key Deliverables

### 1. Trained Model

-   Model weights\
-   Scripts\
-   Config files

### 2. Performance Report

-   IoU score\
-   Loss graphs\
-   Failure case analysis

------------------------------------------------------------------------

## iii. Sample Judging Criteria

### Model Performance

-   IoU Score → 80 points

### Report Quality

-   Documentation clarity → 20 points

------------------------------------------------------------------------

## iv. Important Links

-   Falcon Signup: https://falcon.duality.ai/auth/sign-up\
-   Dataset Download:
    https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert\
-   Discord: https://discord.com/invite/dualityfalconcommunity

------------------------------------------------------------------------

# 2. Task Instructions

## i. AI Engineering

### 1. Create Falcon Account

-   Sign up and log in

### 2. Download Dataset

Includes: - RGB + segmented images\
- Train/Val folders\
- Test images\
- Training scripts\
- Visualization tools

### 3. Setup Environment

-   Install Anaconda/Miniconda\
-   Run `setup_env.bat` (Windows)\
-   Use `.sh` for Mac/Linux

### 4. Training Workflow

-   `train.py` → trains model\
-   `test.py` → evaluates unseen data

### 5. Train Model

``` bash
conda activate EDU
python train.py
```

### 6. Benchmark Results

-   Predictions\
-   Loss metrics\
-   IoU score

------------------------------------------------------------------------

## ii. Documentation & Presentation

### Guidelines

-   Keep it clear and structured\
-   Use visuals and graphs\
-   Avoid overly complex language

### Report Structure (Max 8 pages)

1.  Title\
2.  Methodology\
    3-4. Results & Metrics\
    5-6. Challenges & Solutions\
3.  Conclusion & Future Work

------------------------------------------------------------------------

# 3. Submission Instructions

## Required Files

### 1. Final Folder

-   Training & testing scripts\
-   Config files

### 2. Report

-   Methodology\
-   Challenges\
-   Optimizations\
-   Performance

### 3. README

-   Steps to run project\
-   Dependencies\
-   Expected outputs

------------------------------------------------------------------------

## Rules

-   Use only provided dataset\
-   Do NOT train on test data

------------------------------------------------------------------------

## After Submission

-   Share results\
-   Get feedback\
-   Explore advanced topics:
    -   Self-supervised learning\
    -   Domain adaptation\
    -   Multi-view detection

------------------------------------------------------------------------

# 4. Common Issues

### Setup Script

-   `.bat` → Windows only\
-   Use `.sh` for Mac/Linux

### Slow Training Fixes

-   Reduce batch size\
-   Close background apps\
-   Monitor GPU usage

### Data Sharing

-   Use Google Drive / Dropbox / Git

------------------------------------------------------------------------

# 5. Glossary

  Term                    Definition
  ----------------------- -------------------------------------------
  Digital Twin            Virtual replica of real-world system
  Semantic Segmentation   Pixel-wise classification
  IoU                     Overlap between prediction & ground truth
  NMS                     Removes duplicate detections
  Class Imbalance         Uneven data distribution
  Ground Truth            Correct labeled data

------------------------------------------------------------------------

## Benchmarks

-   Training Loss → Should decrease\
-   Inference Speed → \< 50 ms/image

------------------------------------------------------------------------

## Final Note

Share your work on LinkedIn and tag Duality AI to showcase your project.
