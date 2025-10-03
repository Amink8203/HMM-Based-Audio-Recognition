# HMM-Based Audio Recognition Project

## Overview

This project implements Hidden Markov Models (HMM) for audio recognition, specifically focusing on digit recognition (0-9) and speaker identification. The implementation includes both a library-based approach using `hmmlearn` and a custom implementation from scratch.

**Author:** Amin Aghakasiri  
**Student ID:** 810101381  
**Course:** Introduction to Artificial Intelligence - Computer Assignment 2

## Project Description

The goal of this project is to build HMM models that can:
1. **Digit Recognition**: Recognize spoken digits from 0 to 9
2. **Speaker Identification**: Identify speakers based on their voice characteristics

The project uses audio recordings from 6 different speakers, each repeating numbers 0-9 multiple times, creating a comprehensive dataset for training and testing the HMM models.

## Dataset

- **Speakers**: 6 speakers (george, jackson, lucas, nicolas, theo, yweweler)
- **Numbers**: Digits 0-9
- **Total Recordings**: 300 audio files (50 recordings per speaker)
- **Format**: WAV audio files
- **Naming Convention**: `{number}_{speaker}_{iteration}.wav`

## Features and Methodology

### Audio Feature Extraction

The project implements comprehensive audio feature extraction techniques:

#### 1. MFCC (Mel-Frequency Cepstral Coefficients)
- Primary feature used for HMM training
- 13 MFCC coefficients extracted per audio frame
- Normalized to 25 time frames for consistency

#### 2. Audio Preprocessing
- **Silence Removal**: Uses librosa's trim function with 38dB threshold
- **Normalization**: Features normalized between 0 and 1
- **Padding**: Arrays padded to uniform shape for model consistency

#### 3. Additional Features Analyzed
- **Zero Crossing Rate (ZCR)**: Time domain feature for frequency analysis
- **Mel-spectrograms**: Frequency domain representation
- **Chroma Features**: Pitch class distribution analysis

### Model Architecture

#### HMM Implementation Approaches

1. **Library-based Implementation** (`hmmlearn`)
   - Uses Gaussian HMM with optimized parameters
   - Efficient training with built-in algorithms
   - Better numerical stability

2. **Custom Implementation** (From Scratch)
   - Forward-Backward algorithm implementation
   - Expectation-Maximization (EM) training
   - Custom Gaussian emission probabilities

#### Model Parameters

- **Numbers Recognition**:
  - Hidden States: 6
  - Training Data: 80% (240 samples per digit)
  - Test Data: 20% (60 samples per digit)

- **Speaker Recognition**:
  - Hidden States: 10  
  - Training Data: 80% (400 samples per speaker)
  - Test Data: 20% (100 samples per speaker)

## Installation and Requirements

### Dependencies

```python
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import math
from hmmlearn import hmm
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import scipy
```

### Installation

```bash
pip install librosa hmmlearn numpy matplotlib seaborn scikit-learn scipy
```

## Usage

### 1. Data Preprocessing

```python
# Load and preprocess audio data
audio = []
numbers_dict_train = {"0": [], "1": [], ..., "9": []}
speakers_dict_train = {"george": [], "jackson": [], ..., "yweweler": []}

# Preprocess recordings
preprocess(audio, numbers_dict_train, speakers_dict_train, "./recordings/")

# Split train/test data
numbers_dict_train, numbers_dict_test = seperate_test_train(numbers_dict_train)
speakers_dict_train, speakers_dict_test = seperate_test_train(speakers_dict_train)
```

### 2. Model Training

#### Library-based Approach
```python
# Train number recognition models
hmm_models_numbers = make_models(numbers_dict_train, NUM_OF_HIDDEN_STATES_NUMBERS)

# Train speaker recognition models  
hmm_models_speakers = make_models(speakers_dict_train, NUM_OF_HIDDEN_STATES_SPEAKERS)
```

#### Custom Implementation
```python
# Custom HMM class
hmm_models_custom = {}
for key in numbers_dict_train:
    model = HMM(num_hidden_states=NUM_OF_HIDDEN_STATES_NUMBERS)
    model.train(numbers_dict_train[key], num_iterations=10)
    hmm_models_custom[key] = model
```

### 3. Model Evaluation

```python
# Make predictions
true_values, predicts = predict_model(numbers_dict_test, hmm_models_numbers)

# Generate confusion matrix
confusion_matrix = produce_confusion_matrix(true_values, predicts)

# Calculate metrics
calc_measurements(confusion_matrix, labels)
```

## Results and Performance

### Number Recognition Results

#### Library Implementation (hmmlearn)
- **Total Accuracy**: 88.166%
- **Macro-Average Precision**: 88.502%
- **Performance**: Excellent consistency between metrics

#### Custom Implementation (From Scratch)
- **Total Accuracy**: 70.16%
- **Macro-Average Precision**: 74.37%
- **Performance**: Good but lower than library implementation

### Speaker Recognition Results

#### Library Implementation (hmmlearn)
- **Total Accuracy**: 97.5%
- **Macro-Average Precision**: 97.5%
- **Performance**: Excellent recognition accuracy

#### Custom Implementation (From Scratch)
- **Total Accuracy**: 85.66%
- **Macro-Average Precision**: 88.22%
- **Performance**: Very good speaker identification

### Key Observations

1. **Speaker Recognition vs. Digit Recognition**: Speaker recognition consistently outperforms digit recognition due to:
   - More training data per speaker (450 vs 240 samples)
   - Fewer classes to distinguish (6 vs 10)

2. **Library vs. Custom Implementation**: Library implementation shows superior performance due to:
   - Optimized hyperparameter tuning
   - Better numerical stability
   - Advanced algorithmic variations

## Evaluation Metrics

The project implements comprehensive evaluation metrics for multi-class classification:

### Core Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: Quality of positive predictions per class
- **Recall**: Ability to find all positive instances per class  
- **F1-Score**: Harmonic mean of precision and recall

### Multi-class Adaptations
- **Macro-averaging**: Equal weight to each class
- **Micro-averaging**: Equal weight to each instance
- **Per-class metrics**: Individual performance analysis

## Project Structure

```
├── CA2.ipynb                 # Main Jupyter notebook with implementation
├── recordings/               # Directory containing WAV audio files
│   ├── 0_george_0.wav
│   ├── 0_jackson_0.wav
│   └── ...
├── Description/
│   └── AI-A2.pdf            # Project description document
└── README.md                # This file
```

## Technical Implementation Details

### Audio Processing Pipeline
1. **Loading**: Audio files loaded using librosa
2. **Trimming**: Silence removal with 38dB threshold
3. **Feature Extraction**: 13 MFCC coefficients
4. **Normalization**: Min-max scaling to [0,1]
5. **Padding**: Uniform 25-frame sequences

### HMM Training Process
1. **Initialization**: Random parameter initialization
2. **EM Algorithm**: Iterative parameter optimization
3. **Forward-Backward**: Probability computation
4. **Convergence**: Training until parameter stability

### Prediction Methodology
1. **Likelihood Computation**: Score calculation for each model
2. **Maximum Likelihood**: Best matching model selection
3. **Classification**: Assignment to highest-scoring class
