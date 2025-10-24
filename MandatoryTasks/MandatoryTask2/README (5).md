# Regression Task: Predicting y from w and x

## Overview

This project tackles a regression problem involving three variables: w, x, and y. The objective is to develop a predictive model that accurately estimates y based on the input features w and x. The dataset, while structurally simple, exhibits complex non-linear patterns that require careful feature engineering and model selection and it contains a huge dataset (100M rows).

**NOTE: The model (NN here), that I have trained does not much coincide with the actual graph, due to constraint in computation power, and time I was not able to improve it much. But I will in future will come again to this project and imrove as much as possivle.**

## Dataset Description

The dataset consists of three columns:
- **w**: First input feature
- **x**: Second input feature  
- **y**: Target variable to be predicted

Initial exploratory data analysis revealed a non-trivial relationship between the predictors and the target. The data distribution suggested that simple linear models would be insufficient for capturing the underlying patterns.

## Methodology

### 1. Data Exploration

The initial phase involved loading and examining the dataset structure. A sample of 30,000 points was visualized to understand the relationship between x and y. The scatter plot revealed a complex, non-linear pattern with what appeared to be periodic components and significant variance across different regions of the feature space.



### 2. Data Preprocessing

Due to the large dataset size (100 million rows), I employed a chunk-based processing strategy:

- Computed scaling parameters (mean and standard deviation) from a 100,000-row subset
- Saved StandardScaler objects for both features and target variable
- Applied standardization to ensure numerical stability during training

### 3. Model Architecture

A neural network was constructed with the following specifications:

**Architecture:**
- Input layer: 16 dimensions (after Fourier feature transformation)
- Hidden layer 1: 512 neurons with ReLU activation
- Hidden layer 2: 512 neurons with ReLU activation
- Hidden layer 3: 256 neurons with ReLU activation
- Hidden layer 4: 128 neurons with ReLU activation
- Output layer: 1 neuron (regression output)

**Training Configuration:**
- Optimizer: Adam with learning rate of 0.0001
- Loss function: Mean Squared Error
- Batch size: 8192
- Training approach: 15 epochs per chunk, processing 50 chunks total

**Model Parameters:**
- Total parameters: 435,713
- Trainable parameters: 435,713

### 4. Training Process

The model was trained incrementally on chunks of data to handle the large dataset size efficiently. This approach allowed for:
- Manageable memory footprint
- Continuous learning across the entire dataset
- Regular model updates without requiring full dataset in memory


## Key Observations

### Pattern Recognition

The complexity of the relationship between w, x, and y became apparent through the analysis:

- The target variable y exhibits periodic behavior that correlates with the input features
- The variance in y changes across different regions of the input space
- Simple polynomial relationships are insufficient to capture the full complexity

### Feature Importance

Both w and x contribute meaningfully to the prediction of y, though their interaction appears non-additive. The periodic nature of the relationship suggests that the underlying process may involve trigonometric or wave-like functions.




## Results Interpretation

<img width="1152" height="667" alt="image" src="https://github.com/user-attachments/assets/d023680c-20b4-4abe-88b1-3e1fde677cb6" />


The residual analysis shows that errors are generally unbiased and follow an approximately normal distribution, suggesting that the model has successfully captured the systematic component of the relationship without significant underfitting or overfitting.

## Challenges I Faced in this task

1.  **Dataset Size**: Processing 100 million rows required careful memory management and incremental training approaches.

2. **Non-linearity**: Initial attempts with simpler models revealed the need for more sophisticated feature engineering.

3. **Computational Resources**: Training deep networks on large datasets required optimization of batch sizes and training schedules.

## Usage

1. Place your dataset in the appropriate directory
2. Run the training script to process data and train the model
3. Use the trained model for predictions on new data
4. Preprocessing objects (StandardScalers) are saved for consistent transformation


## Future improvements could explore:
- Additional feature engineering approaches
- Ensemble methods combining multiple model architectures
- Hyperparameter optimization for the Fourier feature transformation
- Alternative neural network architectures such as residual networks

