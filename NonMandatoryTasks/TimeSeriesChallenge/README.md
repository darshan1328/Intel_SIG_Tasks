# The Time Series Challenge

## Overview
Your task is to explore how **foundation models for time series forecasting** perform in **zero-shot** and **few-shot** settings — and compare them against **classical statistical models** like ARIMA or Exponential Smoothing.  
The goal: understand how large pre-trained models generalize to unseen data with little or no training, and evaluate whether traditional models can still compete!



## Important Guidelines
- You may use any **pre-trained time series foundation models**, such as:
  - [Chronos](https://huggingface.co/papers/2403.07815) (zero-shot only)
  - IBMs Granite and TinyTimeMixer models
  - [Moirai](https://huggingface.co/Salesforce/moirai-1.0-R-large) (fine-tuning supported)
  - [TimeGPT](https://www.nixtla.io/docs) (zero-shot only)
- For classical baselines, try any of these (you are not limited to these):
  - ARIMA / SARIMA  
  - Holt-Winters (Exponential Smoothing)  
  - Prophet *(optional)*  
- Use any **open dataset**, from:
  - [Monash Time Series Forecasting Repository](https://forecastingdata.org/)  
  - M4 / M5 competition subsets  
- You are also free to use python libraries and other packages which contain python APIs to these models. 
- If you are feeling super enthusiastic, you are free to even create your own pipeline and implement the code for these models from the source code repositories. This will definitely carry bonus points!

**NOTE: You are not limited by any of the above mentioned models and datasets as long as it fits the guidelines.**

Explain your approach clearly, and include references to any papers or repos you use.  
Creativity, depth of analysis, and clear visualization will be rewarded.


## Requirements

### **Input**
- A time series dataset (univariate or multivariate).  
- Optionally, a small subset of data for few-shot adaptation.  

### **Output**
- **Forecasted values** for the test set.  
- **Evaluation metrics** (MAE, RMSE, MAPE, etc.) for each model.  
- **Comparison report** covering:
  - Zero-shot vs. few-shot performance  
  - Foundation vs. statistical models  
  - Key observations and insights  

## Experiments

### **Exp 1: Zero-Shot Forecasting**
Use at least two foundation models (e.g., Chronos, Moirai) *without fine-tuning* to forecast on your chosen dataset.  
Compare their zero-shot performance on standard metrics



### **Exp 2: Classical Baseline Comparison**
Implement at least two classical forecasting methods (ARIMA, Exponential Smoothing, Prophet).  
Compare their results and discuss where foundation models excel or fail.



## Bonus Experiments

### **Exp 3: Few-Shot Forecasting**
Choose one of the following approaches:

**Option A — True Few-Shot Fine-Tuning:**  
Fine-tune an open model like **Moirai** or **PatchTST** on a *small subset* (≈5–10%) of the data and evaluate improvements.  

**Option B — Simulated Few-Shot Adaptation:**  
For models like **Chronos** or **TimeGPT**, vary the *context window* or training history length (5–10% of total data) to simulate low-data adaptation.  
Compare results to the full-data zero-shot baseline.

### **(Optional) Exp 4: Visualization & Insights**
Visualize forecasts, residuals, and attention patterns (if available).  
Discuss what “generalization” means for time series models.


## 🎯 Submission
Submit:
- A clean **Colab / Jupyter notebook or GitHub repo** with your code and results.  
- A **short report (2–3 pages)** summarizing:
  - Approach & dataset  
  - Experimental results  
  - Insights and takeaways  



**Good luck!**