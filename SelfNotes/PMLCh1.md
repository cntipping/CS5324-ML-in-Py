# PML CH1

Topic: ML, Reading Notes
Date: August 25, 2025

![](https://i.pinimg.com/originals/26/77/87/2677876fdeffafde5e89c5c1e92c2edf.gif)

# Giving Computers the Ability to Learn from Data

---

<aside>
💡

Machine Learning is a **subset** of artificial intelligence (AI) 

</aside>

# The Three Types of Machine Learning

## Supervised Learning

→ Labeled data

→ Direct Feedback

→ Predict Outcome/Future

![Screenshot 2025-08-26 at 11.18.47.png](PML%20CH1%2025b626961bb4809a92ded704886e209b/Screenshot_2025-08-26_at_11.18.47.png)

- **classification** and **regression** are subcategories of supervised learning

### Classification

<aside>
💡

**Goal:** predict the categorical class labels of new instances, based on past observations

</aside>

- class labels are discrete, unordered values
- e.g. spam emails (binary)
- use the ML algo to learn the classification rule
- binary vs multi-class classification
- e.g. handwritten character recognition

### Regression

<aside>
💡

**Goal:** given a number of predictor (**explanatory)** variables and a continuous response (**outcome**), find the relationship between the two

</aside>

- predictor variables are called ‘features’
- response variables are called ‘target variables’

## Unsupervised Learning

→ No labels

→ No feedback

→ Find hidden structure in data

- using unlabeled data or data of unknown structure

### Clustering

- allows organization of information in subgroups ( **clusters** )
- also referred to as **unsupervised classification**

### Dimensionality Reduction

## Reinforcement Learning

→ Decision process

→ Reward System

→ Learn series of actions

![Screenshot 2025-08-26 at 12.24.40.png](PML%20CH1%2025b626961bb4809a92ded704886e209b/Screenshot_2025-08-26_at_12.24.40.png)

<aside>
💡

**Goal:** develop a system (**agent**) that improves its performance based on interactions with tne environment

</aside>

- includes a **reward signal**
- think of as a field related to supervised learning
- success is measured against reward function
- e.g. chess game where board state is the environment and win/lose is the reward function

---

## Notation

![Screenshot 2025-08-26 at 12.35.41.png](PML%20CH1%2025b626961bb4809a92ded704886e209b/Screenshot_2025-08-26_at_12.35.41.png)

![Screenshot 2025-08-26 at 12.36.12.png](PML%20CH1%2025b626961bb4809a92ded704886e209b/Screenshot_2025-08-26_at_12.36.12.png)

---

# Roadmap for Building ML Systems

![Screenshot 2025-08-26 at 12.52.06.png](PML%20CH1%2025b626961bb4809a92ded704886e209b/Screenshot_2025-08-26_at_12.52.06.png)

- reducing dimensionality → less storage space required making learning algo faster
- dimensionality reduction can improve predictive performance of a model as the dataset contains a large number of irrelevant features (or noise)  (if dataset has low signal-to-noise ratio)
- randomly divide dataset into test and train data
- “it is essential to compare at least a handful of different algorithms in order to train and select the best performing model”
    - must decide upon the metric to measure performance
    - accuracy is a common metric
- ask *how do we know which model performs well on the final test dataset and real-world data is we don’t use this test dataset for the model selection, but keep it for the final model evaluation?*

> **cross-validation :**  further divide dataset into training and validation subsets in order to estimate the generalization performance of each model
> 
- frequently use hyperparameter optimization techniques to fine-tune performance of model
- after selecting a model, evaluate the model against unseen data to estimate the ***generalization error***
