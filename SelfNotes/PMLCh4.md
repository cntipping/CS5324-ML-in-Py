# PML CH4

Topic: ML, Reading Notes
Date: August 27, 2025

![](https://i.pinimg.com/originals/cb/25/d5/cb25d556c94dde722c37d5cad1c924a6.gif)

# Building Good Training Datasets - Data Preprocessing

---

# Dealing with Missing Data

- Remove the corresponding features (columns) or training examples (rows) from the dataset entirely

```python
>>> df.dropna(axis=0) # drops rows with NaNs

>>> df.dropna(axis=1) # drop columns with min 1 NaNs

# only drop rows where all columns are NaN
# (returns the whole array here since we don't
# have a row with all values NaN)
>>> df.dropna(how='all')

# drop rows that have fewer than 4 real values
>>> df.dropna(thresh=4)

# only drop rows where NaN appear in specific columns (here: 'C')
>>> df.dropna(subset=['C'])
```

- Imputing missing values
    - **mean imputation** is most common (replace val w/ mean of column)
        
        ```python
        >>> from sklearn.impute import SimpleImputer
        >>> import numpy as np
        >>> imr = SimpleImputer(missing_values=np.nan, strategy='mean')
        >>> imr = imr.fit(df.values)
        >>> imputed_data = imr.transform(df.values)
        >>> imputed_data
        ```
        
    - can also use `median` or `most_frequent`
    - use pandas `fillna`
        
        ```python
        >>> df.fillna(df.mean())
        ```
        
    
    ![Screenshot 2025-08-28 at 10.38.10.png](PML%20CH4%2025c626961bb480d6b028e0e65b504a40/Screenshot_2025-08-28_at_10.38.10.png)
    
    ![Screenshot 2025-08-28 at 10.39.50.png](PML%20CH4%2025c626961bb480d6b028e0e65b504a40/Screenshot_2025-08-28_at_10.39.50.png)
    

---

## Handling Categorical Data

- categorical data is split between **ordinal** and **nominal** features
    - ordinal → sorted or ordered
        - these must be mapped to define their order
    - nominal → unordered
        - use one hot encoding to prevent order bias

---

## Test/Train Data Split

- test/train split is important as you’re withholding info the algo could benefit from but having too little test data may provide a more inaccurate generalization error estimate
- Common splits:
    - 60:40
    - 70:30
    - 80:20
- For larger data sets 90:10 or 99:1 are acceptable (>100,000 training points)
- [Model evaluation, model selection, and algorithm selection in machine learning](https://arxiv.org/pdf/1811.12808.pdf)

---

# Feature Scaling

- **Decision trees** and **random forests** do not have to worry about feature scaling
- approaches to feature scaling:
    - **normalization**
        - special case of min-max scaling
        - $x_{norm}^{(i)} = \frac{{x^{i}} - {x_{min}}}{{x_{max}} - {x_{min}}}$
    - **standardization** (more practical)
        - $x_{std}^{i}=\frac{{x^{(i)}}-u_x}{o_x}$

---

# Selecting meaningful features

- model performs better on training dataset than test dataset → overfitting
- **overfitting:** model fits the parameters too closely with regard to the particular observations in the training dataset, but does not generalize well to new data (**high variance**)
    - model is too complex for given training data
- common solutions:
    - collect more training data
    - introduce penalty for complexity via regularization
    - choose a simpler model w fewer parameters
    - reduce dimensionality

## L1 and L2 regularization as penalties against model complexity (Sparse Solutions)

- **L2 regularization**
    - $L2: ||w||_2^2 = \sum{^m_{j=1}w^2_j}$
- **L1 regularization**
    - $L1: ||w||_1=\sum{^m_{j=1}|w_j|}$

## Sequential feature selection algos

- can avoid overfitting using **dimensionality reduction**
- two techniques:
    - **feature selection**
        - select subset of original features
        - greedy search algos
            
            [**sequential backward selection (SBS)**](PML%20CH4%2025c626961bb480d6b028e0e65b504a40/sequential%20backward%20selection%20(SBS)%20262626961bb480eb955fca48a47c0c0b.md)
            
    - **feature extraction**
        - derive info from feature set to construct a new feature subspace