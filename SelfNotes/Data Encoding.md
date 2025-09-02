# Data Encoding

Topic: ML
Date: August 27, 2025

![](https://i.pinimg.com/originals/18/22/8d/18228d33fd29cea21515e098105fb721.gif)

# Table Data in Pandas

- **Columns** → referred to as attributes, variables, fields, characteristics, features
- **Rows** → points, samples, instances
- **Outcome** → target, class, label
- Table data storage is not memory efficient
    
    It depends on the backend (saving and accessing)
    

---

## Feature Vector Representation

### Discrete

### Continuous

- Nominal or categorical
    - stored as bool, one hot encoding (binary rep), hash func
    - e.g. employee id
    - no ordering
    - allowed transforms: permuting values
- Ordinal
    - allowed transforms: $Vnew = fmono (Vold)+b$
    - ordering matters
    - stored as int or bool
    - e.g. star ratings (1-5)

- Interval or Numeric
    - cont numerical val
    - could be specified range
    - e.g. BMI, temp
    - allowed transforms: $Vnew = fmono(Vold)+b$
    - stored as float
- Ratio or Numeric
    - zero is meaningful
    - e.g. sea level elevation
    - allowed transforms: $Vnew = fmono(Vold)$

<aside>
💡

bias → adding val to something

</aside>

<aside>
💡

monotonic → does not change order

</aside>

<aside>
💡

one hot prevents ordering

</aside>

[MachineLearningNotebooks/01_Numpy and Pandas Intro.ipynb at master · eclarson/MachineLearningNotebooks](https://github.com/eclarson/MachineLearningNotebooks/blob/master/01_Numpy%20and%20Pandas%20Intro.ipynb)

> Also assume data was entered wrong - Dr. Ip Lin
> 

## Handling Issues w Data Quality

- **Eliminate** instance or feature
- **Ignore**
- **Impute** (with what)

### Split-Impute-Combine

→ Used when most of the subgroup has valid data