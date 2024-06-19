#### <div align="center"> <h1> Video Game Sales Regression Project </h1> </div>

<p align="center">
  <img src="https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/ec4646d0-d4ca-44c0-aa2d-219d2e1b9963" alt="">
</p>

Welcome to VideoGameSalesRegressionProject repositary, In this project we aim to analyze and predict video game sales using regression models. The dataset includes various attributes related to video games, such as sales figures across different regions, genre, and publisher. The goal is to build a regression model that can accurately predict sales based on these attributes. 

### Overview
This project aims to analyze and predict video game sales using regression models. The dataset includes various attributes related to video games such as sales figures across different regions, genre, and publisher. The goal is to build a regression model that can accurately predict sales based on these attributes.

### Objectives

- Develop a robust regression model to predict video game sales.
- Use exploratory data analysis and data visualization to uncover insights.
- Improve understanding of factors that significantly impact video game sales.

### Key Findings and Insights

**No Duplicate Values:** The dataset was clean with no duplicate entries.

**Handling Missing Values:** Missing values were successfully filled using median and mode.

**Effective Feature Engineering:** Categorical and numerical data were separated and one-hot encoded.

**High Model Accuracy:** The regression model achieved a high accuracy with a Mean Absolute Percentage Error (MAPE) of less than 12%.

### Dataset

- Source: VideoGameSales.csv
- Size: 16,598 records and 11 variables.
- Variables: Includes Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales.

## Let's Get Started :

**First of all Import all the necessary modules in the Jupyter Notebook.**

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt          

import datetime as dt 

import scipy.stats as stats

import statsmodels.formula.api as smf

**Now Import the Dataset.**

VGSales = pd.read_csv('VideoGameSales.csv')

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/9cbe3368-a4d6-423d-b252-56cb992114d2)

VGSales is our DataFrame Name. There are 16598 records and 11 variables in our DataFrame

### Check if there are any Missing Values

VGSales.isna().sum()

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/67a55373-30b4-4b95-94da-e631221e6446)

There are missing values in our data. So we need to fill them.

VGSales['Year'] = VGSales.Year.fillna(VGSales['Year'].median())

VGSales['Publisher'] = VGSales.Publisher.fillna(VGSales['Publisher'].mode()[0])

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/887872fd-ad6c-47cd-84ad-8bb69fab8df8)

Now there are no missing values in our data and no duplicate values. So we can continue with further procedure.

### Separating the DataFrame into Two Different DataFrames : Num_Data and Cat_Data

Num_Data = VGSales.select_dtypes(['int','float'])

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/ad2ac582-4261-4d32-8fcc-3a240e7edce8)

There are 16598 records and 7 variables in our Num_Data

Cat_Data = VGSales.select_dtypes(['object'])

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/a96ab755-e3fb-4b4f-b92d-34709e57452c)

There are 16598 records and 4 variables in our Num_Data

**We need to assign an index to the data so that we can merge both datasets.**

Num_Data['index'] = Num_Data.index

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/b909018a-2168-4195-88e6-88d89ccdedc6)

Cat_Data['index'] = Cat_Data.index

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/be5c141e-60bb-4686-a501-caec2503b00e)

Now there are 16598 records, 8 variables in our Num_Data and 16598 records, 5 variables in our Cat_Data

## Data Visualization

- Checking the unique and value counts of Genre variable.

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/effbe9d2-ffbd-4cde-9c6e-b22bdd083530)

The dataset contains 12 unique values in the Genre variable with Action games being the most prevalent genre.

- Checking the unique and value counts of Platform variable.

![image](https://github.com/Swagath123Koyada/VideoGameSalesRegressionProject/assets/164196153/0c6f756c-d891-40f6-8714-53f0b9fe9487)

Out of the 31 unique values in the dataset for the Platform variable "Dual Screen (DS)" emerges as the most predominant platform, closely followed by "PS2".

### One-Hot Encoding

**Categorical data is converted into numerical data using one-hot encoding. This is essential for regression analysis which requires numerical input.**

Cat_Data = pd.get_dummies(Cat_Data,columns =['Genre','Platform'],dtype=int)

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/fc636c56-3aff-44bb-8658-491a1232e024)

Now, we have 46 columns in our Categorical DataSet.

**We have Categorical Data in our Dataframe but we only need numerical, so we need to remove these variables from our Dataframe.**

### Copying the Dataframe and drop the variables

Cat_Data_copy = Cat_Data.copy()

Copy the DataFrame so that the Original data will not get lost.

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/d82a23d2-d162-456d-9c4e-783d2c3552af)

### Now Merge both Num_Data and Cat_Data

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/33a4e37b-d72a-4710-9ccf-affa83bb7277)

After merging both the Num_Data and Cat_Data_copy datasets, we have assigned the name VGSales_1 to the merged dataset and the shape of the merged dataset is 16598 records and 51 variables.

## Feature Engineering

After merging the Datasets, we have done some Feature Engineering and Correlation analysis on our Data to understand which features are most strongly correlated with the target variable (Global_Sales).

corr = VGSales_1.corrwith(VGSales_1.Global_Sales).abs().sort_values(ascending = False)

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/aed39b4a-652d-4a20-9392-bfe01d95c827)

According to our data here in correlation analysis, we have taken the cut off upto 0.05.

This would mean that whatever the variables have been excluded that have weaker correlations with the target variable.

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/c9d1fd25-753c-4f55-8002-58038055a603)

These are the variables that have strong correlations with the target variable.

**NOTE :**

Selecting correlation coefficients between features is based on the dataset being analyzed. Different datasets may exhibit different correlation patterns, and the choice of which correlations to consider significant (e.g., greater than 0.05) depends on the specific characteristics and goals of the analysis.

### Splitting the final dataset intoTraining and Testing dataset


























