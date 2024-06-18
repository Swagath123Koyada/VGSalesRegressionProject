#### <div align="center"> <h1> VGSales Regression Project </h1> </div>

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

















































































