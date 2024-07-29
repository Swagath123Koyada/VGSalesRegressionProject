#### <div align="center"> <h1> Video Game Sales Regression Project </h1> </div>

<p align="center">
  <img src="https://github.com/user-attachments/assets/dc2c35bc-f3b6-4c7e-bd41-3ea2a4187339" width="350"/>
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
- Variables Includes : Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales.

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

- It's crucial to ensure that the data is split randomly to prevent any biases in the training or testing datasets.

- The choice of test_size depends on the size of your dataset and the desired trade-off between training and testing data.

- After splitting, the training set (X_train and y_train) is used to train your machine learning model, and the testing set (X_test and y_test) is used to evaluate its performance.

**So to start with Training and Testing Data we need to import Some libraries and modules.**

import sklearn

from sklearn.model_selection import train_test_split

log_train,log_test = train_test_split(VG_1_copy,test_size = 0.3, random_state = 123)

By using this code, we can split a DataFrame into training and testing sets using the train_test_split function from the sklearn.model_selection module.

The shape of the train and test datasets are 11618 records, 12 variables and 4980 records, 12 variables respectively.

## Building the Regression model

To build a regression model, first of all import library and module.

from sklearn import datasets, linear_model

By using formula of Ordinary Least Squares (OLS) regression models using the statsmodels library.

formula = 'Global_Sales~' + '+'.join(train.columns.difference(['Global_Sales’]))

formula2 = 'Global_Sales~' + '+'.join(train.columns.difference(['Global_Sales', 'Platform_GB’]))

formula3 = 'Global_Sales~' + '+'.join(train.columns.difference(['Global_Sales', 'Platform_GB', 'Genre_Platform’]))

formula4 = 'Global_Sales~' + '+'.join(train.columns.difference(['Global_Sales', 'Platform_GB', 'Genre_Platform', 'Platform_NES’]))

formula5 = 'Global_Sales~' + '+'.join(train.columns.difference(['Global_Sales', 'Platform_GB', 'Genre_Platform', 'Platform_NES', 'Year’]))

formula6 = 'Global_Sales~' + '+'.join(train.columns.difference(['Global_Sales', 'Platform_GB', 'Genre_Platform', 'Platform_NES', 'Year', 'Genre_Adventure']))

Fitting the OLS Regression Model using the formula specified in formula6.

model = smf.ols(formula6, train).fit()

And then Printing the Model’s detailed Summary of the regression model by using.

print(model.summary())

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/07ba50b4-83fb-46fd-953f-8184503aafd0)

Making Predictions by using the fitted model to predict the target variable (Global_Sales) for the training data.

pred_Global_Sales = model.predict(train)

pred_Global_Sales

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/a237f55a-3d9c-43b9-9872-28bdef1db1d7)

train['Global_Sales']

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/a306b8a1-9529-4157-bd9b-01d447b96a4e)

## Model Results

### Comparison of actual vs. predicted Global_Sales

Result = pd.DataFrame(pd.concat([train['Global_Sales'],pred_Global_Sales],axis = 1))

Result = Result.rename(columns = {0:'Prediction'})

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/be11c973-1556-49f6-a03e-fd4af155f72a)

Result['Error'] = np.abs(Result['Global_Sales'] - Result['Prediction'])

Result = Result.rename(columns = {0:'Prediction’})

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/08eb9ab9-2870-45a7-bdb1-94788aa48b17)

Result['Percent_Error'] = (Result['Error']/Result['Global_Sales'])*100

Result = Result.rename(columns = {0:'Prediction’})

![image](https://github.com/Swagath123Koyada/VGSalesRegressionProject/assets/164196153/9afd4f36-1840-4ad4-b154-d0ec54e7c027)

MAPE = np.round(np.mean(Result['Percent_Error']),2)

**MAPE error is coming out to be 3.29 which is less than 10-12%. So we can say it is a very good regression model.**

## Conclusion

The project successfully developed a regression model to predict global video game saleswith a high level of accuracy, as evidenced by the low MAPE of 3.29%. This indicates that the model’s predictions are, on average, very close to the actual sales values.The insights gained from the correlation analysis and feature importance can helpstakeholders in making informed decisions to maximize video game sales.
