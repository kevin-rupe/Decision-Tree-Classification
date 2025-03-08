# Decision-Tree-Classification
Project completed in pursuit of Master's of Science in Data Analytics.

## PART I: RESEARCH QUESTION

### PROPOSAL OF QUESTION 

Is it possible to predict if a patient will be readmitted to the hospital given certain key contributing factors using decision tree classification method?

### DEFINED GOAL

The primary goal of this analysis is to develop a machine learning algorithm model using decision tree classification to help the medical facility identify patients who are more at risk for readmission. 

## PART II: METHOD JUSTIFICATION

### EXPLANATION OF PREDICTION METHOD

For this analysis, I chose to use decision tree classification method. Decision Tree classification is a method that can predict both categorical data and numerical data. The algorithm uses nodes and branches to make decisions. For example, does the patient suffer from chronic back pain? Two branches would branch out of this node, following either the “yes” branch or the “no” branch. This method can be built to a specific depth of nodes, where the final node is often called the leaf node. (Thorn, 2020)

There are several methods for how to best train the model to fit the data. We can use hyperparameter tuning to find the best parameters, and then use these to learn how accurately our model will predict outcomes. More of how these steps are done will follow in later sections of this document. 

The expected outcome of this analysis is that we will know how accurately this model is at predicting if a patient will be readmitted to the hospital. We will also know which variables in our dataset are used in the decision tree to determine the outcome. I also expect that this model will perform well and will give us high accuracy, therefore, enabling us to offer valuable insight to the hospital regarding patient readmission. 

### SUMMARY OF METHOD ASSUMPTION

One assumption of Decision Tree classification is that each node can be split into binary branches (Saini, 2024). This could be a false assumption in that not every decision can be broken down into just two choices. Many decisions could be split into many more branches than just two. For example, in our dataset, a node could be ‘Complication Risk’ and the branches would be low, medium, and high. 

### PACKAGES OR LIBRARIES LIST

![IMG_1616](https://github.com/user-attachments/assets/d568dab1-997d-48f4-90e5-350ec4460bf3)

## PART III: DATA PREPARATION

### DATA PREPROCESSING
One important preprocessing goal is to convert the categorical variables to numeric values using dummy variables. In order to use categorical variables you have to convert, for example, responses such as ‘Yes’ and ‘No’ to 1 or 0. 

In regression, we need to use k-1 dummies for each variable to counter against multicollinearity, but in machine learning algorithms such as Decision Tree Classifier, we need to keep all dummies because we don’t need to assume that there is a linear relationship between our target and feature variables. (Shmueli, 2015) 

### DATA SET VARIABLES 

![IMG_1617](https://github.com/user-attachments/assets/9c78673e-3bbf-412b-b151-5c54b590204f)

### STEPS FOR ANALYSIS

First, I need to import the Pandas library and load the dataset into Jupyter Notebook (Pandas, 2023).
```python
#import CSV file
import pandas as pd
med_df = pd.read_csv("C:/Users/e0145653/Documents/WGU/D209 - Data Mining/medical_clean.csv")
```

Next, I will view the data to get a sense of what variables we have in our dataset by viewing the dataframe’s first few rows. There appear to be some variables here that I will not need for this analysis; that are useless based on my research question. 
```python
#view dataset
med_df.head()
```
![IMG_1618](https://github.com/user-attachments/assets/a574b756-69b0-4ee5-b56f-688315d14d7a)

After evaluating all of the variables, I decide on dropping the variables that I feel are not significant to my research question. 
```python
## drop unused columns
med_df = med_df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State',
                      'County', 'Zip', 'Lat', 'Lng', 'Population', 'Area', 'TimeZone', 
                      'Job', 'Children', 'Age', 'Income', 'Marital', 'Gender', 'VitD_levels', 
                      'Doc_visits', 'Full_meals_eaten', 'vitD_supp', 'Soft_drink', 
                      'TotalCharge', 'Additional_charges', 'Item1', 'Item2', 'Item3', 
                      'Item4','Item5', 'Item6', 'Item7', 'Item8'], axis=1)
```

Next, I will check for duplicated rows and for missing values in the dataset. I find none, so there is no need to clean the data for these issues. 
```python
#check for duplicated/missing values
print(med_df.duplicated().value_counts())
print("")
print('Variables        Missing Values')
print('---------        --------------')
print(med_df.isna().sum())
```
![IMG_1619](https://github.com/user-attachments/assets/553630f1-39d8-4160-8de8-9dd2dc4a811b)


Next, I need to verify if there are any outliers in the variable Initial_days since it’s a continuous numeric variable. The best way to check for outliers is to use a boxplot which will display outliers on the outside of each of the whiskers. (Waskom, 2012-2022) 
```python
#check for outliers
import seaborn as sns
sns.boxplot(med_df.Initial_days, orient='h').set(title='Initial_days')
```
![IMG_1620](https://github.com/user-attachments/assets/3b7e2d18-c7ef-484b-8b25-f384d8177aaa)

Before performing any machine learning, we need to convert all of our categorical variables to dummy variables. For all of the variables with “Yes” or “No” values, I used one-hot encoding method of replacing “Yes” with 1, and “No” with 0. 

```python
#One-Hot Encoding for (Yes/No) variables
prefix_list1 = ['ReAdmis','HighBlood', 'Stroke', 'Arthritis', 
               'Diabetes', 'Anxiety', 'Asthma',
               'Overweight', 'Hyperlipidemia', 'BackPain',
               'Allergic_rhinitis', 'Reflux_esophagitis']

prefix_dict = {'Yes': 1, 'No': 0}

for col in prefix_list1:
    med_df[col] = med_df[col].replace(prefix_dict)
```

For the other variables with more categorical values, I used Pandas get_dummies function to convert these to 1’s and 0’s. To make these variable names shorter, I renamed each one in the step of get_dummies, and then renamed them one time further to remove white spaces. 
```python
#Get dummies for variables
ia = pd.get_dummies(med_df['Initial_admin'], prefix='IA', prefix_sep='_', drop_first=False)
cr = pd.get_dummies(med_df['Complication_risk'], prefix='CompRisk', prefix_sep='_', drop_first=False)
svc = pd.get_dummies(med_df['Services'], prefix='Svc', prefix_sep='_', drop_first=False)
med_df = pd.concat([med_df, ia, cr, svc], axis=1)
med_df = med_df.drop(['Initial_admin', 'Complication_risk', 'Services'], axis=1)

#rename columns
med_df.rename(columns = {'IA_Observation Admission': 'IA_Observation'}, inplace=True)
med_df.rename(columns = {'IA_Emergency Admission': 'IA_Emergency'}, inplace=True)
med_df.rename(columns = {'IA_Elective Admission': 'IA_Elective'}, inplace=True)
med_df.rename(columns = {'Svc_Intravenous': 'Svc_IV'}, inplace=True)
med_df.rename(columns = {'Svc_CT Scan': 'Svc_CT'}, inplace=True)
med_df.rename(columns = {'Svc_Blood Work': 'Svc_BW'}, inplace=True)
```

I would like to continue to evaluate the variables to determine which ones are best suited to my target variable. For this step, I am going to use a feature selection method called SelectKBest from scikit-learn which will evaluate all the variables and can provide me with the best variables based on significant p-values (i.e., < 0.05). 
```python
X = med_df.drop(['ReAdmis', 'Initial_days'],1)
y = med_df['ReAdmis']

from sklearn.feature_selection import SelectKBest, f_classif
skbest = SelectKBest(score_func=f_classif, k='all')

X_new = skbest.fit_transform(X,y)

p_values = pd.DataFrame({'Feature': X.columns,
                        'p_value': skbest.pvalues_}).sort_values('p_value')
#p_values[p_values['p_value']<0.5]

features_to_keep = p_values['Feature'][p_values['p_value']<0.5]
print(features_to_keep)
```
![Image 3-8-25 at 11 00 AM](https://github.com/user-attachments/assets/757cb580-b0cc-4d80-aab1-1ddc46eb46b4)


## PART IV: ANALYSIS

### SPLITTING THE DATA
```python
#split the data into train/test sets
from sklearn.model_selection import train_test_split

X = med_df.drop(['ReAdmis'], axis=1)
y = med_df['ReAdmis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
```

### OUTPUT AND INTERMEDIATE CALCULATIONS

The technique I am using for this project is Decision Tree Classification. This algorithm is a non-parametric supervised learning method that breaks nodes (i.e., decisions) into different paths (called branches) based on certain criteria. At the end of a branch is a leaf node which represents a classification or decision. The goal is to identify whether the dependent variable can be predicted given various significant factors by allowing the algorithm to decide which branch is taken. 

![IMG_1624](https://github.com/user-attachments/assets/c15a74fa-922d-49a4-b5ff-f34cb8d9b0ff)

Before starting this classification, I needed to convert all of my categorical variables to dummy variables so that they would be numeric. I then split the data into two sets: training and testing, which I split 80%/20% respectively. I then used scikit-learn’s SelectKBest function to find the variables which are significant based on their p-values. 

![Image 3-8-25 at 11 00 AM](https://github.com/user-attachments/assets/9b46feb7-e119-41bd-af76-98ac93a3fa3e)

To run the best model possible, I used hyperparameter tuning via scikit-learn’s GridSearchCV function to find the best parameters. I validated this function using max_depth, criterion, min_samples_leaf, and min_samples_split parameters. I then instantiated the Decision Tree Classifier, and then ran GridSearchCV. I then fit this to the training dataset in order to find the best parameters found, which are shown below along with the best score from this model. 

```python
#Grid Search to find best parameters
from sklearn.model_selection import GridSearchCV
import numpy as np

params_dt = {
    'max_depth': np.arange(2,10),
    'criterion': ('gini', 'entropy', 'log_loss'),
    'min_samples_leaf': np.arange(0.1, 0.24, 0.01),
    'min_samples_split': np.arange(2,32,2)
}

dt = DecisionTreeClassifier(random_state=24)
grid_dt = GridSearchCV(estimator=dt, 
                       param_grid=params_dt, 
                       cv=10,
                       scoring='accuracy',
                       n_jobs=-1)
grid_dt.fit(X_train, y_train)

print('Best Parameters: {}'.format(grid_dt.best_params_))
print('Best Score: {:.4f}'.format(grid_dt.best_score_))
```
> Best Parameters: {'criterion': 'gini', 'max_depth': 2, 'min_samples_leaf': 0.1, 'min_samples_split': 2}
>
> Best Score: 0.9785

Without hyperparameter tuning, the initial model shows to be 96.95% accurate. After running GridSearchCV’s best parameters into the model, the accuracy improves to 97.6%. The mean squared error (MSE), root mean squared error (RMSE), r-squared, and mean absolute error (MAE) all improved as well, as shown below. 

```python
#Initial DecisionTree classification
dt = DecisionTreeClassifier(random_state=24)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

#Initial model accuracy 
initial_accuracy = accuracy_score(y_test, y_pred)
print("The initial model accuracy is  ", np.round((initial_accuracy * 100),2), "%")

#Initial MSE, RMSE, R^2, MAE
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
r2_dt = R2(y_test, y_pred)
mae_dt = MAE(y_test, y_pred)
print("The initial model MSE is       ", np.round((mse_dt),4))
print("The initial model RMSE is      ", np.round((rmse_dt),4))
print("The initial model R-Squared is ", np.round((r2_dt),4))
print("The initial model MAE is       ", np.round((mae_dt),4))
```
> The initial model accuracy is 96.95 %
>
> The initial model MSE is 0.0305
>
> The initial model RMSE is 0.1746
>
> The initial model R-Squared is 0.8687
>
> The initial model MAE is 0.0305

```python
## Decision Tree model with best parameters

#dtc model
dt = DecisionTreeClassifier(max_depth=2,
                            criterion='gini',
                            min_samples_leaf=0.1,
                            min_samples_split=2,
                            random_state=24)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

#accuracy_score
best_accuracy = accuracy_score(y_test, y_pred)
print("The best params model accuracy is  ", np.round((best_accuracy * 100),2), "%")

#MSE, RMSE, R^2, MAE scores
mse_dt = MSE(y_test, y_pred)
rmse_dt = mse_dt**(1/2)
r2_dt = R2(y_test, y_pred)
mae_dt = MAE(y_test, y_pred)
print("The best params model MSE is       ", np.round((mse_dt),4))
print("The best params model RMSE is      ", np.round((rmse_dt),4))
print("The best params model R-Squared is ", np.round((r2_dt),4))
print("The best params model MAE is       ", np.round((mae_dt),4))
```
> The best params model accuracy is 97.6 %
>
> The best params model MSE is 0.024
>
> The best params model RMSE is 0.1549
>
> The best params model R-Squared is 0.8967
>
> The best params model MAE is 0.024 

The main feature that is the most predictive according to my Decision Tree Classifier is the Initial_days variable, which provides more than 90% of the predictive power. 
```python
# Feature Importance

#create a series
importances_dt = pd.Series(dt.feature_importances_, index=X_train.columns)

#sort the values
sorted_importances_dt = importances_dt.sort_values()

#plot the sorted values
sorted_importances_dt.plot(kind='barh',
                          color='lightblue')
plt.show()
```
![IMG_1625](https://github.com/user-attachments/assets/538e343d-7c77-430f-bd38-0e8bd9acef2d)


## PART V: DATA SUMMARY AND IMPLICATIONS

### ACCURACY AND MSE

![IMG_1626](https://github.com/user-attachments/assets/a5c71800-c56d-4725-8396-9f8849ab432c)

The mean squared error (MSE) is the average difference squared between the predicted values and actual values (Stewart, n.d.). This gives a good measurement for determining how well the model is at predicting. The root mean squared error (RMSE) is basically just the square root of the MSE. It is used in the same way of providing a measurement for model effectiveness. The R-squared metric represents the proportion of variance between predicted and actual residuals, on a scale of 0 to 1, where 1 is a perfect model (Fernando, 2023). 

The table above also shows the same metrics from both the training and testing sets which shows nearly congruent metrics. This proves a very well fit model that is good at predicting. 

### RESULTS AND IMPLICATIONS

For my final model, the accuracy is 97.6%. The MSE is 0.024 which shows that there is very negligible error between predictions and actual values from the training set. Likewise, the RMSE is 0.1549 which is very low. The R-Squared value is 0.8967 which means that this model is fit nearly 90%. Using SelectKBest, I learned that medical conditions of asthma, chronic back pain, arthritis, obesity all played a significant role in patient readmission. Services such as CT Scans, MRI’s, and IV’s also played a role, as did how the patient was admitted to the hospital. The largest factor though was how long the patient was initially admitted to the hospital. I then used hyperparameter tuning to find the optimal values for max_depth, min_samples_leaf, min_samples_split, and criterion. Based on GridSearchCV the best model provided the values for max_depth = 2, criterion = gini, min_samples_leaf = 0.1, and min_samples_split = 2. The model accuracy could be further improved by adding more parameters in GridSearchCV and then retraining the model using these tuned parameters to predict the data. 

### LIMITATION
  	
Decision Tree Classification is unstable due to changes in data over time. For this reason, models cannot be used over long periods of time, so they are not sustainable and therefore, would not be advisable to use over “other tree-based algorithms such as random forest or various boosting algorithms.” (Kapil, 2022)

### COURSE OF ACTION

The recommended course of action for the medical facility would be to use the medical factors from this model to determine whether a patient will be readmitted or not. The medical staff should be consulted for ways the patients could be treated more effectively for these medical conditions, services they received, and even the means by which the patient was initially admitted to the hospital. Given more precise care for these certain factors, the hospital could certainly lower their patient readmission rate. This would be a positive move not only for the overall well-being of their patients, but also improve the company’s financial position. I would also advise the decision makers using this model to only use this model for no longer than 12 months given that decision tree classification is unstable. 

## PART VI: SUPPORTING DOCUMENATION

### SOURCES FOR THIRD-PARTY CODE

Pandas (2023, June 28). Retrieved September 27, 2023, from https://pandas.pydata.org/docs/reference/index.html.

Waskom, M. (2012-2022). Seaborn Statistical Data Visualization. Retrieved September 27, 2023, from https://seaborn.pydata.org/index.html.

Scikit Learn (2007-2024). scikit-learn: Machine Learning in Python. Retrieved March 6, 2024, from https://scikit-learn.org/stable/index.html. 

### SOURCES 

Thorn, J. (2020, Mar. 8) Decision Trees Explained. Retrieved March 9, 2024, from https://towardsdatascience.com/decision-trees-explained-3ec41632ceb6. 

Datacamp. (n.d.) Decision-Tree for Classification. Retrieved March 9, 2024, from https://campus.datacamp.com/courses/machine-learning-with-tree-based-models-in-python.

Jordan, J. (2017, Nov. 2). Hyperparameter tuning for machine learning models. Retrieved March 6, 2024, from https://www.jeremyjordan.me/hyperparameter-tuning/.

Saini, A. (2024, Jan. 5). Decision Tree – A Step-by-Step Guide. Retrieved March 9, 2024, from https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/.

Shmueli, G. (2015, Aug. 19). Categorical predictors: how many dummies to use in regression vs. k-nearest neighbors. Retrieved March 5, 2024, from https://www.bzst.com/2015/08/categorical-predictors-how-many-dummies.html 

Stewart, K. (n.d.) mean squared error. Retrieved March 10, 2024, from https://www.britannica.com/science/mean-squared-error.

Fernando, J. (2023, Dec. 13). R-Squared: Definition, Calculation Formula, Uses and Limitations. Retrieved March 10, 2024, from https://www.investopedia.com/terms/r/r-squared.asp.

Kapil, A.R. (2022, Oct. 1). Decision Tree Algorithm in Machine Learning: Advantages, Disadvantages, and Limitations. Retrieved March 10, 2024, from https://www.analytixlabs.co.in/blog/decision-tree-algorithm.
