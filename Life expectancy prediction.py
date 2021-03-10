#!/usr/bin/env python
# coding: utf-8

# # Welcome to the final project!
# During this course we covered linear regression and SVM for data analysis. In this notebook we will be working with data about life expectancy across different countries. We use descriptive features based on statistical data to predict life expectancy as a continuous value, and later classify countries to either have a "short" or "long" life expectancy. You will be asked to implement your own Linear Regression model based on the materials given in the lectures. For classification, you will be using an already implememnted SVM classifier from the `sklearn` library.

# In[1]:


#some necessary imports we'll use later
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sys
sys.path.append("..")
import grading
grader = grading.Grader(assignment_key="-8r0HMXGTwqC5yKZHJrcbw", 
                      all_parts=["HPkMz", "xOP97", "cMnZI"])


# In[113]:


# token expires every 30 min
COURSERA_TOKEN = 'NicXjm0C50Adz07w' # YOUR COURSERA TOKEN HERE (can be found in Programming section)
COURSERA_EMAIL = 'Ahmetova.RN@phystech.edu' # YOUR COURSERA EMAIL HERE


# ## Looking at the data
# First, we need to read the data from a `csv` file. This portion has been done for you using `pandas` library. For more information check out the [documentation](https://pandas.pydata.org/pandas-docs/stable/). 

# In[3]:


df = pd.read_csv('dataset.csv')
print("Dataset dimesions: ", df.shape)
# dataset preview
df.head()


# In[4]:


# list all available features
df.columns


# **Feature list:**
# 1. Year
# 2. Status - Developed (1) or Developing (0) status
# 3. Life expectancy - Life Expectancy in age
# 4. Adult Mortality - Adult Mortality Rates of both sexes (probability of dying between 15 and 60 years per 1000 population)
# 5. Infant deaths - Number of Infant Deaths per 1000 population
# 6. Alcohol - Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
# 7. Percentage expenditure - Expenditure on health as a percentage of Gross Domestic Product per capita(%)
# 8. Hepatitis B - Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
# 9. Measles - Measles, number of reported cases per 1000 population
# 10. BMI - Average Body Mass Index of entire population
# 11. Under-five deaths - Number of under-five deaths per 1000 population
# 12. Polio - Polio (Pol3) immunization coverage among 1-year-olds (%)
# 13. Total expenditure - General government expenditure on health as a percentage of total government expenditure (%)
# 14. Diphtheria - Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
# 15. HIV/AIDS - Deaths per 1 000 live births HIV/AIDS (0-4 years)
# 16. GDP - Gross Domestic Product per capita (in USD)
# 17. Population - Population of the country
# 18. Thinness 1-19 years - Prevalence of thinness among children and adolescents for Age 10 to 19 (% )
# 19. Thinness 5-9 years - Prevalence of thinness among children for Age 5 to 9(%)
# 20. Income composition of resources - Human Development Index in terms of income composition of resources (index ranging from 0 to 1)
# 21. Schooling - Number of years of Schooling(years)
# 
# 
# **Target**: Life expectancy

# In[40]:


target_feature = 'life_expectancy'


# In[6]:


plt.figure(figsize=(15,15))
a, b = 5, 5
for i, col in enumerate(df.columns[df.columns !=target_feature]):
    plt.subplot(b, a, i+1)
    plt.scatter(df[col], df[target_feature])
    plt.title(col)
plt.tight_layout()


# Based on the data plots above, which features do you think will contribute to good results of linear regression the most?

# ## Linear Regression

# ### Split data to train and test

# We want our model to not be biased towards ceratin data, so we will train the model on one set of data and test on another. This is done in order to evaluate how well the model performs on previously "unseen" values. This data separation has been done for you using the `train_test_split` method. The size of the test dataset is 20% of the total, and we define `random_state` to get the same consistent results when running this code.

# In[115]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)


# ### Writing linear regression
# 
# Linear regression is a linear approximation $$f(x)=w_0 + w_1x_1 + ... +w_nx_n$$
# 
# As we recall from the lecture the analytical solution for an inconsistent system of $Xw=y$ is
# 
# $$\hat{w} = (X^TX)^{-1}X^Ty$$
# 
# In our case the matrix $X$ is the matrix, where each row is an object (person) and each column is a feature. $y$ is the target value - vector of life expectancy values. $\hat{w}$ is the approximate solution.
# 
# In the class below create a linear regression class. `AlgLinearRegression` has to have methods for training on dataset and predicting values. Those methods have been outlined for you. Don't forget to add the dummy variable for the scalar intercept ($w_0$)!
# 
# **Hint**: You can use `np.hstack` to append the mock variable (vector of ones) for scalar intercept. Vector of ones can be created using `np.ones` method. Do not forget to do this in both `fit` and `predict` methods.

# In[116]:


class AlgLinearRegression():
    def __init__(self, fit_intercept = True):
        self.coef_ = []
    def fit(self, X, y):
        epochs=200000 
        learning_rate=0.001
        '''
        This method takes the training data and calculate the approximate solution w (self.coef).
        It will later be used to predict values for new data.
        
        self - reference parameter to class instance.
        X - matrix of features.
        y - vector of target varibles.
        
        Returns - self.
        '''
        #your code here
        n=X.shape[1]
        self.coef_=np.zeros(n+1)
        m = float(len(y))
        x = np.hstack((np.ones((int(m), 1)), X))
        
        for i in range(epochs):
            
            prediction =x.dot(self.coef_)
            delta=prediction-y
            current_gradient=delta.dot(x)
            current_gradient*=1/m
            self.coef_=self.coef_-current_gradient*learning_rate
            '''if i%1000==0:
                print(i)'''

        
        return self

    def predict(self, X):
        '''
        This method takes new data and applies the self.coef (calculated in fit) to it to get the new target predictions.
        
        self - reference parameter to class instance.
        X - matrix of features.
        
        Returns - predicted vector of target values.
        '''
        #your code here
        m=X.shape[0]
        x = np.hstack((np.ones((int(m), 1)), X))
        y_pred=np.dot(x, self.coef_)
        return y_pred


# Train and test your regressor using one feature, `schooling`, first. `X_train` and `y_train` are for fitting the regressor, `X_test` for predicting values, and finally `y_test` is for assessing quality. Use mean squared error as the quality metric (see [here](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)).
# $$MSE (y\_true, y\_pred) = \frac{1}{n}\sum_{i=1}^n (y\_true_i - y\_pred_i)^2$$

# In[117]:


X_train = df_train['schooling'].values.reshape(-1,1)
X_test = df_test['schooling'].values.reshape(-1,1)
y_train = df_train[target_feature].values
y_test = df_test[target_feature].values


# In[118]:


#your code here
reg =AlgLinearRegression()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)


# In[119]:


from sklearn.metrics import mean_squared_error
#calculate error (your code here)
ans1 = mean_squared_error(y_test, y_pred)
ans1


# In[120]:


#visualize the constructed line
plt.scatter(X_test, y_test, alpha=0.8)
plt.plot(X_test, y_pred, c='r')
plt.title('One feature AlgLinearRegression')
plt.show()


# Is it a good result? Think of ways we can improve the predictions.

# In[121]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("HPkMz", ans1)


# In[122]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# ### Now let's use other features and see if it helps decrease error
# Use the list of features (+ traget value) defined in `features` variable below to train you regressor. 

# In[106]:


#features = ['BMI', 'life_expectancy']
features = ['status', 'BMI', 'total_expenditure', 
            'HIV/AIDS', ' thinness 5-9 years', 'income_composition_of_resources', 'schooling', 'life_expectancy']


# In[107]:


df_train, df_test = train_test_split(df[features], test_size = 0.2, random_state = 42)
X_train = df_train.drop([target_feature], axis=1).values
X_test = df_test.drop([target_feature], axis=1).values
y_train = df_train[target_feature].values
y_test = df_test[target_feature].values


# In[108]:


# your code here
from sklearn import linear_model 
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
#y_pred=reg.predict(X_test)


# In[109]:


y_pred=reg.predict(X_test)


# In[110]:


# calcualte error (your code here)
ans2 = mean_squared_error(y_test, y_pred)
ans2


# In[111]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("xOP97", ans2)


# In[114]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# Let's plot the difference between our predictions and true values. What does it say about our regression model? How can the plot below suggest ways to improve the predictions?

# In[25]:


#plot difference in predictions on all objects
plot_range = len(y_pred)
plt.scatter(np.arange(0, plot_range), y_pred-y_test, alpha=0.8)
plt.plot(np.arange(0, plot_range), np.zeros((plot_range,)), c='g')
plt.title('Differenece in prediction and true value for multi feature AlgLinearRegression')
plt.show()


# In[26]:


#plot underpredictions vs. over predictions distribution
plt.hist(y_pred-y_test, bins=20)
plt.axvline(0, c='y')
plt.show()


# Plotting feature importance

# In[27]:


plt.figure(figsize=(20,10))
names = features[:-1] + ['free var']
plt.bar(np.arange(len(names)), reg.coef_)
plt.xticks(range(len(names)), names, rotation='25')
plt.show()


# ## SVM

# In this section of the notebook you will be asked to use `SVM` classifier to split our data in two classes. In the lectures we covered SVM with linear kernel. 
# 
# <img src="SVM_illustration.png" style="width:60%">
# 
# The idea of SVM is to draw a hyperplane so that the separation between classes (two, in our case) is maximum, and then use this hylerplane to determine the class for new objects.
# 
# [Image source and more info](https://scikit-learn.org/stable/modules/svm.html).

# ### Now let's train a classifier to divide our data into two categories as established below:

# Let's say that a person is expected to live a long life if his life expectancy is more than 80 years.

# In[9]:


TSLD = 80


# Our new target value is the class: let us put the class to 0 if the life expectancy is short, and to 1 if it is long.

# In[13]:


df_class = df[features].copy()
df_class['long_life'] = np.where(df_class[target_feature] >= TSLD, 1, 0)
short_life = df_class[target_feature][df_class[target_feature] < TSLD]
long_life = df_class[target_feature][df[target_feature] >= TSLD]
# drop the old target variable
df_class = df_class.drop([target_feature], axis=1)
print("Short to long life expectancy ratio: ", np.round(short_life.shape[0]/df.shape[0],2),':', 
      np.round(long_life.shape[0]/df.shape[0],2))


# In[14]:


#new data preview
df_class.head()


# ### Training and evaluating model

# In this part of the notebook we will be using an already implemented SVM model from `sklearn`. Use linear kernel.

# In[15]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Same as with linear regression we need to split our data into train and test samples to later evaluate the quality of the classifier. Use [accuracy score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score) as the quality metric here.
# 
# $$accuracy(y\_true, y\_pred) = \frac{1}{n}\sum_{i=1}^n [y\_true=y\_pred],$$ where $[a=b]=1$, if $a=b$, and $0$ otherwise.

# In[16]:


df_train, df_test = train_test_split(df_class, test_size = 0.2, random_state = 42)


# In[17]:


#separating target variable from features
X_train = df_train.drop(['long_life'], axis=1)
X_test = df_test.drop(['long_life'], axis=1)
y_train = df_train['long_life']
y_test = df_test['long_life']


# In[27]:


# train classifier (your code here)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)


# In[28]:


y_pred=clf.predict(X_test)


# In[29]:


# evaluate results (your code here)
ans3 = accuracy_score(y_test, y_pred)
ans3


# In[30]:


## GRADED PART, DO NOT CHANGE!
grader.set_answer("cMnZI", ans3)


# In[31]:


# you can make submission with answers so far to check yourself at this stage
grader.submit(COURSERA_EMAIL, COURSERA_TOKEN)


# For linear kernel we can extract feature importances:

# In[32]:


# plot impact for each feature to the classification
plt.figure(figsize=(20,10))
names = X_train.columns
plt.bar(np.arange(len(names)), clf.coef_[0])
plt.xticks(range(len(names)), names)
plt.title('Feature importnaces for SVM')
plt.show()


# What can you tell from the plot above? Which features are the most important in predicting the label for an object? 
