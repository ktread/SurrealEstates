
################ AMES HOUSING PROJECT 

########## EDA AND PREPROCESSING ################

# Import Data and Packages

import pandas as pd
import numpy as np

import scipy.stats as stats

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('bmh')                    
sns.set_style({'axes.grid':False}) 
%matplotlib inline

from sklearn.linear_model import LinearRegression, HuberRegressor

# Data

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

##1.0 EDA
### Checking Dimensions of datasets

print('Dimensions of train data:', train.shape)
print('Dimensions of test data:', test.shape)


### Remove Outliers
## FirstFlSR
threshold = 2000
train = train[train.loc[:,'1stFlrSF'] < threshold]

## GrLivArea
threshold = 2500
train = train[train.loc[:,'GrLivArea'] < threshold]

## GarageArea
threshold = 1000
train = train[train.loc[:,'GarageArea'] < threshold]

## TotalBsmntSF
threshold = 2000
train = train[train.loc[:,'TotalBsmtSF'] < threshold]

##2.0 Missing variables

#### Combine Data

ntrain = train.shape[0]
ntest = test.shape[0]

housing_merge = pd.merge(train, test, how='outer').reset_index(drop=True)

##### Extracting 'SalePrice' and 'ID'
### ID is extracted so it is not transformed during the normalization process
y_sale = pd.DataFrame(train['SalePrice'])
id_temp = pd.DataFrame(housing_merge['Id'])

#### Dropping unwanted
housing_merge.drop(['Id'], axis=1, inplace=True)
housing_merge.drop(['SalePrice'], axis=1, inplace=True)

print("Train data size is : {}".format(train.shape))
print("Test data size is : {}".format(test.shape))
#print("Combined dataset size is : {}".format(housing_merge.shape))


### Total Missing
total = housing_merge.isna().sum()/housing_merge.isna().count()
sum=housing_merge.isna().sum()

missing=pd.concat([total,sum],axis=1,keys=['Perc','Sum']).sort_values(by='Perc',ascending=False)

colstodrop = missing[missing['Sum']>0].index

missing[missing['Sum']>0]

## Drop high Missing

housing_merge.drop(['PoolQC', 'FireplaceQu'], axis=1, inplace=True)

### Iputing Mode 

## MSZoning
housing_merge['MSZoning']=housing_merge['MSZoning'].fillna('RL')
## Electrical
housing_merge['Electrical']=housing_merge['Electrical'].fillna('SBrkr')
## Utilities
housing_merge['Utilities']=housing_merge['Utilities'].fillna('AllPub')
## Exterior1st
housing_merge['Exterior1st']=housing_merge['Exterior1st'].fillna('VinylSd')
## Exterior2nd
housing_merge['Exterior2nd']=housing_merge['Exterior2nd'].fillna('VinylSd')
## KitchenQual
housing_merge['KitchenQual']=housing_merge['KitchenQual'].fillna('TA')
#Functional
housing_merge['Functional']=housing_merge['Functional'].fillna('Typ')
##SaleType
housing_merge['SaleType']=housing_merge['SaleType'].fillna('WD')

### Imputing Random Number 

## LotFrontage
housing_merge['LotFrontage'] = housing_merge['LotFrontage'].apply(lambda x: np.random.
# MasVnrArea
housing_merge['MasVnrArea'] = housing_merge['MasVnrArea'].apply(lambda x: np.random.choice(housing_merge['MasVnrArea'].dropna().values) if np.isnan(x) else x)
# BsmtFinSF1
housing_merge['BsmtFinSF1'] = housing_merge['BsmtFinSF1'].apply(lambda x: np.random.choice(housing_merge['BsmtFinSF1'].dropna().values) if np.isnan(x) else x)
# BsmtFinSF2
housing_merge['BsmtFinSF2'] = housing_merge['BsmtFinSF2'].apply(lambda x: np.random.choice(housing_merge['BsmtFinSF2'].dropna().values) if np.isnan(x) else x)
# BsmtUnfSF
housing_merge['BsmtUnfSF'] = housing_merge['BsmtUnfSF'].apply(lambda x: np.random.choice(housing_merge['BsmtUnfSF'].dropna().values) if np.isnan(x) else x)
# TotalBasmtSF
housing_merge['TotalBsmtSF'] = housing_merge['TotalBsmtSF'].apply(lambda x: np.random.choice(housing_merge['TotalBsmtSF'].dropna().values) if np.isnan(x) else x)
# GarageArea
housing_merge['GarageArea'] = housing_merge['GarageArea'].apply(lambda x: np.random.choice(housing_merge['GarageArea'].dropna().values) if np.isnan(x) else x)
# Total Bsmt Sf
housing_merge['TotalBsmtSF'] = housing_merge['TotalBsmtSF'].apply(lambda x: np.random.choice(housing_merge['TotalBsmtSF'].dropna().values) if np.isnan(x) else x)

### Imputing median

##BedroomAbvGr
housing_merge['BedroomAbvGr'].fillna(housing_merge['BedroomAbvGr'].median(),inplace = True)
# BsmtFullBath
housing_merge['BsmtFullBath'].fillna(housing_merge['BsmtFullBath'].median(), inplace=True)
#BsmtHalfBath
housing_merge['BsmtHalfBath'].fillna(housing_merge['BsmtHalfBath'].median(), inplace=True)
#Garage Cars
housing_merge['GarageCars'].fillna(housing_merge['GarageCars'].median(), inplace=True)
#GarageYrBlt
housing_merge['GarageYrBlt'].fillna(housing_merge['GarageYrBlt'].median(), inplace=True)
#Fireplaces
housing_merge['Fireplaces'].fillna(housing_merge['Fireplaces'].median(), inplace=True)
#FullBath
housing_merge['FullBath'].fillna(housing_merge['FullBath'].median(), inplace=True)
#HalfBath 
housing_merge['HalfBath'].fillna(housing_merge['HalfBath'].median(), inplace=True)
#KitchenAbvGr 
housing_merge['KitchenAbvGr'].fillna(housing_merge['KitchenAbvGr'].median(), inplace=True)
#MSSubClass 
housing_merge['MSSubClass'].fillna(housing_merge['MSSubClass'].median(), inplace=True)
#MoSold 
housing_merge['MoSold'].fillna(housing_merge['MoSold'].median(), inplace=True)
#OverallCond 
housing_merge['OverallCond'].fillna(housing_merge['OverallCond'].median(), inplace=True)
#OverallQual
housing_merge['OverallQual'].fillna(housing_merge['OverallQual'].median(), inplace=True)
#YrSold
housing_merge['YrSold'].fillna(housing_merge['YrSold'].median(), inplace=True)

### Imputing None
impute_none = housing_merge.loc[:, [ 'MiscFeature', 'Alley', 'Fence',
                                    'GarageType', 'GarageCond','GarageFinish','GarageQual',
                                    'BsmtFinType2','BsmtExposure','BsmtQual','BsmtCond','BsmtFinType1',
                                    'MasVnrType']]
for i in impute_none.columns:
	housing_merge[i].fillna('None', inplace = True)


###  Feature Engineering
housing_merge['TotalSF'] = (housing_merge['BsmtFinSF1'] + housing_merge['BsmtFinSF2'] 
                           + housing_merge['1stFlrSF'] + housing_merge['2ndFlrSF'])
housing_merge['TotalBathrooms'] = (housing_merge['FullBath'] + (0.5*housing_merge['HalfBath']) +
                              housing_merge['BsmtFullBath'] + (0.5*housing_merge['BsmtHalfBath']))
housing_merge['TotalPorchSF'] = (housing_merge['OpenPorchSF'] + housing_merge['3SsnPorch'] +
                             housing_merge['EnclosedPorch'] + housing_merge['ScreenPorch'] +
                              housing_merge['WoodDeckSF'])

housing_merge['Pool'] = housing_merge['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
housing_merge['Has2flor'] = housing_merge['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
housing_merge['hasGarage'] = housing_merge['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
housing_merge['hasBsmt'] = housing_merge['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
housing_merge['hasFireplace'] = housing_merge['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#### Dropping Other features
housing_merge.drop(['FullBath', 'BsmtFullBath', 'HalfBath', 'BsmtHalfBath'], axis=1, inplace=True)
housing_merge.drop(['BsmtFinSF1', 'BsmtFinSF2', '1stFlrSF'], axis=1, inplace=True)
housing_merge.drop(['3SsnPorch', 'EnclosedPorch', 'ScreenPorch', 'WoodDeckSF', 'OpenPorchSF'], axis=1, inplace=True)
housing_merge.drop(['GarageCars'], axis=1, inplace=True)
housing_merge.drop(['LotFrontage'], axis=1, inplace=True)

numeric_cols = housing_merge.dtypes[housing_merge.dtypes != "object"].index
categori_cols = housing_merge.dtypes[housing_merge.dtypes == object]

#### Dummify Categorical Features
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
from numpy import argmax

cols = ( 'BldgType', 'BsmtCond', 'BsmtExposure','Fence', 'MiscFeature', 'Alley',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Condition1', 'Condition1',
       'Condition2', 'Electrical', 'ExterCond', 'ExterQual',
       'Exterior1st', 'Exterior2nd', 'Foundation',
       'GarageCond', 'Functional', 'GarageFinish', 'GarageQual', 'GarageType',
       'Heating', 'HeatingQC', 'KitchenQual', 'LandContour', 'LandSlope',
       'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType',
       'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle',
       'SaleCondition', 'SaleType', 'Street', 'Utilities')

##### process and encode to make it easier for the machine learning algorithm
### to read cat var
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(housing_merge[c].values))
    housing_merge[c] = lbl.transform(list(housing_merge[c].values))
    
print('Shape housing_merge: {}'.format(housing_merge.shape))

### hanging back to category

housing_merge[['BldgType', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Condition1', 'Condition1',
       'Condition2', 'Electrical', 'ExterCond', 'ExterQual',
       'Exterior1st', 'Exterior2nd', 'Foundation',
       'GarageCond', 'Functional', 'GarageFinish', 'GarageQual', 'GarageType',
       'Heating', 'HeatingQC', 'KitchenQual', 'LandContour', 'LandSlope',
       'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 
       'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle',
       'SaleCondition', 'SaleType', 'Street', 'Utilities','Fence', 'MiscFeature', 'Alley']] = housing_merge[['BldgType', 'BsmtCond', 'BsmtExposure',
       'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'Condition1', 'Condition1',
       'Condition2', 'Electrical', 'ExterCond', 'ExterQual',
       'Exterior1st', 'Exterior2nd',  'Foundation',
       'GarageCond', 'Functional', 'GarageFinish', 'GarageQual', 'GarageType',
       'Heating', 'HeatingQC', 'KitchenQual', 'LandContour', 'LandSlope',
       'LotConfig', 'LotShape', 'MSZoning', 'MasVnrType', 
       'Neighborhood', 'PavedDrive', 'RoofMatl', 'RoofStyle',
       'SaleCondition', 'SaleType', 'Street', 'Utilities','Fence', 'MiscFeature', 'Alley']].astype('category')

housing_merge = pd.get_dummies(housing_merge, drop_first = True)

#### Normality assesment

num_housing_merge = housing_merge.select_dtypes(include = ['int64', 'float64'])
display(num_housing_merge.head())
display(num_housing_merge.columns.values)

train_num_std = [col for col in num_housing_merge if abs(housing_merge[col].skew()) <= 1]
train_num_yjt = [col for col in num_housing_merge if abs(housing_merge[col].skew()) > 1]

########### Transform Continuous input
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

### Num
scaler = StandardScaler().fit(housing_merge[train_num_std].values)
housing_merge[train_num_std] = scaler.transform(housing_merge[train_num_std].values)

### power transform
pt = PowerTransformer().fit(housing_merge[train_num_yjt].values)
housing_merge[train_num_yjt] = pt.transform(housing_merge[train_num_yjt].values)


### SalePrice Transform
y_sale = np.log1p(y_sale)

### TO CSV

housing_merge = pd.concat([housing_merge, id_temp], axis=1)

housing_train = pd.DataFrame(housing_merge.iloc[0:1353, :])
housing_test = pd.DataFrame(housing_merge.iloc[1353:,:])

y_train = pd.DataFrame( y_sale)

housing_train.to_csv('housing_train.csv', index = False)
housing_test.to_csv('housing_test.csv', index = False)



y_train.to_csv('y_housing.csv', index = False)


################# LINEAR MODEL ################

import pandas as pd
import numpy as np

housing_train = pd.read_csv('housing_train.csv')
housing_test = pd.read_csv('housing_test.csv')

y_housing = pd.read_csv('y_housing.csv')

from sklearn import linear_model
from sklearn.linear_model import ElasticNet, Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.decomposition import PCA



import sklearn.model_selection as ms

from sklearn import ensemble
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
from sklearn.svm import SVR

X_train, X_test, y_train, y_test = train_test_split(housing_train, 
	y_housing, test_size = 0.30, random_state = 42)


ridge = linear_model.Ridge(alpha = 25, normalize=False, copy_X= True, fit_intercept = True)
#lasso = linear_model.Lasso(alpha= 0, normalize = True)
#elasticnet = linear_model.ElasticNet(alpha = 0.01, l1_ratio=0.5, normalize=False)
#clf = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)

#gbm = ensemble.GradientBoostingRegressor()
#rf = ensemble.RandomForestRegressor(n_estimators = 400, min_samples_split=2,
                                    #min_samples_leaf = 1, max_features = 'sqrt',
                                    #max_depth = None, bootstrap = False, random_state=42)


modelList =[ridge]
modelSeries= pd.Series(modelList, index =[ 'Ridge'])

modelSeries.apply(lambda x:x.fit(X_train, y_train))

ans = pd.concat([modelSeries.apply(lambda x: x.score(X_train,y_train)),modelSeries.apply(lambda x: x.score(X_test,y_test)), 
                                    modelSeries.apply(lambda x: np.sqrt(mean_squared_error(x.predict(X_test), y_test)))],axis=1)

ans.columns = ['train score', 'test score', 'rmse']
ans


##### Prediction for Kaggle

test = pd.read_csv('test.csv')

import math

housing_test['SalePrice'] = ridge.predict(housing_test)

housing_test['SalePrice'] = housing_test['SalePrice'] .apply(lambda x: math.exp(x))

submission =housing_test[['Id', 'SalePrice']].sort_values(ascending = True, by ='Id')

submission.to_csv('submission.csv', index = False) 








