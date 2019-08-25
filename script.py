import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
import numpy as np
from scipy.stats import norm, skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy.stats import zscore

# Read files
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
# Remove outliers
df_train = df_train.drop(df_train[(df_train.GrLivArea>4000) & (df_train.SalePrice<300000)].index)

# Using z-score to remove outliters
# num_col = df_train.dtypes[df_train.dtypes != object].index
# z = np.abs(zscore(df_train[num_col]))
# row, col = np.where(z > 4)
# df = pd.DataFrame({"row": row, "col": col})
# rows_count = df.groupby(['row']).count()

# outliers = rows_count[rows_count.col > 2].index
# df_train.drop(outliers, inplace=True)

# Normalize Sale Price
df_train.SalePrice = np.log1p(df_train.SalePrice)

# Combine training and test to do feature engineering
train_id = df_train.Id
test_id  = df_test.Id

nrow_train = df_train.shape[0]
nrow_test  = df_test.shape[0]

df_train.drop('Id',axis=1,inplace = True)
df_test.drop('Id',axis=1,inplace = True)
y_train = df_train.SalePrice.values
df_train.drop('SalePrice',axis=1,inplace = True)
full_data = pd.concat((df_train, df_test),sort=False).reset_index(drop=True)

# Missing data
full_data.loc[(full_data.PoolArea > 0) & (full_data.PoolQC.isnull()), 'PoolQC'] = 'TA'
full_data.PoolQC.fillna('None',inplace=True)
full_data.MiscFeature.fillna('None',inplace=True)
full_data.Alley.fillna('None',inplace=True)
full_data.Fence.fillna('None',inplace=True)
full_data.FireplaceQu.fillna('None',inplace=True)

# Two options here
full_data.LotFrontage = full_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#full_data.LotFrontage.fillna(0, inplace=True)
# Garage features part I
# There are two special cases
# Impute those two special cases with OverallCond

full_data.loc[(full_data.GarageCond.isnull()) & (full_data.OverallCond==8) & (full_data.GarageArea==360.0),\
      ['GarageFinish','GarageQual','GarageCond','GarageYrBlt']] =\
[full_data[full_data['OverallCond']==8]['GarageFinish'].mode()[0],
full_data[full_data['OverallCond']==8]['GarageQual'].mode()[0],
full_data[full_data['OverallCond']==8]['GarageCond'].mode()[0],
1910]
full_data.loc[(full_data.GarageCond.isnull()) & (full_data.OverallCond==6) &\
            (full_data.YearBuilt==1923) & (full_data.GarageType=='Detchd'),\
            ['GarageCond','GarageYrBlt','GarageFinish','GarageQual','GarageArea','GarageCars']] =\
[full_data[full_data['OverallCond']==6]['GarageCond'].mode()[0],
 1923,
 full_data[full_data['OverallCond']==6]['GarageFinish'].mode()[0],
 full_data[full_data['OverallCond']==6]['GarageQual'].mode()[0],
 full_data[full_data['OverallCond']==6]['GarageArea'].median(),
 full_data[full_data['OverallCond']==6]['GarageCars'].median()]

for col in ('GarageCond','GarageType','GarageFinish','GarageQual'):
    full_data[col].fillna('None',inplace=True)

full_data.GarageYrBlt.fillna(0.0,inplace=True)
for col in ('BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtCond','BsmtQual'):
    full_data[col].fillna('None',inplace=True)
    
for col in ('BsmtFullBath','BsmtFinSF2','BsmtFinSF1','TotalBsmtSF','BsmtUnfSF','BsmtHalfBath'):
    full_data[col].fillna(0.0,inplace=True)

full_data.MasVnrType.fillna('None',inplace=True)
full_data.MasVnrArea.fillna(0.0,inplace=True)
full_data.loc[(full_data.MSSubClass==20)&(full_data.MSZoning.isnull()),'MSZoning'] = 'RL'
full_data.loc[(full_data.MSSubClass==30)&(full_data.MSZoning.isnull()),'MSZoning'] = 'RM'
full_data.loc[(full_data.MSSubClass==70)&(full_data.MSZoning.isnull()),'MSZoning'] = 'RM'

# There are only 1 NoSeWa and 2 Nan. Also, the NoSeWa is in the test set, so this feature won't help in prediction
full_data = full_data.drop(['Utilities'], axis=1)

full_data.Functional.fillna('Typ',inplace=True)
full_data.SaleType.fillna(full_data.SaleType.mode()[0],inplace=True)
full_data.Exterior1st.fillna(full_data.Exterior1st.mode()[0],inplace=True)
full_data.Exterior2nd.fillna(full_data.Exterior2nd.mode()[0],inplace=True)
full_data.KitchenQual.fillna(full_data.KitchenQual.mode()[0],inplace=True)
full_data.Electrical.fillna(full_data.Electrical.mode()[0],inplace=True)

# There is a strange GarageYrBlt, modify it
full_data.loc[full_data['GarageYrBlt'] == 2207, 'GarageYrBlt'] = 2007

#############################################
# Transform nominal columns to correct type #
#############################################
full_data.OverallCond = full_data.OverallCond.astype(str)
full_data.MSSubClass = full_data.MSSubClass.astype(str)
full_data.YrSold = full_data.YrSold.astype(str)
full_data.MoSold = full_data.MoSold.astype(str)

from sklearn.preprocessing import LabelEncoder
catecols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')

# process columns, apply LabelEncoder to categorical features
for i in catecols:
    lbl = LabelEncoder() 
    lbl.fit(list(full_data[i].values)) 
    full_data[i] = lbl.transform(list(full_data[i].values))

###########################
# Add Additional Features #
###########################
full_data['TotalLivSF'] = full_data['BsmtFinSF1'] + full_data['BsmtFinSF2'] + full_data['1stFlrSF'] + full_data['2ndFlrSF']
full_data['Total_Bath'] = (full_data['FullBath'] + (0.5*full_data['HalfBath']) +\
                       full_data['BsmtFullBath'] + (0.5*full_data['BsmtHalfBath']))

#full_data['2ndFL'] = full_data['2ndFlrSF'].apply(lambda x: 'Y' if x > 0 else 'N')
#full_data['bsmt'] = full_data['TotalBsmtSF'].apply(lambda x: 'Y' if x > 0 else 'N')

#full_data['TotalPorchSF'] = (full_data['OpenPorchSF'] + full_data['3SsnPorch'] +\
#                              full_data['EnclosedPorch'] + full_data['ScreenPorch'] +\
#                             full_data['WoodDeckSF'])
#drop_fea1 = ['BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath',
#            'HalfBath','BsmtFullBath','BsmtHalfBath']
#drop_fea2 = ['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF']
#full_data.drop(drop_fea1,axis=1,inplace=True)
#full_data.drop(drop_fea2,axis=1,inplace=True)

# Reduce Skewness using Box Cox
num_cols = full_data.dtypes[full_data.dtypes != "object"].index
skewed_cols = full_data[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_cols = skewed_cols[abs(skewed_cols) > 0.75]
skewed_features = skewed_cols.index
lam = 0.15
for feat in skewed_features:
#    full_data[feat] = boxcox1p(full_data[feat], boxcox_normmax(full_data[feat]+1))
    full_data[feat] = boxcox1p(full_data[feat], lam)
    
#full_data[skewed_features] = np.log1p(full_data[skewed_features])

#Scale numeric columns
#scaler = StandardScaler()
#scaler = preprocessing.RobustScaler()
#full_data[num_cols] = scaler.fit_transform(full_data[num_cols])

#cate_col = full_data.dtypes[full_data.dtypes == object].index

# Dummify columns
#dummies_drop = [i + '_'+ full_data[i].value_counts().index[0] for i in cate_col]
full_data = pd.get_dummies(full_data)
#print(full_data.shape)
#full_data.drop(dummies_drop,axis=1)

# Split training and test set
x_train = full_data[:nrow_train]
x_test = full_data[nrow_train:]

x_train.to_csv('train_mod.csv', index=False)
x_test.to_csv('test_mod.csv', index=False)
y_train =pd.DataFrame(y_train,columns=['SalePrice'])
y_train.to_csv('y_train.csv',index=False)
##############################################################################################################
# Modeling part 

from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
from sklearn.svm import SVR

# Ridge
ridge = Ridge()
para_ridge = {"alpha": np.logspace(0.8,1.1,100)}
# Grid Search
gs_ridge = GridSearchCV(ridge, para_ridge, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
gs_ridge.fit(x_train, y_train)
gs_ridge.best_params_
#{'alpha': 9.930467558623954}

# Use RobustScaler to reduce the influence of outliers
ridge = make_pipeline(RobustScaler(), Ridge(alpha = gs_ridge.best_params_['alpha'], random_state=1))
ridge.fit(x_train, y_train)
score_ridge = ridge.score(x_train, y_train)
rmse_ridge = np.sqrt(mean_squared_error(y_train,ridge.predict(x_train)))
ridge_pred = np.expm1(ridge.predict(x_test))

# Lasso
lasso = Lasso()
para_lasso = {"alpha": np.logspace(-4,-3.3,100)}
lasso.set_params(max_iter = 10000)
gs_lasso = GridSearchCV(lasso, para_lasso, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
gs_lasso.fit(x_train, y_train)
gs_lasso.best_params_
#{'alpha': 0.0003503034741265338}

lasso = make_pipeline(RobustScaler(), Lasso(alpha =gs_lasso.best_params_['alpha'], random_state=1))
lasso.fit(x_train, y_train)
score_lasso = lasso.score(x_train, y_train)
rmse_lasso = np.sqrt(mean_squared_error(y_train,lasso.predict(x_train)))
lasso_pred = np.expm1(lasso.predict(x_test))

# ElasticNet
ElNet = ElasticNet()
para_ElNet = {"alpha": np.logspace(-3.8,-3.3,10),"l1_ratio": np.linspace(0.7,0.9,10)}
ElNet.set_params(max_iter = 5000)
gs_ElNet = GridSearchCV(ElNet, para_ElNet, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
gs_ElNet.fit(x_train, y_train)
gs_ElNet.best_params_
#{'alpha': 0.0003880510732210184, 'l1_ratio': 0.9}

ElNet.set_params(alpha = gs_ElNet.best_params_['alpha'],l1_ratio=gs_ElNet.best_params_['l1_ratio'])
ElNet.fit(x_train, y_train)
score_ElNet = ElNet.score(x_train, y_train)
rmse_ElNet = np.sqrt(mean_squared_error(y_train,ElNet.predict(x_train)))
ElNet_pred = np.expm1(ElNet.predict(x_test))

# Gradient Boosting
gbm = GradientBoostingRegressor(max_depth=4, max_features='log2',
                                min_samples_leaf=15, min_samples_split=10, 
                                loss='huber')

para_gbm = {"n_estimators":np.arange(1000,10000,1000),'learning_rate':[0.05,0.1,0.5]}
gs_gbm = GridSearchCV(gbm, para_gbm, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
gs_gbm.fit(x_train, y_train)
gs_gbm.best_params_
#{'learning_rate': 0.05, 'n_estimators': 2000}

gbm.set_params(learning_rate = gs_gbm.best_params_['learning_rate'],\
                  n_estimators=gs_gbm.best_params_['n_estimators'])
gbm.fit(x_train, y_train)
score_gbm = gbm.score(x_train, y_train)
rmse_gbm = np.sqrt(mean_squared_error(y_train,gbm.predict(x_train)))
gbm_pred = np.expm1(gbm.predict(x_test))

# Show the top 10 important features from gbm
sorted_importance = sorted(zip(x_train.columns, gbm.feature_importances_), key=lambda t:t[1], reverse=True)
plt.barh([x[0] for x in sorted_importance[:10]][::-1],[x[1] for x in sorted_importance[:10]][::-1])
plt.title('GBM Top 10 Feature Importance', fontsize=20)
plt.figure(figsize=(10,8))

# XGBoost
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, min_child_weight=1.7817,
                             reg_alpha=0.4640,
                             reg_lambda=0.8571, subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

para_xgb = {"n_estimators":np.arange(1000,8000,1000),'learning_rate':[0.05,0.1,0.5],'gamma':[0.03,0.04,0.05]}
gs_xgb = GridSearchCV(model_xgb, para_xgb, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
gs_xgb.fit(x_train, y_train)
gs_xgb.best_params_
#{'gamma': 0.03, 'learning_rate': 0.05, 'n_estimators': 1000}

model_xgb.set_params(learning_rate = gs_xgb.best_params_['learning_rate'],\
                       n_estimators=gs_xgb.best_params_['n_estimators'],\
                     gamma=gs_xgb.best_params_['gamma'])
model_xgb.fit(x_train, y_train)
score_xgb = model_xgb.score(x_train, y_train)
rmse_xgb = np.sqrt(mean_squared_error(y_train,model_xgb.predict(x_train)))
xgb_pred = np.expm1(model_xgb.predict(x_test))

# Show the top 10 important features from xgboost
sorted_importance_xgb = sorted(zip(x_train.columns, model_xgb.feature_importances_), key=lambda t:t[1], reverse=True)
plt.barh([x[0] for x in sorted_importance_xgb[:10]][::-1],[x[1] for x in sorted_importance_xgb[:10]][::-1])
plt.title('XGB Top 10 Feature Importance', fontsize=20)
plt.figure(figsize=(10,8))

# Support vector regression
svr = SVR(epsilon = 0.1, gamma=1e-8,C=100000)
svr.fit(x_train,y_train) 
score_svr = svr.score(x_train, y_train)
rmse_svr = np.sqrt(mean_squared_error(y_train,svr.predict(x_train)))
svr_pred = np.expm1(svr.predict(x_test))

# Stack model
ridge_b = make_pipeline(RobustScaler(), Ridge(alpha = 9.930467558623954, random_state=1))
lasso_b = make_pipeline(RobustScaler(), Lasso(alpha = 0.0003503034741265338, random_state=1))
ElNet_b = ElasticNet(alpha = 0.0003880510732210184, l1_ratio = 0.9)
svr_b = SVR(epsilon = 0.1, gamma=1e-8,C=100000)
model_xgb_b = xgb.XGBRegressor(gamma = 0.03, learning_rate = 0.05, n_estimators = 1000,
                             colsample_bytree=0.4603, min_child_weight=1.7817,
                             reg_alpha=0.4640,
                             reg_lambda=0.8571, subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
gbm_b = GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 2000,
                                  max_depth=4, max_features='log2',
                                  min_samples_leaf=15, min_samples_split=10, 
                                  loss='huber')

stackmodel = StackingRegressor(
        regressors=[ElNet_b, lasso_b, ridge_b, svr_b, model_xgb_b, gbm_b],
        meta_regressor=Lasso(alpha=0.00035)
    )

stackmodel.fit(x_train, y_train)

stacked = stackmodel.predict(x_test)
rmse_stacked = np.sqrt(mean_squared_error(y_train,stackmodel.predict(x_train)))
stacked_pred = np.expm1(stacked)


# Averaged model
ensembled = np.expm1((0.25 * ridge.predict(x_test).reshape(-1,1)) +
                     (0.2 * ElNet.predict(x_test).reshape(-1,1)) +
                     (0.2 * lasso.predict(x_test).reshape(-1,1)) +
                     (0.15 * model_xgb.predict(x_test).reshape(-1,1)) +
                     (0.2 * GBoost.predict(x_test).reshape(-1,1)))

# Print the performance of each model
obj = pd.DataFrame([[score_ridge,rdg_trainRMSE,rdg_testRMSE,rmse_ridge_test-rmse_ridge_train,0.11866],
                    [score_lasso,lso_trainRMSE,lso_testRMSE,lso_testRMSE-lso_trainRMSE,0.11938],
                    [score_ElNet,net_trainRMSE,net_testRMSE,net_testRMSE-net_trainRMSE,0.11926],
                    [score_gbm,gbm_trainRMSE,gbm_testRMSE,gbm_testRMSE-gbm_trainRMSE,0.12485],
                    [score_xgb,xgb_trainRMSE,xgb_testRMSE,xgb_testRMSE-xgb_trainRMSE,0.12386],
                    [score_svr,svr_trainRMSE,svr_testRMSE,svr_testRMSE-svr_trainRMSE,0.12388]],
                   columns = ['score_grid','train_RMSE','test_RMSE','diff_RMSE','Kaggle_score'],
                   index=['Ridge','Lasso','ElNet','GBM','XGB','SVR'])
print(obj)

# Output the submission file
sub = pd.DataFrame()
sub['Id'] = df_test.Id
sub['SalePrice'] = ensembled
sub.to_csv('submission.csv',index=False)







