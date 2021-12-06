## Commit 1 : Boosting model without Feature Selection


                    """ Import LIBRARIES"""
##1-GENERAL
import pandas as pd
import numpy as np
import random

##2-Visualization Library
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

##4-scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler

## ENCODING
from sklearn.preprocessing import LabelEncoder

##TRANSFORMATION
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer

##Imputation
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

##STATISTICS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels
import statsmodels.api as sm
import scipy.stats as stats
import statistics
from scipy import stats
from statsmodels.stats import weightstats as stests
from scipy.stats import shapiro
from statsmodels.stats import power
import statsmodels.formula.api as smf
from scipy.stats.mstats import winsorize

##train-test split
from sklearn.model_selection import train_test_split

##MACHINE LEARNING
#Feature selection
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

#Sampling Techniques - Classification
from imblearn.under_sampling import RandomUnderSampler

#Feature Importance
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm  import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.ensemble import RandomForestClassifier

# performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import fbeta_score

#Hyperparamter Tuning
from sklearn.model_selection import GridSearchCV
from sklearn import tree

#STACKING  MODELS
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import VotingClassifier

## WARNNINGS
from warnings import filterwarnings
filterwarnings('ignore')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
pd.options.display.float_format = '{:.6f}'.format
pd.options.display.max_columns = None

                             """2 - DATA PREPROCESSING"""
'Source data import'
train = pd.read_csv('F:/PROJECTS/Hackathon/Loan Defaulter Prediction/Data/train.csv')
test = pd.read_csv('F:/PROJECTS/Hackathon/Loan Defaulter Prediction/Data/test.csv')
print(train.head())

'Sanity Check'
print(data.dtypes)
print(data.shape)
print(data.size)


##TRAIN
catdata = train[['Term','Batch Enrolled', 'Grade', 'Sub Grade', 'Employment Duration',
       'Verification Status', 'Payment Plan', 'Loan Title',
       'Initial List Status', 'Application Type','Loan Status']]
numdata = train[train.columns.difference(['Term','Batch Enrolled', 'Grade', 'Sub Grade', 'Employment Duration',
       'Verification Status', 'Payment Plan', 'Loan Title',
       'Initial List Status', 'Application Type','Loan Status'])]

##TEST
catdatatest = test[['Term','Batch Enrolled', 'Grade', 'Sub Grade', 'Employment Duration',
       'Verification Status', 'Payment Plan', 'Loan Title',
       'Initial List Status', 'Application Type']]
numdatatest = test[test.columns.difference(['Term','Batch Enrolled', 'Grade', 'Sub Grade', 'Employment Duration',
       'Verification Status', 'Payment Plan', 'Loan Title',
       'Initial List Status', 'Application Type'])]


'Datatype Conversion'
for i in catdata:
    catdata[i] = catdata[i].astype('str')

for i in catdatatest:
    catdatatest[i] = catdatatest[i].astype('str')


'Feature understanding'
# 5 POINT summary - NUMERICAL
ipl.describe()
numdata.describe(include = 'all')
catdata.describe(include = 'all')

#Info
ipl.info()
catdata.info()
numdata.info()

## Numercialessentials
def numessentials(x):
    return pd.DataFrame([x.skew(), x.sum(), x.mean(), x.median(),  x.std(), x.var(),x.min(),x.max()],
                  index=['skew', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN','MAX'])


## Numerical Attributes - My Function
def numericalattributes(X):
    Output = pd.DataFrame()
    Output['Variables'] = X.columns
    Output['Skewness'] = X.skew().values
    Output ['Kurtosis'] = X.kurt().values
    Output ['Standarddeviation'] = X.std().values
    Output ['Variance'] = X.var().values
    Output ['Mean'] = X.mean().values
    Output ['Median'] = X.median().values
    Output ['Minimum'] = X.min().values
    Output ['Maximum'] = X.max().values
    Output ['Sum'] = X.sum().values
    Output ['Count'] = X.count().values
    return Output

## Variable Summary
def variablesummary(x):
    uc = x.mean()+(2*x.std())
    lc = x.mean()-(2*x.std())
    for i in x:
        if i<lc or i>uc:
            count = 1
        else:
            count = 0
    outlier_flag = count
    return pd.Series([x.corr(),x.count(), x.isnull().sum(), x.sum(), x.mean(), x.median(),  x.std(), x.var(), x.min(), x.quantile(0.01), x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),x.quantile(0.50),x.quantile(0.75), x.quantile(0.90),x.quantile(0.95), x.quantile(0.99),x.max() , lc , uc,outlier_flag],
                  index=['N', 'NMISS', 'SUM', 'MEAN','MEDIAN', 'STD', 'VAR', 'MIN', 'P1' , 'P5' ,'P10' ,'P25' ,'P50' ,'P75' ,'P90' ,'P95' ,'P99' ,'MAX','LC','UC','outlier_flag'])
# UC = MEAN + 2 STD

'CATEGORICAL'
## VALUE - COUNTS
for i in catdata.columns:
    print(catdata[i].value_counts())


                    '2.7 - Null value Finding'
Total = train.isnull().sum().sort_values(ascending=False)
Percent = (train.isnull().sum()*100/len(train)).sort_values(ascending=False)
missingdata = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])    
print(missingdata)

Total = test.isnull().sum().sort_values(ascending=False)
Percent = (test.isnull().sum()*100/len(test)).sort_values(ascending=False)
missingdata = pd.concat([Total, Percent], axis = 1, keys = ['Total', 'Percentage of Missing Values'])
print(missingdata)

                                      """ OUTLIERS"""
#1-Fix Outlier Range
q1 = numdata.quantile(0.25)
q3 = numdata.quantile(0.75)
IQR = q3 - q1
upper_range = q3 + (IQR*1.5)
lower_range = q1 - (IQR*1.5)
extreme_upper_range = q3 + (IQR*3)
extreme_lower_range = q1 - (IQR*3)

#2-Find Count of Outliers
pd.DataFrame(((numdata < extreme_lower_range) | (numdata > extreme_upper_range)).sum(),
             columns = ['No. of Outliers']).sort_values(by = 'No. of Outliers', ascending = False)

#3-Find Percentage of Outliers
pd.DataFrame(((numdata < extreme_lower_range) | (numdata > extreme_upper_range)).sum(),
             columns = ['No. of Outliers']).sort_values(by = 'No. of Outliers', ascending = False) / len(numdata)*100

                              'Outlier Treatment'
#1-Capping (Winzorization)
for i in numdata.columns:
    q1 = numdata[i].quantile(0.25)
    q3 = numdata[i].quantile(0.75)
    IQR = q3-q1
    UB = q3 + 1.5*IQR
    LB = q1 - 1.5*IQR
    UC = numdata[i].quantile(0.99)
    LC = numdata[i].quantile(0.01)
    for ind1 in numdata[i].index:
        if numdata.loc[ind1,i] > UB:
            numdata.loc[ind1,i] = UC
        if numdata.loc[ind1,i] < LB:
            numdata.loc[ind1,i] = LC

                            'Scaling'
#1-MINMAX SCALER
mm = MinMaxScaler()
numdatamm = mm.fit_transform(numdata)
numdatamm = pd.DataFrame(numdatamm,columns = numdata.columns)

#2-ROBUST SCALER
rs = RobustScaler()
numdatars = rs.fit_transform(numdata)
numdatars = pd.DataFrame(numdatars,columns = numdata.columns)

#3-STANDARD SCALER
sc = StandardScaler()
numdatasc = sc.fit_transform(numdata)
numdatasc = pd.DataFrame(numdatasc,columns = numdata.columns)

#4-MaxAbs Scaler
scaler = MaxAbsScaler()
df_scaled = scaler.fit_transform(numdata.values)
df_scaled = pd.DataFrame(df_scaled,columns = numdata.columns)

#5-Quantile Transformer Scaler
scaler = QuantileTransformer()
df_scaled = scaler.fit_transform(numdata.values)
df_scaled = pd.DataFrame(df_scaled,columns = numdata.columns)

#6-Unit Vector Scaler/Normalizer
from sklearn.preprocessing import Normalizer
scaler = Normalizer(norm = 'l2')
"norm = 'l2' is default"
df_scaled[col_names] = scaler.fit_transform(features.values)


                     """Variable Inflation Factor"""
def variableinflation(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)



                       """ CORRELATION"""
correlation = X.corr()
sns.heatmap(correlation, annot = True)

#Absolute correlation with Target variable
abs(df.corr()['Variable Name']).sort_values(ascending = False)

def correlation(X):
    correlation = X.corr()
    heatmap = sns.heatmap(correlation,annot = True)
    return correlation,heatmap

def visual(X):
        distplot = sns.distplot(X.columns)
        scatterplot = sns.scatterplot(X.columns)
        return distplot,scatterplot


                         'TRANSFORMATION'
#Log Transformation
np.log2(numdata1['Item_Outlet_Sales']+0.000000001).skew()

#Power Transformation
np.power(numdata1['Item_Outlet_Sales'],0.39).skew()

#Squareroot Transformation
np.sqrt(df.iloc[:,0])

#Boxcox Transformation
sales_box,lam= stats.boxcox(numdata1['Item_Outlet_Sales'])
pd.DataFrame(sales_box).skew()

#Custom Transformer
transformer = FunctionTransformer(np.log2, validate = True)
df_scaled[col_names] = transformer.transform(features.values)

#PowerTransformer
scaler = PowerTransformer(method = 'box-cox')
df_scaled[col_names] = scaler.fit_transform(numdata.values)
#NOTE-method = 'box-cox' or 'yeo-johnson''


                     '2.11 - Encoding Categorical variable'
#2 - Factorize Encoding
for i in catdata:
    catdata[i] = catdata[i].factorize()[0]

for i in catdatatest:
    catdatatest[i] = catdatatest[i].factorize()[0]


                            'Concatenating Data'
# 'axis=1' concats the dataframes along columns 
CONCAT = pd.concat([numdata,catdata],axis = 1)
XTEST = pd.concat([numdatatest,catdatatest],axis = 1)


                         """VISUALIZATION"""
                     """ Univariate Analysis"""
#1 - Distplot Analysis
row = 3
col = 1
count = 1
for i in numdata:
    plt.subplot(row,col,count)
    plt.title(i)
    sns.distplot(data[i])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(20,20))



# 2 -Countplot
row = 3
col = 1
count = 1
for i in numdata:
    plt.subplot(row,col,count)
    plt.title(i)
    sns.distplot(data[i])
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10,10))


# 3 - Boxplot Analysis
plt.figure(figsize = (15,8))
numdata.boxplot()
plt.title('Distribution of all Numeric Variables', fontsize = 15)
plt.xticks(rotation = 'vertical', fontsize = 15)
plt.show()

                           """Train Test Split"""
X = CONCAT.drop(['Loan Status'],axis =1)
Y = CONCAT['Loan Status']
xtrain, xtest, ytrain, ytest = train_test_split(X,Y,test_size=0.33)


                        'TARGET CLASS BALANCING - CLASSFICATION'
#Random Under Sampling
rus = RandomUnderSampler(return_indices =  True)
xresampled,yresampled,idxresampled  =  rus.fit_sample(X,Y)

#SMOTE
s=SMOTE()
imbal=smote.fit_transform(x,y)



                          'FEATURE SELECTION'
rf = RandomForestClassifier()
rf_backward = sfs(estimator = rf, k_features = 'best', forward = False,
                     verbose = 2, scoring = 'f1')
sfs_backward = rf_backward.fit(X,Y)
print('Features selelected using backward elimination are: ')
print(sfs_backward.k_feature_names_)
print('\nR-Squared: ', sfs_backward.k_score_)


                            "ENSEMBLE MODEL BUILDING"
#1-Boosting
r1 = lgb.LGBMClassifier(bagging_fraction=0.9, bagging_freq=3, boosting_type='gbdt',
               class_weight=None, colsample_bytree=1.0, feature_fraction=0.4,
               importance_type='split', learning_rate=0.001, max_depth=-1,
               min_child_samples=11, min_child_weight=0.001, min_split_gain=0,
               n_estimators=10, n_jobs=-1, num_leaves=60, objective=None,
               random_state=6122, reg_alpha=10, reg_lambda=0.4, silent=True,
               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

r2 = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                           learning_rate=0.05, loss='deviance', max_depth=3,
                           max_features=1.0, max_leaf_nodes=None,
                           min_impurity_decrease=0.01,
                           min_samples_leaf=3, min_samples_split=4,
                           min_weight_fraction_leaf=0.0, n_estimators=50,
                           n_iter_no_change=None,
                           random_state=6122, subsample=0.4, tol=0.0001,
                           validation_fraction=0.1, verbose=0,
                           warm_start=False)

r3 = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=0.0001,
                   n_estimators=50, random_state=6122)

model1 = VotingClassifier([('ada', r3), ('lgbm', r1),('gbc',r2)])
model1.fit(X,Y)
ypred=model1.predict(XTEST)
ypred = pd.DataFrame(ypred)
ypred.to_excel('F:/PROJECTS/Hackathon/Loan Defaulter Prediction/SUBMISSION/Boosting.xlsx')



