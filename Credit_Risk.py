#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# In[2]:


pd.set_option('display.max_columns' , None)


# In[3]:


df = pd.read_csv(r'C:\Users\Farhad.lotfi\Desktop\dataset\Credit Risk\train.csv')
df


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isna().sum()


# In[7]:


df['loan_amnt'].dtypes


# In[8]:


dict(df.dtypes)
object_columns = []
for i in df.columns:
    if df[i].dtypes == 'object':
        object_columns.append(i)
object_columns


# In[9]:


df[object_columns].head(20)


# In[10]:


import seaborn as sns

object_columns.append('bad_loans')

sns.catplot(x="home_ownership", kind="count", palette="ch:.25", data=df)


# In[11]:


for i in object_columns :
    print(i , ':::' , dict(df[i].value_counts()))


# In[ ]:





# In[12]:


df = pd.concat([df,pd.get_dummies(df['grade'],drop_first=True)], axis=1)
df = pd.concat([df,pd.get_dummies(df['home_ownership'],drop_first=True)], axis=1)
df = pd.concat([df,pd.get_dummies(df['purpose'],drop_first=True)], axis=1)


# In[13]:


df.drop(['home_ownership','pymnt_plan','grade','purpose'] , axis = 1 , inplace = True)


# In[14]:


df


# for i in df.columns:
#     plt.figure()
#     sns.distplot(df[i],hist=True)
#     plt.xlabel(i)

# In[15]:


boxplot_columns = ['loan_amnt','funded_amnt','emp_length_num','payment_inc_ratio','open_acc']
normal_columns = ['dti' , 'revol_util']


# In[16]:


def outlier_normal(x):
    return df[((df[x] - df[x].mean() ) / df[x].std() ).abs() < 3]


# In[17]:


def outlier_boxplot(x):
    q1 = df[x].quantile(0.25)
    q3 = df[x].quantile(0.75)
    IQR = q3 - q1
    return df[~ ( (df[x] < (q1-1.8*IQR) ) | ((df[x] > (q3+1.8*IQR)) ))]


# In[18]:


outlier_normal('revol_util')


# In[19]:


df.shape


# for i in normal_columns:
#     df = outlier_normal(i)

# In[20]:


df.shape


# for i in boxplot_columns:
#     df = outlier_boxplot(i)

# In[21]:


df.shape


# In[22]:


df.isna().sum()


# In[23]:


na_columns = ['delinq_2yrs' , 'delinq_2yrs_zero' , 'inq_last_6mths' , 'open_acc' , 'pub_rec' ,'pub_rec_zero' ,'payment_inc_ratio']
              
for i in na_columns :
    plt.figure()
    plt.hist(df[i])
    plt.xlabel(i)
    print(i , ' mean : ', df[i].mean())
    print(i , ' median : ', df[i].median())
    print(i , ' min : ', df[i].min())
    print(i , ' max : ', df[i].max())


# In[24]:


df['delinq_2yrs'] = df['delinq_2yrs'].fillna(df['delinq_2yrs'].median())
df['delinq_2yrs_zero'] = df['delinq_2yrs_zero'].fillna(df['delinq_2yrs_zero'].median())
df['inq_last_6mths'] = df['inq_last_6mths'].fillna(df['inq_last_6mths'].median())
df['open_acc'] = df['open_acc'].fillna(df['open_acc'].median())
df['pub_rec'] = df['pub_rec'].fillna(df['pub_rec'].median())
df['pub_rec_zero'] = df['pub_rec_zero'].fillna(df['pub_rec_zero'].median())
df['payment_inc_ratio'] = df['payment_inc_ratio'].fillna(df['payment_inc_ratio'].median())


# In[25]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
MinMaxScaler = MinMaxScaler()
StandardScaler = StandardScaler()

df = pd.DataFrame(MinMaxScaler.fit_transform(df), columns = df.columns)
df.head()


# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)
# df

# In[26]:


df.isna().sum()


# In[27]:


for i in df.columns:
    plt.figure()
    plt.hist(df[i])
    plt.xlabel(i)


# df.dtypes
# 
# for i in ['bad_loans','B','C','D','E','F','G','OTHER','OWN','RENT']:
#     df[i] = df[i].astype(bool)

# In[28]:


df.dtypes


# In[29]:


x = df.drop(['bad_loans'] , axis = 1)
y = pd.DataFrame(df['bad_loans'])


# In[30]:


x


# In[31]:


y


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


xtrain , xtest , ytrain , ytest = train_test_split(x,y,test_size = 0.2 , random_state = 600)


# In[34]:


xtrain.shape


# In[35]:


xtest.shape


# In[36]:


ytrain.shape


# In[37]:


ytest.shape


# In[38]:


from xgboost import XGBClassifier


# In[39]:


model = XGBClassifier()
model.fit(xtrain,ytrain)


# In[40]:


pred = model.predict(xtest)


# In[41]:


model.score(xtrain,ytrain)


# In[42]:


model.score(xtest,ytest)


# In[43]:


from sklearn.metrics import mean_squared_error , mean_absolute_error , confusion_matrix


# In[44]:


mae = mean_absolute_error(ytest, pred)
mae


# In[45]:


confusion_matrix(ytest, pred)


# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report

clf = DecisionTreeClassifier(criterion='gini', max_depth=7, min_samples_split=15, min_samples_leaf=10,
                            class_weight={0:0.25 , 1:0.90}, random_state= 2)


pred = cross_val_predict(clf, x, y, cv=25)

print(accuracy_score(y, pred))
print(confusion_matrix(y, pred))
print(f1_score(y, pred))


# In[47]:


clf.fit(x,y)


# In[ ]:





# In[48]:


df_test = pd.read_csv(r'C:\Users\Farhad.lotfi\Desktop\dataset\Credit Risk\test.csv')
df_test


# In[49]:


df_test = pd.concat([df_test,pd.get_dummies(df_test['grade'],drop_first=True)], axis=1)
df_test = pd.concat([df_test,pd.get_dummies(df_test['home_ownership'],drop_first=True)], axis=1)
df_test = pd.concat([df_test,pd.get_dummies(df_test['purpose'],drop_first=True)], axis=1)
df_test


# In[50]:


df_test.drop(['home_ownership','pymnt_plan','grade','purpose'] , axis = 1 , inplace = True)


# In[51]:


from sklearn.preprocessing import MinMaxScaler,StandardScaler
MinMaxScaler = MinMaxScaler()
StandardScaler = StandardScaler()

df_test = pd.DataFrame(MinMaxScaler.fit_transform(df_test), columns = df_test.columns)




# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# df_test = pd.DataFrame(imputer.fit_transform(df_test),columns = df_test.columns)
# 
# df_test

# In[52]:


df_test['delinq_2yrs'] = df_test['delinq_2yrs'].fillna(df_test['delinq_2yrs'].median())
df_test['delinq_2yrs_zero'] = df_test['delinq_2yrs_zero'].fillna(df_test['delinq_2yrs_zero'].median())
df_test['inq_last_6mths'] = df_test['inq_last_6mths'].fillna(df_test['inq_last_6mths'].median())
df_test['open_acc'] = df_test['open_acc'].fillna(df_test['open_acc'].median())
df_test['pub_rec'] = df_test['pub_rec'].fillna(df_test['pub_rec'].median())
df_test['pub_rec_zero'] = df_test['pub_rec_zero'].fillna(df_test['pub_rec_zero'].median())
df_test['payment_inc_ratio'] = df_test['payment_inc_ratio'].fillna(df_test['payment_inc_ratio'].median())


# In[53]:


df_test.isna().sum()


# In[54]:


df_test.dtypes


# In[55]:


pred = clf.predict(df_test)
pred = pd.DataFrame(pred)
pred


# In[56]:


pred.to_csv(r'C:\Users\Farhad.lotfi\Desktop\dataset\credit_risk3.csv',index = False, header = None)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




