
# coding: utf-8

# In[9]:


import numpy as np,pandas as pd
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import auc,roc_curve,classification_report,accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import scipy,pickle,math


# In[2]:


def compute_ks(proba,target): 
    get_ks = lambda proba, target: ks_2samp(proba[target == 1],proba[target != 1]).statistic
    return get_ks(proba, target)


# In[39]:


def grid_serch_log(data,lables,param={'penalty':['l1','l2'],'C':[0.5,1.0,1.2]}):
    clf=LogisticRegression(penalty='l2',C=1.0, fit_intercept=True,random_state=123)
    model=GridSearchCV(clf,param_grid=param,cv=5)
    model.fit(data,lables)
    model=par=model.best_estimator_
    model.fit(data,lables)
    proba_shape=model.predict_proba(data)
    prob=proba_shape.max(axis=1)
    coe=model.coef_ #逻辑回归的参数
    const=model.intercept_#逻辑回归的常数项
    coe_all=const.append(coe)
    ks=compute_ks(prob,lables) #模型最后的KS值
    fpr, tpr, thresholds = roc_curve(lables,prob)
    auc_z = auc(fpr, tpr)
    plt.plot(fpr, tpr, c = 'r', lw = 2)
    plt.show()
    return coe,const,ks


# In[10]:


def get_score(coe,woe_dic,pdo=20,good_bad=[20,50]): #PDO为20（每高20分好坏比翻一倍），好坏比取20。
    A = pdo / math.log(2)
    B =good_bad[1]+(p*math.log(1/good_bad[0]))
    base_score=round(q + p * coe[0], 0)
    data=pd.DataFrame(np.zeros((len(woe_dic))))
    data['base']=base_score
    for k,v in zip(woe_dic.items(),coe[1:]):
        for item ,woe in v.items():
            data['item']=b*woe*coe
    print(data.T)
    return data.T

