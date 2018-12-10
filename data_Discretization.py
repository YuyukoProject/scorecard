
# coding: utf-8

# In[6]:


'''
评分卡模型
==============
连续特征往往是非线性的,因此预测模型有两种思路

* 先做一些特征工程,通常工业上会将连续数据分段,把它转化为一组离散数据.
    (如何分段也就是这种思路下预测模型准确度的根本了) 通常采用的LR模型
* 使用深度学习,深度学习自带特征抽象能力.
本模型就是第一种思路的实践.
'''


# In[7]:


'''
难点:
    1.特征的选取,往往会有很多的特征,这就要求在众多的特征中选取相对高效的特征,防止最终的评分卡过于繁琐
    2.对连续型数据的处理会极大程度上影响后期评分卡的构建,常见的离散化处理有简单的分箱(等深、等宽),
    但这样的分箱会在极大程度依赖于对业务的了解,因此,此处选择利用算法的方式进行分箱(决策树)
    3.理算woe与iv值.
    4.构建模型并分类训练模型,常用的评分卡分类模型是LR模型(并不局限于此),此处决定选择LR模型
    5.构建评分卡并保存
'''


# In[8]:


import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import  auc,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import os
from itertools import  combinations


# In[9]:


# 数据的读取
def get_data(filename,path='F:/data/scorecard/',encodeing='utf-8',header=None):
    '''根据文件的名称与地址,导入数据,以datafr格式返回'''
    try:
        file_path=path+filename
        data=pd.read_csv(file_path,encoding=encodeing,header=header)
        return data
    except:
        print('文件导入错误了')


# In[146]:


#数据的保存
def save_data(data,filename,path='F:/data/scorecard/',encodeing='utf-8',header=None,index=False):
    '''
    根据文件的名称与地址,将数据结果以csv格式导出
    '''
    try:
        data=pd.DataFrame(data)
        file_path=path+filename
        data.to_csv(file_path,encoding=encodeing,header=header,index=index)
    except:
        return ('文件保存错误')


# In[11]:


#特征选取可采用的是逐步回归的策略,也可根据之后的iv值选取对应的字段(此处仅写出相应的逐步回归策略)
def get_grad_feature(data,target=-1):
    '''将数据按照逐步回归的方式,依次auc值,以此选择出相对重要的,默认数据的最后一列为target'''
    model=LinearRegression()
    col_tar=data.columns[target]
    cols=data.columns-col_tar
    cols_need=[]
    auc=0
    for col in cols:
        col_m=cols_need.append(col)
        data_choice=data[:,col_m]
        model.fit(data_choice,data[:,col_tar])
        auc_r=model.auc
        if auc_r>auc:
            auc=auc
            cols_need.append(col)
    return data[:,cols_need],cols_need


# In[122]:


def entropy(lable,n): #信息熵的计算方法
    p1=(np.sum(lable))/n
    p2=(len(lable)-p1)/n
    if p1==0 or p2==0:
        entropy=0
    else:
        entropy=-p1*np.log(p1)-p2*np.log(p2) #p1有可能为0,这也就意味着不会完美分类
    return entropy


# In[13]:


def gini(lable,n):#gini系数的计算方法
    p1=np.sum(lable)
    p2=len(lable)-p1
    gini_sco=(p1/n)**2+(p2/n)**2
    return gini_sco


# In[14]:


# # 计算节点分割后的信息熵
# def cac_entropy(data,lable,node=[]):
#     #数据分割(此时的节点数可能不止一个,因此需要进行切割)
#     node.sort
#     data_spilt=[]
#     entropy_total=0
#     n=len(lable)
#     data_spilt.append(lable[(data<node[0])])
#     for i in range(1,len(node)):#切割数据       
#         data_spilt.append(lable[(data>=node[i-1])&(data<node[i])])
#     data_spilt.append(lable[(data>=node[-1])])
#     #计算按节点分割数据之后的信息增益
#     for j in range(len(data_spilt)):
#         entropy_total+=entropy(data_spilt[i],n)
#     return entropy_total


# In[56]:


def cac_info(data,lable,criterion='gini',node=[]):
    node.sort
    data=np.array(data)
    data_spilt=[]
    total_info= 0
    n=len(lable)
    data_spilt.append(lable[(data<node[0])])
    for i in range(1,len(node)):#切割数据       
        data_spilt.append(lable[(data>=node[i-1])&(data<node[i])])
    data_spilt.append(lable[(data>=node[-1])])
    if criterion=='gini':
        for j in range(len(data_spilt)):
            total_info+=gini(data_spilt[j],n)
    else:
        for j in range(len(data_spilt)):
            total_info+=entropy(data_spilt[j],n)
    return total_info


# In[16]:


# #计算节点分割之后的gini系数
# def cac_gini(data,lable,node=[]):
#     node.sort
#     data_spilt=[]
#     gini_total=0
#     n=len(lable)
#     data_spilt.append(lable[(data<node[0])])
#     for i in range(1,len(node)):#切割数据       
#         data_spilt.append(lable[(data>=node[i-1])&(data<node[i])])
#     data_spilt.append(lable[(data>=node[-1])])
#     #计算按节点分割数据之后的gini不纯度
#     for j in range(len(data_spilt)):
#         gini_total+=gini(data_spilt[j],n)
#     gini_total=1-gini_total
#     return gini_total


# In[17]:


#寻找节点,为避免数据量较大,决定设定一个阈值,当值两个值之间的差距小于这个阈值,则不考虑节点在这两个值之间
def find_node(data,threshold=0.01):#
    n,node=len(data),[]
    data=np.sort(data)
    size,gap=data[-1]-data[0],data[1:]-data[:-1]
    if threshold:
        thres_hold=threshold
    else: 
        thres_hold=size/n
    node_can=data[:-1][gap>thres_hold]
    node=[]
    for _ in range(len(node_can)-1):
        node.append((node_can[_]+node_can[_+1])/2)
    return node      


# In[125]:


#利用决策树的思想找出各变量分箱的节点
def tree_nodes(data,labels,criterion='gini',n_th=[4,4,4,4],threshold=0.01):
    node_result=[]
    n_t=0
    for col in data.columns:
        nodes=find_node(data[col],threshold=threshold)
        node_res=[]
        for node in combinations(nodes,n_th[n_t]):
            node=list(node)
            score=cac_info(data[col],labels,criterion=criterion,node=node)
            node_res.append([score,node])  
        n_t+=1
        node_res.sort(key=lambda x:x[0],reverse=False)
        node_result.append(node_res[0][1])
    return node_result


# In[19]:


# def tree_nodes(data,labels,criterion='gini',n_th=None,threshold=0.01,score=0.2):
#     node_result=[]
#     if criterion=='gini':
#         for col in data.columns:
#             node=[]
#             nodes=find_node(data[col],threshold=threshold)
#             for _ in nodes:
#                 node.append(_)
#                 gini_total=cac_gini(data[col],labels,node)
#                 if gini_total>score:
#                     node.pop()
#             node_result.append(node)
#     else:
#         for col in data.columns:
#             nodes=find_node(data[col],n_th=n_th)
#             for _ in nodes:
#                 node.append(_)
#                 entropy_total=cac_entropy(data[col],labels,node)
#                 entropy_node=cac_entropy(data[col],labels,_)
#                 if gini_total>score:
#                     node.pop()
#             node_result.append(node)
#     return node_result


# In[20]:


#利用决策树的节点数据对数据进行离散化处理(可以采用Dataframe自带的cut,也可以用numpy)
def col_dis(data,col_name,bins):
    data_max,data_min=data.max(),data.min()
    bins.insert(0,data_min-0.1),bins.append(data_max )
    bins.sort()
    data=pd.DataFrame(data,columns=[col_name])
    data_dis=pd.cut(data[col_name],bins=bins)
    return data_dis


# In[129]:


# 将上述方法总结起来,确定数据的导入并直接生成最终的离散化结果(data输入为dataframe格式)
def discretization(data,labels,criterion='gini',n_th=[4,4,4,4],threshold=0.1,save=True):
    data_dis=pd.DataFrame(np.zeros(data.shape),columns=data.columns)
    i=0
    bins=tree_nodes(data,labels,criterion,n_th=n_th,threshold=threshold)
    for col in data.columns: #得出所有的字段的节点数据
        data_dis[col]=col_dis(data[col],col,bins[i])
        i+=1
    if save:
        save_data(data_dis,'discre',path='F:/data/scorecard/',encodeing='utf-8',header=None)
    return data_dis

