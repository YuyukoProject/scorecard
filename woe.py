
# coding: utf-8

# In[11]:


import pandas as pd,numpy as np


# In[12]:


def event_dic(data,lables):#将数据按值根据离散类进行统计(为woe的计算做准备)
    val=np.unique(data)
    event_total = lables.sum()
    non_event_total = len(lables) - event_total
    eve_dic={}
    for i in val:
        event=lables[np.where(data==i)] 
        event_count=event.sum()
        non_event_count=len(event)-event_count
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        eve_dic[i]=[rate_event,rate_non_event]
    return eve_dic


# In[13]:


def woe_iv(data,lables,woe_min=-20,woe_max=20):#根据数据,计算woe与iv值
    woe_dic,iv_dic={},{}    
    for col in data.columns:
        woe_dic1={}
        iv=0
        col_dic=event_dic(data[col],lables)
        print('='*100)
        for k,(rate_event, rate_non_event) in col_dic.items():
            if rate_event == 0:
                woek = woe_min
            elif rate_non_event == 0:
                woek = woe_max
            else:
                woek = np.log(rate_event / rate_non_event)
            woe_dic1[k]=woek
            iv += (rate_event - rate_non_event) * woek
        woe_dic[col]=woe_dic1
        iv_dic[col]=iv
        col_table=pd.DataFrame([woe_dic1],index=['woe'])
        print(col_table.T)
    return woe_dic,iv_dic


# In[14]:


#woe 编码
def data_woe_encode(data,woe_dic):
    data_woe=pd.DataFrame(np.zeros(data.shape),columns=data.columns)
    for col in data.columns:
        col_dic=woe_dic[col]
        data_woe[col]=data[col].map(col_dic)
    return data_woe

