
# coding: utf-8

# In[ ]:


import data_Discretization as dd
import woe
import grid_search as gs


# In[ ]:


if __name__ == '__main__':
    data=dd.get_data('data')
    x=data[:,:-1]
    y=data[:,-1]
    data_dis=discretization(x,y,criterion='gini',n_th=[4,4,4,4],threshold=0.1,save=True)
    woe_dic,iv_dic=woe_iv(data,lables,woe_min=-20,woe_max=20)
    coe,const,ks=grid_serch_log(data,lables,param={'penalty':['l1','l2'],'C':[0.5,1.0,1.2]})
    score_data=get_score(coe,woe_dic,pdo=20,good_bad=[20,50])

