# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 15:11:57 2021

@author: vmcx3
"""

import csv
import pandas as pd
import numpy as np
from scipy.stats import hmean
from scipy.special import boxcox, inv_boxcox
from scipy import stats
import random
from sklearn.cluster import KMeans
from pandas.plotting import scatter_matrix
from scipy.stats import norm
from matplotlib import pyplot as plt
import math
#import tensorflow as tf
import seaborn as sns
sns.set()
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import re
import os
import datetime
import ast
from statsmodels import robust
import networkx as nx
import kmeans1d

#import pickle5 as pickle

def count_values_in_range(series, range_min, range_max):

    # "between" returns a boolean Series equivalent to left <= series <= right.
    # NA values will be treated as False.
    return series.between(left=range_min, right=range_max).sum()

def GetKey(dict1,val):
    for key, value in dict1.items():
        for TMCid in value:
            if val == TMCid:
                return key
    return -1
def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)
 
def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta
        
        
#os.chdir('C:\\Users\\vmcx3\\Desktop\\Data sets\\Vehicular Datasets\\monthlydata')
#os.chdir('C:\\Users\\vmcx3\\Desktop\\Data sets\\Vehicular Datasets\\monthlydata')
#data_dir = 'C:\\Users\\vmcx3\\Desktop\\Data sets\\Vehicular Datasets\\monthlydata'
os.chdir('C:\\Users\\vmcx3\\Desktop\\Data sets\\Vehicular Datasets\\monthlydata')

#fp = os.path.join(data_dir,'G.pkl')
#G = nx.read_gpickle(fp)
G = nx.read_gml('new_G.gml')


#delete geometry attribute
'''
att_list = list(list(G.edges(data=True))[0][-1].keys()) #list of all attributes
for n1, n2, d in G.edges(data=True):
    if 'geometry' in d:
        del d['geometry']
'''

def find_adj_tmc(G, tmc):
#tmc = '13546+0.29498'
    edge_list = [(u,v) for u,v,att in G.edges(data=True) if att['tmc_id'] == tmc]
    
    source_nodes = list(set([x[0] for x in edge_list]))
    des_nodes = list(set([x[1] for x in edge_list]))
    
    
    in_tmc = []
    out_tmc = []
    for node in source_nodes:
        in_edges = G.in_edges(node)
        for e in in_edges:
            in_tmc.append([att['tmc_id'] for u,v,att in G.edges(data= True) if (u,v) == e][0])
    
    for node in des_nodes:
        out_edges = G.out_edges(node)
        for e in out_edges:
            out_tmc.append([att['tmc_id'] for u,v,att in G.edges(data= True) if (u,v) == e][0])
        
    return list(set(in_tmc)), list(set(out_tmc))

#example
#tmc = '13546+0.29498'
#in_tmc, out_tmc = find_adj_tmc(G, tmc)


with open('rsu_clusters.txt') as f:
    new_dict = f.read()
new_dict = ast.literal_eval(new_dict)
       
print('here1: ', datetime.datetime.now().time())



"""
train_df = pd.read_csv("aprildata.csv", header = 0, nrows = 6292442)

print('here1: ', datetime.datetime.now().time())

#train_df['tmc'] = train_df['tmc_id'].str[:5]
#train_df['new_tmc'] = train_df['tmc'].map(lambda x: re.sub(r'\-', '', x))

#train_df['new_tmc'] = train_df['new_tmc'].map(lambda x: re.sub(r'\+', '', x))

train_df['datetime_str'] = train_df['datetime'].str[:26] #upto microsec

train_df['datetime_ms'] =  pd.to_datetime(train_df['datetime'].str[:26], format='%Y-%m-%d %H:%M:%S.%f') #upto microsec
train_df['datetime_min'] = pd.to_datetime(train_df['datetime'].str[:16], format='%Y-%m-%d %H:%M') #upto min
train_df['datetime_sec'] = pd.to_datetime(train_df['datetime'].str[:19], format='%Y-%m-%d %H:%M:%S') #upto sec
print('here2: ', datetime.datetime.now().time())
"""



####################### new approach #########################
#date and time range
dt_7am = datetime.datetime.strptime("7:00:00","%H:%M:%S").time()
dt_9pm = datetime.datetime.strptime("21:00:00","%H:%M:%S").time()
start_date = datetime.date(2018, 4, 4)
end_date = datetime.date(2018, 4, 7)

#tmc_in_rsu = new_dict['rsu_32']

tmc_in_rsu =  []

#RSUs having >= 20 tmcs
rsu_list = ['rsu_15','rsu_16','rsu_17','rsu_18','rsu_19','rsu_20','rsu_23','rsu_24']
#rsu_list = ['rsu_2','rsu_4','rsu_5','rsu_6','rsu_8','rsu_9','rsu_11','rsu_15','rsu_16','rsu_17','rsu_18','rsu_19','rsu_20','rsu_23','rsu_24','rsu_26','rsu_29','rsu_30','rsu_31','rsu_32','rsu_33']
for rsu in rsu_list:
#for rsu in ['rsu_30', 'rsu_31','rsu_32']:
    tmc_in_rsu = tmc_in_rsu + new_dict[rsu]
    #tmc_in_rsu.append(new_dict[rsu])



dict_of_org_dfs = {}
dict_of_ratio_df = {}
dict_of_ratio = {}
daywise_tmc_mean ={}
for single_date in daterange(start_date, end_date):
    print(single_date.strftime("%Y-%m-%d"))
    df = pd.read_csv("DayWise/aprildata_" + str(single_date)+ ".csv", header = 0)
    df = df[df.tmc_id.isin(tmc_in_rsu)]
    
    #temp_df = df[['tmc_id', 'SP']].groupby(['tmc_id']).mean().reset_index()
    #tmc_in_sp = list(temp_df[temp_df.SP > 40]['tmc_id'])
    #df = df[df.tmc_id.isin(tmc_in_sp)]
    #del temp_df
    df['datetime_min'] = pd.to_datetime(df['datetime'].str[:16], format='%Y-%m-%d %H:%M') #upto min
    df['datetime_sec'] = pd.to_datetime(df['datetime'].str[:19], format='%Y-%m-%d %H:%M:%S') #upto sec
    df['datetime_time'] = df['datetime_sec'].dt.time
    df = df[(df.datetime_time >= dt_7am) & (df.datetime_time <= dt_9pm)]
    
    dict_of_org_dfs[single_date] = df[['datetime','datetime_min','datetime_sec','datetime_time', 'SP', 'tmc_id', 'CN']]
    
train_df = pd.concat(dict_of_org_dfs[single_date] for single_date in dict_of_org_dfs if single_date != datetime.date(2018, 4, 6))

## confidence checking
#train_df = train_df[train_df.CN >= .85]


#filter by SL
temp_df = train_df[['tmc_id', 'SP']].groupby(['tmc_id']).mean().reset_index()
tmc_in_sp = list(temp_df[(temp_df.SP >= 38) & (temp_df.SP < 50)]['tmc_id'])
train_df = train_df[train_df.tmc_id.isin(tmc_in_sp)]
del temp_df

test_df =  dict_of_org_dfs[datetime.date(2018, 4, 6)]
#test_df = test_df[(test_df.datetime_time >= datetime.datetime.strptime("7:00:00","%H:%M:%S").time()) & (test_df.datetime_time >= datetime.datetime.strptime("9:00:00","%H:%M:%S").time())]
test_df = test_df[test_df.tmc_id.isin(tmc_in_sp)]
test_df = test_df[(test_df.datetime_time >= datetime.datetime.strptime("10:00:00","%H:%M:%S").time()) & (test_df.datetime_time < datetime.datetime.strptime("11:00:00","%H:%M:%S").time())]
####################### new approach: end #########################

print('here2: ', datetime.datetime.now().time())


#dict_of_tmc_df = {}
#mean_std_info = pd.DataFrame(columns = ['tmc_id','count', 'mean', 'std'])

"""
dict_of_tmc_time_sep = {}
for tmc in set(train_df['new_tmc']):
    df = pd.DataFrame(columns = ['tmc_id','time_min', 'mean', 'std'])
    temp_df = train_df[train_df.tmc == tmc]
    for minute in set(temp_df['datetime_min']):    
        all_val = temp_df[(temp_df.datetime_min >= minute) & (temp_df.datetime_min <= minute + datetime.timedelta(minutes=15))]['SP']
        df.loc[len(df)] = [tmc, minute, all_val.mean(), all_val.std()]
    
    dict_of_tmc_time_sep[tmc] = df

print('here4: ', datetime.datetime.now().time())
"""
#tmc_in_rsu = new_dict['rsu_32']

    
#train_df1 = train_df[train_df.tmc_id.isin(tmc_in_rsu)]
train_df1 = train_df.copy()

"""
mean_std_whole = {}
for key in new_dict:
    mean_std_df = pd.DataFrame(columns = ['time','count', 'mean', 'std'])
    tmc_in_cluster = new_dict[key]
    temp_df = train_df1[train_df1.tmc_id.isin(tmc_in_cluster)]

    for minute in set(temp_df['datetime_min']): 
        all_val = temp_df[(temp_df.datetime_min >= minute) & (temp_df.datetime_min <= minute + datetime.timedelta(minutes=15))]['SP']
        mean_std_df.loc[len(mean_std_df)] = [minute, len(all_val), all_val.mean(), all_val.std()]
    
    mean_std_whole[key] = mean_std_df
"""

mean_std_whole = pd.DataFrame(columns = ['rsu', 'startpoint', 'endpoint','starttime', 'endtime', 'mean', 'std'])
for rsu in rsu_list:
    print('rsu: ', rsu, datetime.datetime.now().time())
    startpoint = datetime.datetime(2018, 4, 4, 7, 0, 0)
    count = 0
    tmc_in_rsu1 = new_dict[rsu]
    while startpoint < max(train_df1['datetime_sec']):
#        if count%20 == 0: 
#            print('count: ', count, 'at: ', datetime.datetime.now().time(), startpoint.date(), startpoint.time())
#        count +=1
        df_T = train_df1[(train_df1.tmc_id.isin(tmc_in_rsu1)) & (train_df1.datetime_sec >= startpoint) & (train_df1.datetime_sec < startpoint + datetime.timedelta(minutes = 15))]
        if len(df_T) > 0:
            mu_MR = df_T['SP'].mean()
            #sigma_MR = df_T['SP'].std()
            sigma_MR = robust.scale.mad(df_T['SP'], c=0.6744897501960817, axis=0)
        else:
            mu_MR = 0
            sigma_MR = 0
        
        #mean_std_whole.loc[len(mean_std_whole)] = [startpoint,startpoint + datetime.timedelta(minutes = 15),startpoint.strftime("%H:%M:%S"),(startpoint + datetime.timedelta(minutes = 15)).strftime("%H:%M:%S"),  mu_MR, sigma_MR]
        mean_std_whole.loc[len(mean_std_whole)] = [rsu, startpoint,startpoint + datetime.timedelta(minutes = 15),startpoint.time(),(startpoint + datetime.timedelta(minutes = 15)).time(),  mu_MR, sigma_MR]
        startpoint = startpoint + datetime.timedelta(minutes = 15)
        
        #if startpoint >= datetime.datetime(2018, 4, 1, 21, 0, 0):
        #    startpoint =  datetime.datetime(2018, 4, 2, 7, 0, 0)
        if (startpoint + datetime.timedelta(minutes = 15)).time() == datetime.time(21, 0):
            startpoint = startpoint + datetime.timedelta(minutes = 15) + datetime.timedelta(hours = 10)
    print(rsu)

mean_std_whole['mean'] = pd.to_numeric(mean_std_whole['mean'])
mean_std_whole['std'] = pd.to_numeric(mean_std_whole['std'])   
mean_std_avg = mean_std_whole.groupby(['rsu','starttime', 'endtime']).mean().reset_index()

print('here3: ', datetime.datetime.now().time())


deltaAvg = 13
false_df = test_df.copy()
false_df = false_df.reset_index()

rand_4 = pd.DataFrame(np.random.randint(0,5,size=(len(false_df), 1)), columns=['rand'])
 
false_df['SP'] = false_df['SP'] + deltaAvg + rand_4['rand']

final_df = test_df.groupby('tmc_id')['SP'].apply(list)

false_final_df = false_df.groupby('tmc_id')['SP'].apply(list)
print('here5: ', datetime.datetime.now().time())


datetime_min_df = test_df.groupby('tmc_id')['datetime_time'].apply(list)


frames = [final_df[:40][:], false_final_df[40:][:]]
#frames = [meterwise_original[:150][:], meterwise_falsified[150:][:]]
TMCwise = pd.concat(frames)

print('here6: ', datetime.datetime.now().time())

 
N = len(TMCwise)             #number of TMCs


temp = np.zeros(shape=(N,max([len(x) for x in TMCwise])))
theta = pd.DataFrame(temp)
l_original = pd.DataFrame(temp)
x = l_original.copy()
cw = l_original.copy()
w = l_original.copy()

tmc_info = np.zeros(shape=N)

for i in range(N):
    if GetKey(new_dict, TMCwise.index[i]) != -1:
        
        #rsu_info = GetKey(new_dict, TMCwise.index[i])
        #temp_mean_std = mean_std_whole[rsu_info]
        tmc_id = TMCwise.index[i]
        rsu = GetKey(new_dict, tmc_id)
        
        
        
        for j in range(len(TMCwise[i])):
            time_min = datetime_min_df[i][j]
            
            
            #theta.iloc[i,j] = abs(TMCwise[i][j] - temp_mean_std[temp_mean_std.time == date_min]['mean'].values[0])
            #std_dev = temp_mean_std[temp_mean_std.time == date_min]['std'].values[0]
            mean_val = mean_std_avg[(mean_std_avg.rsu == rsu) & (time_min >= mean_std_avg.starttime) & (time_min < mean_std_avg.endtime)]['mean'].values[0]
            mad_val = mean_std_avg[(mean_std_avg.rsu == rsu) & (time_min >= mean_std_avg.starttime) & (time_min < mean_std_avg.endtime)]['std'].values[0]
            theta.iloc[i,j] = abs(TMCwise[i][j] -mean_val)
            
            #print('std_dev: ', std_dev)
            if theta.iloc[i,j]<mad_val:
                l_original.iloc[i,j]=4
            elif theta.iloc[i,j]<2*mad_val:
                l_original.iloc[i,j]=3
            elif theta.iloc[i,j]<3*mad_val:
                l_original.iloc[i,j]=2
            else:
                l_original.iloc[i,j]=1
    
    else:
        tmc_info[i] = 1
                
            
print('here7: ', datetime.datetime.now().time())
l1_original = l_original.apply(np.sort, axis = 1)
 
K=4
 
for i in range(N):
    for j in range(len(TMCwise[i])):
        x.iloc[i,j] = 1 + ((K-1)*j)/len(TMCwise[i])
        
temp1 = np.zeros(shape=N)
std_dr = pd.DataFrame(temp1)
   
for i in range(N):
    std_dr.iloc[i] = np.std(l1_original[i])    
    
for i in range(len(std_dr)):
    if int(std_dr.iloc[i]) == 0:
       std_dr.iloc[i] = np.mean(std_dr) 
print('here8: ', datetime.datetime.now().time())
M_BR = 4

for i in range(N):
    for j in range(len(TMCwise[i])):
        cw.iloc[i,j] = (1/(math.sqrt(2*3.1415)*std_dr.iloc[i,0]))*(math.exp((-1*math.pow((x.iloc[i,j]-M_BR),2))/(2*math.pow(std_dr.iloc[i,0],2))))
        

for i in range(N):
    for j in range(len(TMCwise[i])):
        w.iloc[i,j] = cw.iloc[i,j]/np.sum(cw.iloc[i,:])
        
eeta = 2;
R = np.zeros(shape=N)
print('here9: ', datetime.datetime.now().time())

for TM in range(N):
    temp2 = np.zeros(shape=(4,len(TMCwise[TM])))
    I = pd.DataFrame(temp2)
    

    for j in range(len(TMCwise[TM])):
        if l_original.iloc[TM,j] == 1:
            I.iloc[0,j] = 1
        elif l_original.iloc[TM,j] == 2:
            I.iloc[1,j] = 1
        elif l_original.iloc[TM,j] == 3:
            I.iloc[2,j] = 1
        else:
            I.iloc[3,j] = 1
            
    temp3 = np.zeros(shape=4)
    wd = pd.DataFrame(temp3)
    
    for i in range(4):
        for j in range(len(TMCwise[TM])):
            wd.iloc[i,0] = wd.iloc[i,0] + I.iloc[i,j]*w.iloc[TM,j]
            
    for i in range(4):
        R[TM] = R[TM] + (i+1)*wd.iloc[i,0]

            
print('here10: ', datetime.datetime.now().time())

TR = np.zeros(shape=N)

for TM in range(N):
    TR[TM] = (1/math.pow(K,eeta))*(math.pow(R[TM],eeta))


xax = np.zeros(shape=N)

for i in range(N):
    xax[i] = i


#plt.hold(True)
fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
axes = plt.gca()
plt.hold(True)
for i in range(N):
    if i<37:
        plt.plot(xax[i],TR[i],'bo')
    elif i==37:
        plt.plot(xax[i],TR[i],'bo', label = "Non-Anomalous TMC")
    elif i==62:
        plt.plot(xax[i],TR[i],'r*', label = "Anomalous TMC")
    else:
        plt.plot(xax[i],TR[i],'r*')
        
plt.axhline(y=0.472, color='g', linestyle='-.')

plt.xlabel('TMC ID',fontsize = 20)
plt.ylabel('Trust Score',fontsize = 20)
plt.legend(fontsize = 20)
#plt.savefig('Deductive_deltaAvg15.eps')


Q = 24
Z1 = []
Z2 = random.sample(list([x for x in TR[1:30] if 0.9 > x > .55]),np.int(Q/2)) + random.sample(list([x for x in TR[30:]
 if .1 < x < .45]),np.int(Q/2))
k =2
clusters1, centroids1 = kmeans1d.cluster(Z2, k)
print(np.mean(centroids1), np.mean(Z2))
#TH = np.mean(Z2)
TH =np.mean(centroids1)
Slc = np.zeros(len(TR))
while (Z1 != Z2):
    for i in range(len(TR)):
        Slc[i] = abs(TH - TR[i])
    #print('Slc: ', sorted(Slc)) 
    #print('TR: ', TR)
    Z1 = Z2.copy()
    ind = sorted(range(len(Slc)), key=lambda k: Slc[k])
    for i in range(Q):
        Z2[i] = TR[ind[i]]


    #TH = np.mean(Z2)

    clusters, centroids = kmeans1d.cluster(Z2, k)
    TH = np.mean(centroids)
    print(np.mean(centroids))



cost = 0    
mis_clas = 0
for i in range(N):
    if i<30:
        if TR[i] < TH:
            mis_clas +=1
            cost = cost + abs(TH - TR[i])
    else:
        if TR[i] > TH:
            mis_clas +=1
            cost = cost + abs(TH - TR[i])

total_cost = mis_clas/N + cost/sum([abs(TH - i) for i in TR])
print('total_cost: ', total_cost)

'''    
Loss = 0    
for i in range(N):
    if i<30:
        if TR[i] > TH:
            Loss = Loss - abs(TH - TR[i])
        else:
            Loss = Loss + abs(TH - TR[i])
    else:
        if TR[i] < TH:
            Loss = Loss - abs(TH - TR[i])
        else:
            Loss = Loss + abs(TH - TR[i])
print(Loss)
'''


'''
tmc_series = pd.DataFrame(columns = ['tmc', 'count', 'avg_SP'])
tmc_list=[]
for tmc in tmc_in_sp[40:45]:
    startpoint = datetime.datetime(2018, 4, 4, 7, 0, 0)
    endpoint = datetime.datetime(2018, 4, 4, 21, 0, 0)
    tmc_list.append(tmc)
    while startpoint < endpoint:
        T_series = train_df1[(train_df1.tmc_id.isin(tmc_list)) & (train_df1.datetime_sec >= startpoint) & (train_df1.datetime_sec < startpoint + datetime.timedelta(minutes = 15))]
        avg_SP = np.mean(T_series.SP)
        #T_series_9 = train_df1[(train_df1.tmc_id.isin(tmc_list)) & (train_df1.datetime_sec >= startpoint) & (train_df1.datetime_sec < startpoint + datetime.timedelta(minutes = 15)) & (train_df1.CN>= 0.9)]
        #T_series_8 = train_df1[(train_df1.tmc_id.isin(tmc_list)) & (train_df1.datetime_sec >= startpoint) & (train_df1.datetime_sec < startpoint + datetime.timedelta(minutes = 15)) & (train_df1.CN>= 0.8) & (train_df1.CN < 0.9)]
        #T_series_7 = train_df1[(train_df1.tmc_id.isin(tmc_list)) & (train_df1.datetime_sec >= startpoint) & (train_df1.datetime_sec < startpoint + datetime.timedelta(minutes = 15)) & (train_df1.CN>= 0.7) & (train_df1.CN < 0.8)]
        
        
        if len(T_series) > 0:
            cnt = T_series['SP'].count()
        else:
            cnt=0
        tmc_series.loc[len(tmc_series)] = [tmc, cnt, avg_SP]
        startpoint = startpoint + datetime.timedelta(minutes = 15)
    tmc_list=[]
        
tmc_series_final = tmc_series.groupby('tmc')['avg_SP'].apply(list)

for i in range(len(tmc_series_final)):
    plt.plot(tmc_series_final[i],label = "TMC"+str(i+1))
    plt.xlabel('Timeslot (15 min)')
    plt.ylabel('Average Traffic Factor')
    plt.legend()
'''

'''
mean_std_whole['rownum'] = [x for x in range(len(mean_std_whole))]
mean_std_whole['rownum'] = mean_std_whole['rownum']/56 
mean_std_whole['rownum'] = mean_std_whole['rownum'].apply(math.floor) 
mean_std_whole['rownum'] = mean_std_whole['rownum'] + 1

fig = plt.figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
axes = plt.gca()      
for day in set(mean_std_whole['rownum']):
    #X = mean_std_whole[mean_std_whole.rownum == day]['starttime']
    X = [x for x in range(56)]
    Y = mean_std_whole[mean_std_whole.rownum == day]['mean']
    #plt.plot(X,Y, label =  'day' + str(day))    
    ax = Y.plot.kde()
#plt.plot(X, mean_std_avg['mean'], color = 'black', label = 'mean')
#ax = mean_std_avg['mean'].plot.kde()
ax.grid(True)
plt.ylabel( 'Values',fontsize=12)
plt.xlabel('avg speed (15 min window)',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Optimization of T')
plt.legend(fontsize=12, ncol = 2)
plt.show() 
'''

 
'''
fig = plt.figure(num=None, figsize=(8, 5), dpi=150, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111)
axes = plt.gca()  
Q = 20
Z2 = random.sample(list([x for x in TR[1:60] if .9 > x > .5]),np.int(Q/2))
Z3 = random.sample(list([x for x in TR[60:] if .1 < x < .5]),np.int(Q/2))
count1=0
count2=0
count3=0
plt.hold(True)
for i in range(N):
    if TR[i] in Z2:
        if count1<1:
            plt.plot(xax[i],TR[i],'bo', label = "Labeled Honest TMC")
            count1 += 1
        else:
            plt.plot(xax[i],TR[i],'bo')
    elif TR[i] in Z3:
        if count2<1:
            plt.plot(xax[i],TR[i],'r*', label = "Labeled Anomalous TMC")
            count2 += 1
        else:
            plt.plot(xax[i],TR[i],'r*')
    else:
        if count3<1:
            plt.plot(xax[i],TR[i],'g^', label = "Unlabeled TMC")
            count3 += 1
        else:
            plt.plot(xax[i],TR[i],'g^')
plt.xlabel('TMC ID', fontsize=20)
plt.ylabel('Trust Score',fontsize=20)
plt.xlim([0,120])
plt.legend(fontsize=12, loc='upper right', frameon = False)
plt.savefig('labels.eps')
'''