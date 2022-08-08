#reference: https://proceedings.neurips.cc/paper/4017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
#goss+efb

import sys
import time
import pprint
import numpy as np

import lightgbm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

def calculate_mape(y,predictions):
    diff=np.abs(y-predictions)
    return np.average(diff/y)

#minimise loop
def find_initial_regressor(x,y):
    return np.random.choice(y)

def find_best_regressor(x,y,weights):
    found_loss=None
    found_m=None
    found_p=None
    for p in np.arange(-1,1.1,0.1):
        m=DecisionTreeRegressor(max_depth=1)
        m.fit(x,y,sample_weight=weights)
        train_predictions=m.predict(x)
        current_loss=np.average((y-p*train_predictions)**2)
        if found_loss==None or current_loss<found_loss:
            found_loss=current_loss
            found_m=m
            found_p=p
    return m,found_p,found_loss

def goss(x,y,y_hats,a,b):
    sort_indices=np.argsort(np.abs(y_hats))
    l=int(a*len(y_hats))
    indices_small_gradients=np.arange(l,len(y))
    indices_small_gradients=np.random.choice(indices_small_gradients,int(b*len(y_hats)),replace=False)
    weights=np.ones(len(y_hats))
    weights[indices_small_gradients]=(1-a)/b
    goss_indices=np.concatenate((np.arange(0,l),indices_small_gradients))
    goss_x=x[goss_indices]
    goss_y=y[goss_indices]
    goss_y_hats=y_hats[goss_indices]
    weights=weights[goss_indices]
    return goss_x,goss_y,goss_y_hats,weights

def feature_conflicts(x,bundle_indices,d):
    x_features=x[:,bundle_indices]
    x_features=np.sum(x_features,axis=1)
    dot_products=x_features*x[:,d]
    conflicts=dot_products[dot_products!=0]
    return len(conflicts)

def efb(x,k):
    feature_bundles=[]
    current_feature_bundle=[]
    indices=np.arange(np.shape(x)[1])
    while len(indices)>0:
        ind_to_remove=[]
        for dim in indices:
            conflict=feature_conflicts(x,current_feature_bundle,dim)
            if conflict<k:
                current_feature_bundle.append(dim)
                ind_to_remove.append(np.argwhere(indices==dim))
        feature_bundles.append(current_feature_bundle)
        indices=np.delete(indices,ind_to_remove)
        current_feature_bundle=[]
    return feature_bundles

def merge_efb(x,feature_bundles,r):
    x=quantise_x(x,r)
    merged_x=np.empty([np.shape(x)[0],0])
    for feature_bundle in feature_bundles:
        current_x=x[:,feature_bundle]
        current_merged_x=np.zeros(np.shape(current_x)[0])
        for i in range(np.shape(current_x)[1]):
            current_merged_x=current_merged_x+current_x[:,i]+i*r
        current_merged_x=np.reshape(current_merged_x,(np.shape(current_merged_x)[0],1))
        merged_x=np.concatenate((merged_x,current_merged_x),axis=1)
    return merged_x

def quantise_x(x,r):
    max_values=np.repeat([np.max(x,axis=0)],np.shape(x)[0],axis=0)
    min_values=np.repeat([np.min(x,axis=0)],np.shape(x)[0],axis=0)
    bin_sizes=(max_values-min_values)/r
    x=np.floor((x-min_values)/bin_sizes)
    return x

def build_lgbm_classifier(x,y,num_stages):
    ps=[]
    lgbm_classifier=[]
    current_predictions=find_initial_regressor(x,y)
    ps.append(1)
    weights=np.ones(len(y))
    lgbm_classifier.append(current_predictions)
    for stage in range(num_stages):
        previous_predictions=batch_predict(lgbm_classifier,ps,x)
        y_hats=y-previous_predictions
        current_regressor,p_min,loss=find_best_regressor(x,y_hats,weights)
        ps.append(p_min)
        lgbm_classifier.append(current_regressor)
#        print('stage:',stage,'p:',p_min,'loss:',loss,'number of points:',len(weights))
        x,y,y_hats,weights=goss(x,y,y_hats,0.2,0.2)
        if len(y)==0:
            break
    return lgbm_classifier,ps

def batch_predict(lgbm_classifier,ps,x):
    predictions=np.repeat(lgbm_classifier[0],np.shape(x)[0])
    for m in range(1,len(lgbm_classifier)):
        predictions=predictions+ps[m]*lgbm_classifier[m].predict(x)
    return predictions

if __name__=='__main__':
    np.random.seed(0)

    #generate a dataset
    x=np.random.rand(80000,16)
    y=np.random.rand(80000)
    x[x<0.8]=0 #try to make it sparse manually to take advantage of efb

    no_samples=np.shape(x)[0]

    x_train=x[:40000,:]
    x_test=x[40000:,:]
    y_train=y[:40000]
    y_test=y[40000:]
    x_efb_train=x_train

    start_time=time.time()
    feature_bundles=efb(x_train,2000)
    x_efb_train=merge_efb(x_train,feature_bundles,16)
    lgbm_classifier,ps=build_lgbm_classifier(x_efb_train,y_train,16)
    end_time=time.time()
    print('training time:',end_time-start_time)

#    pprint.pprint(lgbm_classifier)
#    pprint.pprint(ps)

    train_predictions=batch_predict(lgbm_classifier,ps,x_efb_train)
    print('training mape',calculate_mape(y_train,train_predictions))

    print('******from the implementation in lightgbm******')
    start_time=time.time()
    m=lightgbm.LGBMRegressor()
    m.fit(x_train,y_train)
    end_time=time.time()
    print('time taken:',end_time-start_time)
    predictions=m.predict(x_train)
    print('mape:',calculate_mape(y_train,predictions))

    print('******from the implementation of gradient boosting in sklearn******')
    start_time=time.time()
    m=GradientBoostingRegressor()
    m.fit(x_train,y_train)
    end_time=time.time()
    print('time taken',end_time-start_time)
    predictions=m.predict(x_train)
    print('mape:',calculate_mape(y_train,predictions))

