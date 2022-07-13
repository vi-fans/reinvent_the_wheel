#reference: http://luthuli.cs.uiuc.edu/~daf/courses/Opt-2017/Papers/2699986.pdf

import sys
import pprint
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

def calculate_mape(y,predictions):
    diff=np.abs(y-predictions)
    return np.average(diff/y)

def find_initial_regressor(x,y):
    mape=None
    initial_y=None
    for unique_y in set(y):
        current_predictions=np.repeat(unique_y,len(y))
        current_mape=calculate_mape(y,current_predictions)
        if mape==None or current_mape<mape:
            mape=current_mape
            initial_y=unique_y
    return unique_y

def find_best_regressor(x,y):
    found_loss=None
    found_m=None
    found_p=None
    for p in np.arange(-1,1.1,0.1):
        m=DecisionTreeRegressor(max_depth=1)
        m.fit(x,y)
        train_predictions=m.predict(x)
        current_loss=np.average((y-p*train_predictions)**2)
        if found_loss==None or current_loss<found_loss:
            found_loss=current_loss
            found_m=m
            found_p=p
    return m,found_p,found_loss

def build_gradient_boost_classifier(x,y,num_stages):
    ps=[]
    gradient_boost_classifier=[]
    current_predictions=find_initial_regressor(x,y)
    ps.append(1)
    gradient_boost_classifier.append(current_predictions)
    for stage in range(num_stages):
        previous_predictions=batch_predict(gradient_boost_classifier,ps,x)
        y_hats=y-previous_predictions
        current_regressor,p_min,loss=find_best_regressor(x,y_hats)
        ps.append(p_min)
        gradient_boost_classifier.append(current_regressor)
        print('stage:',stage,'p:',p_min,'loss:',loss)
    return gradient_boost_classifier,ps

def single_predict(gradient_boost_classifier,ps,x):
    return prediction

def batch_predict(gradient_boost_classifier,ps,x):
    predictions=np.repeat(gradient_boost_classifier[0],np.shape(x)[0])
    for m in range(1,len(gradient_boost_classifier)):
        predictions=predictions+ps[m]*gradient_boost_classifier[m].predict(x)
    return predictions

if __name__=='__main__':
    np.random.seed(0)

    #generate a dataset
    x_train=np.random.rand(1000,32)*2
    x_test=np.random.rand(1000,32)*4
    x=np.concatenate((x_train,x_test),axis=0)

    y=np.random.rand(2000,1)
    y[:1000]=y[:1000]*2
    y[1000:]=y[1000:]*4

    xy=np.concatenate((x,y),axis=1)
    np.random.shuffle(xy)
    x=xy[:,0:32]
    y=xy[:,32]

    x_train=x[:1000,:]
    x_test=x[1000:,:]
    y_train=y[:1000]
    y_test=y[1000:]

    gradient_boost_classifier,ps=build_gradient_boost_classifier(x_train,y_train,1024)

    pprint.pprint(gradient_boost_classifier)
    pprint.pprint(ps)

    train_predictions=batch_predict(gradient_boost_classifier,ps,x_train)
    plt.figure(figsize=(20,10))
    plt.plot(y_train[0:100],'r')
    plt.plot(train_predictions[0:100])
    plt.savefig(sys.argv[1])
    print('training mape',calculate_mape(y_train,train_predictions))

    #compare fitting with sklearn
    m=GradientBoostingRegressor()
    m.fit(x_train,y_train)
    predictions=m.predict(x_train)
    print('for comparison:',calculate_mape(y_train,predictions))

