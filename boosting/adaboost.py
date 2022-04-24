import pprint
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def calculate_loss(y,predictions,w):
    loss_array=w*(predictions!=y)
    loss=np.sum(loss_array)/np.sum(w)
    return loss

def update_weights(w,predictions,y,alpha):
    exponential_item=alpha*(predictions!=y)
    w=w*np.exp(exponential_item)
    w=w/np.sum(w)
    return w

def find_best_decision_stump(x,y,w):
    m=DecisionTreeClassifier(random_state=0,max_depth=1)
    m.fit(x,y,sample_weight=w)
    current_predictions=m.predict(x)
    loss=calculate_loss(y,current_predictions,w)
    alpha=np.log((1-loss)/loss)
    return m,current_predictions,loss,alpha

def build_adaboost_classifier(x,y,num_stages):
    alpha=[]
    adaboost_classifier=[]
    w=np.ones(np.shape(x)[0])/np.shape(x)[0]
    for stage in range(num_stages):
        current_decision_stump,current_predictions,current_loss,current_alpha=find_best_decision_stump(x,y,w)
        print('stage:',stage,'loss:',current_loss)
        alpha.append(current_alpha)
        w=update_weights(w,current_predictions,y,current_alpha)
        adaboost_classifier.append(current_decision_stump)
    return adaboost_classifier,alpha

def batch_predict(adaboost_classifier,alpha,x):
    predictions=[]
    for i in range(np.shape(x)[0]):
        current_x=x[i,:]
        current_prediction_positive=0
        current_prediction_negative=0
        for m in range(len(adaboost_classifier)):
            current_prediction=alpha[m]*adaboost_classifier[m].predict([current_x])
            if current_prediction>0:
                current_prediction_positive=current_prediction_positive+current_prediction
            else:
                current_prediction_negative=current_prediction_negative+current_prediction
        current_prediction_negative=np.abs(current_prediction_negative)
        if current_prediction_positive>current_prediction_negative:
            predictions.append(1)
        else:
            predictions.append(-1)
    return np.array(predictions)

if __name__=='__main__':
    np.random.seed(0)
    #generate a dataset
    x_train=np.floor(np.random.rand(5000,8)/0.5)*2
    x_test=np.floor(np.random.rand(5000,8)/0.5)*4
    x=np.concatenate((x_train,x_test),axis=0)

    y=np.concatenate((np.ones((5000,1)),np.zeros((5000,1))-1),axis=0)

    xy=np.concatenate((x,y),axis=1)
    np.random.shuffle(xy)
    x=xy[:,0:8]
    y=xy[:,8]

    x_train=x[:5000,:]
    x_test=x[5000:,:]
    y_train=y[:5000]
    y_test=y[5000:]

    adaboost_classifier,alpha=build_adaboost_classifier(x_train,y_train,20)

    pprint.pprint(adaboost_classifier)
    print(alpha)
    train_predictions=batch_predict(adaboost_classifier,alpha,x_train)
    print('training accuracy',len(train_predictions[train_predictions==y_train])/len(y_train))

    test_predictions=batch_predict(adaboost_classifier,alpha,x_test)
    print('testing accuracy',len(test_predictions[test_predictions==y_test])/len(y_test))

    train_stump_predictions=batch_predict([adaboost_classifier[0]],alpha,x_test)
    print('training accuracy (first stump only)',len(train_stump_predictions[train_stump_predictions==y_train])/len(y_train))

    stump_predictions=batch_predict([adaboost_classifier[0]],alpha,x_test)
    print('testing accuracy (first stump only)',len(stump_predictions[stump_predictions==y_test])/len(y_test))

