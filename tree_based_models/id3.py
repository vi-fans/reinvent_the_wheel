import numpy as np

def calculate_probability(values):
    probabilities={}
    unique_values=list(set(values))
    for unique_value in unique_values:
        probabilities[str(unique_value)]=len(values[values==unique_value])/len(values)
    return probabilities

def calculate_entropy(values):
    probabilities=calculate_probability(values)
    entropy=0
    for probability_key in probabilities.keys():
        probability_value=probabilities[probability_key]
        entropy=entropy-1*probability_value*np.log(probability_value)
        entropy=entropy/np.log(np.exp(1))
    return entropy

def calculate_information_gain(x,y):
    information_gain=0
    unique_x_values=set(x)
    for unique_x_value in unique_x_values:
        sub_y=y[x==unique_x_value]
        sub_ratio=len(sub_y)/len(y)
        information_gain=information_gain+sub_ratio*calculate_entropy(sub_y)
    return information_gain

def create_decision_tree(x,y,dimension_level):
    unique_rows=np.unique(x,axis=0)
    #print(x,y)
    if np.shape(unique_rows)[0]<2 or dimension_level==0:
        return y[0]
    entropy=calculate_entropy(y)
    base_information_gain=-1
    selected_dimension=-1
    for i in range(np.shape(x)[1]):
        current_information_gain=entropy-calculate_information_gain(x[:,i],y)
        if base_information_gain<current_information_gain:
            base_information_gain=current_information_gain
            selected_dimension=i
    #print('entropy:',entropy,'information gain:',base_information_gain)
    #print('splitting based on dimension',selected_dimension)
    unique_x_values=set(x[:,selected_dimension])
    #sub-tree routine
    decision_tree={}
    for unique_x_value in unique_x_values:
        #print('unique_x_value:',unique_x_value,'out of',unique_x_values)
        decision_tree['dimension']=selected_dimension
        sub_mask=x[:,selected_dimension]==unique_x_value
        sub_x=x[sub_mask]
        sub_y=y[sub_mask]
        decision_sub_tree=create_decision_tree(sub_x,sub_y,dimension_level-1)
        decision_tree[str(unique_x_value)]=decision_sub_tree
    return decision_tree

def check_fit(x,y,decision_tree):
    matches=0
    for i in range(np.shape(x)[0]):
        current_x=x[i,:]
        decision_sub_tree=decision_tree
        for j in range(0,np.shape(x)[1]):
            if type(decision_sub_tree) is dict:
                current_dimension=int(decision_sub_tree['dimension'])
                decision_sub_tree=decision_sub_tree[str(x[i,current_dimension])]
            else:
                break
        prediction=decision_sub_tree
        if prediction==y[i]:
            matches=matches+1
    accuracy=matches/len(y)
    return accuracy

if __name__=='__main__':
    for i in range(10):
        #generate random dataset
        x=np.floor(np.random.rand(100,32)/0.2)
        y=np.floor(np.random.rand(100)/0.2)

        print(x,y)

        decision_tree=create_decision_tree(x,y,np.shape(x)[1])
        print('decision tree:',decision_tree)
        
        print('training accuracy:',check_fit(x,y,decision_tree))
    

