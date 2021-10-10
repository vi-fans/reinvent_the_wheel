import sys
import numpy as np
import matplotlib.pyplot as plt

#x: data, k: number of clusters
def basic_kmeans(x,k):
    centre_indices=np.random.choice(range(np.shape(x)[0]),k,replace=False)
    centres=x[centre_indices,:]
    previous_membership=np.zeros(np.shape(x)[0])-1
    membership_change=1
    while membership_change>0:
        membership=[]
        for i in range(0,np.shape(x)[0]):
            distances=np.sum(np.square(x[i,:]-centres),axis=1)
            membership.append(np.argmin(distances))
        membership=np.array(membership)
        membership_change=np.sum(membership!=previous_membership)
        previous_membership=membership
        print('current loss: '+str(membership_change))
        for i in range(0,k):
            if sum(membership==i)>0:
                centres[i,:]=np.average(x[membership==i],axis=0)
    return centres,membership

if __name__=='__main__':
    #generate random data for clustering, fix seed so that results are reproducible
    np.random.seed(0)
    number_of_points=int(sys.argv[1])
    dimensions=int(sys.argv[2])
    k=int(sys.argv[3])
    x=np.random.rand(number_of_points,dimensions)
    centres,membership=basic_kmeans(x,k)
    #simple visualisation
    if np.shape(x)[1]==2 and k==3:
        plt.figure(figsize=(10,10))
        for i in range(0,3):
            current_x=x[membership==i,:]
            if np.shape(current_x)[0]>0:
                plt.scatter(current_x[:,0],current_x[:,1],label='cluster '+str(i))
                plt.scatter(centres[i,0],centres[i,1],label='centre '+str(i),linewidths=6)
        plt.legend()
        plt.savefig('basic_kmeans.jpg')

