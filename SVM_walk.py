# script to produce SVM model
#find best parameters for SVModel by de-commenting the GridSearchCV found at end of code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import itertools


#split dataset into train and test
svm_ds=pd.read_csv('large_svm_dataset.csv',index_col=0)# create a dataframe of the labelled dataset file
#print(svm_ds.head())
x_var=svm_ds.drop(['interval(30s)','pure walking','volunteer'],axis=1)# Leave only the EN filtered rows columns as features
#print(x_var.head())
y_var=svm_ds['pure walking']# select pure walking as our label (target)
x_train,x_test,y_train,y_test=train_test_split(x_var,y_var,test_size=0.3, random_state=0) #random state allows reproducibility of split, dataset is split into 70% train set and 30% test set
#print(len(x_train))
#print(len(y_train))
#print(len(x_test))


#fit the model
SVModel=SVC(kernel='rbf',C=10, gamma='scale')# best parameters were found using 5-fold gridsearch (see end of code)
SVModel.fit(x_train,y_train)

#accuracy_score for trained ser
print(confusion_matrix(y_train,SVModel.predict(x_train)))
print(accuracy_score(y_train,SVModel.predict(x_train)))

#test the model on test set and get metrics
y_pred=SVModel.predict(x_test)

#print('accuracy:' + str(accuracy_score(y_test,y_pred)))
#print('Precision Score : ' + str(precision_score(y_test,y_pred)))
#print('Recall Score : ' + str(recall_score(y_test,y_pred)))
#print('F1 Score : ' + str(f1_score(y_test,y_pred)))
cf=confusion_matrix(y_test,y_pred)
print('Confusion Matrix : \n' + str(confusion_matrix(y_test,y_pred)))
print('Classification report:\n', classification_report(y_test,y_pred))


#create a plot of confusion Matrix
plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
tick_marks = np.arange(len(set(y_test))) # length of classes
class_labels = ['0','1']
plt.xticks(tick_marks,class_labels)
plt.yticks(tick_marks,class_labels)
thresh = cf.max() / 2. #plotting text value inside cells
for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
    plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
plt.show()



#Finding best parameters by looping(testing version, use gridsearch below to find best parameters)

#krn=['linear','poly','rbf','sigmoid']
#rng_C=np.arange(1,52,10)
#rng_deg=np.arange(3,8)
#rng_co=np.arange(0.001,10,0.5)
#rng_gam=['auto','scale']


#best_score=0
#for i in krn:
    #for j in rng_C:
        #for k in rng_deg:
            #for z in rng_co:
                #for x in rng_gam:
                    #SVModel=SVC(kernel=i,C=j,degree=k,coef0=z,gamma=x)
                    #SVModel.fit(x_train,y_train)
                    #acc_score=accuracy_score(y_test,SVModel.predict(x_test))
                    #if best_score<acc_score:
                        #best_score=acc_score
                        #bi=i
                        #bj=j
                        #bk=k
                        #bz=z
                        #bx=x
#print(best_score,bi,bj,bk,bz,bx)

#Finding best parameters by 5-fold GridSearchCV:
#param={'kernel':('linear','poly','rbf','sigmoid'),'C':[1,52,10], 'degree':[3,8],'coef0':[0.001,10,0.5],'gamma':('auto','scale')}
#SVModel=SVC()
#GridS=GridSearchCV(SVModel,param,cv=5)
#GridS.fit(x_train,y_train)#find best parameters on train dataset
#print(GridS.best_params_)
