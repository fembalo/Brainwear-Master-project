#script to create a learning curve
import matplotlib.pyplot as plt
import pandas as pd
import mlxtend
from mlxtend.plotting import plot_learning_curves
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



#split dataset into train and test
svm_ds=pd.read_csv('large_svm_dataset.csv',index_col=0)
#print(svm_ds.head())
x_var=svm_ds.drop(['interval(30s)','pure walking','volunteer'],axis=1)
#print(x_var.head())
y_var=svm_ds['pure walking']
x_train,x_test,y_train,y_test=train_test_split(x_var,y_var,test_size=0.3, random_state=0)

SVModel=SVC(kernel='rbf',C=10, gamma='scale')# model was tuned using k-fold gridsearch (see end of code)
plot_learning_curves(x_train, y_train, x_test, y_test, SVModel)
plt.show()
