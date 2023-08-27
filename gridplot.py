import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

#split dataset into train and test
svm_ds=pd.read_csv('large_svm_dataset.csv',index_col=0)
#print(svm_ds.head())
x_var=svm_ds.drop(['interval(30s)','pure walking','volunteer'],axis=1)
#print(x_var.head())
y_var=svm_ds['pure walking']
x_train,x_test,y_train,y_test=train_test_split(x_var,y_var,test_size=0.3, random_state=0) #random state allows reproducibility of split, dataset is split into 70% train set and 30% test set

params = {'C': [0.1, 1, 10, 100],
               'gamma': ['auto','scale']}

grid_search = GridSearchCV(SVC(random_state=0), params, cv=5,verbose=10)
grid_search.fit(x_train, y_train)

results_df = pd.DataFrame(grid_search.cv_results_)
scores = np.array(results_df.mean_test_score).reshape(4, 2)

sns.heatmap(scores, annot=True,
            xticklabels=params['gamma'], yticklabels=params['C'])#create a grid plot of the different parameters combination score


plt.show()
