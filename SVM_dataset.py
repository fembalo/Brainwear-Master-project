#pipeline to produce a labelled dataset for SVM model
#1:create a folder for all your extracted walking files
#2: if re-run the newly created file needs to be removed from folder

import pandas as pd
import matplotlib.pyplot as plt
import numpy  as np
from os import listdir
from os.path import isfile, join
import os

def listdir_nohidden(mypath):
    for f in os.listdir(mypath):
        if not f.startswith('.'):
            yield f #to not include hidden files present in the folder

mypath = "./walk_vol/"
files = [f for f in listdir_nohidden(mypath) if isfile(join(mypath, f))] #make list of all files in the folder

os.chdir(mypath)


chunk_size=3000 # 3000 rows represent one interval of 30s
label=[]
walk_chunks=[]
name_tag=[]
df=[]

#read each walk extracted file and create graphs for each epoch
for file in files:
    #print(files)
    count=1
    for chunk in pd.read_csv(file, sep=',',iterator=True,chunksize=chunk_size,index_col=0):
        #print(count)
        if count<=1000:#number of graphs we want to label for each file to produce our labelled dataset
            initial=file.split("_")[0]
            name_tag.append(initial)
            epoch= chunk.iloc[0]['interval(30s)']
            plt.plot(chunk['EN_filt'])
            plt.title(initial + '_' + str(epoch))
            plt.ylim((0,8))
            plt.yticks(np.arange(0,8, 0.5))
            plt.axhline(y=1.04, color='r', linestyle='-')# walking threahold line calculated as the mean of light tasks
            plt.show(block=False)
            count+=1

        else:
            break

#label each graph as walking (1) or non-walking (0)
        while True:
            try:
                input_user =int(input("mixed or pure walking (0 or 1) ?"))
                break

            except ValueError:
                 print("Sorry, invalid input try again.")#
                 continue
            break
        label.append(input_user)
        walk_chunks.append(chunk)
        plt.savefig('graphs/graph_epoch-' + str(epoch) + '_'+ initial + '.png')
        plt.close()

#reshape the dataframe so that the EN filtered for each epoch (interval 30s) is in columns and not rows i.e 3000 columns for EN plus 1 column for interval(30s)
df=[]
for n in walk_chunks:
    dl=pd.DataFrame(n)
    dl=dl[['interval(30s)','EN_filt']]
    dl = dl.set_index(['interval(30s)',dl.groupby('interval(30s)').cumcount()])['EN_filt'].unstack()
    dl = dl.reset_index()#otherwise the index will be interval (30s)
    df.append(dl)

#create one dataframe by concatenating all epochs
dataset=pd.concat(df,axis=0,ignore_index=True)
labelled_walk=pd.DataFrame(label)
name_tag=pd.DataFrame(name_tag)
dataset=pd.concat([dataset,labelled_walk], axis=1, ignore_index=True)#concate features with labels
dataset.columns = [*dataset.columns[:-1], 'pure walking']
dataset.rename(columns={dataset.columns[0]: 'interval(30s)'},inplace=True)
switch = dataset.pop('interval(30s)')
dataset['interval(30s)']=switch #move interval (30s) to end of dataframe to keep it neat
dataset['Volunteer']=name_tag#add a column with volunteer initials to dataframe
dataset.to_csv('dataset_svm.csv')
print('done')
