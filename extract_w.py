#Script that merges timeseries with raw data to extract walking for a large raw file that was split in chunks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1)create a timeseries dataframe and create a column labelling each row as a 30s epoch and a centisecond
V_timeSeries = pd.read_csv('MW_20181107_LEFT_100Hz-timeSeries.csv')
V_timeSeries.columns = ['acceleration',  'imputed', 'moderate', 'sedentary', 'sleep', 'tasks-light', 'walking', 'MET'] #re-label columns to make things easier
V_timeSeries['30s epochs'] = np.arange(len(V_timeSeries)) #label each row as an interval
V_timeSeries['centiseconds'] = V_timeSeries['30s epochs'] * 3000 #identify the centisecond of each interval by multiplying by 3000 (there are 3000 centiseconds in 30 seconds).


# 2) read each raw file and label each row with the respective centisecond
whole_rawfile=['MW_20181107_LEFT_100Hz1-raw.csv','MW_20181107_LEFT_100Hz2-raw.csv','MW_20181107_LEFT_100Hz3-raw.csv','MW_20181107_LEFT_100Hz4-raw.csv','MW_20181107_LEFT_100Hz5-raw.csv','MW_20181107_LEFT_100Hz6-raw.csv']

old_centiseconds=0
number_of_files=1
labelled_raw=[]
for file in whole_rawfile:
    V_raw = pd.read_csv(file)

    if number_of_files==1:
       V_raw['centiseconds'] = np.arange(len(V_raw))#label each centisecond  for first file
       #print(V_raw['centiseconds'].head())
       #print(V_raw['centiseconds'].tail())
    else:
        V_raw['centiseconds'] = np.arange(old_centiseconds,old_centiseconds+len(V_raw)) #label each centisecond for all files except the first

   #update old_centiseconds so the centiseconds are labeled from the last centisecond of the previous file
    old_centiseconds = len(V_raw)+old_centiseconds
    print(old_centiseconds)
    labelled_raw.append(V_raw)
    #V_raw.to_csv('labelled_raw' + str(number_of_files)+'.csv')
    number_of_files+=1

#4)merge each file with time series on centiseconds and extract walking
walking_extracted=[]
for file in labelled_raw :
    V_raw=pd.read_csv(file,index_col=0)
    V_raw['EN'] = np.linalg.norm(V_raw[['x','y','z']].values,axis=1) #calculate euclidean norm
    V_walk = V_raw.merge(V_timeSeries, on = 'centiseconds', how =
                   'outer')
    V_walk = V_walk.fillna(method='ffill') #where the dataframes have been merged, many rows will have NA, so the value is taken from the previous filled row and copied down. Now each centisecond is labelled with an activyt rather than only once ever 3000 centisecond.
    V_walk = V_walk.loc[(V_walk['centiseconds'] < len(V_raw))] #because we are merging one file at a time, to extract only the portion of the timeseries that matches the raw file
    V_walk = V_walk.loc[(V_walk['walking'] == 1) & (V_walk['imputed'] == 0) & (V_walk['moderate'] == 0) & (V_walk['sedentary'] == 0) & (V_walk['sleep'] == 0) & (V_walk['tasks-light'] == 0)] #select the walking from the timeseries
    V_walk = V_walk.drop(['acceleration', 'imputed', 'moderate', 'sedentary', 'sleep', 'tasks-light','MET'], axis=1)#drop the activities that arenot walking
    V_walk[['date','time']] = V_walk.time.str.split(expand=True)
    walking_extracted.append(V_raw)

#concatenate walking files into one
walking_extracted=pd.concat(walking_extracted,ignore_index=True)
#filtered_walking.to_csv('filtered_walking.csv')

print('walking extracted')

#5) Filter (moving average filter, window size 10)
n=0
N=len(walking_extracted)
x = np.linspace(0, n, N)
fig, ax = plt.subplots()
window_lst =[10]
y_avg = np.zeros((len(window_lst) , N))
for i, window in enumerate(window_lst):
    avg_mask = np.ones(window) / window
    y_avg = np.convolve(walking_extracted['EN'], avg_mask, 'same')
    ax.plot(y_avg, label=window)
ax.legend()
plt.savefig("filt_walk.png")
plt.close()
tmp = pd.DataFrame(y_avg)

#merge back with original dataframe
tmp.columns = ['EN_filt']
tmp['sample_no1'] = np.arange(len(tmp))
walking_extracted['sample_no1'] = np.arange(len(walking_extracted))
filtered_walking = walking_extracted.merge(tmp, on='sample_no1', how='outer')
filtered_walking.to_csv('filtered_walking.csv')
print('walking filtered')


# 7)creates graphs of 30s epochs (3000 centiseconds)
 chunk_size=3000

 for chunk in pd.read_csv('filtered_walking.csv', sep=',',iterator=True,chunksize=chunk_size):
     epoch= chunk.iloc[0]['30s epochs']
     plt.plot(chunk['EN_filt'])
     plt.title(epoch)
     plt.savefig('graphs/graph_epoch' + str(epoch) + '.png')
     plt.close()

print(done)
