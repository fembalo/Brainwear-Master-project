#To create chunks of big raw files

import pandas as pd
chunk_size=33000000 #reads the file in chunks of 33M rows. This produces a total of 6 files.
chunkname=1

for chunk in pd.read_csv('MW_20181107_LEFT_100Hz-raw.csv', sep=',',iterator=True,chunksize=chunk_size):

    #create files for each chunk:note that the chuck as not in a dataframe format but as a textfilereader.
    chunk.to_csv('MW_20181107_LEFT_100Hz' +str(chunkname)+'-raw'+'.csv',index=False)
    chunkname+=1
