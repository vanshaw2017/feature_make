import numpy as np
import pandas as pd
label_path="./TrainLabels.csv"
feature_path = "F:/DaiSEE/FinalFeature/Train"
df = pd.read_csv(label_path)
video_list=[]
for video in df['ClipID']:
    videoID = video.split(".")[0]
    video_list.append(videoID)

for index in video_list[2000:5001]:#5358
    df = pd.read_csv(feature_path + '/' + index + '.csv')
    #print(feature_path + '/' + index + '.csv')
    rows=df.iloc[:,0].size
    #print(rows)
    if(rows>300):
        range(0,rows-300)
        df=df.drop(df.index[list(range(0,rows-300))])
        df.to_csv(feature_path + '/' + index + '.csv', index=0)
    #print(df.iloc[:,0].size)