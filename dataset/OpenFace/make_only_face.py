import numpy as np
import pandas as pd
label_path="F:/DaiSEE/DAiSEE/DAiSEE/Labels/TrainLabels.csv"
feature_path = "F:/DaiSEE/OpenFaceData/Train"
write_path="F:/DaiSEE/onlyface/Train"
list_less=[]
list_more=[]
df = pd.read_csv(label_path)
video_list=[]
for video in df['ClipID']:
    videoID = video.split(".")[0]
    video_list.append(videoID)

for index in video_list:#5358
    df = pd.read_csv(feature_path + '/' + index + '.csv')
    #print(feature_path + '/' + index + '.csv')
    rows=df.iloc[:,0].size
    #print(rows)
    if(rows>300):
        df=df.drop(df.index[list(range(0,rows-300))])
        df.to_csv(write_path + '/' + index + '.csv', index=0)
        list_more.append(index)
    if(rows<300):
        #df = df.drop(df.index[list(range(rows, 300))])
        for i in range(rows,300):
            df = df.append(df.iloc[rows-1],ignore_index=True)
        df.to_csv(write_path + '/' + index + '.csv', index=0)
        list_less.append(index)
print("list_more=",list_more)
print("list_less=",list_less)