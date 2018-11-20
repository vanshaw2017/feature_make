import numpy as np
import pandas as pd
label_path="F:/DaiSEE/DAiSEE/DAiSEE/Labels/ValidationLabels.csv"
feature_path = "F:/DaiSEE/FacialFeature/Validation"
list_less=[]
list_more=[]
df = pd.read_csv(label_path)
video_list=[]
data1 = np.zeros((1,35)) #tatal data
for video in df['ClipID']:
    videoID = video.split(".")[0]
    video_list.append(videoID)
#print(video_list)

for index in video_list:#5358
    df = pd.read_csv(feature_path + '/' + index + '.csv')
    #print(feature_path + '/' + index + '.csv')
    rows=df.iloc[:,0].size
    #print(rows)
    if(rows==300):
        df = df.as_matrix()
        data1 = np.concatenate((data1, df), axis=0)

    elif(rows>300):
        df=df.drop(df.index[list(range(0,rows-300))])
        #print(df)
        #df.to_csv(feature_path + '/' + index + '.csv', index=0)
        #list_more.append(index)
        df = df.as_matrix()
        data1 = np.concatenate((data1, df), axis=0)
    else:
        #df = df.drop(df.index[list(range(rows, 300))])
        for i in range(rows,300):
            df = df.append(df.iloc[rows-1],ignore_index=True)
        #df.to_csv(feature_path + '/' + index + '.csv', index=0)
        #list_less.append(index)
        df = df.as_matrix()
        data1 = np.concatenate((data1, df), axis=0)
data1 = np.delete(data1, 0, axis=0)
print('data1_shape_before_reshape:',data1.shape)
data1 = data1.reshape((-1, 300, 35))#需要修改维度
print('data1_shape_after_reshape:', data1.shape)
np.save("F:/DaiSEE/Segment/Validation/facial_validation_data.npy", data1)


# print("list_more=",list_more)
# print("list_less=",list_less)
