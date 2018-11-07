import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split

def get_data(feature_path,label_path):
    """
    :param feature_path:
    :param label_path:
    :return: train_data,train_label,test_data,test_label shape(-1,300,86)
    """
    video_list = []#video in trainlabel.csv
    engagement_label = []#the total labels
    # data1 = np.zeros((1,86)) #tatal data
    # data2 = np.zeros((1, 86))
    # data3 = np.zeros((1, 86))
    # data4 = np.zeros((1, 86))
    # data5 = np.zeros((1, 86))
    df = pd.read_csv(label_path)
    # for video in df['ClipID']:
    #     videoID = video.split(".")[0]
    #     video_list.append(videoID)
    for video_label in df['Engagement']:
        engagement_label.append(video_label)
    engagement_label = np.array(engagement_label)
    engagement_label=engagement_label[0:5000]
    # to get the whole data
    # for index in video_list[0:1000]:#5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data1=np.concatenate((data1,df),axis=0)
    # data1=np.delete(data1, 0, axis=0)
    # np.save("F:/DaiSEE/data1.npy",data1)
    # print(data1.shape)
    #
    # for index in video_list[1000:2000]:#5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data2=np.concatenate((data2,df),axis=0)
    # data2 = np.delete(data2, 0, axis=0)
    # np.save("F:/DaiSEE/data2.npy", data2)
    # print(data2.shape )

    # for index in video_list[2000:3000]:  # 5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data3 = np.concatenate((data3, df), axis=0)
    # data3 = np.delete(data3, 0, axis=0)
    # np.save("F:/DaiSEE/data3.npy", data3)
    # print(data3.shape)
    #
    # for index in video_list[3000:4000]:  # 5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data4 = np.concatenate((data4, df), axis=0)
    # data4 = np.delete(data4, 0, axis=0)
    # np.save("F:/DaiSEE/data4.npy", data4)
    # print(data4.shape)
    #
    # for index in video_list[4000:5000]:  # 5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data5 = np.concatenate((data5, df), axis=0)
    # data5 = np.delete(data5, 0, axis=0)
    # np.save("F:/DaiSEE/data5.npy", data5)
    # print(data5.shape)
    data1 = np.load("F:/DaiSEE/data1.npy")
    data2 = np.load("F:/DaiSEE/data2.npy")
    data3 = np.load("F:/DaiSEE/data3.npy")
    data4 = np.load("F:/DaiSEE/data4.npy")
    data5 = np.load("F:/DaiSEE/data5.npy")
    data=np.concatenate((data1,data2,data3,data4,data5))
    data=data.reshape((-1,300,86))



    #spilt the train set、validation set and test set ,the ratio is 6:2:2
    X_train_validation, X_test, y_train_validation, y_test = train_test_split(data, engagement_label, test_size=0.2,
                                                                  random_state=42,shuffle=True)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation,
                                                                    test_size=0.2,random_state=42, shuffle=True)
    return X_train,y_train,X_validation,y_validation,X_test,y_test


label_path="./TrainLabels.csv"
feature_path = "F:/DaiSEE/FinalFeature/Train"
X_train,y_train,X_validation,y_validation,X_test,y_test=get_data(feature_path,label_path)

print(X_train.shape,X_test.shape,X_validation.shape)
print(y_train.shape,y_test.shape,y_validation.shape)
np.save("F:/DaiSEE/X_train.npy", X_train)
np.save("F:/DaiSEE/X_test.npy", X_test)
np.save("F:/DaiSEE/X_validation.npy", X_validation)
np.save("F:/DaiSEE/y_train.npy", y_train)
np.save("F:/DaiSEE/y_test.npy", y_test)
np.save("F:/DaiSEE/y_validation.npy", y_validation)