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
    data1 = np.zeros((1,86)) #tatal data
    # data2 = np.zeros((1, 86))
    # data3 = np.zeros((1, 86))
    # data4 = np.zeros((1, 86))
    # data5 = np.zeros((1, 86))
    df = pd.read_csv(label_path)
    for video in df['ClipID']:
        videoID = video.split(".")[0]
        video_list.append(videoID)
    for video_label in df['Engagement']:
        engagement_label.append(video_label)
    engagement_label = np.array(engagement_label)
    np.save("F:/DaiSEE/Segment/Train/train_labels.npy",engagement_label)
    # to get the whole data
    for index in video_list[5000:]:#5358
        df = pd.read_csv(feature_path + '/' + index + '.csv')
        df = df.as_matrix()
        data1=np.concatenate((data1,df),axis=0)
    data1=np.delete(data1, 0, axis=0)
    print(data1.shape)
    np.save("F:/DaiSEE/Segment/Train/data1.npy",data1)


    # for index in video_list[500:1000]:#5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data2=np.concatenate((data2,df),axis=0)
    # data2 = np.delete(data2, 0, axis=0)
    # np.save("F:/DaiSEE/Segment/Validation/data2.npy", data2)
    # print(data2.shape )
    #
    # for index in video_list[1000:]:  # 5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data3 = np.concatenate((data3, df), axis=0)
    # data3 = np.delete(data3, 0, axis=0)
    # np.save("F:/DaiSEE/Segment/Validation/data3.npy", data3)
    # print(data3.shape)

    # for index in video_list[1500:]:  # 5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data4 = np.concatenate((data4, df), axis=0)
    # data4 = np.delete(data4, 0, axis=0)
    # np.save("F:/DaiSEE/Segment/Test/data4.npy", data4)
    # print(data4.shape)

    # for index in video_list[4000:5000]:  # 5358
    #     df = pd.read_csv(feature_path + '/' + index + '.csv')
    #     df = df.as_matrix()
    #     data5 = np.concatenate((data5, df), axis=0)
    # data5 = np.delete(data5, 0, axis=0)
    # np.save("F:/DaiSEE/data5.npy", data5)
    # print(data5.shape)
    data1 = np.load("F:/DaiSEE/Segment/Train/data1.npy")
    data2 = np.load("F:/DaiSEE/Segment/Train/total_feature_5000.npy")
    # data2 = np.load("F:/DaiSEE/Segment/Validation/data2.npy")
    # data3 = np.load("F:/DaiSEE/Segment/Validation/data3.npy")
    #data4 = np.load("F:/DaiSEE/Segment/Validation/data4.npy")
    #data5 = np.load("F:/DaiSEE/data5.npy")
    data1 = data1.reshape((-1, 300, 86))
    data2 = data2.reshape((-1, 300, 86))
    data=np.concatenate((data1,data2))
    data=data.reshape((-1,300,86))
    np.save("F:/DaiSEE/Segment/Train/train_data.npy", data)




    #spilt the train set、validation set and test set ,the ratio is 6:2:2
    # X_train, X_test, y_train, y_test = train_test_split(data, engagement_label, test_size=0.2,
    #                                                               random_state=42,shuffle=True)
    # #X_train, X_validation, y_train, y_validation = train_test_split(X_train_validation, y_train_validation,
    #                                                                 #test_size=0.2,random_state=42, shuffle=True)
    # return X_train,y_train,X_test,y_test



label_path="F:/DaiSEE/DAiSEE/DAiSEE/Labels/TrainLabels.csv"
feature_path = "F:/DaiSEE/FinalFeature/Train"
get_data(feature_path,label_path)


# np.save("F:/DaiSEE/X_train.npy", X_train)
# np.save("F:/DaiSEE/X_test.npy", X_test)
# #np.save("F:/DaiSEE/X_validation.npy", X_validation)
# np.save("F:/DaiSEE/y_train.npy", y_train)
# np.save("F:/DaiSEE/y_test.npy", y_test)
# #np.save("F:/DaiSEE/y_validation.npy", y_validation)
