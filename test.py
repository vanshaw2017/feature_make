import pandas as pd
import numpy as np
import os

face_index_array=[' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z',' gaze_angle_x',
                 ' gaze_angle_y',
             ' eye_0_distace',' eye_1_distace',' eye_lmk_Z_11',' eye_lmk_Z_17',' eye_lmk_Z_39',' eye_lmk_Z_45',
             ' pose_Tx',
             ' pose_Ty',' pose_Tz',' pose_Rx', ' pose_Ry', ' pose_Rz',' X_0',' X_4',' X_8',' X_12',' X_16',' Y_0',
             ' Y_4',' Y_8',' Y_12',' Y_16',' Z_0',' Z_4',' Z_8',' Z_12',' Z_16',' AU01_r', ' AU02_r', ' AU04_r',
             ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' U10_r', ' AU12_r', ' AU14_r',' AU15_r', ' AU17_r',
             ' AU20_r',
             ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r',' AU01_c', ' AU02_c', ' AU04_c',' AU05_c', ' AU06_c',
             ' AU07_c', ' AU09_c', ' AU10_c', ' AU12_c', ' AU14_c',' AU15_c', ' AU17_c', ' AU20_c', ' AU23_c',
             ' AU25_c', ' AU26_c',' AU28_c', ' AU45_c'
            ]

pose_index_array=['Nose_x','Nose_y','Neck_x','Neck_y','RShoulder_x','RShoulder_y','RElbow_x','RElbow_y','RWrist_x',
                  'RWrist_y','LShoulder_x','LShoulder_y','LElbow_x','LElbow_y','LWrist_x','LWrist_y']

def openface_feature_process(face_feature_input_path,pose_feature_input_path,face_index_array,pose_index_array,
                             final_feature_path):
    '''
    :param feature_input_path:
    the file path for the feature
    :return:
    '''
    for csv_file in os.listdir(face_feature_input_path):
        df_face = pd.read_csv(face_feature_input_path + '/' +csv_file)
        df_pose = pd.read_csv(pose_feature_input_path + '/' + csv_file)
        #合成新的眼睛特征
        df_face[' eye_0_distace'] = df_face[' eye_lmk_Y_17'] - df_face[' eye_lmk_Y_11']
        df_face[' eye_1_distace'] = df_face[' eye_lmk_Y_45'] - df_face[' eye_lmk_Y_39']
        new_face_index=df_face.reindex(columns=face_index_array,fill_value=0)
        final_face_df = new_face_index.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        new_pose_index = df_pose.reindex(columns=pose_index_array,fill_value=0)
        final_pose_df = new_pose_index.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        final_df=final_face_df.join(final_pose_df)
        final_df.fillna(0,inplace=True)
        final_df.to_csv(final_feature_path+'/'+csv_file,index=0)

face_feature_input_path="F:/DaiSEE/OpenFaceData/Validation"
pose_feature_input_path="F:/DaiSEE/AlphaPoseData/Validation"
final_feature_path="F:/DaiSEE/FinalFeature/Validation"
openface_feature_process(face_feature_input_path,pose_feature_input_path,face_index_array,pose_index_array,
                         final_feature_path)















