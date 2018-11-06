import pandas as pd
import numpy as np
import os
import math
index_array=[' gaze_0_x',' gaze_0_y',' gaze_0_z',' gaze_1_x',' gaze_1_y',' gaze_1_z',' gaze_angle_x',' gaze_angle_y',
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

#  def cal_distance(df):
#     df[' eye_0_distace'] = math.sqrt((df[' eye_lmk_X_11']-df[' eye_lmk_X_17'])**2+
#                                      (df[' eye_lmk_Y_11']-df[' eye_lmk_Y_17'])**2 )
#     df[' eye_1_distace'] = math.sqrt((df[' eye_lmk_X_39'] - df[' eye_lmk_X_45']) ** 2 +
#                                      (df[' eye_lmk_Y_39'] - df[' eye_lmk_Y_45']) ** 2)
#     return df

for csv_file in os.listdir("./dataset/OpenFace"):
    df=pd.read_csv("./dataset/OpenFace/"+csv_file)
    #合成新的眼睛特征
    df[' eye_0_distace'] = df[' eye_lmk_Y_17'] - df[' eye_lmk_Y_11']
    df[' eye_1_distace'] = df[' eye_lmk_Y_45'] - df[' eye_lmk_Y_39']
    new_index=df.reindex(columns=index_array)
    final_df = new_index.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    final_df.to_csv("./dataset/OpenFace/"+csv_file,index=0)














