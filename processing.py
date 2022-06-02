import cv2
import os, sys
import numpy as np
#import tensorflow as tf


# from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization
# from tensorflow.keras.layers import UpSampling2D, Input, Concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.metrics import Recall, Precision
# from tensorflow.keras import backend as K 

def generate_outputs():
    
    # sys.setrecursionlimit(10**5)
    # # classification model 
    # modelc = pickle.load(open('classification_model.sav', 'rb'))
    # #segmentation model
    # IMAGE_SIZE = 256
    # smooth = 1e-15
    
    # #loss functions
    # def dice_coef(y_true, y_pred):
    #     y_true = tf.keras.layers.Flatten()(y_true)
    #     y_pred = tf.keras.layers.Flatten()(y_pred)
    #     intersection = tf.reduce_sum(y_true * y_pred)
    #     return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    # def dice_loss(y_true, y_pred):
    #     return 1.0 - dice_coef(y_true, y_pred)

    
    # # image processing
    # def read_image(path):
    #     path = path.decode()
    #     x = cv2.imread(path, cv2.IMREAD_COLOR)
    #     x = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    #     x = x/255.0
    #     return x

    # #loading the model
    # models = tf.keras.models.load_model('./segmentation_model_2-5-22', custom_objects={'dice_coef':dice_coef,'dice_loss':dice_loss})

    # #noise removal

    # def noise_rem(df):
    #     visited = [[0 for _ in range(len(df[0]))] for _ in range(len(df))]
    #     dp = [[0 for _ in range(len(df[0]))] for _ in range(len(df))]
    #     par = [[[i, j] for j in range(len(df[0]))] for i in range(len(df))]

    #     dx = [-1,1,-1,0,1]
    #     dy = [0,0,1,1,1]

    #     def dfs(x, y, scc):
    #         if visited[x][y]:
    #             return dp[x][y]
                
    #         visited[x][y] = scc

    #         for dir in range(5):
    #             u = x + dx[dir]
    #             v = y + dy[dir]
    #             if 0 <= u < len(df) and 0 <= v < len(df[0]):
    #                 if df[u][v] > 0:
    #                     new_length = (dx[dir]*dx[dir] + dy[dir]*dy[dir])**0.5 + dfs(u, v, scc)
    #                     if new_length > dp[x][y]:
    #                         dp[x][y] = new_length
    #                         par[x][y] = [u,v]

    #         return dp[x][y]

    #     G = 1
    #     for i in range(len(df)):
    #         for j in range(len(df[0])):
    #             if visited[i][j] == 0 and df[i][j] > 0:
    #                 dfs(i, j, G)
    #                 G += 1
                    

    #     DSU = {}
    #     for i in range(1, G):
    #         DSU[i] = 0

    #     for i in range(len(df)):
    #         for j in range(len(df[0])):
    #             if visited[i][j]:
    #                 DSU[visited[i][j]] += 1


    #     threshold = 5
    #     scc_to_remove = []
    #     for i in range(1, G):
    #         if DSU[i] <= threshold:
    #             scc_to_remove.append(i)

    #     for i in range(len(df)):
    #         for j in range(len(df[0])):
    #             if visited[i][j] in scc_to_remove:
    #                 df[i][j] = 0

    #     df = np.array(df,dtype = 'uint8')
    #     return df


    # #locations
    # sp ='shots'
    # dp1 = 'static/images/'
    # dp2 = 'static/results/'
    # num_images = len(os.listdir(sp))
    # props=[0]*num_images

    # #main code
    
    # for idx , file in enumerate(os.listdir(sp)):
    #     flag=1
    #     image = tf.keras.preprocessing.image.load_img(sp+'/'+file,target_size=(224,224))
    #     input_arr = tf.keras.preprocessing.image.img_to_array(image)
    #     input_arr = input_arr*(1./255)
    #     input_arr = np.array([input_arr])  # Convert single image to a batch.
    #     predictions = modelc.predict(input_arr)
    #     predictions=np.argmax(predictions, axis=1)
        
    #     if predictions==0:
    #         flag=0
    #         props[idx] = 'No Crack Detected'
            

    #     image = cv2.imread(sp+'/'+file)
    #     image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    #     cv2.imwrite(dp1+file,image)     
    #     image = read_image(sp+'/'+file)
    #     img = np.uint8(models.predict(np.expand_dims(image, axis=0))[0] > 0.5)   
    #     image = image*255
        
    #     A=0
    #     for i in range(256):
    #         for j in range(256):
    #             if img[i][j]>0:
    #                 image[i][j][0] = 0
    #                 image[i][j][1] = 0
    #                 image[i][j][2] = 150
    #                 A+=1
                    
    #     cv2.imwrite(dp2+file,image) 
         
    #     img = noise_rem(img.reshape(256,256).tolist())
    #     size = np.size(img)
    #     skel = np.zeros(img.shape,np.uint8)
    #     element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    #     done = False

    #     while( not done):
    #         eroded = cv2.erode(img,element)
    #         temp = cv2.dilate(eroded,element)
    #         temp = cv2.subtract(img,temp)
    #         skel = cv2.bitwise_or(skel,temp)
    #         img = eroded.copy()

    #         zeros = size - cv2.countNonZero(img)
    #         if zeros==size:
    #             done = True
    #     L=0
    #     for i in range(256):
    #         for j in range(256):
    #             if skel[i][j]>0:
    #                 L+=1
                    
    #     string = 'Length = {} pixels ; Width = {} pixels'.format(L,round(A/L,2))
    #     if flag:
    #         props[idx]=string

    sp ='shots'
    dp1 = 'static/images/'
    dp2 = 'static/results/'
    for idx , file in enumerate(os.listdir(sp)):
        img = cv2.imread(sp+'/'+file)
        cv2.imwrite(dp1+'/'+file,img)
        cv2.imwrite(dp2+'/'+file,img)
        
    return ['dss']*len(os.listdir(sp))
   
