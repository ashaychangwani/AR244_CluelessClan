# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:21:16 2020

@author: Ashay
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from PIL import Image
import cv2


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from segmentation_models import get_preprocessing



json_file = open("content/drive/My Drive/severstal_february/severstal_model/model_binary.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_binary = model_from_json(loaded_model_json)
model_binary.load_weights("content/drive/My Drive/severstal_february/severstal_model/model_binary.h5")




json_file = open("content/drive/My Drive/severstal_february/severstal_model/model_multi.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_multi = model_from_json(loaded_model_json)
model_multi.load_weights("content/drive/My Drive/severstal_february/severstal_model/model_multi.h5")




json_file = open("content/drive/My Drive/severstal_february/severstal_model/model_segment_1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_segment_1 = model_from_json(loaded_model_json)
model_segment_1.load_weights("content/drive/My Drive/severstal_february/severstal_model/model_segment_1.h5")




json_file = open("content/drive/My Drive/severstal_february/severstal_model/model_segment_2.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_segment_2 = model_from_json(loaded_model_json)
model_segment_2.load_weights("content/drive/My Drive/severstal_february/severstal_model/model_segment_2.h5")




json_file = open("content/drive/My Drive/severstal_february/severstal_model/model_segment_3.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_segment_3 = model_from_json(loaded_model_json)
model_segment_3.load_weights("content/drive/My Drive/severstal_february/severstal_model/model_segment_3.h5")




json_file = open("content/drive/My Drive/severstal_february/severstal_model/model_segment_4.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model_segment_4 = model_from_json(loaded_model_json)
model_segment_4.load_weights("content/drive/My Drive/severstal_february/severstal_model/model_segment_4.h5")


preprocess = get_preprocessing('efficientnetb1')







class test_DataGenerator_3(keras.utils.Sequence):
    def __init__(self, df, batch_size = 1, image_path = 'content/raw_data/',
                 preprocess=None, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.preprocess = preprocess
        self.info = info
        self.data_path = image_path
        self.on_epoch_end()
    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
    def __getitem__(self, index): 
        '''
        The DataGenerator takes ImageIds of batch size 1 and returns Image array to the model.
        With the help of ImageIds the DataGenerator locates the Image file in the path, the image is read and resized from
        256x1600 to 256x800.
        '''
        X = np.empty((self.batch_size,256,800,3),dtype=np.float32)
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['ImageId'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = Image.open(self.data_path + f).resize((800,256))      
        if self.preprocess!=None: X = self.preprocess(X)
        return X

def area(i):
    '''
    Input: EncodedPixels (str)
    Output: number of pixels having the defect
    '''
    return sum([int(k) for k in i.split(' ')[1::2]])


def pred_classification(X):
    '''
    Input: ImageIds in form of a dataframe
    Return: Predictions of classification models
    '''
    data_generator = ImageDataGenerator(rescale=1./255).flow_from_dataframe(dataframe=X, directory='content/raw_data/', 
                                                                            x_col="ImageId", class_mode = None, 
                                                                            target_size=(256,512), batch_size=1, shuffle=False)

    data_preds_binary = model_binary.predict_generator(data_generator,verbose=0)
    data_preds_multi_label = model_multi.predict_generator(data_generator,verbose=0)
    data_classification = pd.DataFrame(data_preds_multi_label, columns = ['defect_1','defect_2','defect_3','defect_4'])
    data_classification['hasDefect'] = data_preds_binary
    data_classification['ImageId'] = X['ImageId']
    return data_classification[['ImageId','hasDefect','defect_1','defect_2','defect_3','defect_4']]

def rle2mask(mask_rle, shape=(1600,256)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def pred_segmentation(X):
    
    
    # Train dataset prediction visualization
    ep1 = model_segment_1.predict_generator(test_DataGenerator_3(X,preprocess=preprocess),verbose=1)
    ep2 = model_segment_2.predict_generator(test_DataGenerator_3(X,preprocess=preprocess),verbose=1)
    ep3 = model_segment_3.predict_generator(test_DataGenerator_3(X,preprocess=preprocess),verbose=1)
    ep4 = model_segment_4.predict_generator(test_DataGenerator_3(X,preprocess=preprocess),verbose=1)
    
    '''
    ep1 = mask2rle(np.array((Image.fromarray((test_preds_1[0,:,:,0])>=0.5)).resize((1600,256))).astype(int))
    ep2 = mask2rle(np.array((Image.fromarray((test_preds_2[0,:,:,0])>=0.5)).resize((1600,256))).astype(int))
    ep3 = mask2rle(np.array((Image.fromarray((test_preds_3[0,:,:,0])>=0.5)).resize((1600,256))).astype(int))
    ep4 = mask2rle(np.array((Image.fromarray((test_preds_4[0,:,:,0])>=0.5)).resize((1600,256))).astype(int))
    
    #tmp=[X.ImageId.iloc[0],ep1,ep2,ep3,ep4]
    

    #return pd.DataFrame(tmp,columns=['ImageId','EncodedPixels_1','EncodedPixels_2','EncodedPixels_3','EncodedPixels_4'])
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20, 13))
    c1 = Image.fromarray(test_preds_3[0,:,:,0])
    ax3.imshow(np.array(c1.resize((1600,256)))>0.5)
    ax3.set_title('Predicted Mask')
    plt.show()
    '''
    tmp=[X.ImageId[0],ep1,ep2,ep3,ep4]

    return pd.DataFrame([tmp],columns=['ImageId','EncodedPixels_1','EncodedPixels_2','EncodedPixels_3','EncodedPixels_4'])
   

def pred_combined(X):
    '''
    Input: ImageId (dataframe)
    Return: Comdined dataframe of output of pred_classification function and pred_segmentation function
    '''
    return  (pred_classification(X).merge(pred_segmentation(X),on=['ImageId']))


p=[]

def steel_prediction(X):
    global p
    '''
    Function-1:
    Input: ImageId(dataframe)
    Process: Calls pred_combined which calls pred_classification and pred_segmentation
            Applies thresholds -> area and classification probability
    Return: DataFrame (columns = ImageId_ClassId,EncodedPixels)
    
    '''
    p = pred_combined(X).iloc[0]
    tmp = []      
    j, b, m1, m2, m3, m4, ep1, ep2, ep3, ep4 = p
    # randomly selected classification threshold values to get high recall 
    # for no defect binary classifier and high precision for multi-label classifier 
    # while not compromising much on other metrics
    
    hasDef = []
    print('content/raw_data/'+j)
    ogImage=cv2.imread('content/raw_data/'+j)            #PRATIK CHANGE THIS TO THE PATH WHERE THE IMG WILL BE 
    
    
    # area thresholds are determined from EDA performed only on train dataset
    aEp1=area(mask2rle(np.array((Image.fromarray((ep1[0,:,:,0])>=0.5)).resize((1600,256))).astype(int)))
    if aEp1>=500 and b>=0.5 and m1>=0.3: 
        c1 = Image.fromarray(ep1[0,:,:,0])
        img=(np.array(c1.resize((1600,256))))
        img = img[..., np.newaxis]
        blank_image = np.zeros((256,1600,1), np.uint8)
        for i in range(256):
            for j in range(1600):
                t=img[i][j]*255
                if(t<0):
                    blank_image[i][j]=0
                else:
                    blank_image[i][j]=t
                    
        ret, thresh = cv2.threshold(blank_image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ogImage=cv2.drawContours(ogImage, contours, -1, (0,255,0), 3)


    aEp2=area(mask2rle(np.array((Image.fromarray((ep2[0,:,:,0])>=0.5)).resize((1600,256))).astype(int)))
    if aEp2>=700 and b>=0.5 and m2>=0.3:
        c1 = Image.fromarray(ep2[0,:,:,0])
        img=(np.array(c1.resize((1600,256))))
        img = img[..., np.newaxis]
        blank_image = np.zeros((256,1600,1), np.uint8)
        for i in range(256):
            for j in range(1600):
                t=img[i][j]*255
                if(t<0):
                    blank_image[i][j]=0
                else:
                    blank_image[i][j]=t
                    
        ret, thresh = cv2.threshold(blank_image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ogImage=cv2.drawContours(ogImage, contours, -1, (255,0,0), 3)


    aEp3=area(mask2rle(np.array((Image.fromarray((ep3[0,:,:,0])>=0.5)).resize((1600,256))).astype(int)))
    if aEp3>=1100 and b>=0.5 and m3>=0.45:
        c1 = Image.fromarray(ep3[0,:,:,0])
        img=(np.array(c1.resize((1600,256))))
        img = img[..., np.newaxis]
        blank_image = np.zeros((256,1600,1), np.uint8)
        for i in range(256):
            for j in range(1600):
                t=img[i][j]*255
                if(t<0):
                    blank_image[i][j]=0
                else:
                    blank_image[i][j]=t
                    
        ret, thresh = cv2.threshold(blank_image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ogImage=cv2.drawContours(ogImage, contours, -1, (0,0,255), 3)

        
    aEp4=area(mask2rle(np.array((Image.fromarray((ep4[0,:,:,0])>=0.5)).resize((1600,256))).astype(int)))
    if aEp4>=2800 and b>=0.5 and m4>=0.3:
        c1 = Image.fromarray(ep4[0,:,:,0])
        img=(np.array(c1.resize((1600,256))))
        img = img[..., np.newaxis]
        blank_image = np.zeros((256,1600,1), np.uint8)
        for i in range(256):
            for j in range(1600):
                t=img[i][j]*255
                if(t<0):
                    blank_image[i][j]=0
                else:
                    blank_image[i][j]=t
                    
        ret, thresh = cv2.threshold(blank_image, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ogImage=cv2.drawContours(ogImage, contours, -1, (0,0,0), 3)


    cv2.putText(ogImage, 'Bruises/Roll marks', (0, 0), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,255,0), 5);
    cv2.putText(ogImage, 'Buckle', (20, 0), cv2.FONT_HERSHEY_PLAIN, 1.0, (255,0,0), 5);
    cv2.putText(ogImage, 'Edge wave', (40, 0), cv2.FONT_HERSHEY_PLAIN, 1.0, (0,0,255), 5);
    cv2.putText(ogImage, 'Pinchers', (60, 0), cv2.FONT_HERSHEY_PLAIN, 1.0,(0,0,0), 5);
    cv2.imshow('test',ogImage)
    cv2.waitKey(0)
    
    #return pd.DataFrame(tmp, columns = ['ImageId_ClassId','EncodedPixels'])






    '''
    
     fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20, 13))
    img = cv2.imread(str("content/A1_train/" + X.ImageId[0]))
    ax1.imshow(img)
    ax1.set_title(X.ImageId.values[0])

    ax2.imshow(rle2mask(X.EncodedPixels.values[0]))
    ax2.set_title('Ground Truth Mask')

    c1 = Image.fromarray(new_pred[0,:,:,0])
    ax3.imshow(np.array(c1.resize((1600,256)))>0.5)
    ax3.set_title('Predicted Mask')
    plt.show()
    
    
import cv2
img=(np.array(c1.resize((1600,256))))
img = img[..., np.newaxis]
blank_image = np.zeros((256,1600,1), np.uint8)
for i in range(256):
    for j in range(1600):
        t=img[i][j]*255
        if(t<0):
            blank_image[i][j]=0
        else:
            blank_image[i][j]=t
            
 
ogImage=cv2.imread('content/raw_data/005d86c25.jpg')

ret, thresh = cv2.threshold(blank_image, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ogImage=cv2.drawContours(ogImage, contours, -1, (0,255,0), 3)

cv2.imshow('test', ogImage)
cv2.waitKey(0)


ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)



img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
value, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
# opencv 4.x:
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)












l="331 18 587 53 843 89 1099 124 1355 159 1611 177 1867 177 2123 177 2380 177 2636 177 2892 177 3148 177 3404 177 3660 177 3916 177 4172 177 4428 177 4684 177 4940 177 5196 177 5452 177 5708 177 5964 177 6220 177 6477 176 6733 176 6989 176 7245 176 7501 176 7757 176 8013 176 8269 176 8525 176 8781 176 9037 176 9293 176 9549 176 9805 176 10061 176 10317 176 10574 174 10830 174 11086 174 11342 174 11598 174 11854 174 12110 174 12366 174 12622 174 12878 174 13134 174 13390 174 13646 174 13902 174 14158 174 14414 174 14671 173 14927 173 15183 173 15439 173 15695 173 15951 173 16207 173 16463 173 16719 173 16975 173 17231 173 17487 173 17743 173 17999 173 18255 173 18512 172 18768 172 19024 172 19280 172 19536 172 19792 172 20048 172 20304 172 20560 172 20816 172 21072 172 21328 172 21584 172 21840 172 22096 172 22352 172 22609 171 22865 171 23121 171 23377 171 23595 2 23633 171 23851 4 23889 171 24106 8 24145 171 24362 11 24401 171 24618 13 24657 171 24874 16 24913 171 25129 19 25169 171 25385 22 25425 171 25641 25 25681 171 25896 28 25937 170 26152 30 26193 170 26408 30 26449 170 26663 31 26706 169 26919 30 26962 169 27175 30 27218 169 27431 30 27474 169 27686 31 27730 169 27942 30 27986 169 28198 30 28242 169 28453 31 28498 169 28709 31 28754 169 28965 30 29009 170 29220 31 29265 170 29476 31 29521 170 29732 31 29776 171 29988 30 30032 171 30243 31 30288 171 30499 31 30543 172 30755 31 30799 172 31010 31 31055 172 31266 31 31310 173 31522 31 31566 173 31777 32 31821 174 32033 31 32077 174 32289 31 32333 174 32545 31 32588 175 32800 32 32844 175 33056 31 33100 175 33312 31 33355 176 33567 32 33611 176 33823 32 33867 176 34079 31 34122 177 34334 32 34378 177 34590 32 34634 177 34846 32 34889 178 35102 31 35145 178 35357 32 35401 178 35613 32 35656 179 35869 32 35912 179 36124 32 36168 179 36380 32 36423 180 36636 32 36679 180 36891 33 36935 180 37147 32 37190 181 37403 32 37446 181 37659 32 37701 182 37914 33 37957 182 38170 32 38213 182 38426 32 38468 183 38681 33 38724 183 38937 33 38980 183 39193 32 39235 184 39448 33 39491 184 39704 33 39747 184 39960 33 40002 185 40216 32 40258 185 40471 33 40514 185 40727 33 40769 186 40983 33 41025 186 41238 33 41281 186 41494 33 41536 186 41750 33 41792 186 42005 34 42048 186 42261 33 42303 187 42517 33 42559 187 42773 33 42814 188 43028 34 43070 188 43284 33 43326 188 43540 33 43581 189 43795 34 43837 189 44051 34 44093 189 44307 33 44348 190 44562 34 44604 190 44818 34 44860 190 45074 34 45115 191 45330 33 45371 191 45585 34 45627 191 45841 34 45883 191 46097 34 46139 191 46352 34 46395 191 46608 34 46651 191 46864 34 46907 191 47119 35 47162 192 47375 34 47418 192 47631 34 47674 192 47887 34 47930 192 48142 35 48186 192 48398 34 48442 192 48654 34 48698 192 48909 35 48954 192 49165 35 49210 192 49421 34 49466 192 49676 35 49722 192 49932 35 49978 192 50188 35 50234 192 50444 34 50490 192 50699 35 50745 193 50955 35 51001 193 51211 35 51257 193 51466 35 51513 193 51722 35 51769 193 51978 35 52025 193 52233 36 52281 193 52489 35 52537 193 52745 35 52793 193 53001 35 53049 193 53256 36 53305 193 53512 35 53561 193 53768 35 53817 193 54023 36 54073 193 54279 36 54328 194 54535 35 54584 194 54790 36 54840 194 55046 36 55096 194 55302 36 55352 194 55558 35 55608 194 55813 36 55864 194 56069 36 56120 194 56325 36 56376 194 56581 35 56632 194 56837 35 56888 193 57093 35 57144 193 57349 35 57400 193 57605 34 57656 193 57861 34 57911 194 58117 34 58167 194 58373 34 58423 194 58629 33 58679 194 58885 33 58935 194 59141 33 59191 194 59397 33 59447 194 59653 32 59703 194 59909 32 59959 194 60165 32 60215 194 60421 32 60471 194 60677 31 60727 194 60933 31 60983 194 61189 31 61239 194 61445 31 61494 195 61701 30 61750 195 61957 30 62006 195 62213 30 62262 195 62469 30 62518 195 62725 29 62774 195 62981 29 63030 195 63237 29 63286 195 63493 29 63542 195 63749 28 63798 195 64005 28 64054 195 64262 27 64310 195 64518 27 64566 195 64774 26 64822 195 65030 26 65077 196 65286 26 65333 196 65542 26 65589 196 65798 25 65845 196 66054 25 66101 196 66310 25 66357 196 66566 25 66613 196 66822 24 66869 196 67078 24 67125 196 67334 24 67381 196 67590 24 67637 196 67846 23 67893 196 68102 23 68149 196 68358 23 68405 196 68614 23 68660 197 68870 22 68916 197 69126 22 69172 197 69382 22 69428 197 69638 22 69684 197 69894 21 69940 197 70150 21 70196 197 70406 21 70451 198 70662 19 70707 198 70918 16 70962 199 71174 13 71218 199 71430 10 71473 200 71686 8 71729 200 71942 5 71984 201 72198 2 72240 201 72495 202 72751 202 73006 203 73262 203 73517 204 73773 204 74028 205 74284 205 74539 206 74795 206 75050 207 75306 207 75561 208 75817 208 76072 209 76328 209 76583 210 76839 210 77094 211 77350 211 77605 212 77860 213 78116 213 78371 214 78627 214 78882 215 79138 215 79393 216 79649 216 79904 217 80160 217 80415 218 80671 218 80926 219 81182 219 81437 220 81693 220 81948 221 82204 221 82459 222 82715 222 82970 223 83226 223 83481 224 83737 224 83992 225 84248 225 84503 226 84759 226 85014 227 85270 227 85525 228 85781 228 86036 229 86292 229 86547 230 86803 230 87058 231 87314 231 87569 232 87825 232 88080 233 88336 233 88591 234 88847 234 89103 234 89358 235 89614 235 89869 236 90125 236 90380 237 90636 237 90891 238 91147 238 91402 239 91658 239 91913 240 92169 240 92425 240 92680 241 92936 241 93191 242 93447 242 93702 243 93958 243 94213 244 94469 244 94724 245 94980 245 95235 246 95491 246 95746 247 96002 247 96258 247 96514 247 96770 247 97026 247 97282 247 97538 247 97794 247 98050 247 98306 247 98562 247 98818 247 99074 247 99330 247 99586 247 99842 247 100098 247 100354 247 100610 247 100866 246 101122 246 101378 245 101634 245 101890 244 102146 244 102402 243 102658 243 102914 242 103170 242 103426 241 103682 241 103938 240 104194 240 104450 239 104706 239 104962 238 105218 238 105474 237 105730 237 105986 236 106242 236 106498 236 106754 235 107010 235 107266 235 107522 234 107778 234 108034 233 108290 233 108546 233 108802 232 109058 232 109314 232 109570 231 109826 231 110082 231 110338 231 110594 231 110850 231 111106 231 111362 231 111618 231 111874 231 112130 231 112386 231 112642 231 112898 231 113154 231 113410 230 113666 230 113922 230 114178 230 114434 230 114690 230 114946 230 115202 230 115458 230 115714 230 115970 230 116226 230 116482 230 116738 230 116994 230 117250 230 117506 230 117762 230 118018 230 118274 230 118530 230 118786 230 119042 230 119298 230 119554 230 119810 230 120066 230 120322 229 120578 229 120834 229 121090 229 121346 229 121602 229 121858 229 122114 229 122370 229 122626 229 122882 229 123138 229 123394 229 123650 229 123906 229 124162 228 124418 228 124674 228 124930 227 125186 227 125442 227 125698 226 125954 226 126210 226 126466 225 126722 225 126978 225 127234 224 127490 224 127746 224 128002 223 128258 223 128514 223 128770 222 129026 222 129282 222 129538 221 129794 221 130050 221 130306 220 130562 220 130818 221 131074 221 131330 222 131586 222 131842 223 132098 224 132354 224 132610 225 132866 226 133122 226 133378 227 133634 227 133890 228 134146 229 134402 229 134658 230 134914 231 135170 231 135426 232 135682 232 135938 233 136194 234 136450 234 136706 235 136961 237 137217 237 137473 238 137729 238 137985 239 138241 240 138497 240 138753 241 139009 242 139265 242 139521 243 139777 243 140033 244 140289 245 140545 245 140801 246 141057 247 141313 247 141569 248 141825 248 142081 249 142337 250 142593 250 142849 251 143105 252 143361 252 143617 253 143873 253 144129 254 144385 254 144641 254 144897 254 145153 254 145409 254 145665 254 145921 254 146177 254 146433 254 146689 254 146945 254 147201 254 147457 254 147713 254 147969 254 148225 254 148481 254 148737 254 148993 254 149249 254 149505 254 149761 254 150017 254 150273 254 150529 254 150785 254 151041 255 151297 255 151553 255 151809 255 152065 255 152321 255 152577 255 152833 255 153089 255 153345 255 153601 255 153857 255 154113 255 154369 255 154625 255 154881 255 155137 255 155393 255 155649 255 155905 255 156161 255 156417 255 156673 255 156929 255 157185 255 157441 255 157697 255 157953 255 158209 255 158465 255 158721 255 158977 255 159233 255 159489 255 159745 255 160001 255 160257 255 160513 255 160769 255 161025 255 161281 255 161537 255 161793 255 162049 255 162305 255 162561 255 162817 255 163073 255 163329 255 163585 255 163841 255 164097 255 164353 10940 175298 241 175556 229 175813 224 176070 223 176328 221 176585 220 176842 219 177100 217 177357 216 177614 215 177872 49 177922 163 178129 48 178179 162 178386 47 178436 161 178644 45 178693 160 178901 44 178949 160 179158 43 179206 159 179416 41 179463 158 179673 40 179720 157 179930 39 179977 156 180188 37 180234 155 180445 36 180491 154 180702 35 180748 153 180960 33 181005 152 181217 32 181261 152 181474 31 181518 151 181732 29 181775 151 181989 28 182032 150 182246 27 182289 149 182504 25 182546 148 182762 23 182803 147 183021 20 183061 145 183280 17 183318 144 183539 14 183575 143 183798 11 183832 142 184057 8 184089 141 184316 5 184347 139 184575 2 184604 138 184861 137 185118 136 185375 135 185633 133 185890 132 186147 131 186404 130 186662 128 186919 127 187176 126 187433 125 187690 109 187948 92 188205 90 188462 88 188719 86 188977 83 189234 81 189491 79 189748 77 190005 75 190263 72 190520 70 190777 68 191034 66 191291 64 191549 61 191806 59 192063 57 192320 55 192578 52 192835 50 193092 48 193349 46 193606 44 193864 39 194121 33 194378 27 194635 21 194892 16 195150 9 195407 3"



temp=pred_combined(X)



'''


data = 	[['0025bde0c.jpg']]
X = pd.DataFrame(data, columns= ['ImageId'])

steel_prediction(X)



ogImage=cv2.imread('content/raw_data/'+'0025bde0c.jpg')  
cv2.putText(ogImage, 'Bruises/Roll marks', (0, 25), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,255,0), 1);
cv2.putText(ogImage, 'Buckle', (0, 50), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255,0,0), 1);
cv2.putText(ogImage, 'Edge wave', (0, 75), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,255), 1);
cv2.putText(ogImage, 'Pinchers', (0, 100), cv2.FONT_HERSHEY_SIMPLEX , 0.5,(0,0,0), 1);
cv2.imshow('test',ogImage)
cv2.waitKey(0)
    