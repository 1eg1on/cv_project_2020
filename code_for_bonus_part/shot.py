from keras.models import load_model
from keras.preprocessing import image
import cv2
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import time





class TakeShot():
    
    def __init__(self,model_h5_file):
        '''
            model_h5_file - is a .h5 file in the directory prepared to read with keras.load_model method      
        '''
        
        self.model = load_model(model_h5_file)
        self.objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        
        #self.model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
        
    def emotion_analysis(self,emotions):
        
        y_pos = np.arange(len(self.objects))
        plt.bar(y_pos, emotions, align='center', alpha=0.9)
        plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
        plt.xticks(y_pos, self.objects)
        plt.ylabel('percentage')
        plt.title('emotion') 
        plt.show()
        
    def take_picture(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            raise Exception("Could not open video device")

        ret, frame = video_capture.read()

        video_capture.release()
        frameRGB = frame[:,:,::-1] # BGR => RGB
        
        cut_pic = frameRGB[200:400, 300:500] #adjust the middle to find the face approximately
        plt.imshow(cut_pic)
        return cut_pic[:,:,0].reshape(cut_pic.shape[0],cut_pic.shape[1])
    
    def compress_to_input(self,picture):
        
        '''
        picture - the return of the take_picture method
        
        '''
    
        im = Image.fromarray(pic)
        im.save("img/your_file.jpeg")
        img = Image.open('img/your_file.jpeg').resize((48, 48)).convert('RGBA')
        plt.imshow(img);
        img = np.array(img)[:,:,0]
        return img
        
    def emotion(self,img):
        
        '''
        img - compressed image48 output by compress_to_input method
        '''
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)

        x /= 255

        custom = self.model.predict(x)
        
        #self.emotion_analysis(custom[0])

        x = np.array(x, 'float32')
        x = x.reshape([48, 48]);
        m=1e-22
        a=custom[0]
        for i in range(0,len(a)):
            if a[i]>m:
                m=a[i]
                ind=i

        return self.objects[ind]



'''
WORK PART
'''

instanсe = TakeShot('emotion_recognition_2_batch.h5') #initialize

for i in range(100):
    pic = instanсe.take_picture() #take a picture
    comp_pic = instanсe.compress_to_input(pic) #compress the picture
    em = instanсe.emotion(comp_pic)
    with open("tmp.txt", "w") as file:
        print(em, file=file)
    time.sleep(0.5)





# 