import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #to suppress some warnings

# Define data generators
train_directory = './data/train'
val_directory = './data/test'

num_train = 28709 # number of training set
num_val = 7178 # number of validation set
batch_size = 64 

trainDataGenrator = ImageDataGenerator(rescale=1./255) # rescale the image to 0-1
valDataGenrator = ImageDataGenerator(rescale=1./255) # rescale the image to 0-1


train_generator = trainDataGenrator.flow_from_directory(
        train_directory,
        target_size=(48,48), 
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical'
    ) # Define the CNN Model

validation_generator = valDataGenrator.flow_from_directory(
        val_directory,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale", 
        class_mode='categorical'
    ) # Define the CNN Model

# Create the model
model = Sequential() # Plot the training and validation loss + accuracy
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1))) #
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# If you want to train the same model or try other models, go for this
def Train_Model(num_epoch):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                  metrics=['accuracy'])
    model_info = model.fit(
            train_generator,
            steps_per_epoch=num_train // batch_size,
            epochs=num_epoch,
            validation_data=validation_generator,
            validation_steps=num_val // batch_size) #builds the model
    model.save_weights('./modules/model.h5')
    return True

class VideoCamera(object):
    isOpened = True  
    def __init__(self):
        self.video = cv.VideoCapture(0, cv.CAP_DSHOW) # 0 -> index of camera
    def __del__(self):
        self.video.release() # release the camera
    def get_frame(self):
        isOpened = True # check if camera is open
        model.load_weights('./modules/model.h5') # load the weights
        cv.ocl.setUseOpenCL(False) # to avoid error
        emotion_dict = {0: "Angry", 1: "Disgusted", 
                        2: "Fearful", 3: "Happy",
                        4: "Neutral", 5: "Sad",
                        6: "Surprised"} # dictionary of emotions
        ret, frame = self.video.read() # read the camera
        if not ret: # if not return the frame
            print("Unable to capture video")
        facecasc = cv.CascadeClassifier('./modules/haarcascade_frontalface_default.xml') # load the cascade
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # convert to grayscale
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5) # detect the faces and store the positions 
        for (x, y, w, h) in faces: # frame, x, y, w, h
            cv.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 0), 2) # draw rectangle to main frame 
            roi_gray = gray[y:y + h, x:x + w] # crop the region of interest i.e. face from the frame
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray, (48, 48)), -1), 0) # resize the image
            prediction = model.predict(cropped_img) # predict the emotion
            maxindex = int(np.argmax(prediction)) # get the index of the largest value
            #TODO: make the color of the text change with the emotion
            cv.putText(frame, emotion_dict[maxindex], (x+20, y-60), 
                       cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA) # write the emotion text above rectangle
        ret, jpeg = cv.imencode('.jpg', frame) # encode the frame into jpeg
        if isOpened:
            return jpeg.tobytes()
    def close_camera(self):
        isOpened = False
        self.video.release() # release the camera