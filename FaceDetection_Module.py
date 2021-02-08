import cv2
import numpy as np
from PIL import Image
import torch

def feats_extract(face_crop, model, device):
    face_96 = Image.fromarray(face_crop).resize((96,96),Image.ANTIALIAS)
    
    # convert back to manipulate
    test_face = np.array(face_96)
    test_face_copy = test_face.copy() 

    # We convert it to torch domain so we can use it in our model
    test_face_torch = torch.from_numpy(test_face).float().to(device)
    test_face = test_face_torch.reshape(1,1,96,96) 
    
    # Using the model to predict the coordinates in the face we are dealing in this iteration
    test_predictions = model(test_face)
    test_predictions = test_predictions.cpu().data.numpy()

    # This is the list with the face keypoints we are detecting
    #keypts_labels_plantilla = train_data.columns.tolist() 

    # We pair the coordinates and pile then in columns for coord x and coord y
    coord = np.vstack(np.split(test_predictions[0],15))

    for (x, y) in coord:
            cv2.circle(test_face_copy, (x, y), 2, (0, 255, 0), -1) # the coordinates are plotted directly onto the cropped image

    cv2.imshow("Prova", test_face_copy)


    # Le due classi qui sotto sono obsolete, le mantengo perché potrebbero risultare utili in futuro

class FaceDetection():
    
    def __init__(self, path2img = 'AbccEAc.jpg', path2class = 'haarcascade_frontalface_default.xml'):
        
        #Load image
        self.img_original = cv2.imread(path2img)
        
        # Convert to RGB colorspace
        self.img_original = self.convertToRGB(self.img_original)
        
        # copy original image
        self.img_with_detections = np.copy(self.img_original)
        
        #convert image to gray (opencv expects gray images)
        self.gray_img = self.convertToGray(self.img_original)

        #load cascade classifier (haarcascade) training file
        self.haar_face_cascade = cv2.CascadeClassifier(path2class)

        #Detect multiscale images 
        self.faces = self.haar_face_cascade.detectMultiScale(self.gray_img, scaleFactor=1.1, minNeighbors=5);

    def number_faces(self):
        #print the number of faces found 
        print('Faces found: ', len(self.faces))

    def convertToGray(self, img):
        # Convert the RGB  image to grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def convertToRGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    def detection(self):
    
        faces_crop = []
        for (x, y, w, h) in self.faces:  
            obj = self.img_original[y:y + h, x:x + w]
            faces_crop.append(obj)
            cv2.rectangle(self.img_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return faces_crop

class FaceDetection2():
    
    def __init__(self, img = [] , path2class = 'haarcascade_frontalface_default.xml'):
        
        #Load image
        self.img_original = img
        
        # Convert to RGB colorspace
        self.img_original = self.convertToRGB(self.img_original)
        
        # copy original image
        self.img_with_detections = np.copy(self.img_original)
        
        #convert image to gray (opencv expects gray images)
        self.gray_img = self.convertToGray(self.img_original)

        #load cascade classifier (haarcascade) training file
        self.haar_face_cascade = cv2.CascadeClassifier(path2class)

        #Detect multiscale images 
        self.faces = self.haar_face_cascade.detectMultiScale(self.gray_img, scaleFactor=1.1, minNeighbors=5);

    def number_faces(self):
        #print the number of faces found 
        print('Faces found: ', len(self.faces))

    def convertToGray(self, img):
        # Convert the RGB  image to grayscale
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    def convertToRGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    
    def detection(self):
    
        faces_crop = []
        for (x, y, w, h) in self.faces:  
            obj = self.img_original[y:y + h, x:x + w]
            faces_crop.append(obj)
            cv2.rectangle(self.img_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        return faces_crop