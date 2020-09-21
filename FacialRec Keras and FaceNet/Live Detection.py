from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir
from os.path import isdir
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
from random import choice
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import load_model
import cv2

def get_embedding(model, face_pixel):

    face_pixel=face_pixel.astype('float32')

    mean, std = face_pixel.mean(), face_pixel.std()
    face_pixel = (face_pixel-mean)/std

    samples = expand_dims(face_pixel,axis=0) #transform face into one sample

    yhat = model.predict(samples)
    return yhat[0]

# load the face dataset
# data = load('5-celebrity-faces-dataset.npz')
# trainX, trainy, testX, testy = data['arr_0'], data['arr_1'] ,data['arr_2'] ,data['arr_3']
# model = load_model('facenet_keras.h5')
#
# newTrainX =[]
#
# for face_pixels in trainX:
#     embedding = get_embedding(model, face_pixels)
#     newTrainX.append(embedding)
# newTrainX = asarray(newTrainX)
#
# newTestX = list()
#
# for face_pixels in testX:
#     embedding=get_embedding(model, face_pixels)
#     newTestX.append(embedding)
#
# savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, trainy, newTestX, testy)


data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy = data['arr_0'], data['arr_1']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)

model = SVC(kernel='linear')
model.fit(trainX, trainy)

# extract a single face from a given photograph

def extract_face(image, required_size=(160,160)):

    pixels = asarray(image) # convert image to array

    detector = MTCNN() # This is the Face detector

    results = detector.detect_faces(pixels) # detect faces in the image and returns a list results

    x1, y1, width, height = results[0]['box'] # result[0] is a dict which has key 'box' and box has four elements

    x1, y1 = abs(x1), abs(y1)

    x2, y2 = x1+width, y1+height

    face=pixels[y1:y2, x1:x2]  # indexing like this can be done on numpy arrays not normal lists each comma gives entry to deeper list

    image = Image.fromarray(face) #image from the face pixels
    image= image.resize(required_size) #resize the image
    face_array = asarray(image)

    return face_array, x1,x2,y1,y2



video_capture=cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    cv2.imshow("Video", frame)

    face , x1, x2, y1, y2= extract_face(frame)

    cv2.rectangle(frame, (x1, y1), (x2 ,y2), (0, 255, 0), 2)

    k=cv2.waitKey(1)

    if k & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()