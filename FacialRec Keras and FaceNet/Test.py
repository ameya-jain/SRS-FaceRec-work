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



def extract_face(filename, required_size=(160,160)):

    image = Image.open(filename)  # load image file

    image = image.convert('RGB') # convert to RGB

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

    return face_array


def load_faces(directory):
    faces=[]
    face = extract_face(directory)
    print("Return type of extract face is : ", type(face))
    faces.append(face)

    return faces

path = 'ben_afflec pic.jpg'

faces=load_faces(path)

print("Return type of load faces is : ", type(faces))
print("Return type of load dataset is : ", type(asarray(faces)))

