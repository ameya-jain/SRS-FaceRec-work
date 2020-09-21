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



# image = Image.open("ben_afflec pic.jpg")  # load image file
#
# image = image.convert('RGB')  # convert to RGB
#
# pixels = asarray(image)  # convert image to array
#
# detector = MTCNN()  # This is the Face detector
#
# results = detector.detect_faces(pixels)  # detect faces in the image
#
# print(results)
#
# x1, y1, width, height = results[0]['box']  # NOT SURE what the 'box' does
#
# x1, y1 = abs(x1), abs(y1)
#
# x2, y2 = x1 + width, y1 + height
#
# face = pixels[y1:y2, x1:x2]  # The notation

# image = Image.fromarray(face)  # image from the face pixels
# image = image.resize(required_size)  # resize the image
# face_array = asarray(image)



# extract a single face from a given photograph

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
    for file_name in listdir(directory):

        path = directory + file_name #path of each picture
        face = extract_face(path)

        faces.append(face)

    return faces


def load_dataset(directory):

    X, y = list(), list()

    for subdir in listdir(directory):

        path = directory+subdir+'/'

        if not isdir(path):
            continue

        faces=load_faces(path)

        labels = [subdir for _ in range(len(faces))] # all the labels

        #summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))

        X.extend(faces)
        y.extend(labels)

    return asarray(X), asarray(y)

# directory = '5-celebrity-faces-dataset/train/'
# # load train dataset
# trainX, trainy = load_dataset(directory)
# print(trainX.shape, trainy.shape)
# # load test dataset
# testX, testy = load_dataset('5-celebrity-faces-dataset/val/')
# # save arrays to one file in compressed format
# savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)



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


#load faces
data = load('5-celebrity-faces-dataset.npz')
testX_faces = data['arr_2']

# load dataset
data = load('5-celebrity-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# test model on a random example from the test dataset
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()