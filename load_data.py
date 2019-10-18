import os, random, pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATADIR = '/Users/tkim/Dev/pyth/tensorflow/kagglecatsanddogs_3367a/PetImages'
CATEGORIES = ['Dog','Cat']
training_data = []
IMG_SIZE = 50

def show_image():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
           img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
           plt.imshow(img_array, cmap="gray")
           plt.show()
           break
        break

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

# show_image()

create_training_data()
random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# save the data, reduce rebuilding the model
pickle_out = (open('X.pickle', 'wb'))
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = (open('Y.pickle', 'wb'))
pickle.dump(Y, pickle_out)
pickle_out.close()

pickle_in = open('X.pickle', 'rb')
X = pickle.load(pickle_in)

print(X[1])