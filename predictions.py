import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation
from keras.models import load_model
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, f1_score, recall_score, classification_report
import itertools
import glob

%reload_ext autoreload
%autoreload 2
%matplotlib inline


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



TrainAnnotations = pd.read_csv("TrainAnnotations.csv")
#TrainAnnotations = TrainAnnotations.drop(TrainAnnotations[TrainAnnotations['annotation'] == 0].sample(frac=0.6).index)
filenames = ["TrainData/" + fname for fname in TrainAnnotations['file_name'].tolist()]
labels = TrainAnnotations['annotation'].tolist()

train_images=[]
for i in range(len(labels)):
    img = image.load_img(filenames[i], target_size=(128,128,3))
    img = image.img_to_array(img)
    img = img/255
    train_images.append(img)
    
X = np.array(train_images)
y = np.array(labels)
print(X.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, labels, random_state=25, test_size=0.25)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 25)
Y_train = np_utils.to_categorical(y_train, 5)
Y_val = np_utils.to_categorical(y_val, 5)
y = np_utils.to_categorical(y, 5)
model = None
model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(5, activation='softmax'))
epochs = 15
batch_size = 32
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=2, save_best_only=True)
es = EarlyStopping(monitor='val_accuracy', patience = 3)
callbacks=[mc]

hs = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), verbose=2, epochs=epochs, batch_size=batch_size, callbacks=callbacks)

saved_model = load_model('best_model.h5')
val_acc = saved_model.evaluate(X_val, Y_val, verbose=1)
#print("val accuracy = ", val_acc[1])
print("%s: %.2f%%" % (saved_model.metrics_names[1], val_acc[1]*100))

model_history=list()
model_history.append(hs)

x=0
for i, history in enumerate(model_history):
    x=x+1
    fig = plt.figure(figsize=(15,20))
    plt.subplot(5,2,x)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model '+str(i)+' accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    x=x+1
    plt.subplot(5,2,x)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    

y_pred = saved_model.predict(X_val)
print(classification_report(np.argmax(Y_val, axis=1), np.argmax(y_pred, axis=1), digits=5))
cnf_matrix = confusion_matrix(np.argmax(Y_val, axis=1), np.argmax(y_pred, axis=1))

# Plot normalized confusion matrix
fig = plt.figure()
fig.set_size_inches(14, 12, forward=True)
#fig.align_labels()

# fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
plot_confusion_matrix(cnf_matrix, classes=np.asarray(np.unique(y)), normalize=True,
                      title='Normalized confusion matrix')


test_path = "TestData/*.jpg"
#TrainAnnotations = pd.read_csv("../input/train-annotations/TrainAnnotations.csv")

test_filenames = glob.glob(test_path)
#filenames = ["../input/train-annotations/TrainData-20200215T214719Z-001/TrainData/" + fname for fname in TrainAnnotations['file_name'].tolist()]
#labels = TrainAnnotations['annotation'].tolist()
test_images=[]
for i in range(len(test_filenames)):
    img = image.load_img(test_filenames[i], target_size=(128,128,3))
    img = image.img_to_array(img)
    img = img/255
    test_images.append(img)
    
X_test = np.array(test_images)
y_pred_test = model.predict(X_test)
predictions_test = np.argmax(y_pred_test, axis = 1)
one_hot_predictions = to_categorical(predictions_test)
result = pd.DataFrame(one_hot_predictions)
result.to_csv("predict.csv", header=None, index=False)
