import cv2  #It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.
import os,glob #These modules is used to retrieve files/pathnames

import epochs as epochs
import numpy as np #NumPy is a python library used for working with arrays.
import tensorflow as tf
from skimage.feature import hog , local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

""" The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide,
returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane,
you can then feed some features to your classifier to see what the "predicted" class is."""
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
path='Dataset/*'
data_path = os.path.join(path,'*g')
imagePaths = glob.glob(data_path)
print('Total dataset images length:')
print(len(imagePaths))

data = []
LBP = []
labels = []
hogFeatures = []
featurevector = [[0 for x in range(0)] for y in range(0)]  #2d arrayg

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath,0)
	image = cv2.resize(image, (100,100),interpolation=cv2.INTER_AREA)
	#cv2.imshow('a',image)
	#cv2.waitKey(0)
	#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
						cells_per_block=(1, 1), visualize=True)
	lbp = local_binary_pattern(image, 10,15,  method= "uniform")
	#hogFeatures.append(fd)
	#LBP.append(lbp)
	llb = lbp.ravel()
	hogFeatures = fd
	features = np.concatenate((hogFeatures,llb))
	featurevector.append(features)
	#extract labels
	#data.append(image)
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
print('\n')
print('********** Dataset labels *********''\n')
print(labels)
print('\n')
print('********** Total hogFeatures *********')
print(len(hogFeatures))
print('\n')
X_train, X_test, y_train, y_test = train_test_split(featurevector, labels, test_size = 0.30)
print('train images ',len(X_train) )
print('train labels ',len(y_train))
print('\n')
print('test images ',len(X_test))
print('test labels',len(y_test))
print('\n')
print('********** Total LBP_Features *********')
print(len(llb))

########## Classifiers #############
##########   SVM       #############
svclassifier = svm.SVC()
svclassifier = SVC(kernel='linear')
svclassifier.probability = True
svclassifier.fit(X_train, y_train)
#Now making prediction
y_pred = svclassifier.predict(X_test)
print('\n')
print("Actual Labels :    "'\n',y_test)
print('\n')
print("Predicted Labels : "'\n',y_pred)
print('\n')
# accuracy
accuracy = svclassifier.score(X_test, y_test)
print('SVM Accuracy = ',accuracy*100)
print('\n')

##########  Classifiers ##########
##########   KNN       ############
classifier= KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#Now making prediction
y_pred = classifier.predict(X_test)
print('\n')
print("Actual Labels :    ", y_test)
print('\n')
print("Predicted Labels : ",y_pred)

# accuracy
accuracy = classifier.score(X_test, y_test)
print('KNN Accuracy',accuracy*100)
print('\n')

####################################




##########  Classifiers ##########
###########decsion three ###########
#Create Decision Tree classifer object
clf_dt = DecisionTreeClassifier()
clf_dt = clf_dt.fit(X_train,y_train)
y_pred = clf_dt.predict(X_test)
print('\n')
print("Actual Labels : ", y_test)
print('\n')
print("Predicted Labels : ",y_pred)
accuracy = clf_dt.score(X_test,y_test)
print('decision tree bayes accuracy',accuracy*100)
print('\n')
 ####################################

##########  Classifiers ##########
################## LogisticRegression ##################
logreg_clf = LogisticRegression()
logreg_clf.fit(X_train, y_train)
y_pred = logreg_clf.predict(X_test)
print('\n')
print("Actual Labels : ", y_test)
print('\n')
print("Predicted Labels : ",y_pred)
accuracy = logreg_clf .score(X_test,y_test)
print('logistic regression',accuracy*100)
print('\n')
##################################################
##########  Random forest ##########
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred=rf.predict(X_test)

#accuary
accuracy = rf.score(X_test,y_test)
print('Random forest',accuracy*100)
print('\n')


##########  Classifiers ##########
##########   GNB       ############
#Create a Gaussian Classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = nb_clf.predict(X_test)
print('\n')
print("Actual Labels : ", y_test)
print('\n')
print("Predicted Labels : ",y_pred)
accuracy = nb_clf.score(X_test,y_test)
print('navie bayes accuracy',accuracy*100)
print('\n')
######################			#########################

#############Hybrid CNN-SVM#######################
# Load a pre-trained CNN model (e.g., VGG16)
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Train an SVM classifier on the CNN-extracted features
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions using the SVM classifier
svm_predictions = svm_classifier.predict(X_test)
y_pred = nb_clf.predict(X_test)
print('\n')
print("Actual Labels : ", y_test)
print('\n')
print("Predicted Labels : ",y_pred)
# Evaluate the accuracy of the hybrid classifier
accuracy = accuracy_score(y_test, svm_predictions)
print('Hybrid CNN-SVM Accuracy = ', accuracy * 100)

###########################################################

# confusion matrix
print('********** Confusion Matrix *********')
print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))
joblib.dump(svclassifier,'SVM train model.pkl')
print('Model saved...!')
