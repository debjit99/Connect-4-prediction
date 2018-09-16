#---------------------------------------------------------------------#
#      The program reads the data file, builds three classifiers      #
# (decision tree, naive Bayesian classifier, support vector machine)  #
#            and outputs results of k-fold cross validation           #
#		and plots a graph of accuracy in i-th cross validation        #
#  Submitted by Ashwani Anand(BMC201605) and Debjit Paria(BMC201704)  #
#---------------------------------------------------------------------#

import time										#for calculating time taken
import numpy as np 								#for arrays etc
from sklearn import tree						#for decision tree
from sklearn.naive_bayes import MultinomialNB	#for Naive Bayes
from sklearn.svm import LinearSVC				#for SVM
from sklearn.model_selection import KFold		#for K-fold cross validation
from sklearn.metrics import accuracy_score		#for classification accuracy
import matplotlib.pyplot as plt					#for plotting accuracies for all tests

#---------Loading data from a file----------------------------------------------
OpenX = open('connect-4.data',"r")       		#Opening the file and assigning variable OpenX to it

b1 = 1
x1 = 0
o1 = 3

x = OpenX.read()								#Reading opened file

X = []
y = []

w = []
i = 0

while i < len(x) - 1:							# clean the data

	s = []

	while x[i] != '\n':
		s.append(x[i])
		i = i + 1

	w = []

	for j in range(0, 42):
		if s[2*j] == 'b':
			w.append(b1)
		elif s[2*j] == 'x':
			w.append(x1)
		elif s[2*j] == 'o':
			w.append(o1)


	X.append(w)

	if(s[84] == 'w'):
		y.append(1)
	elif s[84] == 'l':
		y.append(-1)
	elif s[84] == 'd':
		y.append(0)

	i = i + 1

X = np.array(X)
y = np.array(y)

kf = KFold(n_splits = 10, shuffle = False)	#Making sets from k fold cross validation
kf.get_n_splits(X)



#---------decision tree---------------------------------------------------------
opendt = open('decision_tree.txt', 'w')

accuracy_dt = []

dt_time_start = time.clock()

n = 1
for train_index, test_index in kf.split(X):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	clf = tree.DecisionTreeClassifier(random_state=0)						# decision tree classifier
	clf = clf.fit(X_train, y_train)											# training current train data

	opendt.write('Indices used to train the ' + str(n)+ ' model: \n' )
	for j in train_index:
		opendt.write(str(j)+ ', ')

	opendt.write(' \n \n')

	opendt.write('\nIndices used to validate this model: \n')

	for j in test_index:
		opendt.write(str(j) + ', ')

	opendt.write(' \n \n')

	y_pred = clf.predict(X_test)
	y_true = y_test

	accuracy_dt.append(accuracy_score(y_true, y_pred))			# calculating error on the ith cross validation

	opendt.write('\nPredictions using this model on the validation set\n' )
	for j in y_pred:
		opendt.write(str(X[j]) + ": "+  str(j) + '\n')

	opendt.write('\n \n')

	opendt.write('\nAccuracy on this set: ' + str(accuracy_dt[-1]))

	opendt.write('\n \n \n \n \n')

	n = n + 1

accuracy_dt = np.array(accuracy_dt)

dt_time_total = time.clock() - dt_time_start

opendt.write(" \n Total time taken by Decision Tree: " + str(dt_time_total))

opendt.write(" \n Average accuracy: " + str(accuracy_dt.sum()*10))# mean accuracy across 10 cross validation

opendt.write(" \n Highest accuracy: " + str(accuracy_dt.max()*100))

opendt.write('\n \n \n \n \n')

opendt.close()

#---------end decision tree-----------------------------------------------------


#---------naive bayes classifier------------------------------------------------
opennb = open('Naive_Bayes.txt', 'w')

accuracy_nb = []

n = 1
nb_time_start = time.clock()

for train_index, test_index in kf.split(X):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	mnb = MultinomialNB(alpha = 0.5) 					        # naive bayes classifier
	mnb = mnb.fit(X_train, y_train,)							# training current train data

	opennb.write('Indices used to train the ' +str(n)+ ' model: \n' )
	for j in train_index:
		opennb.write(str(j)+ ', ')

	opennb.write('\n \n')

	opennb.write('\nIndices used to validate this model: \n')

	for j in test_index:
		opennb.write(str(j) + ', ')

	opennb.write('\n \n')

	y_pred = mnb.predict(X_test)
	y_true = y_test


	accuracy_nb.append(accuracy_score(y_true, y_pred))				# calculating error on the ith cross validation

	opennb.write('\nPredictions using this model on the validation set\n' )
	for j in y_pred:
		opennb.write(str(X[j])+ ': '+ str(j) + '\n ')

	opennb.write('\n \n')

	opennb.write('\nAccuracy on this set: ' + str(accuracy_nb[-1]*100))

	opennb.write('\n \n \n \n \n')
	n = n + 1

accuracy_nb = np.array(accuracy_nb)


nb_time_total = time.clock() - nb_time_start

opennb.write("\nTotal time taken by Naive Bayes:" + str(nb_time_total))

opennb.write("\nAverage accuracy: " + str(accuracy_nb.sum()*10))	# mean accuracy across 10 cross validation

opennb.write("\nHighest accuracy: " + str(accuracy_nb.max()*100))

opennb.write('\n \n \n \n \n')

opennb.close()

#---------end naive bayes classifier--------------------------------------------


#---------SVM-------------------------------------------------------------------
opensvm = open('SVM.txt', 'w')

n = 1

accuracy_svm = []

svm_time_start = time.clock()

for train_index, test_index in kf.split(X):

	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	sv = LinearSVC() 												# SVM
	sv = sv.fit(X_train, y_train)							        # training current train data

	opensvm.write('Indices used to train the ' + str(n) + ' model: \n' )
	for j in train_index:
		opensvm.write(str(j)+ ', ')

	opensvm.write('\n \n')

	opensvm.write('\nIndices used to validate this model: \n')

	for j in test_index:
		opensvm.write(str(j) + ', ')

	opensvm.write('\n \n')

	y_pred = sv.predict(X_test)
	y_true = y_test


	accuracy_svm.append(accuracy_score(y_true, y_pred))				# calculating error on the i-th cross validation

	opensvm.write('\nPredictions using this model on the validation set\n' )
	for j in y_pred:
		opensvm.write(str(X[j]) + ': ' + str(j) + '\n')

	opensvm.write('\n \n')

	opensvm.write('\nAccuracy on this set: ' + str(accuracy_svm[-1]*100))

	opensvm.write('\n \n \n \n \n')
	n = n + 1


accuracy_svm = np.array(accuracy_svm)

svm_time_total = time.clock() - svm_time_start

opensvm.write("\nTotal time taken by SVM:" + str(svm_time_total))

opensvm.write("\nAverage accuracy :" + str(accuracy_svm.sum()*10))	# mean accuracy across 10 cross validation

opensvm.write("\nHighest accuracy: " + str(accuracy_svm.max()*100))

opensvm.write('\n \n \n \n \n')

opensvm.close()
#---------end SVM---------------------------------------------------------------

#---------for plotting the Accuracies of classifier-----------------------------
plt.plot([1,2,3,4,5,6,7,8,9,10], accuracy_dt, color ='r', label ='Decision Tree Error')
plt.plot([1,2,3,4,5,6,7,8,9,10], accuracy_nb, color ='g', label ='Naive Bayes Error')
plt.plot([1,2,3,4,5,6,7,8,9,10], accuracy_svm, color ='b', label ='SVM Error')

plt.xlabel('Test Data number')
plt.ylabel('Accuracy')
plt.title("Error")
plt.legend(loc='upper right')
plt.show()


plt.show()
#---------end Plotter-----------------------------------------------------------
