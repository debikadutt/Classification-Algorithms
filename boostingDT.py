import numpy as np
#from scipy.optimize import minimize
#from scipy.io import loadmat
import math 
from random import shuffle
from StringIO import StringIO
from numpy  import array
import csv
import operator
from collections import Counter
import random
import pandas as pd
from statistics import mean
from scipy.stats import mode

import math


class DecisionTree:

	def __init__(self, data, parent):#change as required
		self.dataset = data
		self.isLeaf = None #check
		self.label = None
		self.height = None
		self.parentNode = parent
		self.column = None
		self.children= dict()
		self.id = -1
		self.splitValue = []
		self.cr_visited = list()

def checkLabel(dataset,class_label):
	same_label = -1
	dataset = np.array(dataset)
	for i in dataset:
		#print "blah blah"
		#print dataset
		#print str(i) + " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
		if i[dataset.shape[1]-1] == class_label:
			same_label = class_label
		else:
			same_label = -1
			break
	return same_label

def calculateEntropy(dataset):
	# -sigma (c1/(c1+c2) *log2(c1/(c1+c2)))
	#c1, c2 are counts so will be number of rows in dataset
	last_column = dataset.shape[1]
	total_count = dataset.shape[0]
	last_column = dataset.shape[1]

	class_1 = dataset[0][last_column - 1]
	class1_count = 0
	class2_count = 0
	#print dataset[:,last_column - 1]
	for i in dataset:
		#print i[last_column - 1:]
		if i[last_column - 1:] == class_1:
			class1_count += 1
		else:
			class2_count += 1


	#print class2_count
	#print class1_count

	#if total_count == 0:
		#do something
	#else:
	total_count = class1_count+class2_count
	if total_count == 0: 
		first = 0
		second = 0
	else:
		first = float (class1_count) / float (total_count)
		second = float (class2_count) / float (total_count)
	if first == 0:
		log_first = 0.0 
	else: 
		log_first = np.log2(first)
	if second == 0: 
		log_second = 0.0
	else: 
		log_second = np.log2(second)
	entropy = 0.0
	#if first !=0 and second != 0:
	entropy = -1 * first * log_first - second * log_second
	return entropy

def forFloatVal(dataset,column,org_info):
	#print "Column: " +str(column)
	dataset = np.array(dataset)
	total_number = dataset.shape[0]
	mean1 = 0.0
	for i in dataset:
		mean1 += float(i[column])
	mean1 = mean1/total_number
	#print mean1
	#print column
	#print total_number
	t1_count = 0
	t2_count = 0
	class1_count = 0
	class2_count = 0
	
	#calculating entropy
	for i in dataset:
		#print i[-1]
		if float(i[column])<=mean1:
			t1_count+=1
			if i[-1]==0: class1_count+=1
			else: class2_count+=1

	#print "Label 2 count for this feature: " + str(class2_count)
	#print "Label 1 count for this feature: " + str(class1_count)
	total_count = class1_count+class2_count
	if total_count == 0: 
		first = 0
		second = 0
	else:
		first = float (class1_count) / float (total_count)
		second = float (class2_count) / float (total_count)
	if first == 0:
		log_first = 0.0 
	else: 
		log_first = np.log2(first)
	if second == 0: 
		log_second = 0.0
	else: 
		log_second = np.log2(second)
	entropy1 = 0.0
	#if first !=0 and second != 0:
	entropy1 = -1 * first * log_first - second * log_second
	
	class1_count = 0
	class2_count = 0
	total_count = 0
	#calculating entropy
	for i in dataset:
		#print i[-1]
		if float(i[column])>mean1:
			t2_count+=1
			if i[-1]==0: class1_count+=1
			else: class2_count+=1
	#print "Label 2 count for this feature: " + str(class2_count)
	#print "Label 1 count for this feature: " + str(class1_count)
	total_count = class1_count + class2_count
	if total_count == 0: 
		first = 0
		second = 0
	else:
		first = float (class1_count) / float (total_count)
		second = float (class2_count) / float (total_count)
	if first == 0:
		log_first = 0.0 
	else: 
		log_first = np.log2(first)
	if second == 0: 
		log_second = 0.0
	else: 
		log_second = np.log2(second)
	entropy2 = 0.0
	#if first !=0 and second != 0:
	#print "L1 & L2"
	#print t1_count
	#print t2_count

	entropy2 = -1 * first * log_first - second * log_second

	info = 0.0
	info = float(t1_count)/float(t1_count+t2_count) * entropy1 + float(t2_count)/float(t1_count+t2_count) * entropy2

	
	column_gain[column] = org_info - info




def startTreeBuild(node, parent, attribute_count):
	#check if it's a leaf node
	node.dataset = np.array(node.dataset)
	#print node.dataset.shape
	class_label = node.dataset[0][attribute_count - 1]
	#print "Classlabel: " + str(class_label)
	org_info = calculateEntropy(node.dataset)
	new_label = checkLabel(node.dataset, class_label)

	#print "KKKKKKKKKKKK "+ str(new_label)
	if new_label != -1:
		#Leaf node
		node.isLeaf = True
		node.label = new_label
		node.column = -1
		
		
	else:
		#print node.dataset.shape[1]-1
		#print int(node.dataset.shape[1]*0.2)
		for i in range(int(node.dataset.shape[1]*0.4)):
			cr = -1
			while cr ==-1 or cr in node.cr_visited:
				cr = random.randint(0,node.dataset.shape[1]-1)
			node.cr_visited.append(cr)
			#print "visiting cr: "
			if isinstance(node.dataset[0][cr], float): 
				#print "YAAYYYY " + str(i)
				forFloatVal(node.dataset,cr,org_info)
			else:
				function(node.dataset,i)
		#print column_gain
		#print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Out of function &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
		#print "Each column gain: "
		
		sorted_gains = sorted(column_gain.items(), key=operator.itemgetter(1), reverse = True)
		minimum_index = sorted_gains[0][0]
		node.column = minimum_index
		#print sorted_gains
		#####MEAN1 PART
		total_number = node.dataset.shape[0]

		mean1 = 0.0
		for j in node.dataset:
			mean1 += float(j[minimum_index])
		mean1 = mean1/total_number
		#print mean1
		features = set(node.dataset[:,minimum_index])
		node.splitValue = mean1

		splitFeature = dict()

		#print uniqueValues
		for f in range(2):
			splitFeature[f] = []

		less = 0
		more = 0
		#print "Mean1 is"+str(mean1)
		for i in range(node.dataset.shape[0]):

			if node.dataset[i][minimum_index]<= mean1:
				#print node.dataset[i][minimum_index]
				less+=1
				splitFeature[0].append(node.dataset[i])
			else:
				more+=1
				splitFeature[1].append(node.dataset[i])
		#print "PPPPPPPPPPSSSSSSSQQQQQQQQQQ$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
		#print less+more
		#print splitFeature
		#for p in splitFeature:
			#print splitFeature[p]
		
		for p in splitFeature:
			r = np.array(splitFeature[p])
			#print "PPPPPPPPPPSSSSSSSQQQQQQQQQQ$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
			#print p
			#print r
			child = DecisionTree(r, node.column) #or me parent
			child.column = -1
			node.children[p] = child
			#print "hey hey "
			#print node.children
			startTreeBuild(child,node,attribute_count)
		#startTreeBuild(child, None, attribute_count)


def crossValidation(data,k):
	x = data
	#np.random.shuffle(x)
	divider = int(x.shape[0]/k)
	if k == 1:
		divider = 50
	#print divider
	j = 0;
	q = divider
	for i in xrange(k):
		#print "LALALALALALALALALALALA!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@:  "+str(i)
		#print q
		#print x[j:q,:]
		test = x[j:q,:]
		
		x_rows = x.view([('', x.dtype)] * x.shape[1])
		test_rows = test.view([('', test.dtype)] * test.shape[1])
		train = np.setdiff1d(x_rows, test_rows).view(x.dtype).reshape(-1, x.shape[1])

		j +=divider
		q +=divider
		yield train, test
		#print "Test"
		#print test
		#print "Training"
		#print train

def findHighest(dataset): 
	dataset = np.array(dataset)
	ct = Counter(dataset[:, len(dataset[0]) - 1])
	return ct.most_common(1)[0][0]

def predictLabelForTest(root, t):
	node = root

	while node.column != -1:
		#print "this is node.dataset "
		#print node.dataset
		nextSplit = float(t[node.column])
		total_number = node.dataset.shape[0]

		mean1 = 0.0
		#print "Nextsplit: " + str(nextSplit)
		for j in node.dataset:
			mean1 += float(j[node.column])
		mean1 = mean1/total_number
		if nextSplit <= mean1:
			node = node.children[0]
		else:
			node = node.children[1]
		#print "nextSplit is " + str(nextSplit)
		#print "popopopop"
		#print "test is "+ str(t)
		#if nextSplit not in node.children:
		#	print "nextSplit NOT IN NODE>CHILDREN"
		#	return findHighest(node.dataset)
		#print "hjhfahskjndmkjhzx node children & dataset"
		#print node.children[nextSplit]
		#node = node.children[nextSplit]
		#print node.dataset
	#print node.label
	
	return node.label

def compare(final1,test):
	#print final1
	row_count = test.shape[0]
	tp_tn = 0
	accuracy = 0

	for i in range(row_count):
		if final1[i] == test[i][-1]:
			tp_tn += 1
	accuracy = float(tp_tn)/row_count
	
	return accuracy*100



def getAccuracy(node,test,weight_list):
	#print "(((((((((((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))))))))))))"
	predict_class = dict()
	row_count = test.shape[0]
	tp_tn = 0
	no = 0
	accuracy = 0
	for i in range(row_count):
		predicted_label = predictLabelForTest(node, test[i])
		predict_class[i] = predicted_label

		if predicted_label == test[i][-1]:
			
			tp_tn += 1
		else:
			no +=1
	accuracy = float(tp_tn)/row_count
	
	return accuracy*100,predict_class, no

def getOthers(test,predict_class,cl):
	#print cl
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	row_count = test.shape[0]
	#print "Test class is: "+str(test[:-1])
	for i in range(row_count):
		#print predict_class[i]
		#print "Test class is: "+str(test[i][-1])
		if predict_class[i] == test[i][-1] and predict_class[i] == cl:
			#print "YYYYYYAAAAAAAAHHHHHHHOOOOOOOOOOOOO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
			tp+=1
		elif predict_class[i] != cl and test[i][-1] == cl:
			fn+=1
		elif predict_class[i] == cl and test[i][-1] != cl:
			fp+=1
	
	if tp == 0 and fp == 0: precision = 0.0
	else:
		precision = (float(tp)/float(tp+fp))*100
	if tp == 0 and fn == 0: recall = 0.0
	else:
		recall = (float(tp)/float(tp+fn))*100
	print "PRESICION, RECALL AND F-MEASURE for "+str(cl)
	print "Precision: "+str(precision)
	print "Recall: "+str(recall)
	if precision == 0 and recall == 0: f_measure = 0
	else:
		f_measure = (2*recall*precision)/(precision+recall)
	print "F measure: "+str(f_measure)
	return precision,recall,f_measure
	
	
def updateWeights(weight_list,predict_class,test,error):
	row_count = test.shape[0]
	for i in range(row_count):
		if predict_class[i] != test[i][-1]:
			#print "BLAAAAAAAAAAAHHHHHHHHHHHHHH"
			#print weight_list[i]
			weight_list[i] *= weight_list[i] * math.exp(error)
			#print weight_list[i]
	denom = sum(weight_list.values())
	#print denom
	for i in range(row_count):
		weight_list[i] = weight_list[i]/ float(denom)
	#print len(weight_list)
	return weight_list


avg_acc = 0
which_dataset = 0
print "Boosting for Decision Tree"
path = raw_input("Enter path of dataset 1: ")
#path = "/Users/falgunibharadwaj/Downloads/CSE601/Project3/project3_dataset1.txt"
#data = np.array(data)
which_dataset = 1
#dt = np.dtype([('f0','S10'),('f1','S10'),('f2','S10'),('f3','S10'),('f4',float)])
data = []
if which_dataset == 2:
	data = np.genfromtxt(path, delimiter=",") #tuple with int
else:
	data = np.genfromtxt(path, delimiter="	")
k = 5
we = 0
dataset = np.array(data) #matrix with string
column_gain = dict()

k = int(raw_input('Enter value of k for k-fold Cross Validation: '))
pr0 = 0
re0 = 0
fm0 = 0
pr1 = 0
re1 = 0
fm1 = 0
global_prediction_list = list()
print "STARTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
print dataset.shape
weight_list = dict()
predict_class = dict()
w = list()

for train, test in crossValidation(dataset,k):
	print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Cross Validation: " + str(we)
	#print len(train)
	#print test.shape
	#print "HAHAHAHA"
	attribute_count = dataset.shape[1]
	
	number_of_trees = 5 #ask user to enter
	rows_remove = 0.3
	col_remove = 0.8
	alpha = 0
	#new_train
	global_prediction_list[:] =[]
	w[:] = []
	for i in range(number_of_trees):
		print "FOR TREEEEE "+str(i)
		random.shuffle(train)
		l = int(train.shape[0]*rows_remove)
		#l = 10
		#print l
		train1 = train[:l,]
		for i in range(train.shape[0]):
			weight_list[i] = float (1)/float(l)
		#print weight_list
		node = DecisionTree(train1, [])
		startTreeBuild(node, None, attribute_count)
		#print "We are done "

		testAccuracy, predict_class, no = getAccuracy(node, train, weight_list)
		#if testAccuracy == 100:
		#	number_of_trees
		print "Accuracy is: "+str(testAccuracy) +"%"
		#print no
		error = float (no)/float(l)
		if error == 0: break
		alpha_m = math.log(float(1-error)/error)
		alpha+=alpha_m
		#print alpha_m

		weight_list = updateWeights(weight_list,predict_class,train,error)
		
		w.append(weight_list)
		#print "KKKK"
		#print len(w)

		global_prediction_list.append(predict_class)
		#p1,r1,f1 = getOthers(node,test,predict_class,0)
		#p2,r2,f2 = getOthers(node,test,predict_class,1)
		#print p1
		#print r1
		#pr0+=p1
		#pr1+=p2
		#fm0+=f1
		#fm1+=f2
		#re0+=r1
		#re1+=r2
		avg_acc += testAccuracy
		#print "Accuracy is " + str(testAccuracy) +"%"
	#print len(global_prediction_list)
	#final_list = dict()
	#last_label = pd.DataFrame(global_prediction_list)
	#numpy_label = last_label.as_matrix()
	#md = numpy_label.T
	#print w
	#for q in md:
	weight = pd.DataFrame(w)
	wm = weight.as_matrix()
	#print wm
	final_list = mode(wm,axis = 1)
	#print final_list
	#ac = 
	#avg_acc = 
	#tree_acc = compare(final_list[0].flatten(),test)
	#p1,r1,f1 = getOthers(test,final_list[0].flatten(),0)
	#p2,r2,f2 = getOthers(test,final_list[0].flatten(),1)
	#print p1
	#print r1
	#pr0+=p1
	#pr1+=p2
	#fm0+=f1
	#fm1+=f2
	#re0+=r1
	#re1+=r2
	#avg_acc+=tree_acc
	#print tree_acc

	#if we > 2: break
	we += 1



#for col in numpy_label.T:





#print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RESULTS!!"
#print "Total Average Accuracy is:			" +str(float(avg_acc)/k) +"%"
