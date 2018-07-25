import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from random import shuffle
from StringIO import StringIO
from numpy  import array
import csv
import operator
from collections import Counter

class DecisionTree:

	def __init__(self, data, parent):#change as required
		self.dataset = data
		self.isLeaf = None #check
		self.label = None
		self.myName = None
		self.parentNode = parent
		self.column = None
		self.children= dict()
		self.id = -1
		self.splitValue = []

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
	mean = 0.0
	for i in dataset:
		mean += float(i[column])
	mean = mean/total_number
	#print mean
	#print column
	#print total_number
	t1_count = 0
	t2_count = 0
	class1_count = 0
	class2_count = 0
	
	#calculating entropy
	for i in dataset:
		#print i[-1]
		if float(i[column])<=mean:
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
		if float(i[column])>mean:
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
		global node_id
		for i in range(node.dataset.shape[1]-1):
			if isinstance(node.dataset[0][i], float): 
				#print "YAAYYYY " + str(i)
				forFloatVal(node.dataset,i,org_info)
			else:
				print " ia am "
				print i
				function(node.dataset,i)
		#print column_gain
		#print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&& Out of function &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
		#print "Each column gain: "
		
		sorted_gains = sorted(column_gain.items(), key=operator.itemgetter(1), reverse = True)
		minimum_index = sorted_gains[0][0]
		node.column = minimum_index
		#print sorted_gains
		#####MEAN PART
		total_number = node.dataset.shape[0]

		mean = 0.0
		for j in node.dataset:
			mean += float(j[minimum_index])
		mean = mean/total_number
		#print mean
		features = set(node.dataset[:,minimum_index])
		node.splitValue = mean
		#if node_id == 0: 
		#	print "######### TREE #########"
		#	print "NODE-ID -> NODE_NAME/FEATURE"
		#	print "[ node children ]"
		#	print " "
		#print str(node_id)+" -> "+str(minimum_index)
		node_id+=1
		#print node_id
		#print list(features)

		splitFeature = dict()

		#print uniqueValues
		for f in range(2):
			splitFeature[f] = []

		less = 0
		more = 0
		#print "Mean is"+str(mean)
		for i in range(node.dataset.shape[0]):

			if node.dataset[i][minimum_index]<= mean:
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
			child.myName = mean
			node.children[p] = child
			#print "hey hey "
			#print node.children
			startTreeBuild(child,node,attribute_count)
		#startTreeBuild(child, None, attribute_count)

def function(dataset,column):
	#creating feature_dict which stores individual values from a feature & number of times it's occurring (T1,5)(T2,4)
	dataset = np.array(dataset)
	features = set(dataset[:,column])
	feature_dict = dict()
	for f in features:
		feature_dict[f] = 0
	
	total_features = 0

	for f in features:
		for i in dataset:
		#print i[0]
		
			#print f
			if i[column] == f:
				#print i
				feature_dict[f] +=1
				total_features += 1

	#print "Total number of features in column " +str(column)+" : " + str(total_features)
	last_column = dataset.shape[1]
	#print "Feature dict() is: " + str(feature_dict)

	entropy_list = dict()
	for f in features:
		#total_count = dataset.shape[0]
		last_column = dataset.shape[1]

		class_1 = dataset[0][last_column - 1]
		class1_count = 0
		class2_count = 0
		#print "Feature Name: "+ str(f)
		for i in dataset:
			if i[column] == f:
				#print "&&&&&&&&&&&&&&&"+i[0]
				#print "Dataset matching feature: " + str(i[last_column - 1:])
				if i[last_column - 1:] == class_1:
					class1_count += 1
				else:
					class2_count += 1


		#print "Label 2 count for this feature: " + str(class2_count)
		#print "Label 1 count for this feature: " + str(class1_count)
		total_count = class1_count + class2_count
		#print "Total (Denominator): " + str(total_count)
		#if total_count == 0:
			#do something
		#else:
		#can also put this in for loop
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
		#print "Entropy for feature subclass " + str(f) +" is: " + str(entropy)
		
		entropy_list[f] = entropy

	#for f in feature_dict
	#print "Entropy List for this column: "+ str(column)
	#print entropy_list
	info = 0.0
	for f in features:
		info += (float (feature_dict[f])/total_features ) * entropy_list[f]

	org_info = calculateEntropy(dataset)
	#print "Original dataset Information Gain: " + str(org_info)
	#print "This column information gain: " + str(info)
	column_gain[column] = org_info - info



def crossValidation(data,k):
	x = data
	#np.random.shuffle(x)
	divider = int(x.shape[0]/k)
	if k == 1:
		divider = 1
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


def predictLabelForTest(root, t):
	node = root

	while node.column != -1:
		#print "this is node.dataset "
		#print node.dataset
		nextSplit = float(t[node.column])
		total_number = node.dataset.shape[0]

		mean = 0.0
		#print "Nextsplit: " + str(nextSplit)
		for j in node.dataset:
			mean += float(j[node.column])
		mean = mean/total_number
		if nextSplit <= mean:
			node = node.children[0]
		else:
			node = node.children[1]
		#print "nextSplit is " + str(nextSplit)
		#print "popopopop"
		#print "test is "+ str(t)
		
		#print node.children[nextSplit]
		#node = node.children[nextSplit]
		#print node.dataset
	#print node.label
	
	return node.label

def getAccuracy(node,test,train):
	#print "(((((((((((((((((((((((((((((((((((((((()))))))))))))))))))))))))))))))))))))))))"
	predict_class = dict()
	row_count = test.shape[0]
	tp_tn = 0
	accuracy = 0
	for i in range(row_count):
		predicted_label = predictLabelForTest(node, test[i])
		predict_class[i] = predicted_label

		if predicted_label == test[i][-1]:
			tp_tn += 1
	accuracy = float(tp_tn)/row_count
	actual_class = set(node.dataset[:,-1])
	#print actual_class
	#print "Predict class is: "+str(predict_class)
	#print "Test is: "
	#print test
	return accuracy*100,predict_class

def getOthers(node,test,predict_class,cl):
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
	
	
avg_acc = 0
which_dataset = 0
print "Decision Tree for dataset 2!"
path = raw_input("Enter path of dataset 2: ")
#path = "/Users/falgunibharadwaj/Downloads/CSE601/Project3/project3_dataset2.txt"
#data = np.array(data)
which_dataset = 2
reader = csv.reader(open(path),delimiter="\t")
data=list(reader)
#print len(data)

for j in data:
	if j[4] == "Absent":
		j[4] = 0
	elif j[4] == "Present":
		j[4] = 1

k = 5
we = 0
dataset = np.array(data) #matrix with string
dataset = dataset.astype(np.float)
column_gain = dict()
k = int(raw_input('Enter value of k for k-fold Cross Validation: '))
pr0 = 0
re0 = 0
fm0 = 0
pr1 = 0
re1 = 0
fm1 = 0
node_id = 0

print "STARTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
print dataset.shape
print dataset[0]

predict_class = dict()

for train, test in crossValidation(dataset,k):
	node_id = 0
	print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Cross Validation: " + str(we)
	#print len(train)
	#print test.shape
	#print "HAHAHAHA"
	attribute_count = dataset.shape[1]
	node = DecisionTree(train, [])
	startTreeBuild(node, None, attribute_count)
	we += 1
	#print node.children[0].column
	#predict for test?train?
	#get accuracy

	testAccuracy, predict_class = getAccuracy(node, test, train)
	#print predict_class
	p1,r1,f1 = getOthers(node,test,predict_class,0)
	p2,r2,f2 = getOthers(node,test,predict_class,1)
	#print p1
	#print r1
	pr0+=p1
	pr1+=p2
	fm0+=f1
	fm1+=f2
	re0+=r1
	re1+=r2
	avg_acc += testAccuracy
	print "Accuracy is " + str(testAccuracy) +"%"
	#if we > 2: break


print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ RESULTS!!"
print "Total Average Accuracy is:			" +str(float(avg_acc)/k) +"%"
print "Total Average Precision for 0 as positive is: 	" +str(float(pr0)/k) +"%"
print "Total Average Recall for 0 as positive is: 	" +str(float(re0)/k) +"%"
print "Total Average F-Measure for 0 as positive is: 	" +str(float(fm0)/k) +"%"
print "Total Average Precision for 1 as positive is: 	" +str(float(pr1)/k) +"%"
print "Total Average Recall for 1 as positive is: 	" +str(float(re1)/k) +"%"
print "Total Average F-Measure for 1 as positive is: 	" +str(float(fm1)/k) +"%"


