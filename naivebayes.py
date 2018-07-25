import numpy as np
from scipy.stats import norm
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cross_validation import KFold

def main():
    filename = open(raw_input('Filename???? : '), 'r')
    #filename = "/Users/debik/Documents/UB/3rd Sem/CSE601/proj3/project3_dataset1.txt"
    dataset = [line.split('\t') for line in filename.readlines()]
    processed_data = [list(x) for x in zip(*dataset)]
    
    for row in processed_data:
        indexing = processed_data.index(row)
        for i in row:
             if any(char.isalpha() for char in i) == True:
                 get_list = preprocessing.LabelEncoder()
                 get_list.fit(list(set(row)))
                 data_temp =  get_list.transform(row).tolist()
                 for i in range(0, len(data_temp)):
                     data_temp[i] = int(data_temp[i]) + 1
                 processed_data[indexing] = data_temp
                 
    dataset = [list(x) for x in zip(*processed_data)]

    for item in dataset:
        item[-1] = item[-1][:-1]
        for i in range(0,len(item)):
            item[i] = float(item[i])

    data = np.array(dataset)
    total_X = data[:,:-1]
    total_Y = data[:,-1:]
    
    accuracy_summary, precision_summary, recall_summary, f1_summary = CrossValidation(total_X, total_Y)
    
    return accuracy_summary, precision_summary, recall_summary, f1_summary
    
def kfold(data,k):
	x = data
	#np.random.shuffle(x)
	divider = int(x.shape[0]/k)
	#print divider
	j = 0;
	q = divider
	for i in xrange(k):
	   test = x[j:q,:]
		
	   x_rows = x.view([('', x.dtype)] * x.shape[1])
	   test_rows = test.view([('', test.dtype)] * test.shape[1])
	   train = np.setdiff1d(x_rows, test_rows).view(x.dtype).reshape(-1, x.shape[1])

	   j +=divider
	   q +=divider
	   yield train, test

def CrossValidation(total_X, total_Y):
    
    # split data into random train and test subsets
    train_X, test_X, train_Y, test_Y = train_test_split(total_X, total_Y, test_size=0.33, random_state=42)

    train_x, train_y, test_x, test_y =  np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

    accuracy_list, precision_list, recall_list, f1_list = ([] for i in range(4))
    
    accuracy_summary = precision_summary = recall_summary = f1_summary = 0
    count = 1
    
    
    # divide all the samples in k groups of samples, called folds
    cross_validate = KFold(total_X.shape[0], n_folds=10, shuffle=False, random_state=0)

    for training_i, testing_j in cross_validate:

        train_x, train_y, test_x, test_y = total_X[training_i], total_Y[training_i], total_X[testing_j], total_Y[testing_j]
        
        print "\n"
        print "************** Iteration:", count, "****************"
        accuracy,precision,recall,f1 = NaiveBayes(train_x, train_y, test_x, test_y)
        count += 1
        accuracy_summary += accuracy
        accuracy_list.append(accuracy)
        
        precision_summary += precision
        precision_list.append(precision)
        
        recall_summary += recall
        recall_list.append(recall)
        
        f1_summary += f1
        f1_list.append(f1)
        
    return accuracy_summary, precision_summary, recall_summary, f1_summary
    


def NaiveBayes(train_x, train_y, test_x, test_y):

    predicted_Y, x_0, x_1 = ([] for i in range(3))

    for i in range(0, train_x.shape[0]):
        if(int(train_y[i][0]) == 1):
            x_1.append(train_x[i])
        else:
            x_0.append(train_x[i])

   # Calculate mean and standard_deviation 
    mean_x_0 = np.array(x_0).mean(axis = 0, dtype = np.float32)
    stddev_x_0 = np.array(x_0).std(axis = 0, dtype = np.float32)
    mean_x_1 = np.array(x_1).mean(axis = 0, dtype = np.float32)
    stddev_x_1 = np.array(x_1).std(axis = 0, dtype = np.float32)

    for t in range(0,test_x.shape[0]):
        zero_prob = one_prob = 1

        # calculating Probability Density Function with Prior probability
        for i in range(0,test_x.shape[1]):
            zero_prob *= norm(mean_x_0[i], stddev_x_0[i]).pdf(test_x[t,i])

        zero_prob *= float(len(x_0)) / (len(x_0) + len(x_1))

        for i in range(0,test_x.shape[1]):
            one_prob *= norm(mean_x_1[i], stddev_x_1[i]).pdf(test_x[t,i])

        one_prob *= float(len(x_1)) / (len(x_0) + len(x_1))
        #print zero_prob
        #print one_prob

        if zero_prob > one_prob:
            predicted_Y.append(0)
        else:
            predicted_Y.append(1)

    accuracy,precision,recall,f1 = PerformanceCalc(predicted_Y , test_y)

    return accuracy,precision,recall,f1
    

def PerformanceCalc(predicted_y , test_y):

    # Evaluate performance metrics: precison, recall, f1_measure 
    totalSum=precisionSum=precisionTotal=recallSum=recallTotal=0

    for i in range(0,len(predicted_y)):
        if predicted_y[i] == test_y[i]:
            totalSum += 1
        #print totalSum

        # Precision
        if predicted_y[i] == 1:
            precisionTotal += 1
            if predicted_y[i] == test_y[i]:
                precisionSum += 1
            #print precisionSum

        # Recall
        if test_y[i] == 1:
            recallTotal += 1
            if predicted_y[i] == test_y[i]:
                recallSum += 1
            #print recallSum
                
    accuracy = float(totalSum)/len(predicted_y)
    print "Accuracy: ", accuracy

    precision = float(precisionSum)/precisionTotal
    print "Precision: ", precision
    
    recall = float(recallSum)/recallTotal
    print "Recall: ", recall
    
    f1_measure = float(2*(precision*recall)/(precision+recall))
    print "F1_measure: ", f1_measure
    
    return accuracy,precision,recall,f1_measure
    
    
   
if __name__ == '__main__':
    avg_accuracy, avg_precision, avg_recall, avg_f1 = main()
    
    print "^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*^*"
    print "Average Accuracy :\t" , avg_accuracy/10
    print "Average Precision :\t" , avg_precision/10
    print "Average Recall :\t" , avg_recall/10
    print "Average F1_measure :\t" , avg_f1/10

   
    