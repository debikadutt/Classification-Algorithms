package kNNAlgo;

import java.io.*;
import java.util.*;
import java.math.*;

public class kNN {
	
	static HashMap<Integer, ArrayList<Double>> data = new HashMap<Integer, ArrayList<Double>>();
	static ArrayList<Integer> training;
	static ArrayList<Integer> testing;
	static HashMap<Integer, Integer> actualClass = new HashMap<Integer, Integer>();
	static HashMap<Integer, Integer> predictedClass;
	static float accuracySum = 0, precisionSum=0, recallSum=0, f1Sum=0;
	
	public static void main(String[] args) {
		
		try{
			BufferedReader br = new BufferedReader(new FileReader("/Users/mtappeta/Desktop/project3_dataset2.csv"));
			int k=7;
			int crossvalidation = 10;

			String line = null;
			
			int id = 0;
			
			
			while((line = br.readLine())!=null){
				
				id++;
				String[] sep = line.split(",");
				ArrayList<Double> attributes = new ArrayList<Double>();
				
				for(int i=0;i<sep.length-1;i++){
					if(sep[i].equals("Present")){
						attributes.add(1.0);
					}else if(sep[i].equals("Absent")){
						attributes.add(0.0);
					}else
						attributes.add(Double.parseDouble(sep[i]));
				}
				
				//STORING ID and attribute vales as <key,value> pair
				data.put(id, attributes);
				
				//STORING ID and the class it belongs to as <key,value> pair
				actualClass.put(id, Integer.parseInt(sep[sep.length-1]));
				
			}
			
			for(int i=1; i<=crossvalidation; i++){		
					
				predictedClass = new HashMap<Integer, Integer>();
				
				splitData(data.size(),i,crossvalidation);
				System.out.println();
				
				for(int j=0;j<testing.size();j++){
						ArrayList<Integer> kNeighbours = euclidean(testing.get(j),k); 
						//for(int i=0;i< kNeighbours.size();i++)
							//System.out.println(kNeighbours.get(i));
						int cluster = findCluster(kNeighbours,k);
						if(cluster == -1){
							System.out.println("Select Different value for k(preferably an odd number)");  //CHECK
							break;
						}else  
							predictedClass.put(testing.get(j), cluster) ;       
						
				}
				calculatePN(i);

			}
			
			
			System.out.println("\nAverage Accuracy = "+accuracySum/(float)crossvalidation);
			System.out.println("Average Precison = "+precisionSum/(float)crossvalidation);
			System.out.println("Average Recall = "+recallSum/(float)crossvalidation);
			System.out.println("Average F1 measure = "+f1Sum/(float)crossvalidation);
			
		}catch(FileNotFoundException e){
			e.printStackTrace();
		}catch(IOException s){
			s.printStackTrace();
		}

	}
	
	//Splits data into training and testing data
	public static void splitData(int size, int partition, int cv){
		
		training = new ArrayList<Integer>();
		testing = new ArrayList<Integer>();;
		int max = (data.size()/cv)*partition ;
		int min = (data.size()/cv)*(partition-1) + 1;
		for(int i=min;i<=max;i++){
			testing.add(i);
		}
		
		for(int j=1;j<data.size();j++){
			if(!testing.contains(j))
				training.add(j);
		}
		/*
		for(int k=0;k<testing.size();k++)
			System.out.println(testing.get(k));*/
	}
	
	public static ArrayList<Integer> euclidean(int testid, int k){
		//System.out.println("called");
		double[] distance = new double[training.size()];
		HashMap<Double, Integer> neighbourDist = new HashMap<Double, Integer>();
		int index = -1;
		
		for(int i=0;i<training.size();i++){	//finding distance of test sample from every point in training sample
			double sum = 0;
			
			for(int j=0;j<data.get(1).size();j++){
				
				double diff = data.get(training.get(i)).get(j) - data.get(testid).get(j);
				double square = diff*diff;
				sum = sum + square;
				
			}
			double sqrt = Math.sqrt(sum);
			distance[++index]  = sqrt;
			neighbourDist.put(sqrt, training.get(i));
			
		}
		
		//SORTING THE ARRAY
		for(int p=0;p<distance.length;p++){

			for(int q=0;q<distance.length;q++){
				if(distance[p]<distance[q]){
					double temp;
					temp = distance[p];
					distance[p] = distance[q];
					distance[q] = temp;
				}
			}
				
		}

		ArrayList<Integer> kNeighbours = new ArrayList<Integer>();
		for(int y=0;y<k;y++){

			kNeighbours.add(neighbourDist.get(distance[y]));
		}
		
		return kNeighbours;
	}
	
	
	public static int findCluster(ArrayList<Integer> neighbours, int k){
		
		int one = 0, zero = 0;
		
		for(int i=0;i<k;i++){
			if(actualClass.get(neighbours.get(i)) == 0)
				zero++;
			else if(actualClass.get(neighbours.get(i)) == 1)
				one ++;
		}
		
		if(zero == one){
			int c = findCluster(neighbours,k-1);
			return c;
		}
		else if(zero>one)
			return 0;
		else if(one>zero) return 1;
		else return -1;
	}
	
	public static void calculatePN(int cv){
		
		int TP=0,TN=0,FN=0,FP=0;
		
		for(int i=0;i<testing.size();i++){
			
			int actual= actualClass.get(testing.get(i));
			int predicted= predictedClass.get(testing.get(i));
			
			if(actual == 0 && predicted == 0){
				TN++;
				
			}
		
			else if(actual == 0 && predicted == 1)
				FP++;
			else if(actual == 1 && predicted == 0)
				FN++;
			else if(actual == 1 && predicted == 1)
				TP++;
		}
		
		calculatePerformance(TP,TN,FP,FN,cv);
	}
	
	public static void calculatePerformance(int tp,int tn,int fp,int fn,int cv){
		
		float TP = (float)tp, TN = (float)tn, FP = (float)fp, FN = (float)fn; 
		float accuracy = 0, precision = 0, recall = 0, f1 = 0;
		
		accuracy = (TP+TN)/(TP+TN+FP+FN);
		accuracySum = accuracySum + accuracy;
		precision = TP/(TP+FP);
		precisionSum = precisionSum + precision;
		recall = TP/(TP+TN);
		recallSum = recallSum + recall;
		f1 = (2*TP)/((2*TP)+TN+FP);
		f1Sum = f1Sum + f1;
		
		System.out.println("------Performance Metrics for Cross-validation "+cv+"-----------");
		System.out.println("Accuracy: "+accuracy);
		System.out.println("Precision: "+precision);
		System.out.println("Recall: "+recall);
		System.out.println("F1-measure: "+f1);
		
		
	}

}



















