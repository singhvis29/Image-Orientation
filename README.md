# Image-Orientation
Finding the orientation (rotation) of the image using  Random Forest, kNN and Adaboost algorithms

### Data:
A dataset of images from the Flickr photo sharing website where each image is converted to 8 x 8 x 3 (the third dimension is because color images are stored as three separate planes – red, green, and blue) = 192 vector i.e. one image per row

### Implementation:
The prediction is done by implementing kNN, Random Forest, and Adaboost Algorithm from scratch (no python ML libraries were used). Implementation of these algorithms have been done as follows-

### Accuracy:
Random Forest - 65.22 <br>
Adaboost - 68.717 <br>
kNN - 71.262 <br>

#### Random Forest:
Reference for the program: https://www.youtube.com/watch?v=LDRbO9a6XPU <br>
Random Forest Classification is a supervised learning algorithm. It is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set. <br>
##### Implementation- <br>
**Training-** <br>
1. After inputting the training data as a pandas dataframe, we convert it to a numpy array <br>
2.  For building a forest of decision trees, the inputs taken from the user are- <br>
a. Max Depth: the depth to which the nodes are split <br>
b. Sample size ratio: the ratio of the dataset which is used to build trees of the forest. This is selected randomly. <br>
c. Number of trees: number of trees which are built for the forest <br>
d. Number of features: the number of features from the full set of 192 features to consider building the tree. These are selected randomly <br>
3. In the program, the features have been set as the best and optimum (running time is reasonable) <br>
4. Before initializing a tree, subsample and features of the dataset are selected for which the tree must be built <br>
5. Building a tree- <br>
a. The subsample, subset of features is passed to the function for building a decision tree along with other parameters <br>
b. Best split is found by checking the medians of all the columns considered (only the column median is considered to reduce the running time). Best split is calculated by calculating the Information Gain from the previous state to the state after the split. The information gain is the difference in the Gini Impurity from the initial state to the state after the split. <br>
c. For all the branches split at the node a recursive tree is built until the maximum depth is reached or if the gain at the node after splitting is zero <br>
d. The tree which is build is saved as reference to all the decision nodes and leaves <br>
6. A forest is built by training number of decision trees. This is the number given by the user. <br>
**Testing-** <br>

1. For each row in the training dataset, the trees which are built in the training step and are traversed to give a decision at the leaf nodes. The class which is maximum at the leaf node is the predicted class for that tree. Similarly, all the trees are traversed for that row and the predicted class is obtained from all the tress. The final predicted class for the row is the class which is voted majority by all the trees. <br>
2. This step is performed for all the rows to predict a class for all the rows. <br>
3. Accuracy is calculated as number of predictions correct by total number of observations in the testing dataset 

#### ADABOOST:
We have taken 200 weak decision stumps <br>
We have implemented one VS one in adaboost. The training data has been converted into 6 different combination, <br>
1. Containing rows where label = 0 or 90 <br>
2. Containing rows where label = 0 or 180 <br>
3. Containing rows where label = 0 or 270 <br>
4. Containing rows where label = 90 or 180 <br>
5. Containing rows where label = 90 or 270 <br>
6. Containing rows where label = 180 or 270 <br>
Each of the 6 combinations calls the adaboost function where the actual algorithm is written <br>
There are 192C2 number of possible stumps, we have chosen randomly 500 stumps as that was the one that gave us the highest accuracy. <br> 
The decision stumps have been taken as given in the pdf, A comparison is made between 2 features and then depending on the results we assign it to one of the two classes. <br>
Initially we assign the weight W=1/N, where n= Num of training points <br>
And k=0 <br>
The hypothesis function is called which does the comparison and returns a list containing +1 and -1  (-1 if the value on left side is greater, +1 if value on right is greater) <br>
Now the value of hypothesis is compared with the labels of all the rows.(we assume value on left to always be class-1 and value on right to always be class 1, for first model 0-> -1 90 ->+1) <br>
If the value of hypothesis does not match with the label, we update the error <br>
error=error+w[j] <br>
This is done for all the rows and the final error for the hypothesis is calculated <br>
Now If the value of hypothesis and labels match we update the weights of the matched label (we reduce it) <br>
 w[m] = w[m]*(error/(1-error))    <br>
Finally we normalize the weights <br>
update the value of a with  a[i]=math.log((1-error)/error) <br>
REPEAT the steps for all the hypothesis     <br>
The “hypothesis” along with value of “a” for all the 6 combinations are stored in a pickle file  <br>
**For testing**, we do one Vs one for all the 6 combinations <br>
For the first combination, we assign the sign (0 or 90) to all the rows of the data depending in the value of a(0 ->-1 , 90- >+1)  <br>
 h[j]+=a[i]*sign[i][j] <br>
if the final value of h[j] is-1 we assign it to class 0 or if it +1 to class 90 <br>
We do this for all the other combinations also. <br>
Finally, we take a mode of the h[j] for all the rows and depending on the mode we assign it to one of the 4 class 0,90,180,270. <br>
We calculate the accuracy by comparing the predicted value with the actual labels. The percentage of properly classified rows gives the accuracy <br>
