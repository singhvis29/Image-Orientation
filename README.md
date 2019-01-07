# Image-Orientation
Finding the orientation (rotation) of the image using  Random Forest, kNN and Adaboost algorithms

### Data:
A dataset of images from the Flickr photo sharing website where each image is converted to 8 x 8 x 3 (the third dimension is because color images are stored as three separate planes – red, green, and blue) = 192 vector i.e. one image per row

### Implementation:
The prediction is done by implementing kNN, Random Forest, and Adaboost Algorithm from scratch (no python ML libraries were used). Implementation of these algorithms have been done as follows-
#### ADABOOST:
We have taken 200 weak decision stumps <br>
We have implemented one VS one in adaboost. The training data has been converted into 6 different combination, <br>
1. Containing rows where label = 0 or 90 <br>
2. Containing rows where label = 0 or 180 <br>
3. Containing rows where label = 0 or 270 <br>
4. Containing rows where label = 90 or 180 <br>
5. Containing rows where label = 90 or 270 <br>
6. Containing rows where label = 180 or 270 <br>
<br>
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
