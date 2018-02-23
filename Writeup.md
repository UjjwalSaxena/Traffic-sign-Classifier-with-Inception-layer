
# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


 
### Load data and Visualize

### Data Set Summary & Exploration

I used the pickle library to load the data and numpy and open cv libraries to preprocess it.

* The size of training set is ?
    - Answer: 34799
* The size of the validation set is ?
    - Answer: 4410
* The size of test set is ?
    - Answer: 12630
* The shape of a traffic sign image is ?
    - (32, 32, 3)
* The number of unique classes/labels in the data set is ?
    - 43


[image1]: ./WriteUpImages/general.png "Some Images from dataset"

![alt text][image1]
[image6]: ./WriteUpImages/trainData.png "Traffic Sign 3"
[image8]: ./WriteUpImages/validdata.png "Traffic Sign 5"
[image63]: ./WriteUpImages/testdata.png "Traffic Sign 3"
![alt text][image6]
![alt text][image8]
![alt text][image63]




### Design and Test a Model Architecture

#### Preprocessing

I grayscaled and normalized the All dataset Images as a part of preprocessing.
I preprocessed and augmented the Images. Trained the model on the images
I normalized the data because it is good for computational point of view. Mathematically it is easy for the network to work on data having variance around 1 and hence normalization was required.
I also decided to convert the images to grayscale to make the calculations easy.
[image4]: ./WriteUpImages/preprocessedtraindata.png "Traffic Sign 1"
![alt text][image4]


#### Augmentation

For adding additional images, I augmented the training data using 3 processes:

**Transform**
shifting the images up, down, left and right, away from the central axis

**Perspective Change**
changing the angles for viewing the image

**Rotating**
Rotating the image by certaing angle right and left.

This Step was necessary to have sufficient number of images for all classes so that the network has it easy to predicat all kinds of classes and not just a few of them on which the network is sufficiently trained.

[image2]: ./WriteUpImages/augTraindata.png "Grayscaling"
![alt text][image2]



#### Architecture

##### My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image     					| 
| Convolution 5x5     	| Valid padding, output 28X28X6             	|
| ELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  output 14X14X6   				|
| Convolution 5x5     	| Valid padding, output 10X10X16           	    |
| ELU					|												|
| Inception Layer		| 1X1, 3X3 and 5X5 parallel layers, Same padding|
|   					| output 	10X10X96							|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  output 5X5X96    				|
| Convolution 3x3	    | output 3X3X150      							|
| ELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  output 2X2X150   				|
| Flatten           	| output 600                            	    |
| Fully Connected		| output 300									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| output 150       						    	|
| RELU					|												|
| Dropout				|												|
| Fully Connected		| output 100        							|
| RELU					|												|
| Dropout				|												|
| Softmax				| output 43										|

 


#### type of optimizer, the batch size, number of epochs , hyperparameters.

    Optimizer: Adam Optimizer
    EPOCHS = 50
    BATCH_SIZE = 128
    prob1=0.7
    prob2=0.5
    rate = 0.0004
    mu = 0
    sigma = 0.1

#### 4. Training Process

##### My final model results were:
* training set accuracy of ?
    99%
* validation set accuracy of ? 
    97.4%
* test set accuracy of ?
    95.4%

[image10]: ./WriteUpImages/training.png "Traffic Sign 4"
![alt text][image10]
I chose an Iterative approach for the architecture. I first chose the normal lenet architecture without augmentation of data, and to my surprize the validation and test accuracy was around 88%.

Also the validation loss was high which could mean that the model was overfitting, since all classes did not have similar no. of Image data.
I augmented the data and increased the dataset. Accuracy rose to 92-93%

Then I searched for the popular architectures of alexnet and googlenet and came accross Inception Layers.
I tried to Implement that in my model and the accuracy rose further but I could see that the accuracy and the loss had very high fluctuations. To reduce overfitting I added dropouts, Increaser augmented data to around 1,20 thousand. And to reduce the fluctuations I lowered the learning rate to 0.0004 for my model. It proved to be good for the model and accuracy rose above 97% with decreased fluctuations with increasing epochs.

Having Inception layer helped to extract features via 1X1, 3X3, and 5X5 window sizes and join them which was the main reason for using it. various window sizes extract various features which the CNN seems important for further processing.


##### Earlier plots for accuracy and losses

[image12]: ./WriteUpImages/accuracyVs.png "Traffic Sign 4"
[image14]: ./WriteUpImages/LossVs.png "Traffic Sign 5"
![alt text][image12] ![alt text][image14]

##### Refined Accuracy and losses

[image11]: ./WriteUpImages/AccuracyFinal.png "Traffic Sign 2"
[image13]: ./WriteUpImages/downloadFinal.png "Traffic Sign 4"
![alt text][image11] ![alt text][image13]


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

[image22]: ./TestImages/1.png "Random Noise"
[image23]: ./TestImages/22.png "Traffic Sign 2"
[image24]: ./TestImages/23.png "Traffic Sign 4"
[image25]: ./TestImages/40.png "Traffic Sign 5"
[image26]: ./TestImages/19.png "Traffic Sign 5"

##### Here are five German traffic signs that I found on the web:

![alt text][image22] ![alt text][image23] ![alt text][image24] 
![alt text][image25] ![alt text][image26]

The first image might be difficult to classify because its brightness and perspective is different from the once system is trained on. Also the image has extra objects in the image except the sign itself.



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).




##### Here are some Test Images I downloaded and visualized
[image7]: ./WriteUpImages/testImages.png "Traffic Sign 4"
![alt text][image7]

##### These are the preprocessed Images before testing them with the Model.
[image9]: ./WriteUpImages/TestPreprocessedImages.png "Traffic Sign 5"
![alt text][image9]



##### Here are the results of the prediction:
The model was able to correctly guess 10 of the 12 traffic signs, which gives an accuracy of 83.33%.


| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)  | Stop sign   									| 
| Keep right     		| Keep right 									|
| Yield					| Yield											|
| Go straight or left	| Go straight or left	 		        		|
| Slippery Road			| Slippery Road      							|


##### Let's have a look at the output

[image3]: ./WriteUpImages/predictedImages.png "Random Noise"
![alt text][image3]
 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

These are the Images of the output probabilities for various possible guesses by the model.
[image35]: ./WriteUpImages/probabilities.png "Random Noise"
![alt text][image35]

[image54]: ./WriteUpImages/prob1.png "Random Noise"
![alt text][image54]

[image55]: ./WriteUpImages/prob2.png "Random Noise"
![alt text][image55]

[image56]: ./WriteUpImages/prob5.png "Random Noise"
![alt text][image56]

[image57]: ./WriteUpImages/prob6.png "Random Noise"
![alt text][image57]

[image58]: ./WriteUpImages/prob8.png "Random Noise"
![alt text][image58]

#### Summary

The System is able to Identify most of the Images correcly, However I believe it could have been better to give better accuracy.
We can use better and known architectures for training data.

We can try experimenting with various activation functions like relu, elu, sigmoid etc. also parellel layers can be added and their efficiencies noted.

I tried to use as many concepts as possible in the architecture and satisfied with the accuracy it is having.
