# **Traffic Sign Recognition**

### **Goal Build a Traffic Sign Recognition Project**

Steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./model_output/Validationaccuracy.png "Validation Accuracy"
[image2]: ./model_output/Trainaccuracy.png "Train Accuracy"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_Images/1.png "Traffic Sign 1"
[image5]: ./test_Images/2.png "Traffic Sign 2"
[image6]: ./test_Images/3.png "Traffic Sign 3"
[image7]: ./test_Images/4.png "Traffic Sign 4"
[image8]: ./test_Images/5.png "Traffic Sign 5"
[image9]: ./test_Images/6.png "Traffic Sign 6"
[image10]: ./test_Images/7.png "Traffic Sign 7"
[image11]: ./test_Images/8.png "Traffic Sign 8"
[image12]: ./model_output/RawandProcessedOutput1.png "Raw and Processed Output"
[image13]: ./model_output/RawandProcessedOutput2.png "Raw and Processed Output"
[image14]: ./model_output/RawandProcessedOutput3.png "Raw and Processed Output"
[image15]: ./model_output/TrainandValidationGreyscale.png "Train and Validation Greyscale"
[image16]: ./model_output/ValidDatasetSignCounts.png "Validation Data Set Counts"
[image17]: ./model_output/TrainDatasetSignCounts.png "Train Dataset Counts"
[image18]: ./model_output/TestDatasetSignCounts.png "Test Dataset Counts"
[image19]: ./model_output/ValidDatasetSignCounts1.png "Validation Data Set Counts"
[image20]: ./model_output/TrainDatasetSignCounts1.png "Train Dataset Counts"
[image21]: ./model_output/TestDatasetSignCounts1.png "Test Dataset Counts"
[image22]: ./model_output/mean_variance.png "Normalization"
[image23]: ./model_output/MyTestImages1.png "Test Images Results 1"
[image24]: ./model_output/MyTestImages2.png "Test Images Results 2"
[image25]: ./model_output/TestImagesbeforeandafter.png "TestImagesbeforeandafter"
[image26]: ./model_output/MyTestImages5.png "Test Images Results 1"
[image27]: ./model_output/MyTestImages6.png "Test Images Results 2"
[image28]: ./model_output/MyTestimagesbar1.png " Test Images Bar 1"
[image29]: ./model_output/MyTestimagesbar2.png " Test Images Bar 2"
[image30]: ./model_output/outputFeatureMapLayer1.png " Layer 1 Feature Map"
[image31]: ./model_output/outputFeatureMapLayer2.png " Layer 2 Feature Map"
[image32]: ./model_output/outputFeatureMapLayer3_1.png " Layer 1 Feature Map Part1"
[image33]: ./model_output/outputFeatureMapLayer3_2.png " Layer 1 Feature Map Part1"
[image34]: ./model_output/outputFeatureMapLayer3_3.png " Layer 1 Feature Map Part1"
[image35]: ./model_output/outputFeatureMapLayer3_4.png " Layer 1 Feature Map Part1"
[image36]: ./model_output/outputFeatureMapLayer3_5.png " Layer 1 Feature Map Part1"
[image37]: ./model_output/outputFeatureMapLayer3_6.png " Layer 1 Feature Map Part1"
[image38]: ./model_output/outputFeatureMapLayer3_7.png " Layer 1 Feature Map Part1"
[image39]: ./model_output/MyTestimagesbar3.png " Test Images Bar 3"
[image40]: ./model_output/MyTestimagesbar4.png " Test Images Bar 4"
[image41]: ./model_output/MyTestImages3.png "Test Images Results 1"
[image42]: ./model_output/MyTestImages4.png "Test Images Results 2"


### External Links References
[Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
[End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/)

## README


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set. The pickled data is a dictionary with 4 key/value pairs:
'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
'labels' is a 1D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
'sizes' is a list containing tuples, (width, height) representing the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. Pickled data is 32*32 images.

Training subset is 80% of initial train.p dataset randomly obtained, validation subset is the rest of the raw train dataset. For testing the full test.p dataset is used.

Training Set:   31367 samples (31367, 32, 32, 3) shape
Validation Set: 7842 samples (7842, 32, 32, 3) shape
Test Set:       12630 samples (12630, 32, 32, 3) shape
Sign Classes Label: 43 samples

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across the Train, Validation and Test sets per class.

| Train Dataset         |     Validation Dataset    |    Test Dataset
|:---------------------:|:-------------------------:|:-------------------------:|
|![alt text][image19]   |![alt text][image20]        |![alt text][image21]          |


Total number of images in the augmented dataset =  31367
Total number of images in the augmented dataset =  7842
Total number of images in the train dataset =  62734
Total number of images in the valid dataset =  15684
Features Normalized
Labels One-Hot Encoded
Training Set:   62734 samples (62734, 32, 32, 1) shape
Validation Set: 15684 samples (15684, 32, 32, 1) shape
Test Set:       12630 samples (12630, 32, 32, 1) shape

Image Shape: (32, 32, 1)
Sign Classes Label: 43 samples


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
Following are the image preprocessing steps:
a) Image Transformation. This step sharpens and increases the contrast of the Images.

| Train Dataset         |     Validation Dataset    |    Test Dataset
|:---------------------:|:-------------------------:|:-------------------------:|
|![alt text][image12]   |![alt text][image13]        |![alt text][image14]          |

Additionally, I build a jittered dataset by adding per image transformed versions of the original training set, yielding __62734__ samples in total. Samples are randomly perturbed.
Below is the split of datasets after augmentation. By adding them synthetically will yield more robust learning to potential deformations in the test set. This also increased the number of train data for the classes which had lesser training data.

| Train Dataset         |     Validation Dataset    |    Test Dataset
|:---------------------:|:-------------------------:|:-------------------------:|
|![alt text][image19]   |![alt text][image20]        |![alt text][image21]          |

I tested the model with and without these synthetic images -

Model with out these images  yielded higher validation accuracy. But the Test Data Accuracy is around 93.8%.

EPOCH 30 ...
Train Accuracy = 1.000
Validation Accuracy = 0.994

Model saved
INFO:tensorflow:Restoring parameters from ./lenet
Test Data Set Accuracy = 0.938

Model with these synthetic images  yielded higher test data accuracy. But the Train and Validation accuracy was 98.4% and 96.8%

EPOCH 30 ...
Train Accuracy = 0.984
Validation Accuracy = 0.968

Model saved
INFO:tensorflow:Restoring parameters from ./lenet
Test Data Set Accuracy = 0.940


b) I decided to convert the images to grayscale because intensity (e.g. edge detection) plays a major role. Grayscale is usually sufficient to distinguish such edges.

Here is an example of a traffic sign images after grayscaling.
![alt text][image15]

c) As a last step, I normalized the image data because to get to zero mean and equal variance.
![alt text][image22]

Normalization is done by Min-Max scaling to a range of a=0.1 and b=0.9. After scaling, the values of the pixels in the input data should range from 0.1 to 0.9. Since the image data is in grayscale, the current values range from a min of 0 to a max of 255.

__Min-Max Scaling:  X′=a+(X−Xmin)/(b−a)Xmax−XminX′=a+(X−Xmin)/(b−a)Xmax−Xmin

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The CNN architecture was inspired by __LeNet__. My final model consisted of the following layers:

| Layer                 |     Description                                                |
|:---------------------:|:-------------------------------------------------------------:|
| Input                 | 32x32x1 Greyscale image                                       |
| Convolution 5x5         | shape=(5, 5, 1, 6) 1x1 stride, same padding, Outputs 28x28x6     |
| RELU                    |                                                                |
| Max pooling              | 2x2 stride,  Outputs 14x14x6                                   |
| Convolution 5x5        | shape=(5, 5, 6, 16) 1x1 stride, same padding, Outputs 10x10x16|
| RELU                    |                                                                |
| Max pooling              | 2x2 stride,  Outputs 5x5x6                                        |
| Convolution 4x4        | shape=(4, 4, 16, 412) Outputs 2x2x412                            |
| RELU                    |                                                                |
| Max pooling              | 2x2 stride,  Outputs 1x1x412                                   |
| Fully connected        | Output 122                                                      |
| Fully connected        | Output 84                                                      |
| Fully connected        | Output 43                                                      |
| Softmax                | 43 Classes                                                    |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used __AdamOptimizer__ basically as a motivation from the LeNet.

The ``tf.train.AdamOptimizer`` uses Adam algorithm to control the learning rate. Adam offers several advantages over the simple ```tf.train.GradientDescentOptimizer```.  Stochastic gradient descent which maintains a single learning rate for all weight updates and the learning rate does not change during training. Foremost is that it uses moving averages of the parameters (momentum). Simply put, this enables Adam to use a larger effective step size, and the algorithm will converge to this step size without fine tuning. The main down side of the algorithm is that Adam requires more computation to be performed for each parameter in each training step (to maintain the moving averages and variance, and calculate the scaled gradient); and more state to be retained for each parameter (approximately tripling the size of the model to store the average and variance for each parameter).
__BATCH_SIZE__ = 156 Batch size is the number of training examples in one forward/backward pass.

__EPOCHS__ = 30 Epoch is one forward pass and one backward pass of __all__ the training examples.

__Learning Rate__ = .00097. The learning rate used is .00097. The proportion that weights are updated (e.g. 0.001). Larger values (e.g. 0.3) results in faster initial learning before the rate is updated. Smaller values (e.g. 1.0E-5) slow learning right down during training.

Variables were initialized with using of a truncated normal distribution with mu = 0.0 and sigma = 0.1. Learning rate was finetuned by try and error process.

Traffic sign classes were coded into one-hot encodings.

As one can observe, at the end of the training process, accuracy stopped increasing and loss oscillated around relatively small value.



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ? __98.3%__
* validation set accuracy of ? __96.7%__
* test set accuracy of ? __94.2%__

| Training Set Accuracy |     Validation Set Acuracy|
|:---------------------:|:-------------------------:|
|![alt text][image1]   |![alt text][image2]            |

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First architecture choosen only had two convolution layer (based of LeNet). I added the third convolution layer with the idea  go through more conv layers, so as to  get activation maps that represent more and more complex features. This resulted in a Test Images which had higher success rates.
* What were some problems with the initial architecture?
Problem with the initial architecture was underfitting. By adding additional layer i was able to get to higher than 95% training accuracy.
I also added Dropouts with keepprob = 0.5 for Training and 1.0 for Validation. This helped overfitting during training.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
As described above added additional convolution layer and dropouts. One additional improvement could be addition of inception modules. Inception Module would increase the performance on such kind of tasks as they allow to not select optimal layer (say, convolution 5x5 or 3x3), by performing different layer types simultaneously and selecting the best one on its own.

* Which parameters were tuned? How were they adjusted and why?
Parameters tuned included Learning Rate and Epochs. Learning Rate tuning parameters and Number of layers in the Model.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

* What architecture was chosen?
LeNet with some additional modifications was chosen.
* Why did you believe it would be relevant to the traffic sign application?
Given that LeNet is able to classify the greyscale handwritten images, it was a good starting point for the Traffic sign classifier.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
When i started the model the speed of training and validation accuracy was very slow. However addition of new convolution layer improved the training and validation accuracy. I also noticed that after adding synthetic images to the data set the model became more robuts to preduict the test images.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that i used to test the model

| __German Traffic Signs__  |                                                 |
|:-------------------------:|:---------------------------------------------:|
|![alt text][image4]         |![alt text][image8]                              |
|![alt text][image5]        |![alt text][image9]                            |
|![alt text][image6]          |![alt text][image10]                             |
|![alt text][image7]        |![alt text][image11]                            |


Below are the test results of the test images classifications.

|   __First Trial__     |
|:---------------------:|
![alt text][image23]
![alt text][image24]

Its is observed that this result could vary run by run. For example, in the below run i observed that the model having difficulty with the "Bumpy Road" sign. This could be resolved having more training data with Image augemented.
One of the test images "Bumpy Road" was predicted to be a "Road Work" sign. "Bump Road" had Low Recall i.e the model has trouble predicting on stop signs.

|   __Second Trial__    |
|:---------------------:|
![alt text][image41]
![alt text][image42]



|   __Third Trial__    |
|:---------------------:|
![alt text][image26]
![alt text][image27]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|   __First Trial__                                                     |
|:---------------------:|:---------------------------------------------:|
| __Image__                |     __Prediction__                             |
|                                                                       |
| 30 Km/h                  | 30 Km/h                                       |
| Bumpy Road            | Bumpy Road                                     |
| Ahead Only            | Ahead Only                                    |
| No Vehicles             | No Vehicles                                     |
| Go strainght or left    | Go strainght or left                            |
| General caution        | General caution                                |
| 30 Km/h                  | 30 Km/h                                         |
| Keep Left                | Keep Left                                      |

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%.

|   __Second Trial__                                                    |
|:---------------------:|:---------------------------------------------:|
| __Image__                |     __Prediction__                            |
|                                                                       |
| 30 Km/h                  | 30 Km/h                                       |
| Bumpy Road            | ~~Bumpy Road~~                                  |
| Ahead Only            | Ahead Only                                    |
| No Vehicles             | No Vehicles                                     |
| Go strainght or left    | Go strainght or left                            |
| General caution        | General caution                                |
| 30 Km/h                  | 30 Km/h                                         |
| Keep Left                | Keep Left                                      |

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 84%.

|   __Third Trial__                                                     |
|:---------------------:|:---------------------------------------------:|
| __Image__                |     __Prediction__                            |
|                                                                       |
| 30 Km/h                  | 30 Km/h                                       |
| Bumpy Road            | Bumpy Road                                     |
| Ahead Only            | Ahead Only                                    |
| No Vehicles             | No Vehicles                                     |
| Go strainght or left    | Go strainght or left                            |
| General caution        | General caution                                |
| 30 Km/h                  | 30 Km/h                                         |
| Keep Left                | Keep Left                                      |


The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of __100__%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below is the result of the model  when predicting on each of the  new images by looking at the softmax probabilities for each prediction. Also visualizations bar chart below.
![alt text][image39]
![alt text][image40]


|   __First Trial__                                                 |
|:---------------------:|:-----------------------------------------:|
| __Prediction__        |     __Probability__                           |
|                                                                   |
| 30 Km/h                  | 98%                                      |
| Bumpy Road            | 100%                                        |
| Ahead Only            | 100%                                        |
| No Vehicles             | 100%                                         |
| Go straight or left    | 100%                                        |
| General caution        | 100%                                        |
| 30 Km/h                  | 100%                                         |
| Keep Left                | 100%                                      |


![alt text][image28]
![alt text][image29]

|   __Second Trial__                                                 |
|:---------------------:|:-----------------------------------------:|
| __Prediction__        |     __Probability__                           |
|                                                                   |
| 30 Km/h                  | 100%                                      |
| Bumpy Road            | 99%                                        |
| Ahead Only            | 100%                                        |
| No Vehicles             | 89%                                         |
| Go straingt or left    | 8%                                        |
| General caution        | 100%                                        |
| 30 Km/h                  | 100%                                         |
| Keep Left                | 100%                                      |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below is the visualization of the Neural Network.

Layer1 is able to look at the diagonal lines and staringht lines. It is able to show that their network's inner weights had high activations to the boundary lines. As visualized here, the first layer of the CNN can recognize -45 degree lines. The first layer of the CNN is also able to recognize +45 degree lines, like the one above. So, the first layer of our CNN clearly picks out very simple shapes and patterns like lines and blobs

|   __Layer1__                                                      |
|:---------------------:|:-----------------------------------------:|
![alt text][image30]

Layer2 is able activate different features such as Arrows. A visualization of the second layer we can see how it ispicking up more complex ideas like circles and arrows.

|   __Layer2__                                                      |
|:---------------------:|:-----------------------------------------:|
![alt text][image31]

Layer 3 tpicks out complex combinations of features from the second layer. These include things like grids.

|   __Layer3__                                                      |
|:---------------------:|:-----------------------------------------:|
![alt text][image32]
![alt text][image33]
![alt text][image34]
![alt text][image35]
![alt text][image36]
![alt text][image37]
![alt text][image38]

As we continue through the last layers it will pick the highest order ideas that we care about for classification, like different traffic classes.epresenting the original width and height the image.
'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. Pickled data is 32*32 images.

Training Set:   31367 samples (31367, 32, 32, 3) shape
Validation Set: 7842 samples (7842, 32, 32, 3) shape
Test Set:       12630 samples (12630, 32, 32, 3) shape
Sign Classes Label: 43 samples
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)
Following are the image preprocessing steps:
a) Image Transformation. This step sharpens the Image. Example ![alt text][.1]

As a first step, I decided to convert the images to grayscale because

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ...

To add more data to the the data set, I used the following techniques because ...

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ...


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
# CarND-Traffic-Sign-Classifier-Project-Ajit
