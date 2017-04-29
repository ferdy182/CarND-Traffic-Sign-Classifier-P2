|#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/classes_visualization.png "Visualization"
[distribution]: ./writeup_images/distribution.png "Distribution"
[preprocess]: ./writeup_images/preprocess.png "Preprocess"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [html](https://github.com/ferdy182/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.html) and [notebook](https://github.com/ferdy182/CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 images
* The size of the validation set is 4410
* The size of test set is 12630 images
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here you can see an example image of each class and the distribution of each class.

![alt text][image1]
![alt text][distribution]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried different approaches, I first converted the image to grayscale, then I tried different techniques on the grayscale version.

* I normalized using the given formula pixel - 128/128.
* To get a mean of 0 and a standard distribution of 1 I also used the z-score formula which is similar to normalization: pixel - mean / std.
* I also tried image equalization on a different copy of the dataset to compare

Here is an example of few traffic signs for each process:

Original
![no preprocess](./writeup_images/nopreprocess.png)
Grayscale
![no preprocess](./writeup_images/grayscale.png)
Normalized
![no preprocess](./writeup_images/normalization.png)
Zscore
![no preprocess](./writeup_images/zscore.png)
Equalized
![no preprocess](./writeup_images/equalization.png)

I did not augment the data because I ran out of time, but some possible data augmentations could be:

* small rotations
* rescale
* add noise
* add blur
* flip signs depending on the symmetry axes, some signs could become a different class (like turn left/right) others are symmetrical in y and some in x and some in both.

However after different tests I decided to use the dataset after applying z-score which gave the best results

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding  outputs 14x14x6 				|
| Convolution 5x5	    |  1x1 stride, valid padding, outputs 10x10x16   |
| Max pooling	      	| 2x2 stride, valid padding  outputs 5x5x16 				|
| Flatten   	| output 400 				|
| Fully connected		| output 120.        									|
| RELU					|												|
| Dropout					|			50%									|
| Fully connected		| output 84.        									|
| RELU					|												|
| Dropout					|			50%									|
| Fully connected		| output 43.        									|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I prepared an array with the data after different preprocesses, and for each set of data I trained the network with different number of epochs from 10 to 250 and batch sizes from 64 to 512. I found out that after hundred batches it does not get better and small batches worked better than bigger batches.
I used Adam optimizer because it is a bit better than Stochastic gradient descent, as it uses momentum to optimize the learning rate.
I also only saved the model when the validation accuracy was better than the previous saved one to avoid overfittin.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.976
* test set accuracy of 0.95

I choose the LeNet architecture beause it works well for classifying images. But plain LeNet was not giving a good accuracy unles it run for more than 200 epochs. I tried different batch sizes such 64, 128, 256 and 512 but a smaller batch size performed better. I also tried with different amount of epochs but after a hundred it did not really improve.

To avoid overfitting I included dropout after the fully connected layers and this increased the accuracy.

I also added early stop, so the model is saved when the validation accuracy is higher than the previous one. This made the training slower and I used it only at the end.


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I downloaded several images from the web to test my network, here they are:

![images downloaded](./writeup_images/web_signals.png)

The first and fourth images are bad quality because they are reescaled from a low quality jpg, so they should be difficult to classify, however the network did fine with those. Other signs have strange perspectives that might also be difficult if the training set did not contain this kind of angles. Also some images with numbers inside might be difficult to classify correctly.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Prediction			        |     Truth	        					| Correct |
|:---------------------:|:---------------------------------------------:|:--:|
| No entry | No entry  | Yes |
| No entry | No entry  | Yes |
| Right-of-way at the next intersection | Right-of-way at the next intersection  | Yes |
| Priority road | Priority road  | Yes |
| Stop | Stop  | Yes |
| Turn left ahead | General caution  | No |
| Road work | Road work  | Yes |
| Road work | Road work  | Yes |
| Yield | Yield  | Yes |
| End of all speed and passing limits | End of all speed and passing limits  | Yes |
| No passing for vehicles over 3.5 metric tons | No passing for vehicles over 3.5 metric tons  | Yes |
| Priority road | Priority road  | Yes |
| Priority road | No vehicles  | No |
| Stop | Stop  | Yes |
| End of all speed and passing limits | End of all speed and passing limits  | Yes |
| Speed limit (30km/h) | Speed limit (30km/h)  | Yes |
| Speed limit (30km/h) | Speed limit (30km/h)  | Yes |
| Speed limit (50km/h) | Speed limit (50km/h)  | Yes |
| Speed limit (30km/h) | Speed limit (60km/h)  | No |
| Speed limit (70km/h) | Speed limit (70km/h)  | Yes |
| Children crossing | Double curve  | No |
| Speed limit (30km/h) | Speed limit (120km/h)  | No |
| Speed limit (30km/h) | Speed limit (30km/h)  | Yes |
| Go straight or left | Go straight or left  | Yes |
| Roundabout mandatory | Roundabout mandatory  | Yes |


The model was able to correctly guess 80% of the images (20 out of 25) and this less than the test set which had 0.95 so that means that the model has some overfitting.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13th cell of the Ipython notebook.

Here is a table showing the softmax probabilities for each of the test images. As you can see, most of the time the model is really certain of the class. It seems to have troubles with numbers such 80 or 120. It is of particular interest the number 13, which is missclassified as the same class as number 12, while it is quite obvious to the eye that they are quite different in shape and colors.

![softmax](./writeup_images/softmax.png)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
