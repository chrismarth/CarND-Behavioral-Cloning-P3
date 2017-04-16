#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals of this project were the following:
* Use a simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

[//]: # (Image References)

[center]: ./examples/center_driving_example.jpg "Example of Center Driving"
[recovery1]: ./examples/recovery_example.jpg "Example of Recovery Driving"
[recovery2]: ./examples/recovery_example_2.jpg "Example of Recovery Driving"
[flipped_before]: ./examples/flipped_before.jpg "Normal Image"
[flipped_after]: ./examples/flipped_after.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* video.mp4 containing a recording of the vehicle driving autonomously in the simulator
* writeup.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. Design Approach and Model Architecture

The model architecture employed in this 

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


| Layer         		|     Description	        					|
|:----------------------|:----------------------------------------------|
| Input         		| 160x320x3 RGB image   					    |
| 2D Cropping      		| 100x320x3 RGB image - top 70 pixels cropped  	|
| Normalize and Center 	| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 2x2 stride, valid padding                  	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding 	                |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding 	                |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding 	                |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding 	                |
| RELU					|												|
| Flatten       		|                                               |
| Fully connected		| outputs 100                                   |
| Fully connected		| outputs 50                                    |
| Fully connected		| outputs 10                                    |
| Fully connected		| outputs 1        							    |

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####2. Model parameter tuning

The following model parameters were tuned

| Parameter        		    | Tuning    	  		|
|:--------------------------|:---------------------:|
| Batch Size                |            32         |
| Learning Rate             |         0.001         |
| L/R Steer Angle offset    |           0.5         |

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####3. Creation of the Training Set & Training Process

The training process was an iterative process, where a baseline training set was first created then augmented based on deficiencies in model performance.

First, a baseline two lap run of the test course was completed. This run was completed trying to maintain the centerline of the road as best and as smoothly as possible. Data from the center, left, and right cameras was used for this (and all subsequent data collections) and using this data, we trained the model and tested performance in autonomous mode. Subjective observation of the results from this stage showed a definite left-bias. That is, the vehicle seemed to be "hugging" the left side of the road.

An example of an image recorded while peforming baseline centerline training is given here:

![alt text][center]

Second, to address the left-bias the image and steering data from the baseline run was flipped using the numpy fliplr function. The resulting image set was effectively doubled, but now we had data for driving what was effectively a mirror image of the baseline track. This definitely helped to minimize the left bias. Performance on straight road and moderate curves was sufficient, but peformance on sharp curves was not acceptable.

An example of an original image and its flipped counterpart are given here:

![alt text][flipped_before]
![alt text][flipped_after]

Third, additional data was collected through the sharp corners at the end of the lap on the test track. Three additional passes were made and recorded to add additional data. While performance in autonomous mode improved, it was still not sufficient enough to satisfy requirements.

Fourth, more data was collected on those corners where the vehicle was having trouble staying on the track. Specifically, more extreme recovery maneuvers were recorded to help the model take more significant action when diverging from the center of the road. After adding this data, the vehicle was able to successfully navigate the entire track.

An example of an original image and its flipped counterpart are given here:

![alt text][recovery1]
![alt text][recovery2]

Finally, once the vehicle was able to successfully navigate the entire track, further refinements to recovery maneuvers were made and added to the baseline training set. It was also at this point that model parameters and network architecture were adjusted to see if performance could be further improved.



I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :



Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:



Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
