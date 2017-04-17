#**Behavioral Cloning** 

###Behavioral Cloning Project

The goals of this project were the following:
* Use a simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around the test track without leaving the road

[//]: # (Image References)

[center]: ./examples/center_driving_example.jpg "Example of Center Driving"
[recovery1]: ./examples/recovery_example.jpg "Example of Recovery Driving"
[recovery2]: ./examples/recovery_example_2.jpg "Example of Recovery Driving"
[flipped_before]: ./examples/flipped_before.jpg "Normal Image"
[flipped_after]: ./examples/flipped_after.jpg "Flipped Image"
[loss_plot]: ./lossPlot.jpg "Mean Squared Error Plot"

###Files Included in this project 

Beyond this writeup, the following files are part of this project submission:
* *model.py* Contains the script to create and train the model
* *drive.py* Drives the car in autonomous mode (this script was unmodified)
* *model.h5* Contains a trained convolution neural network
* *video.mp4* Contains a recording of the vehicle driving autonomously in the simulator

Using the Udacity provided simulator and my *drive.py* file, the car can be driven autonomously around the track by executing:
```sh
python drive.py model.h5
```

The *model.py* file contains the code for training and saving the neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Training can be performed by executing:
```sh
python model.py
```

###Model Architecture and Training Strategy

####1. Design Approach and Model Architecture

The final model architecture employed in this project is very similar to the Nvidia architecture described in the paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) and is summarized in the following table:


| Layer         		|     Description	        					|
|:----------------------|:----------------------------------------------|
| Input         		| 160x320x3 RGB image   					    |
| 2D Cropping      		| 100x320x3 RGB image - top 70 pixels cropped  	|
| Normalize and Center 	| 32x32x1 RGB image   					        |
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


While this was the final architecture used, it certainly was not where we started. 

We first started with a very simple, single fully-connected layer with the intent of checking the basic functionality of the training pipeline. With the pipeline verified, a more complicated network architecture was developed. First, a cropping layer was added to remove unnecessary background detail from each image. Second, a normalization and centering layer was added to help condition the data as ideally as possible for the optimizer. From there, various combinations of convolutional layers were tried with varying levels of success. Ultimately, it was decided to implement the Nvidia architecture given the success that this architecture had within a similar problem space. With the architecture settled, the focus shifted to adjusting model parameters and producing appropriate training data.

Regardless of the architecture, model training was performed by splitting the training data into both a training and validation set in hopes of reducing over-fitting. This was done using the sklearn *train_test_split* function. Training was then performed in batches with the data shuffled within each batch. Training was setup using an ADAM optimizer with the object of minimizing the mean squared error of the steering angle loss. Examining the mean squared error loss during training did not show any significant difference in loss between the training and validation sets so no modifications were made to the network to further minimize over-fitting (e.g. adding a dropout layer). For the final iteration of the architecture and training data, the training and validation set loss is shown in the following plot:

![alt text][loss_plot]

####2. Model parameter tuning

The model parameters that were considered for tuning are shown in the following table, along with their final tuned values.

| Parameter        		    | Value     	  		|
|:--------------------------|:---------------------:|
| Batch Size                |            32         |
| Learning Rate             |         0.001         |
| L/R Steer Angle offset    |           0.5         |

Of these parameters, the only one which was explored in any great detail was the steer angle offset. This value was the difference in steering angle that was applied to the left and right camera images. Without knowing the geometry of the vehicle it is impossible to know what the exact value for this parameter should have been. Thus, different values were tried. Subjectively, it seemed that as this value became larger, the vehicle was able to stay centered within the lane better. However, if the value was made too large, the steering seemed to become too 'jerky' - as if someone was 'sawing' on the steering wheel. Thus, the final value chosen here provided the best compromise between these extremes.

####3. Creation of the Training Set & Training Process

The training process was an iterative process, where a baseline training set was first created then augmented based on deficiencies in model performance.

First, a baseline two lap run of the test course was completed. This run was completed trying to maintain the centerline of the road as best and as smoothly as possible. Data from the center, left, and right cameras was used for this (and all subsequent data collections) and using this data, we trained the model and tested performance in autonomous mode. Subjective observation of the results from this stage showed a definite left-bias. That is, the vehicle seemed to be "hugging" the left side of the road. So much so that the vehicle was driving off the left side of the road in some relatively simple corners.

An example of an image recorded while performing baseline centerline training is given here:

![alt text][center]

Second, in order to address the left-bias, the image and steering data from the baseline run was flipped using the numpy *fliplr* function. The resulting image set was effectively doubled, and we now had data for driving what was effectively a mirror image of the baseline track. This definitely helped to minimize the left bias. Performance on straight road and moderate curves was sufficient, but performance on sharp curves was still not acceptable.

An example of an original image and the flipped counterpart that would have been generated in this enhancement step are given here:

![alt text][flipped_before]
![alt text][flipped_after]

Third, additional data was collected through the sharp corners at the end of the lap on the test track. Three additional passes were made and recorded to add additional data. While performance in autonomous mode improved, it was still not sufficient to satisfy requirements.

Fourth, more data was collected on the first of two corners where the vehicle was having trouble staying on the track. Specifically, more extreme recovery maneuvers were recorded to help the model take more significant action when diverging from the center of the road. After adding this data, the vehicle was able to successfully navigate this particular corner, but still was not able to stay on the track during the entire lap.

An example of camera views during recovery events are given here:

![alt text][recovery1]
![alt text][recovery2]

Finally, additional recovery data was recorded for the last problematic corner. After this data was recorded, the vehicle was able to successfully navigate the entire track without leaving the road at any location. It should be pointed out that for the last two recovery data sets we did not include the flipped/mirror images so that we could focus the model on handling the specifics of these problem cases. It was also at this point that model parameters were adjusted to see if performance could be further improved.

| Test Set        		    | Flipped    	| Total Number of images |
|:--------------------------|:-------------:|:----------------------:|
| Baseline 2 laps           |    Yes        |  15086                 |
| Extra Curves              |    Yes        |  8528                  |
| Recovery                  |    Yes        |  3554                  |
| Recovery Curve 1          |    No         |  1063                  |
| Recovery Curve 2          |    No         |  496                   |
| **TOTAL**          |             |  **28727**                   |

As mentioned above, with the chosen model architecture, model parameters, and training set the vehicle was able to navigate the test track successfully. Video of the vehicle, in its final configuration, driving the test track is contained in the file *video.mp4*
