# Chars74k_Recognition

### Character Recognition of Chars74k dataset - Handwritten and Images

### *using Histogram of Oriented Gradients and Neural Networks*

## Please visit [HERE](https://devendrapratapyadav.github.io/Chars74k_Recognition) for analysis and results.

The code is divided into two parts-

1. C++ - Preprocessing input images and feature vector formation

2. Python/MATLAB - Using the features to train Neural Network


Preprocessing and feature vector formation code is "proj.cpp" in "Code" folder.

"Neural Network" folder contains two codes : in MATLAB and in Python (Scikit-learn)

*"nn-hand.py"* and *"nn-img.py"* are python codes.

*"imgANN.m"* and *"handANN.m"* are MATLAB codes.

`NOTE: Python code requires 'Scikit-learn' library. It is preferable to install 'Anaconda' distribution of python to run it.`


The data (images) used are not provided since they are very large in size.
Download the feature vector .txt files and original images used here:

https://drive.google.com/drive/folders/0B6TI76M3_KiyZUR0UE1WRXJka1E?usp=sharing


##### Preprocessing code: 
Compile:
```sh
$ cmake .
$ make
```

An executable name 'proj' is generated.

Run using:
```sh
$ ./proj
```

`NOTE: To run the code 'data-img' and 'data' directories must be present alongside the 'code' directory. 
'data-img' contains the 'Image Characters'
'data' contains 'Handwritten Characters'
`

These folders are very large. Please download them from the shared folder and place them alongside 'code' folder.

The data used is taken from the 'chars74k' dataset available at: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

This dataset is converted to .png images for each class (26 lowercase, 26 uppercase and 10 digits)

Labels 1-10 = Digits   Labels 11-36= Uppercase   Labels 37-62= Lowercase

We use these images, read them, preprocess them, then form a feature vector which is stored in file "features.txt".

This "features.txt" is given as input to Neural Network implemented in Python using Scikit-learn library.
The neural network trains on some percentage of images and tests on the rest.

After features.txt is generated place it along with neural network code.

*"nn-hand.py"* and *"nn-img.py"* are python codes.

Run using:
```sh
$ python nn-hand.py -- for handwritten characters
```
The code reads data from **'features-handHOG.txt'** and **'features-handPIX.txt'**. Please place them alongwith .py file before running.
Accuracy is displayed after model is trained.

Run using:
```sh
$ python nn-img.py  -- for image characters
```
The code reads data from **'features-imgHOG.txt'** and **'features-imgPIX.txt'** Please place them alongwith .py file before running.


Please download feature-______.txt files from shared link:

https://drive.google.com/drive/folders/0B6TI76M3_KiyZUR0UE1WRXJka1E?usp=sharing


***************************************************************************************************************************
*"imgANN.m"* for image characters and *"handANN.m"* for handwritten characters are MATLAB codes.

Run using MATLAB

Please feature-____.txt files alongwith .m file before running.

