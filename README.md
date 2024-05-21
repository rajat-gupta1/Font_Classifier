# Font Classifier

This repository contains the code for classifying fonts into their respective categories. The training dataset includes 10 different types of fonts, and the model achieves over 90% accuracy on the test set.

## Assumptions

It is assumed that the provided dataset contains sentences in a single line.

## Methodology

To expedite convergence, rather than resizing all images to a uniform dimension (e.g., 500 x 100), each image is divided into square subimages. The model is trained on these subimages, since the font remains consistent within each image. During testing, predictions are made by breaking each test image into subimages, summing the probabilities, and applying argmax to determine the final classification.  

Below is the full image of a training sample from Canterbury text

![Full Image Canterbury](Sample_Images/Canterbury_Full.png)

Below are the submiages for the same sample

![SubImages Canterbury](Sample_Images/Canterbury1.png)
![SubImages Canterbury](Sample_Images/Canterbury2.png)
![SubImages Canterbury](Sample_Images/Canterbury3.png)
![SubImages Canterbury](Sample_Images/Canterbury4.png)
![SubImages Canterbury](Sample_Images/Canterbury5.png)

## Results

### Evaluation Metric

The evaluation metric used in this project is accuracy. This choice is based on the fact that the data is fairly balanced, making accuracy an intuitive and self-explanatory metric for assessing performance.

### Curves

Following are the loss and accuracy curves of the model

![Loss and Accuracy Curves](Sample_Images/Loss_Acc_Curves.png)

The loss and accuracy here refers to the loss and accuracy of the subimages. The "Full Image Test Accuracy" curve is the accuracy curve for the full image.

## Weights

The model_0.pth file containes the weights associated the model. To use these weights and perform prediction on any image, the user can run the part below `8.2 Loading the model`