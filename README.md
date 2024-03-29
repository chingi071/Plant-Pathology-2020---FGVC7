# Plant-Pathology-2020---FGVC7
Kaggle Competition: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

framework: pytorch

## Data preprocessing
Data Augmentation uses RandomHorizontalFlip, RandomSizedCrop, RandomRotation.

## Model design
I used some pre-trained models, such as efficientnet-b0, efficientnet-b3, efficientnet-b6, inception_v3, resnet101.

In addition, I set 5 k-fold to help increase accuracy.

-Learning rate setting:

![image](https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/lr.jpg)



-test time augmentation (TTA):
In order to improve the accuracy of the prediction results, I use 8 TTA.

-Parameters:

Image size: 256, 512

Batch size: 16, 32

## Chart of validate result
-Confusion Matrix

![image](https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/Confusion_Matrix.jpg)

-Training & Validation Accuracy

![image](https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/Accuracy.jpg)

-Training & Validation Loss

![image](https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/Loss.jpg)
