# Plant-Pathology-2020---FGVC7
Kaggle Competition: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

framework: pytorch

## Data preprocessing
Data Augmentation uses RandomHorizontalFlip, RandomSizedCrop, RandomRotation.

## Model design
I used some pre-train models, such as efficientnet-b0, efficientnet-b3, efficientnet-b6, inception_v3, resnet101.

In addition, I set 5 k-fold to help increase accuracy.

-Learning rate setting:

![image] (https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/lr.jpg)


<img width="300" height="200" src="https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/lr.jpg"/></div>


-test time augmentation (TTA):
In order to improve the accuracy of the prediction results, I use 8 TTA.

-Parameters:
Image size: 256, 512
Batch size: 16, 32

## Chart of validate result
-Confusion Matrix

<img width="600" height="550" src="https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/Confusion_Matrix.jpg"/></div>

-Training & Validation Accuracy

<img width="600" height="550" src="https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/Accuracy.jpg"/></div>

-Training & Validation Loss

<img width="600" height="550" src="https://github.com/chingi071/Plant-Pathology-2020---FGVC7/blob/master/pix/Loss.jpg"/></div>
