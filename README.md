# Plant-Pathology-2020---FGVC7
Kaggle Competition: https://www.kaggle.com/c/plant-pathology-2020-fgvc7

framework: pytorch

1.	Data preprocessing
Data Augmentation uses RandomHorizontalFlip, RandomSizedCrop, RandomRotation.

2.	Model design
I used some pre-train models, such as efficientnet-b0, efficientnet-b3, efficientnet-b6, inception_v3, resnet101.
In addition, I set 5 k-fold to help increase accuracy.

-Learning rate setting:

-test time augmentation (TTA):
In order to improve the accuracy of the prediction results, I use 8 TTA.

-Parameters:
Image size: 256, 512
Batch size: 16, 32

3.	Chart of validate result
-Confusion Matrix


-Training & Validation Accuracy


-Training & Validation Loss

