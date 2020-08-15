import os
import shutil
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torchvision.datasets as dset
from PIL import Image
from efficientnet_pytorch import EfficientNet
from torchvision.models import Inception3, inception_v3
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import itertools
from sklearn.model_selection import KFold, StratifiedKFold



save_file = 'predict_kfold'
if not os.path.exists(save_file):
    os.mkdir(save_file)

save_model_file = os.path.join(save_file, 'save_model')
if not os.path.exists(save_model_file):
    os.mkdir(save_model_file)

# =======================================================
gpu_device = '0,1'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")

# =======================================================
train_data_path = 'C:/python/kaggle_Plant_Pathology/dataset/Train_4_class'
train_df = pd.read_csv('C:/python/kaggle_Plant_Pathology/dataset/train.csv')
train_label = train_df.iloc[:, 1:].values
train_label = train_label[:, 2] + train_label[:, 3] * 2 + train_label[:, 1] * 3
test_data_path = 'C:/python/kaggle_Plant_Pathology/dataset/Test'

# =======================================================
# parameter
batch_size = 32
SEED = 42
N_FOLDS = 5
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
model_name = 'efficientnet-b0'

lr = 3e-4
num_epochs = 100
TTA = 8

# =======================================================
class kaggle_Dataset(Dataset):
    def __init__(self, data, transforms, mode):
        self.transforms = transforms
        self.mode = mode

        if mode == 'train' or mode == 'valid':
            data_path = []
            data_label = []

            for i in range(len(data)):
                class_eye = np.eye(4)
                if data['healthy'][i] == 1:
                    class_id = class_eye[0]
                    class_ = 'healthy'

                elif data['multiple_diseases'][i] == 1:
                    class_id = class_eye[1]
                    class_ = 'multiple_diseases'

                elif data['rust'][i] == 1:
                    class_id = class_eye[2]
                    class_ = 'rust'

                elif data['scab'][i] == 1:
                    class_id = class_eye[3]
                    class_ = 'scab'

                data_label.append(class_id)

                image_path = os.path.join(train_data_path, class_, data['image_id'][i] + '.jpg')
                data_path.append(image_path)

            self.data_path = data_path
            self.labels = np.array(data_label)

        elif mode == 'test':
            x_test = []
            for image_id in os.listdir(data):
                test_path = os.path.join(data, image_id)
                x_test.append(test_path)

            self.data_path = x_test

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx]).convert("RGB")
        img = self.transforms(img)

        if self.mode == 'test':
            return img

        else:
            labels = self.labels[idx]
            return img, labels


train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.RandomHorizontalFlip(p=0.3),
                                      transforms.RandomSizedCrop(224),
                                      transforms.RandomRotation(10),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      ])

valid_transform = transforms.Compose([transforms.Resize((256, 256)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                      ])

test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.RandomHorizontalFlip(p=0.3),
                                     transforms.CenterCrop(224),
                                     transforms.RandomRotation(10),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])


def test_dataset_loader(test_data_path, TTA):
    if TTA == 1:
        test_dataset = kaggle_Dataset(test_data_path, transforms=valid_transform, mode='test')
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    else:
        test_dataset = kaggle_Dataset(test_data_path, transforms=test_transform, mode='test')
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return test_loader

# =======================================================
def PlantModel():
    if model_name == 'efficientnet-b0':
        model = EfficientNet.from_pretrained(efficientnet - b0)
        model._fc = nn.Linear(in_features=1280, out_features=4)

    elif model_name == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(efficientnet - b3)
        model._fc = nn.Linear(in_features=1536, out_features=4)

    elif model_name == 'efficientnet-b6':
        model = EfficientNet.from_pretrained(efficientnet - b3)
        model._fc = nn.Linear(in_features=2304, out_features=4)

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        model.fc = nn.Linear(in_features=2048, out_features=4)

    elif model_name == 'inception_v3':
        model = inception_v3(pretrained=True)
        model.aux_logits = False
        model.fc = nn.Linear(in_features=2048, out_features=4)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    return model

# =======================================================
# train
def train_one_fold(i_fold, model, exp_lr_scheduler, optimizer, train_loader, valid_loader):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    train_fold_results = []

    for epoch in range(num_epochs):
        ## training
        start = time.time()
        model.train()
        exp_lr_scheduler.step()
        total_train = 0
        correct_train = 0
        total_train_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data).cuda()
            target = Variable(target.float()).cuda()
            target_value = torch.max(target, 1)[1]

            optimizer.zero_grad()  # Clear gradients
            output = model(data)  # Forward propagation
            output_log = F.log_softmax(output, dim=1)
            train_loss_ = F.nll_loss(output_log, target_value, reduction='sum')

            train_loss_.backward()  # Calculate gradients
            optimizer.step()  # Update parameters

            predicted = torch.max(output_log.data, 1)[1]

            total_train += len(target_value)
            correct_train += sum((predicted == target_value).float())
            total_train_loss += train_loss_.item()

            if batch_idx % 10 == 0:
                print(
                    f'Train Epoch: {epoch + 1}/{num_epochs} [iter:{batch_idx + 1}/{len(train_loader)}], acc:{correct_train / float((batch_idx + 1) * batch_size)}, loss:{total_train_loss / float((batch_idx + 1) * batch_size)}')

        train_acc_ = 100 * ((correct_train / total_train).float())
        train_acc.append(train_acc_.item())
        train_loss.append(total_train_loss / total_train)

        ## evaluate
        model.eval()
        total_val = 0
        corrent_val = 0
        total_val_loss = 0
        val_target_list = []
        val_predict_list = []

        with torch.no_grad():
            for data, target in valid_loader:
                data = Variable(data).cuda()
                target = Variable(target.float()).cuda()
                target_value = torch.max(target, 1)[1]
                val_target_list.append(target_value.cpu().numpy())

                val_output = model(data)
                val_output_log = F.log_softmax(val_output, dim=1)
                val_loss_ = F.nll_loss(val_output_log, target_value, reduction='sum')

                val_predicted = torch.max(val_output_log.data, 1)[1]
                val_predict_list.append(val_predicted.cpu().numpy())

                total_val += len(target_value)
                corrent_val += sum((val_predicted == target_value).float())
                total_val_loss += val_loss_.item()

        val_acc_ = 100 * ((corrent_val / total_val).float())
        val_acc.append(val_acc_.item())
        val_loss.append(total_val_loss / total_val)

        if (epoch == 0):
            val_loss_min = val_loss_

        if val_loss_ < val_loss_min:
            torch.save(model.state_dict(), os.path.join(save_model_file, 'model_best.pth'))
            val_loss_min = val_loss_
            val_target_value = val_target_list
            val_predict_value = val_predict_list

        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}% \nVal_Loss: {} Val_accuracy: {:.6f}%'.format(
            epoch + 1, num_epochs, train_loss[epoch], train_acc_, val_loss[epoch], val_acc_))
        end = time.time()
        print('1 epoch spends ', end - start, ' seconds')

    train_fold_results.append({
        'fold': i_fold,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'val_acc': val_acc,
        'val_loss': val_loss,
        'val_target_value': val_target_value,
        'val_predict_value': val_predict_value
    })

    return train_fold_results

def kfold_(TTA, test_data_path):
    train_results = []
    prediction_list = []
    total_predict = []
    test_loader = test_dataset_loader(test_data_path, TTA)

    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df, train_label)):
        start = time.time()
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))

        valid = train_df.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        train = train_df.iloc[train_idx]
        train.reset_index(drop=True, inplace=True)

        dataset_train = kaggle_Dataset(train, transforms=train_transform, mode='train')
        dataset_valid = kaggle_Dataset(valid, transforms=valid_transform, mode='valid')

        train_loader = DataLoader(dataset_train , batch_size, shuffle=True)
        valid_loader = DataLoader(dataset_valid , batch_size, shuffle=False)

        model = PlantModel()
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

        train_fold_results = train_one_fold(i_fold, model, exp_lr_scheduler, optimizer, train_loader, valid_loader)
        train_results += train_fold_results

        with torch.no_grad():
            model.load_state_dict(torch.load(os.path.join(save_model_file, 'model_best.pth')))
            model.eval()
            pred_exp_list = []

            print('***********************')
            print('start predict.')

            for tta in range(TTA):
                pred = []
                print(f'TTA {tta+1} start predict.')

                for inputs in test_loader:
                    pred_output = model(inputs.cuda())
                    pred_output_log = F.log_softmax(pred_output, dim=1)
                    pred_exp = torch.exp(pred_output_log)

                    for pred_exp_idx in pred_exp:
                        pred.append(pred_exp_idx.cpu().numpy())

                pred_exp_list.append(pred)

            tta_ensem = []

            for model_idx in range(len(pred_exp_list)):
                if model_idx == 0:
                        tta_ensem += pred_exp_list[0]

                else:
                    for pred_idx in range(len(pred_exp_list[0])):
                        tta_ensem[pred_idx] += pred_exp_list[model_idx][pred_idx]

            tta_ensem_ = [i/TTA for i in tta_ensem]

            total_predict.append(tta_ensem_)

        end = time.time()
        print(end-start, ' seconds')
        print('********************************')

    fold_ensem = []

    for fold_idx in range(len(total_predict)):
        if fold_idx == 0:
                fold_ensem += total_predict[0]

        else:
            for pred_idx in range(len(total_predict[0])):
                fold_ensem[pred_idx] += total_predict[fold_idx][pred_idx]

    fold_ensem_ = [i/N_FOLDS for i in fold_ensem]

    pred_exp_list = pd.DataFrame(fold_ensem_, columns = ['healthy', 'multiple_diseases', 'rust', 'scab'])

    test_image_id = []
    for image_id in os.listdir(test_data_path):
        test_image_id.append(image_id.replace('.jpg',''))

    test_image_id = pd.DataFrame(test_image_id, columns = ['image_id'])

    predict_df = pd.concat([test_image_id, pred_exp_list], axis = 1)

    return predict_df, train_results

predict_df, train_results = kfold_(TTA, test_data_path)

# =======================================================
# visualization
def result_list(i, name):
    result = []
    for i_idx in i:
        result.append(i_idx)
    return pd.DataFrame(result, columns=[name])

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')

def visualization(train, valid, title, fold):
    plt.plot(range(num_epochs), train, 'b-', label=f'Training_{title}')
    plt.plot(range(num_epochs), valid, 'g-', label=f'validation{title}')
    plt.title(f'Training & Validation {title}')
    plt.xlabel('Number of epochs')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(os.path.join(save_file, f"{title}_{fold}.png"))
    plt.show()

def all_confusion_matrix(val_target_value, val_predict_value, fold):
    labels_classes = ['healthy', 'multiple_diseases', 'rust', 'scab']

    val_target_value_ = []
    val_predict_value_ = []

    for i in val_target_value:
        for j in range(len(i)):
            val_target_value_.append(i[j])

    for i in val_predict_value:
        for j in range(len(i)):
            val_predict_value_.append(i[j])

    val_micro = metrics.precision_score(val_target_value_, val_predict_value_, average='micro')
    val_macro = metrics.precision_score(val_target_value_, val_predict_value_, average='macro')
    print('val_micro', val_micro)
    print('val_macro', val_macro)

    cm = metrics.confusion_matrix(val_target_value_, val_predict_value_)
    plot_confusion_matrix(cm, labels_classes)
    plt.savefig(os.path.join(save_file, f"confusion_matrix_{fold}_TTA{TTA}.png"))
    plt.show()

    return val_micro, val_macro

valid_metric = []
for i in train_results:
    fold = i['fold']
    train_acc = i['train_acc']
    train_loss = i['train_loss']
    val_acc = i['val_acc']
    val_loss = i['val_loss']

    print('Fold:', fold)
    train_acc_result = result_list(train_acc, 'train_acc')
    train_loss_result = result_list(train_loss, 'train_loss')
    val_acc_result = result_list(val_acc, 'val_acc')
    val_loss_result = result_list(val_loss, 'val_loss')
    train_valid_result = pd.concat([train_acc_result, train_loss_result, val_acc_result, val_loss_result], axis=1)
    train_valid_result.to_csv(os.path.join(save_file, f'train_valid_result_{fold}_fold_TTA{TTA}.csv'))

    val_target_value = i['val_target_value']
    val_predict_value = i['val_predict_value']
    val_micro, val_macro = all_confusion_matrix(val_target_value, val_predict_value, fold, 1)
    visualization(train_acc, val_acc, 'Accuracy', fold, 1)
    visualization(train_loss, val_loss, 'Loss', fold, 1)
    valid_metric.append({'fold': fold, 'val_micro': val_micro, 'val_macro': val_macro})

valid_metric_df = pd.DatsFrame(valid_metric, columns=['fold', 'val_micro', 'val_macro'])
valid_metric_df.to_csv(os.path.join(save_file, f'valid_metric_df_TTA{TTA}.csv'), index=False)
# =======================================================

predict_df.to_csv(os.path.join(save_file, f'Result_tta{TTA}.csv'), index=False)