'''Use multiple rounds to get a more robust results'''
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def cal_metrics(csv_path, type_indices, is_binary=False):
    '''
    calculate average accuracy, accuracy per skin type, PQD, DPM, EOM.
    All known skin types
    input val results csv path, type_indices: a list
    output a dic, 'acc_avg': value, 'acc_per_type': array[x,x,x], 'PQD', 'DPM', 'EOM'
    '''
    df = pd.read_csv(csv_path)
    labels_array = np.zeros((6, len(df['label'].unique())))
    correct_array = np.zeros((6, len(df['label'].unique())))
    predictions_array = np.zeros((6, len(df['label'].unique())))
    positive_list = []  # get positive probability for binary classification
    for i in range(df.shape[0]):
        prediction = df.iloc[i]['prediction']
        label = df.iloc[i]['label']
        type = df.iloc[i]['fitzpatrick'] - 1
        labels_array[int(type), int(label)] += 1
        predictions_array[int(type), int(prediction)] += 1
        if prediction == label:
            correct_array[int(type), int(label)] += 1

        if is_binary:
            if prediction == 0:
                positive_list.append(1.0 - df.iloc[i]['prediction_probability'])
            else:
                positive_list.append(df.iloc[i]['prediction_probability'])

    correct_array = correct_array[type_indices]
    labels_array = labels_array[type_indices]
    predictions_array = predictions_array[type_indices]

    # avg acc, acc per type
    correct_array_sumc, labels_array_sumc = np.sum(correct_array, axis=1), np.sum(labels_array, axis=1)  # sum skin conditions
    
    
    eps = 1e-8
    acc_array = correct_array_sumc / (labels_array_sumc + eps)
    avg_acc = np.sum(correct_array) / (np.sum(labels_array) + eps)

    # PQD
    PQD = acc_array.min() / (acc_array.max() + eps)

    # DPM
    demo_array = predictions_array / (np.sum(predictions_array, axis=1, keepdims=True) + eps)
    DPM = np.mean(demo_array.min(axis=0) / (demo_array.max(axis=0) + eps))

    # EOM
    eo_array = correct_array / (labels_array + eps)

    EOM = np.mean(np.min(eo_array, axis=0) / (np.max(eo_array, axis=0) + eps))

    # if is binary classification, output AUC
    if is_binary:
        fpr, tpr, threshold = roc_curve(df['label'], positive_list, drop_intermediate=True)
        AUC = auc(fpr, tpr)
    else:
        AUC = -1

    return {'acc_avg': avg_acc, 'acc_per_type': acc_array, 'PQD': PQD, 'DPM': DPM, 'EOM': EOM, 'AUC': AUC}






def main():
    config = {
        "epoch": 20,
        "label": "high",
        "holdout": "random_holdout",
        "model": "your-model-name",
        "dataset": "your-dataset",
        "type_indices": [0, 1, 2, 3, 4, 5],      # for in domain
        # type_indices = [ 2, 3, 4, 5]        #for outdomain-a12
        # type_indices = [0, 1, 4, 5]          #for outdomain-a34
        # type_indices = [0, 1, 2, 3]             #for outdomain-a56
        # type_indices = [0, 1, 2]         #for ddi dataset- in domain
        "csv_folder_list": ["S62", "S64", "S66", "S68", "S70"],
        "is_binary": True,
    }


    epoch = config["epoch"]
    label = config["label"]
    holdout_set = config["holdout"]
    model_name = config["model"]
    dataset_name = config["dataset"]
    type_indices = config["type_indices"]
    csv_folder_list = config["csv_folder_list"]
    is_binary = config["is_binary"]

    avg_array = np.zeros((len(csv_folder_list)))
    acc_per_type_array = np.zeros((len(csv_folder_list), len(type_indices)))
    PQD_array = np.zeros((len(csv_folder_list)))
    DPM_array = np.zeros((len(csv_folder_list)))
    EOM_array = np.zeros((len(csv_folder_list)))
    AUC_array = np.zeros((len(csv_folder_list)))

    for i in range(len(csv_folder_list)):
        seed = int(csv_folder_list[i][1:])
    
        csv_path = 'path/to/the/result_csv/results_{}_{}_{}_{}_{}_{}_{}.csv'.format(csv_folder_list[i], model_name, epoch, label, holdout_set,dataset_name, seed)

        dic = cal_metrics(csv_path, type_indices, is_binary=is_binary)
        avg_array[i] = dic['acc_avg']
        acc_per_type_array[i, :] = dic['acc_per_type']
        PQD_array[i] = dic['PQD']
        DPM_array[i] = dic['DPM']
        EOM_array[i] = dic['EOM']
        AUC_array[i] = dic['AUC']



    print('acc_avg array')
    print(avg_array)
    print('acc per type')
    print(acc_per_type_array)
    print('PQD')
    print(PQD_array)
    print('DPM')
    print(DPM_array)
    print('EOM')
    print(EOM_array)
    print('AUC')
    print(AUC_array)

    print('average accuracy mean: {}, std: {}'.format(avg_array.mean(), avg_array.std()))
    print('accuracy per skin type mean and std')
    print(np.mean(acc_per_type_array, axis=0), np.std(acc_per_type_array, axis=0))
    print('PQD mean: {}, std: {}'.format(PQD_array.mean(), PQD_array.std()))
    print('DPM mean: {}, std: {}'.format(DPM_array.mean(), DPM_array.std()))
    print('EOM mean: {}, std: {}'.format(EOM_array.mean(), EOM_array.std()))
    print('AUC mean: {}, std: {}'.format(AUC_array.mean(), AUC_array.std()))



