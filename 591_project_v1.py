import pickle
import csv
from sklearn import svm
import pandas as pd
import numpy as np
import pdb
import sklearn

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import sklearn.linear_model
import sklearn.neural_network

root_dir = '/home/myelugo/mhs/'
train_file = root_dir + 'train_v1.pickle'
valid_file = root_dir + 'valid_v1.pickle'
metadata_file = root_dir + 'Project_Dataset_Release/metadata.csv'
test_base_dir = root_dir + 'Project_Dataset_Release/LISTS/'

with open(train_file, 'rb') as handle:
    train_data = pickle.load(handle)

with open(valid_file, 'rb') as handle:
    valid_data = pickle.load(handle)

id_to_gender = {}
id_to_status = {}

with open(metadata_file, newline='\n') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        row_split = row[0].split(' ')
        if row_split[0] == 'SUB_ID':
            continue
        id_to_gender[row_split[0]] = row_split[2]
        id_to_status[row_split[0]] = row_split[1]

train_files = ['train_0.csv', 'train_1.csv', 'train_2.csv', 'train_3.csv', 'train_4.csv']
valid_files = ['val_0.csv', 'val_1.csv', 'val_2.csv', 'val_3.csv', 'val_4.csv']

def file_to_IDs(file):
    file = test_base_dir + file
    res = []

    with open(file, newline='\n') as file:
      reader = csv.reader(file)
      for row in reader:
        res.append(row[0])

    return res

def cross_validation():
    for vid in range(0, 5):
        train_ind = [0, 1, 2, 3, 4]
        train_ind.remove(vid)
        valid_ind = [vid]

        print("Training with: ", end='')
        for tfiles in train_ind:
            print(train_files[tfiles], end='')
            if tfiles != train_ind[-1]:
                print(", ", end='')
        print(" files and ", end='')
        print("Validating with " + valid_files[valid_ind[0]] + " file.")

        train_x = None 
        train_y = []

        for ind in train_ind:
            file = train_files[ind]
            ID_list = file_to_IDs(file)

            for ID in ID_list:
                if train_x is None:
                    train_x = pd.DataFrame(train_data[ID].mean()).T
                else:
                    train_x = pd.concat([train_x, pd.DataFrame(train_data[ID].mean()).T])
                train_y.append(1 if id_to_status[ID] == 'p' else 0)
            
        valid_x = None 
        res_y = []
        for ind in valid_ind:
            file = valid_files[ind]
            ID_list = file_to_IDs(file)

            for ID in ID_list:
                df = pd.DataFrame(valid_data[ID].mean()).T
                if valid_x is None:
                    valid_x = df
                else:
                    valid_x = pd.concat([valid_x, df])

                res_y.append(1 if id_to_status[ID] == 'p' else 0)


        clf = RandomForestClassifier(criterion='gini')
        #clf = svm.SVC()
        #clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
        clf.fit(train_x, train_y)

        pred = clf.predict(valid_x)

        acc = sklearn.metrics.accuracy_score(np.array(res_y), pred)
        print("Accuracy score: " + str(round(acc * 100, 2)) + "%")
        print()

cross_validation()