import tflearn
import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

def put_indexes(data, indexes):
    for n in range(len(data)):
        for a in range(len(data[n])):
            data[n][a] = indexes[data[n][a]]
    return np.array(data, dtype=np.float32)

def validate(validation_set_input, validation_set_target):
    p = 0
    for n in range(len(validation_set_input)):
        if validation_set_input[n] == validation_set_target[n]:
            p += 1
    if p != 0:
        print('accuracy = ' + str(p / len(validation_set_target)))
    else:
        print('accuracy = 0')

def create_model1(data, labels, validation_set_input, validation_set_target):
    net = tflearn.input_data(shape=[None, 4])
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 32)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net)
    model = tflearn.DNN(net, best_checkpoint_path='models/mymodel_ner_')
    model.fit(data, labels, n_epoch=3, validation_set=(validation_set_input, validation_set_target), batch_size=100, show_metric=True)
    return  model


def train_net(datafile):
    indexes = {'A': 0, 'S': 1, 'V': 2, 'PR': 3, 'CONJ': 4, 'ADVPRO': 5, 'ADV-PRO': 6, 'INTJ': 7, 'ADV': 8, 'PART': 9,
               'A-PRO': 10, 'SPRO': 11, 'S-PRO': 12, 'PRAEDIC': 13, 'APRO': 14, 'NUM': 15, 'ANUM': 16, 'PARENTH': 17,
               '??': 18, 'NONLEX': 19, 'INIT': 20}
    all_data = open(datafile, 'r', encoding='utf-8').read().split('\n')
    data = put_indexes([i.split(';')[:-1] for i in all_data], indexes)
    labels = np.array([[0, indexes[i.split(';')[-1]]] for i in all_data], dtype=np.float32)
    validation_set_input = data[17000:]
    validation_set_target = labels[17000:]
    data = data[:17000]
    labels = labels[:17000]
    print(len(data), len(labels))
    print(len(validation_set_input), len(validation_set_target))
    model = create_model1(data, labels, validation_set_input, validation_set_target)
    return model

def train_svm(datafile):
    indexes = {'A': 0, 'S': 1, 'V': 2, 'PR': 3, 'CONJ': 4, 'ADVPRO': 5, 'ADV-PRO': 6, 'INTJ': 7, 'ADV': 8,
                   'PART': 9,
                   'A-PRO': 10, 'SPRO': 11, 'S-PRO': 12, 'PRAEDIC': 13, 'APRO': 14, 'NUM': 15, 'ANUM': 16,
                   'PARENTH': 17,
                   '??': 18, 'NONLEX': 19, 'INIT': 20}
    all_data = open(datafile, 'r', encoding='utf-8').read().split('\n')
    data = put_indexes([i.split(';')[:-1] for i in all_data], indexes)
    labels = np.array([indexes[i.split(';')[-1]] for i in all_data], dtype=np.float32)
    validation_set_input = data[17000:]
    validation_set_target = labels[17000:]
    data = data[:17000]
    labels = labels[:17000]
    print('--SVC--')
    model1 = SVC()
    model1.fit(data, labels)
    val = model1.predict(validation_set_input)
    validate(val, validation_set_target)
    print('--LinearRegression--')
    model2 = linear_model.LinearRegression()
    model2.fit(data, labels)
    val = model2.predict(validation_set_input)
    validate(val, validation_set_target)
    print('--DecisionTreeClassifier--')
    model3 = tree.DecisionTreeClassifier()
    model3.fit(data, labels)
    val = model3.predict(validation_set_input)
    validate(val, validation_set_target)
    print('--GaussianNB--')
    model4 = GaussianNB()
    model4.fit(data, labels)
    val = model4.predict(validation_set_input)
    validate(val, validation_set_target)
    return model1, model2, model3, model4


model_ner = train_net('set_to_train.txt')
model1, model2, model3, model4 = train_svm('set_to_train.txt')
joblib.dump(model1, 'mymodel_SVC.pkl')
#joblib.dump(model2, 'mymodel_LR.pkl')
joblib.dump(model3, 'mymodel_DT.pkl')
joblib.dump(model4, 'mymodel_NB.pkl')