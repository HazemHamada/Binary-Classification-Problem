import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics, svm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("error", message=".*check_inverse*.", category=UserWarning, append=False)


training=pd.read_csv("training.csv",sep=';')
validation=pd.read_csv("validation.csv",sep=';')
def clean(data):
    data=data.drop_duplicates()
    data=data.drop('variable18',axis=1)  # too many null values
    encoder = LabelEncoder()
    cat=['variable1','variable4','variable5','variable6','variable7','variable9','variable10','variable12',
         'variable13','classLabel']
    num=['variable2','variable3','variable8','variable11','variable14','variable15','variable17','variable19']
    for i in cat:
        data[i]=data[i].apply(lambda x:str(x))
    encoded = data[cat].apply(encoder.fit_transform)
    data = data[num].join(encoded)
    data.variable2 = data.variable2.apply(lambda x: str(x))
    data.variable2=data.variable2.apply(lambda x: float(x.replace(',','.')))
    data.variable3=data.variable3.apply(lambda x: float(x.replace(',','.')))
    data.variable8=data.variable8.apply(lambda x: float(x.replace(',','.')))
    data=data.fillna(data.mode())
    data=data.dropna()
    return data,encoder



def preprocess(data):
    num=['variable2','variable3','variable8','variable11','variable14','variable15','variable17','variable19']
    label=training.classLabel
    train=training.drop('classLabel',axis=1)
    train[num] = preprocessing.scale(train[num])
    transformer = FunctionTransformer(np.log1p, validate=True)
    transformer.transform(train[num])
    train[num] = preprocessing.normalize(train[num], norm='l2')
    return train,label


training,encoder=clean(training)
validation,_=clean(validation)

TrainX,TrainY=preprocess(training)
TestX,TestY=preprocess(validation)


def trainSVM(VTrainX,VTrainY, VTestX,  VTestY):
    clf = svm.SVC(gamma='auto')
    clf.fit(VTrainX, VTrainY)
    valid_pred = clf.predict(VTestX)
    valid_score = metrics.roc_auc_score(VTestY, valid_pred)
    valid_score3 = precision_recall_fscore_support(VTestY, valid_pred, average='weighted')
    print(f"Validation precision,recall,f score: ")
    print(valid_score3)
    print(f"Validation AUC score: {valid_score:.4f}")
    accuracy = accuracy_score(VTestY, valid_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return clf


svmm = trainSVM(TrainX,TrainY, TestX,  TestY)
#Validation precision,recall,f score: (0.987647905340582, 0.9872881355932204, 0.9873068679420286)
#Validation AUC score: 0.9887
#Accuracy: 98.73%


def prediction(model, encoder, input):
    input[1] = float(str(input[1]).replace(",","."))
    input[2] = float(str(input[2]).replace(",", "."))
    input[7] = float(str(input[7]).replace(",", "."))
    input[10] = float(input[10])
    input[13] = float(input[13])
    input[14] = float(input[14])
    input[15] = float(input[15])
    input[17] = float(input[17])
    cat=[0,3,4,5,6,8,9,11,12]
    for i in cat:
        input[i] = encoder.fit_transform([input[i]])
    del input[16]
    prediction = model.predict([input])
    p=''
    if prediction == 0:
        p="no"
    else:
        p='yes'
    print('The predicted lable is: ' + p)
    return prediction

input = ['b', '32,33', '0,00075', 'u', 'g', 'e', 'bb', '1,585', 't', 'f', '0', 't', 's', '420', '0', '4.2e+06', 'NaN', '1']
prediction=prediction(svmm, encoder, input)
