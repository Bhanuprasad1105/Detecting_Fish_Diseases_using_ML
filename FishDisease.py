
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import normalize

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns

main = tkinter.Tk()
main.title("A Machine Learning Approach for Early Detection of Fish Diseases by Analyzing Water Quality") #designing main screen
main.geometry("1300x1200")

global filename
global X_train, X_test, y_train, y_test
global X, Y
global classifier
global dataset
global le1, le2, le3, le4, le5

def upload(): #function to upload tweeter profile
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,encoding='iso-8859-1')
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset))

def getAdaptiveFFT(data): #function to calculate FFT on recordings
    return np.fft.fft(data)/len(data)

def preprocess():
    text.delete('1.0', END)
    global X
    global Y
    global dataset
    global before_features
    global le1, le2, le3, le4, le5
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    Y = []
    #taking dataset as input which contains M cities, N indicators and T recordings
    dataset['LOCATION'] = pd.Series(le1.fit_transform(dataset['LOCATION'].astype(str)))
    dataset['STATE'] = pd.Series(le2.fit_transform(dataset['STATE'].astype(str)))
    dataset['MONTH'] = pd.Series(le3.fit_transform(dataset['MONTH'].astype(str)))
    dataset['CIPerf_(cfu/100_ml)'] = pd.Series(le4.fit_transform(dataset['CIPerf_(cfu/100_ml)'].astype(str)))
    dataset['TERMOTOL_COLIFORM_(cfu/100_ml)'] = pd.Series(le5.fit_transform(dataset['TERMOTOL_COLIFORM_(cfu/100_ml)'].astype(str)))
    dataset = normalize(dataset.values) #dataset normalization
    for i in range(len(dataset)): #looping each indicatior from dataset to calculate water quality
        fft = getAdaptiveFFT(dataset[i]) #calculating FFT on indicators
        signal = np.amax(fft) #getting max signal from FFT
        signal = str(signal)
        signal = signal[1:4]
        signal = float(signal)
        T = dataset[i,10] #getting indicator recording from dataset
        if signal < T/2: #if signal < indicator/2 then calculated value will be 1 else0
            Y.append(1)
        else:
            Y.append(0)
    Y = np.asarray(Y)
    X = dataset #dataset normalization
    text.insert(END,"Dataset Preprocessing Completed\n\n")
    text.insert(END,str(X))

def featureSelection():
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    global X, Y
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train and test split details\n")
    text.insert(END,"80% dataset records used for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset records used for testing  : "+str(X_test.shape[0])+"\n")
    coliform = X[:,6]
    ecoli = X[:,7]
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Sampling Data')
    plt.ylabel('Indicator Values')
    plt.plot(coliform, 'ro-', color = 'blue')
    plt.plot(ecoli, 'ro-', color = 'orange')
    plt.legend(['Coliform', 'Ecoli'], loc='upper left')
    #plt.xticks(wordloss.index)
    plt.title('Coliform & Ecoli Virus Found in Dataset')
    plt.show()
    
def runGradientBoosting():
    text.delete('1.0', END)
    global classifier
    global X_train, X_test, y_train, y_test
    rfc = GradientBoostingClassifier()
    rfc.fit(X_train, y_train)
    predict = rfc.predict(X_test)
    classifier = rfc  
    precision = precision_score(y_test, predict,average='macro') * 100
    recall = recall_score(y_test, predict,average='macro') * 100
    fscore = f1_score(y_test, predict,average='macro') * 100
    accuracy = accuracy_score(y_test,predict)*100
    text.insert(END,"Gradient Boosting Accuracy  : "+str(accuracy)+"\n")
    text.insert(END,"Gradient Boosting Precision : "+str(precision)+"\n")
    text.insert(END,"Gradient Boosting Recall    : "+str(recall)+"\n")
    text.insert(END,"Gradient Boosting FSCORE    : "+str(fscore)+"\n\n")
    labels = ['Healthy Fish', 'Diseases Detected']
    conf_matrix = confusion_matrix(y_test, predict)
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title("Gradient Boosting Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()     

def predict():
    global pca
    global le1, le2, le3, le4, le5
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(filename,encoding='iso-8859-1')
    test.fillna(0, inplace = True)
    temp = test.values
    test['LOCATION'] = pd.Series(le1.transform(test['LOCATION'].astype(str)))
    test['STATE'] = pd.Series(le2.transform(test['STATE'].astype(str)))
    test['MONTH'] = pd.Series(le3.transform(test['MONTH'].astype(str)))
    test['CIPerf_(cfu/100_ml)'] = pd.Series(le4.transform(test['CIPerf_(cfu/100_ml)'].astype(str)))
    test['TERMOTOL_COLIFORM_(cfu/100_ml)'] = pd.Series(le5.transform(test['TERMOTOL_COLIFORM_(cfu/100_ml)'].astype(str)))
    test = test.values
    #test = normalize(test)
    y_pred = classifier.predict(test)
    print(y_pred)
    for i in range(len(test)):
        predict = y_pred[i]  #np.argmax(y_pred[i])
        if predict == 0:
            text.insert(END,"X=%s, Predicted = %s" % (temp[i], '=====> Healthy Fish')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (temp[i], '=====> Fish will get affected by Coliform & Ecoli Virus')+"\n\n")

font = ('times', 16, 'bold')
title = Label(main, text='A Machine Learning Approach for Early Detection of Fish Diseases by Analyzing Water Quality')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Water Quality Dataset", command=upload)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess & Normalize Dataset", command=preprocess)
processButton.place(x=290,y=550)
processButton.config(font=font1) 

featureButton = Button(main, text="Features Selection", command=featureSelection)
featureButton.place(x=570,y=550)
featureButton.config(font=font1) 

gbButton = Button(main, text="Train Gradient Boosting Algorithm", command=runGradientBoosting)
gbButton.place(x=770,y=550)
gbButton.config(font=font1)

predictButton = Button(main, text="Predict Fish Condition", command=predict)
predictButton.place(x=50,y=600)
predictButton.config(font=font1) 

main.config(bg='sea green')
main.mainloop()
