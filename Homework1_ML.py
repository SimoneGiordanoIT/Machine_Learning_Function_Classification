import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import *
from sklearn.naive_bayes import *
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix





def multinomialDistribution(d_list,vectorizer):
    TotalDB = []
    TotalLabel = []


    #Iteration on every string that represent an istance of the dataset
    for datasetStr in d_list:

        #json.load works just on a single line of a json value
        result = json.loads(datasetStr)


        label = str(result.get("semantic"))

        #Change the ' with " to work better with the re library 
        stringChanged = str(result.get("lista_asm")).replace("'", '"')

        #Find the single values of lista_asm that are between " ", remember that this includes also the , that are between two values
        singleValues = re.findall('"([^"]*)"', stringChanged)

        TotalDB.append(str(singleValues))
        TotalLabel.append(label)


    dfLabel = pd.Series (TotalLabel)
    dfDB = pd.Series (TotalDB)

    
    print('----------',vectorizer)

    if vectorizer == "count":
        vectorizerForBlind = 'count'
        vectorizer = CountVectorizer() 
    elif vectorizer == "tfid":
        vectorizerForBlind = 'tfid'
        vectorizer = TfidfVectorizer()

    X_all = vectorizer.fit_transform(dfDB)
    y_all = dfLabel


    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
            test_size=0.2, random_state=10)


    model = MultinomialNB().fit(X_train, y_train)



    #Prediction of the 20% remaining in the dataset
    y_pred = model.predict(X_test)


    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


    matrix = plot_confusion_matrix(model, X_test, y_test,
                                    cmap=plt.cm.Greens ,
                                    normalize='true')

    return matrix




def KNNClassification(d_list):

    arrayFeatures = np.zeros(shape = (14397,12))
    arrayLabel = np.chararray((14397,1),itemsize=10)

    counter = 0

    #Iteration on every string that represent an istance of the dataset
    for datasetStr in d_list:

        #json.load works just on a single line of a json value
        result = json.loads(datasetStr)

        label = str(result.get("semantic"))

        #Change the ' with " to work better with the re library 
        stringListaASM = str(result.get("lista_asm")).replace("'", '"')

        #Find the single values of lista_asm that are between " ", remember that this includes also the , that are between two values
        singleValuesLista = re.findall('"([^"]*)"', stringListaASM)


        #Find the features of the function in the "lista_asm" 
        
        numXMM = str(singleValuesLista).count("xmm") 
        numXOR = str(singleValuesLista).count("xor") 
        numCMP = str(singleValuesLista).count('cmp') 
        numMOV = str(singleValuesLista).count('mov')
        numSHL = str(singleValuesLista).count('shl')  
        numRO = str(singleValuesLista).count('ro')
        numAND = str(singleValuesLista).count('and')  
        numRCR = str(singleValuesLista).count('rcr') 
        numRCL = str(singleValuesLista).count('rcl')
        numOR = str(singleValuesLista).count("'or")  
        numSHR = str(singleValuesLista).count('shr') 
        numNOT = str(singleValuesLista).count('not') 

        arrayLabel[counter] = [label]
        arrayFeatures[counter] = [numXMM,numXOR,numCMP,numMOV,numRO,numRCR,numRCL,numSHL,numSHR,numAND,numOR,numNOT]
        counter += 1




    X_train, X_test, y_train, y_test = train_test_split(arrayFeatures, arrayLabel, 
            test_size=0.2, random_state=23)


    model = KNeighborsClassifier(n_neighbors=5)

    workingModel = model.fit(X_train,y_train)

    #Compute the blindtest given the X_train of the current knn classifier

    blindtestComputation(workingModel)



    y_pred= workingModel.predict(X_test) 


    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

    matrix = plot_confusion_matrix(model, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    return matrix





def blindtestComputation (TrainnedModel):

    with open('./homework_classification/blindtest.json','r') as json_file:
        blindtest = list(json_file)



    arrayFeatures = np.zeros(shape = (757,12))

    counter = 0

    #Iteration on every string that represent an istance of the dataset
    for datasetStr in blindtest:

        #json.load works just on a single line of a json value
        result = json.loads(datasetStr)

        label = str(result.get("semantic"))

        #Change the ' with " to work better with the re library 
        stringListaASM = str(result.get("lista_asm")).replace("'", '"')

        #Find the single values of lista_asm that are between " ", remember that this includes also the , that are between two values
        singleValuesLista = re.findall('"([^"]*)"', stringListaASM)


        #Find the features of the function in the "lista_asm" 
        
        numXMM = str(singleValuesLista).count("xmm") 
        numXOR = str(singleValuesLista).count("xor") 
        numCMP = str(singleValuesLista).count('cmp') 
        numMOV = str(singleValuesLista).count('mov')
        numSHL = str(singleValuesLista).count('shl')  
        numRO = str(singleValuesLista).count('ro')
        numAND = str(singleValuesLista).count('and')  
        numRCR = str(singleValuesLista).count('rcr') 
        numRCL = str(singleValuesLista).count('rcl')
        numOR = str(singleValuesLista).count("'or")  
        numSHR = str(singleValuesLista).count('shr') 
        numNOT = str(singleValuesLista).count('not') 


        arrayFeatures[counter] = [numXMM,numXOR,numCMP,numMOV,numRO,numRCR,numRCL,numSHL,numSHR,numAND,numOR,numNOT]
        counter += 1



    y_pred= TrainnedModel.predict(arrayFeatures) 

    
    listOfLabel=np.array([x.decode() for x in y_pred])
    np.savetxt("./homework_classification/1772347.txt", listOfLabel, fmt='%s', delimiter=',')




if __name__ == "__main__":

    with open('./homework_classification/dataset.json','r') as json_file:
        dataset = list(json_file)

    listOfConfusionMatrix = []

    #Compile multinomial distributions with the different vectorizations
    listofVectorizer = ['count','tfid']

    for vectorizer in listofVectorizer:
        listOfConfusionMatrix.append(multinomialDistribution(dataset,vectorizer))

    #Compile KNN classification
    listOfConfusionMatrix.append(KNNClassification(dataset))


    plt.show()