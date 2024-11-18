import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.metrics import accuracy_score 

df = pd.read_csv('preProcessedData.csv');

class DecisionTreeModel:
    def __init__(self):
        pass;
    
    def AddDataSet(self, seed):
        #Check seed in valid form
        if isinstance(seed, pd.DataFrame):
            print("Insert successfully");
            self.dataSet = seed;
        else:
            print("Can't insert. Check input frame again.");
    
    def TrainTestSplit(self):
        if self.dataSet is None or self.preKey is None:
            print("Insert dataset and prediction key first.");
            return;
        preKey = self.preKey;
        df = self.dataSet;
        x = df.drop(preKey, axis=1);
        y = df[preKey];
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.3, random_state=42);

    def SetPredictionKey(self, key):
        if self.dataSet is None:
            print("Insert dataset (pandas data frame) first");
            return;
        if key in self.dataSet.columns:
            self.preKey = key;
            self.TrainTestSplit();
            print("Insert successfully");
        else:
            print("Can't insert. Check input key again.");
    
    def Fit(self):
        if self.dataSet is None or self.preKey is None:
            print("Insert dataset and prediction key first.");
            return;
        x_train, y_train = self.x_train, self.y_train;
        core = DecisionTreeClassifier();
        core.fit(x_train, y_train);
        self.core = core;

    def Prediction(self):
        x_test = self.x_test;
        y_pred = self.core.predict(x_test);
        self.y_pred = y_pred;
        print("Accuracy for this prediction: ", self.Evaluation());

        return y_pred;

    def Evaluation(self):
        y_test, y_pred = self.y_test, self.y_pred;
        self.accuracy = accuracy_score(y_test, y_pred);
        return self.accuracy;