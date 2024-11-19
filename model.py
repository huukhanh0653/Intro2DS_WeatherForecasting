import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.metrics import accuracy_score;
import copy;

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
        self.HighCorrelationRemove();
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
        print("Fit finish successfully!");

    def Prediction(self):
        if self.x_test is None or self.core is None:
            print("Insert dataset, prediction key and fit first.");
            return;
        x_test = self.x_test;
        y_pred = self.core.predict(x_test);
        self.y_pred = y_pred;
        print("Accuracy for this prediction: ", self.Evaluation());

        return y_pred;

    def Evaluation(self):
        if self.x_test is None or self.core is None:
            print("Insert dataset, prediction key and predict first.");
            return;
        y_test, y_pred = self.y_test, self.y_pred;
        self.accuracy = accuracy_score(y_test, y_pred);
        return self.accuracy;

    def HighCorrelationRemove(self):
        threshold = 0.75;
        df = copy.deepcopy(self.dataSet);
        columns = df.columns;
        for label in columns:
            if df[label].dtype == 'object':
                df = df.drop(label, axis=1);
        #corr matrix
        corr_matrix = df.corr().abs();
        #result set
        result = copy.deepcopy(self.dataSet);
        #columns of corr matrix
        columns = corr_matrix.columns;
        col_num = len(columns);
        #loop through the columns
        i = 0;
        while i < col_num:
            for j in range(0, i):
                corr = corr_matrix.iloc[i, j];
                if corr > threshold:
                    #Drop from result
                    result = result.drop(columns[i], axis=1);
                    #Drop from both side of matrix
                    corr_matrix = corr_matrix.drop(columns[i], axis=0);
                    corr_matrix = corr_matrix.drop(columns[i], axis=1);
                    #Drop the current column from list
                    columns = columns.drop(columns[i]);
                    col_num -= 1;
                    #Back at current index
                    i -= 1;
                    break;
            i += 1;
        self.dataSet = result;

df = pd.read_csv('preProcessedDataWithHmdx.csv');
dtm = DecisionTreeModel();
dtm.AddDataSet(df);
dtm.SetPredictionKey('HMDX_label');
dtm.Fit();
print("Prediction Set: ", dtm.Prediction());
print(dtm.dataSet);