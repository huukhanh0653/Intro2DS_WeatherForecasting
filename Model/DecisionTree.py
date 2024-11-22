import pandas as pd;
from sklearn.model_selection import train_test_split;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay;
from imblearn.over_sampling import SMOTE;
import matplotlib.pyplot as plt
import copy;

class DecisionTreeModel:
    def __init__(self):
        self.x_train = None;
        self.y_train = None;
        self.x_test = None;
        self.y_test = None;
        self.accuracy = None;
        self.core = None;
        self.dataSet = None;
        self.higCorCol = None;
        self.preKey = None;
        self.y_pred = None;

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
        key = self.preKey;
        df = self.HighCorrelationRemove();
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42);
        self.x_test = df_test.drop(key, axis=1);
        self.y_test = df_test[key];
        #over sampling using smote
        smote = SMOTE(random_state=42);
        x_train = df_train.drop(key, axis=1);
        y_train = df_train[key];
        self.x_train, self.y_train = smote.fit_resample(x_train, y_train);
        

    def SetPredictionKey(self, key):
        if self.dataSet is None:
            print("Insert dataset (pandas data frame) first");
            return;
        if key in self.dataSet.columns:
            self.preKey = key;
            self.HighCorrelationDetect();
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

    def HighCorrelationDetect(self):
        exception = self.preKey;
        threshold = 0.75;
        df = copy.deepcopy(self.dataSet);
        columns = df.columns;
        for label in columns:
            if df[label].dtype == 'object':
                df = df.drop(label, axis=1);
        #corr matrix
        corr_matrix = df.corr().abs();
        #result set
        result = list();
        #columns of corr matrix
        columns = corr_matrix.columns;
        col_num = len(columns);
        #loop through the columns
        i = 0;
        while i < col_num:
            for j in range(0, i):
                if exception != columns[i] \
                and corr_matrix.iloc[i, j] > threshold \
                and columns[i] not in result:
                    result.append(columns[i]);
            i += 1;
        self.higCorCol = result;

    def HighCorrelationRemove(self):
        result = copy.deepcopy(self.dataSet);
        higCorCol = self.higCorCol;
        for i in higCorCol:
            result.drop(i, axis=1);
        return result;

    def PredictionDetails(self):
        y_test = self.y_test;
        y_pred = self.y_pred;
        print("Mark as possitive label: ", y_test[0]);
        # Precision
        precision = precision_score(y_test, y_pred, pos_label=y_test[0]);
        print("Precision:", precision);

        # Recall
        recall = recall_score(y_test, y_pred, pos_label=y_test[0]);
        print("Recall:", recall);

        # F1-score
        f1 = f1_score(y_test, y_pred, pos_label=y_test[0])
        print("F1-score:",f1);

    def PredictionConfusionMatrix(self):
        y_test = self.y_test;
        y_pred = self.y_pred;
        # Create the confusion matrix
        cm = confusion_matrix(y_test, y_pred);

        # Visualize the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm);
        disp.plot();
        plt.show();

df = pd.read_csv('./preProcessedDataWithHmdx.csv');
print(df);
dtm = DecisionTreeModel();
dtm.AddDataSet(df);
dtm.SetPredictionKey('HMDX_label');
dtm.Fit();
print("Prediction Set: ", dtm.Prediction());
dtm.PredictionDetails();
dtm.PredictionConfusionMatrix();