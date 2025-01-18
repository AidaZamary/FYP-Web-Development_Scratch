import numpy as np
from collections import Counter

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTreeRecommendation:

    def __init__(self, min_samples_split=None, max_depth=None, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        #stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        #greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        #grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        #parent loss
        parent_entropy = entropy(y)

        #generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        #compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        #information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier,plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def accuracy(y_true, y_pred):   
    accuracy = np.sum(y_true == y_pred)/len(y_true)   
    return accuracy

# RECOMMENDATION MODEL USING DECISION TREE
# Load dataset
tr_df = pd.read_csv('data\TREATMENT_RECORDv2.csv')

# Check missing values
col_names = tr_df.columns
print(col_names)
for c in col_names:
    tr_df[c] = tr_df[c].replace("?", np.NaN)

# If have missing values replace with most frequent value in that column
tr_df = tr_df.apply(lambda x:x.fillna(x.value_counts().index[0]))


# dataset
tr_category_col = ['Age', 'Gender', 'BMI', 'Asymptomatic',
       'Increased thirst', 'Polydipsia', 'Polyuria', 'lethargy', 'Weight loss',
       'Blurring of Vision', 'Recurrent infection', 'Obesity',
       'Acanthosis Nigricans', 'A1c', 'FPG', 'RPG', 'OGTT', 'HDL', 'TG', 'CVD',
       'IGT or IFG on previous testing',
       'Women who delivered a baby weighing >= 4kg ', 'Women with GDM ',
       'Women with PCOS',
       'receiving antiretroviral therapy OR atypical antipsychotic OR taking iron supplement',
       'on erythropoietin injections', 'Physical inactivity', 'Smokers ',
       'Not Balanced diet', 'First degree relative with diabetes',
       'Dyslipidaemia', 'Hypertension', 'Pancreatic damage or surgery',
       'genetic, haematologic and illness-related factors', 'Anaemia', 'CKD']
tr_labelEncoder = preprocessing.LabelEncoder()

# dataset dictionary mapping
tr_mapping_dict = {}
for col in tr_category_col:
    tr_df[col] = tr_labelEncoder.fit_transform(tr_df[col])
    tr_le_name_mapping = dict(zip(tr_labelEncoder.classes_, tr_labelEncoder.transform(tr_labelEncoder.classes_)))
    tr_mapping_dict[col] = tr_le_name_mapping
print(tr_mapping_dict)

# drop RecordNumber
tr_df=tr_df.drop(['RecordNumber'], axis=1)


# Assuming the last column is the target variable and the rest are features
X = tr_df.iloc[:, 0:36].values
y = tr_df.iloc[:, 36].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize DecisionTreeClassifier
clf = DecisionTreeClassifier(min_samples_split=2, max_depth=10)

# Perform cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')  # Using 5-fold cross-validation

print("\n Cross Validation")
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", cv_scores.mean())

print("Cross-validation scores:", cv_scores)

# Fit the model to the entire training data
clf.fit(X_train, y_train)

print("RECOMMENDATION MODEL USING DECISION TREE")
print("**************************************************************************************************************\n")
# Training Accuracy
y_pred_train = clf.predict(X_train)
acc_train = accuracy_score(y_train, y_pred_train)
print("Training Accuracy:", acc_train)

# Testing Accuracy
y_pred_test = clf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print("Testing Accuracy:", acc_test)

conf_matrix = confusion_matrix(y_test, y_pred_test)
print('Confusion Matrix:')
print(conf_matrix)

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
# Define custom class labels
class_labels = ['Primary Care', 'Secondary Care']

# Plot confusion matrix with labels and colors
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

# Classification Report
class_report = classification_report(y_test, y_pred_test)
print('Classification Report:')
print(class_report)

# Mean Absolute Error(MAE)
mae = mean_absolute_error(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae}")

# Mean Absolute Percentage Error(MAPE)
# Mean Absolute Percentage Error (MAPE)
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

mape = calculate_mape(y_test, y_pred_test)
print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# Save the trained model to a file
with open('decision_tree_recommendation.pkl', 'wb') as model_file:
    pickle.dump(clf, model_file)

# Test prediction
treatment_result = {
    1: 'Primary Care',
    2: 'Secondary Care'
}

# testing data for recommendation model
data_test = [
    [63, 1, 24, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0 , 0, 1, 0, 0, 0, 0, 0, 0],
    [55,0,20,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [50,1,40,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [49,1,28,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [70,0,21,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [64,0,34,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [70,1,23,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [64,1,28,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [51,0,23,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [50,1,27,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [45,0,38,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [50,0,21,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [63,0,30,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [60,0,35,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [48,1,20,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [63,1,39,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [70,0,34,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [68,1,28,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [64,0,21,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0],
    [48,1,25,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,2.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,0,0.0,1.0,0.0,0,0.0,0.0,0.0,0.0],
    [69,0,31,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1,0.0,0.0,0.0,0,0.0,0.0,0.0,0.0]
]
# Make predictions and interpret the results
for data in data_test:
    prediction = clf.predict([data])[0]
    print(data)
    print(f"Prediction: {prediction}")
    print(f"Result : {treatment_result[prediction]}")

print("**************************************************************************************************************\n")

# testing dataset in other model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier

# Naive Bayes
nb_clf = GaussianNB()

# Train the model
nb_clf.fit(X_train, y_train)

# Evaluate the model
nb_pred = nb_clf.predict(X_test)

# Evaluate model performance
print("RECOMMENDATION MODEL USING NAIVE BAYES \n **************************************************************************************************************\n")
print("Testing Accuracy:", accuracy_score(y_test, nb_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, nb_pred))
print('Classification Report:')
print(classification_report(y_test, nb_pred))

#AdaBoost
# Create AdaBoost classifier with decision trees as base estimators
adaboost_classifier = AdaBoostClassifier(n_estimators=10, random_state=42)

# Train the model
adaboost_classifier.fit(X_train, y_train)

# Evaluate the model
ada_pred = adaboost_classifier.predict(X_test)

# Evaluate model performance
print("RECOMMENDATION MODEL USING ADABOOST \n **************************************************************************************************************\n")
print("Testing Accuracy:", accuracy_score(y_test, ada_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, ada_pred))
print('Classification Report:')
print(classification_report(y_test, ada_pred))