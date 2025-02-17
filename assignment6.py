# Import libraries
from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd
import warnings
from sklearn.exceptions import ConvergenceWarning
from tabulate import tabulate

# Suppress ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Set options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

df_train = pd.read_csv("Titanic_crew_train.csv")
df_test = pd.read_csv("Titanic_crew_test.csv")

# Pre-processing
# Drop URL columns, irrelevant
# Drop Ticket, Fare and Cabin Columns. No data so we can't do anything with them
columns_to_drop = ["URL", "Fare", "Ticket", "Cabin"]
df_train.drop(columns=columns_to_drop, inplace=True)
df_test.drop(columns=columns_to_drop, inplace=True)

# Drop rows with no survival
df_train_survival = df_train.dropna(subset=["Survived?"])
df_test_survival = df_test.dropna(subset=["Survived?"])

# Drop rows with no age, ONLY applies to training set
df_train_age = df_train.dropna(subset=["Age"])
df_test_age = df_test.copy()

# Convert all strings to integers, can't do it sooner because nulls would be
# assigned an integer value
df_train_survival = df_train_survival.apply(preprocessing.LabelEncoder().fit_transform)
df_test_survival = df_test_survival.apply(preprocessing.LabelEncoder().fit_transform)

df_train_age = df_train_age.apply(preprocessing.LabelEncoder().fit_transform)
df_test_age = df_test_age.apply(preprocessing.LabelEncoder().fit_transform)


###################### SURVIVED PREDICTION ###################################

# Build a decision tree for predicting survived
X = df_train_survival.drop(columns=["Survived?"])
y = df_train_survival["Survived?"]

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree = decision_tree.fit(X, y)

X_test = df_test_survival.drop(columns=["Survived?"])
y_test = df_test_survival["Survived?"]

# Predict and evaluate
y_pred = decision_tree.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred)
cm_decision_tree = confusion_matrix(y_test, y_pred)
correct_predictions_decision_tree = cm_decision_tree.diagonal().sum()
mse_decision_tree = mean_squared_error(y_test, y_pred)

# Build a neural network for predicting survived
neural_net_classifier = MLPClassifier()
neural_net_classifier.fit(X, y)

y_pred = neural_net_classifier.predict(X_test)
accuracy_neural_net = accuracy_score(y_test, y_pred)
cm_neural_net = confusion_matrix(y_test, y_pred)
correct_predictions_neural_net = cm_neural_net.diagonal().sum()
mse_neural_net = mean_squared_error(y_test, y_pred)

# Print results
print("*" * 80)
print(" " * 30 + "SURVIVED PREDICTION\n")
print("Survived prediction:")
print(f"Decision trees: {correct_predictions_decision_tree}/{len(y_pred)}")
print(f"Neural networks: {correct_predictions_neural_net}/{len(y_pred)}")

print("\n" + " " * 30 + "DECISION TREE RESULTS")
print(f'Confusion matrix:')
headers = ['Predicted 0', 'Predicted 1']
table = [['Actual 0', cm_decision_tree[0, 0], cm_decision_tree[0, 1]],
         ['Actual 1', cm_decision_tree[1, 0], cm_decision_tree[1, 1]]]
print(tabulate(table, headers=headers, tablefmt='grid'))
print("\nOther metrics:")
decsision_tree_results = [
    ["Accuracy", accuracy_decision_tree],
    ["MSE", mse_decision_tree]
]
print(tabulate(decsision_tree_results, tablefmt="pretty"))

print("\n" + " " * 30 + "NEURAL NETWORK RESULTS")
headers = ['Predicted 0', 'Predicted 1']
table = [['Actual 0', cm_neural_net[0, 0], cm_neural_net[0, 1]],
         ['Actual 1', cm_neural_net[1, 0], cm_neural_net[1, 1]]]
print(tabulate(table, headers=headers, tablefmt='grid'))
print("\nOther metrics:")
neural_net_results = [
    ["Accuracy", f"{accuracy_neural_net:.2f}"],
    ["MSE", f"{mse_neural_net:.2f}"]
]
print(tabulate(neural_net_results, tablefmt="pretty"))
print("\n" + "*" * 80)

# TODO: Still need to print the classification report


########################### AGE PREDICTION #####################################

# Build a linear regression to predict age
X = df_train_age.drop(columns=["Age"])
y = df_train_age["Age"]

X_test = df_test_age.drop(columns=["Age"])
y_test = df_test_age["Age"]

linear_reg_classifier = LinearRegression()
linear_reg_classifier.fit(X, y)

# Predict and evaluate
y_pred = linear_reg_classifier.predict(X_test)
mse_linear_reg = mean_squared_error(y_test, y_pred)

# Build a decision tree to predict age


# print results
print("\n"+ "*" * 80)
print(" " * 30 + "AGE PREDICTION\n")
print(f"Linear Regression MSE: {mse_linear_reg}")

print("\n" + "*" * 80)

