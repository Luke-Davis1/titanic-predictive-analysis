# Import libraries
from sklearn import preprocessing, tree
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
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

decision_tree = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
decision_tree = decision_tree.fit(X, y)

X_test = df_test_survival.drop(columns=["Survived?"])
y_test = df_test_survival["Survived?"]

# Predict and evaluate
y_pred = decision_tree.predict(X_test)
accuracy_decision_tree = accuracy_score(y_test, y_pred)
cm_decision_tree = confusion_matrix(y_test, y_pred)
correct_predictions_decision_tree = cm_decision_tree.diagonal().sum()
classification_report_tree = classification_report(y_test, y_pred)

# Data needs to be scaled for neural network
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X)

# Transform both the training and test data
X_train_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data back to DataFrames to preserve feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Build a neural network for predicting survived
neural_net_classifier = MLPClassifier(max_iter=1000, random_state=42)
neural_net_classifier.fit(X_train_scaled, y)

y_pred = neural_net_classifier.predict(X_test_scaled)
accuracy_neural_net = accuracy_score(y_test, y_pred)
cm_neural_net = confusion_matrix(y_test, y_pred)
correct_predictions_neural_net = cm_neural_net.diagonal().sum()
classification_report_neural_net = classification_report(y_test, y_pred)

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
]
print(tabulate(decsision_tree_results, tablefmt="pretty"))
print(f'\nClassfication Report:\n{classification_report_tree}')

print("\n" + " " * 30 + "NEURAL NETWORK RESULTS")
headers = ['Predicted 0', 'Predicted 1']
table = [['Actual 0', cm_neural_net[0, 0], cm_neural_net[0, 1]],
         ['Actual 1', cm_neural_net[1, 0], cm_neural_net[1, 1]]]
print(tabulate(table, headers=headers, tablefmt='grid'))
print("\nOther metrics:")
neural_net_results = [
    ["Accuracy", f"{accuracy_neural_net:.2f}"],
]
print(tabulate(neural_net_results, tablefmt="pretty"))

print(f'\nClassfication Report:\n{classification_report_neural_net}')

print("\n" + "*" * 80)

########################### AGE PREDICTION #####################################

# LINEAR REGRESSION

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

# DECISION TREE

# Build a decision tree to predict age
decision_tree_regressor = DecisionTreeRegressor(random_state=42)
decision_tree_regressor.fit(X, y)

# predict and evaluate
y_pred = decision_tree_regressor.predict(X_test)
mse_decision_tree = mean_squared_error(y_test, y_pred)

# NEURAL NETWORK

# Data needs to be scaled for neural network
scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X)

# Transform both the training and test data
X_train_scaled = scaler.transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data back to DataFrames to preserve feature names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Build a neural network to predict age
neural_net_regressor = MLPRegressor(max_iter=1000, random_state=42)
neural_net_regressor.fit(X_train_scaled, y)

# predict and evaluate
y_pred = neural_net_regressor.predict(X_test_scaled)
mse_neural_net = mean_squared_error(y_test, y_pred)

# print results
print(" " * 30 + "AGE PREDICTION\n")
print(f"Linear Regression MSE: " + f"{mse_linear_reg:.2f}")
print(f"Decision Tree MSE: " + f"{mse_decision_tree:.2f}")
print(f"Neural Network MSE: " + f"{mse_neural_net:.2f}")
print("\n" + "*" * 80)

