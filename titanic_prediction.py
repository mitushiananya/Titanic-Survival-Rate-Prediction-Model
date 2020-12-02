#Refer to ReadME
# linear algebra (install numpy)
import numpy as np

# data processing (install pandas)
import pandas as pd

# data visualization (install seaborn, matplotlib, and install using both the preferences and pip command)
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Machine Learning Algorithms (install scikit-learn, and install using both the preferences (for Pycharm run) and pip command (for terminal))
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# load the data
test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# exploring and studying the data
print("Brief Summary of the dataset: ")
train_df.info()

print("Summary of the statistics pertaining to the numeric columns: \n", train_df.describe())

print("First 8 rows of the dataset: \n", train_df.head(8))

# dealing with missing values
print("Printing the missing values: \n")
total = train_df.isnull().sum().sort_values(ascending=False) # sort the values so that the missing values don't come first
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100 # calculating the missing values' percent
percent_2 = (round(percent_1, 1)).sort_values(ascending=False) # using the round function the percent is rounded off to the nearest 1st decimal and then sort the values
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))

# from this we have to find out which ones are the important ones that would contribute to the survival rates
# passenger id, ticket and name have got nothing to do with survival rate so we won't study and analyse them
print("See all the features: \n", train_df.columns.values)

# analysis of the features that will contribute to the survival rate
# 1. age and sex
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train_df[train_df['Sex'] == 'female']
men = train_df[train_df['Sex'] == 'male']
ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False, color= "y")
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False, color="k")
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False, color="y")
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False, color="k")
ax.legend()
_ = ax.set_title('Male')
plt.savefig('ageandsex.png')

# 2. Embarked, Pclass and Sex

FacetGrid = sns.FacetGrid(train_df, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette="Greens",  order=None, hue_order=None )
FacetGrid.add_legend()
plt.savefig('embarkedclassandsex.png')
plt.clf() # to prevent the plot figures from merging in single plots

# 3. pclass
sns.barplot(x='Pclass', y='Survived', data=train_df, palette= "rocket")
plt.savefig('pclass.png')
plt.clf()

d = {'color': ['r', 'b']}
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=2, hue_kws= d, hue='Survived' )
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
plt.savefig('pclass2.png')
plt.clf()

# 4. sibsp and parch
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)

print("Count of the not_alone feature with the number of relatives: \n", train_df['not_alone'].value_counts())

axes = sns.factorplot('relatives', 'Survived', data=train_df, aspect= 2.5)
plt.savefig('sibsipandparch.png')
plt.clf()

# drop Passenger Id as it is not related to survival rate

train_df = train_df.drop(['PassengerId'], axis=1)

# dealing with missing data and transforming data
# 1. Cabin
import re
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]

for dataset in data:
    dataset['Cabin'] = dataset['Cabin'].fillna("U0")
    dataset['Deck'] = dataset['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# we can now drop the cabin feature as we created a new feature Deck from the Cabin feature

train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

# 2. Age

data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, standard deviation using numpy

    rand_age = np.random.randint(mean - std, mean + std, size=is_null)

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

print("Number of Missing values in Age now: ", train_df["Age"].isnull().sum())

# 3. Embarked
print("Embarked Describe: \n", train_df['Embarked'].describe())

common_value = 'S'
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)

# now we will converting data types which is also a part of the data transformation
# quickly get a summary of the data-frame before transforming features to see which features require preprocessing
print("Quick Summary of the data frame once again: \n", train_df.info())

# data transformation of features
# 1. fare

data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

# 2. name

data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    # extract titles from the names
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling the missing values or NaN with 0
    dataset['Title'] = dataset['Title'].fillna(0)
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# 3. sex
genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)

# 4. Ticket
print(train_df['Ticket'].describe())

train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

# 4. Embarked
ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# make sub-features within the following features
# 1. Age (we will first convert age from float to int and then create an AgeGroup category)
data = [train_df, test_df]
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# see the count of the different age groups we just created
print("The count of different Age Groups: \n", train_df['Age'].value_counts())

# 2. Fare
print("First 10 rows: \n", train_df.head(10))

data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare'] = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

# create new features from the main features
# 1. Age
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

# 2. Fare per person
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
# final look at the training set, before we start training the models.
print("Frist 10 rows after creating new features: \n", train_df.head(10))

# machine learning models
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()

# 1. Stochastic Gradient Descent (SGD)
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# 2. Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# 3. Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# 4. K Nearest Neighbor
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# 5. Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# 6. Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# 7. Linear Support Vector Machine
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# 8. Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# now we will compare and find the best Machine learning algorithm
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent',
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
print(result_df.head(9))

# we will perform K-Fold Cross Validation on the Random Forest Algorithm using 10 Folds
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("K-FOLD VALIDATION:- ")
print("Scores: ", scores)
print("Mean: ", scores.mean())
print("Standard Deviation: ", scores.std())

# importances feature of Random Forest
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print("Importances feature of Random Forest:- ")
print(importances.head(15))
importances.plot.bar(color=(1, 0.8, 0.3, 0.8))
plt.savefig('importances.png')
plt.clf()

# conclusion not_alone and parch are not important according to Importances bar plot so we will drop them and then train Random Forest again
train_df  = train_df.drop("not_alone", axis=1)
test_df  = test_df.drop("not_alone", axis=1)

train_df  = train_df.drop("Parch", axis=1)
test_df  = test_df.drop("Parch", axis=1)

# Training random forest again
random_forest = RandomForestClassifier(n_estimators=100, oob_score = True)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Random Forest Score calculation after dropping not_alone and Parch: ", round(acc_random_forest,2,), "%")

# calculate out-of-bag samples score
print("OOB Score:", round(random_forest.oob_score_, 4)*100, "%")

# confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, Y_train, cv=3)
print("Confusion Matric/Array:- ")
print(confusion_matrix(Y_train, predictions))

# Precision and Recall
from sklearn.metrics import precision_score, recall_score

print("Precision: ", (precision_score(Y_train, predictions)) * 100)
print("Recall: ",(recall_score(Y_train, predictions))*100)

# f score
from sklearn.metrics import f1_score
print("F-Score: ", (f1_score(Y_train, predictions))*100)

# Precision Recall Curve
from sklearn.metrics import precision_recall_curve

# getting the probabilities of our predictions
y_scores = random_forest.predict_proba(X_train)
y_scores = y_scores[:,1]

precision, recall, threshold = precision_recall_curve(Y_train, y_scores)
def plot_precision_and_recall(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5, color = "k")
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5, color="y")
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])

plt.figure(figsize=(14, 7))
plot_precision_and_recall(precision, recall, threshold)
plt.savefig('precisionrecallcurve.png')