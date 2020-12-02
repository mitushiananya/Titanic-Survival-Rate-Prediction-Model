Survival Rate of Passengers on Titanic

There are two CSV Files. 
1.	Training set: train.csv
2.	Test Set: test.csv

The training set will be used to build our machine learning models. The test set will be used to see how well our model performs on unseen data. 

Data Dictionary:  

| Column Name	| Column Description                                       |	Column Values (if applicable)                     |
| ----------- |:--------------------------------------------------------:| --------------------------------------------------:|                                            
| PassengerId	| IDs of different passengers                              |	                                                  |
| Survived	  | Survival status of the passengers                        |	0 = Not Survived 1 = Survived                     |
| PClass	    | Ticket Class, that is socio-economic status (SES)	       | 1st = Upper, 2nd = Middle, 3rd = Lower             |
| Name	      | Full name of the passengers	                             |                                                    |
| Sex	        | Gender of  the passengers	                               |                                                    |
| Age	        | Age of the passengers. Age is fractional if less than 1. | If the age is estimated, is it in the form of xx.5	|
| SibSp	      | Sibling = brother, sister, stepbrother, stepsister       |                                                    | 
|             | Spouse = husband, wife                                   |                                                    |
|             | The number of siblings/spouses on board	                 |                                                    |
| Parch	      | Parent = mother, father                                  |                                                    |   
|             | Child = daughter, son, stepdaughter, stepson             |                                                    |
|             | The number of parents/children on board	                 |                                                    |  
| Ticket	    | Ticket Number	                                           |                                                    |
| Fare        |	Fare 	                                                   |                                                    | 
| Cabin	      | Cabin Number	                                           |                                                    | 
| Embarked	  | Port of Embarkation	                                     |C = Cherbourg, Q = Queenstown, S = Southampton      |


I. Import all the libraries 

•	Numpy: For linear algebra
•	Pandas, Seaborn amd Matplotlib: for data analysis and visualisation
•	Sklearn: for building machine learning models or algorithms

II. Load the Data

Load both the training and test sets.

III. Data Analysis

We study and explore the data using different summary functions like info(), describe(), head(), column.values(). In this part, we also plot various curves to determine the survival rate. 

IV. Data Transformation

We deal with missing values so that they don’t pose as an obstacle to our machine learning model. In this stage we convert various features to numeric ones so that the machine learning algorithms can process them. 

V. Make sub-features

We create various sub-features withing the following features:
1.	Age
2.	Fare

VI. Create new features from main features

1.	Age times class
2.	Fare per person

VII. Using Machine Learning Models

We will make use of eight different machine learning algorithms namely: Stochastic Gradient Descent, Random Forest, Logistic Regression, KNN, Decision Tree, Pereptron, Gaussian Naïve Bayes, Linear Support Vector Machine.

VIII. Perform K-Folds Cross Validation

We choose K = 10, thus, we use 10 folds. (Read more at: https://machinelearningmastery.com/k-fold-cross-validation/#:~:text=Cross%2Dvalidation%20is%20a%20resampling,is%20to%20be%20split%20into.)

IX. Importances feature of Random Forest

This step measures relative importance of each feature. And then we also plot a bar graph. 

X. OOB Score

We train Random Forest and also calculate OOB samples to estimate the accuracy. 

XI. Confusion Matrix

This gives information about how well our model works. 
Not Survived Prediction: 485 passengers were correctly classified and 64 were wrongly classified as not survived.
Survived Prediction: 100 passengers were wrongly classified and 242 were correctly classified as survived. 

XII. Precision and Recall

The model predicts approximately 78% as the survival rate of the passenger, whereas the recall gives the actual rate of survival which is approximately 70%. 

XIII. F-Score Calculation:

It is the harmonic mean of the precision and recall. The models F-Score is approximately 74%.

XIV. Last step

We plot the precision recall curve using matplotlib. 
