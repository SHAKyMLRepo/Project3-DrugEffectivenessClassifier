# Project: Drug Prescription Prediction using Decision Trees

![DecisionTree](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/3060f55b-8812-4576-b485-ab70d230a5ca)

## Introduction

The Drug Prescription Prediction project employs Decision Trees to analyze a dataset of drug prescriptions. The primary objective is to leverage machine learning to predict the most suitable drug for a patient based on various health and personal characteristics, such as gender, age, blood pressure, and other relevant factors. This project holds significant implications for personalized medicine and optimizing drug prescription practices.

## Project Overview

**Objective**: The main goal is to develop a Decision Tree model capable of accurately predicting the most appropriate drug for a patient based on their health and personal characteristics.

## Key Components

### Data Sourcing
<br>
 The data collection process utilized a dataset from Kaggle designed to train such models. Privacy concerns and regulations regarding real patient information led to the use of this dataset for generating a proof of concept algorithm. The dataset from Kaggle serves as a foundation for showcasing how such a model could be applied in the future using real patient data.

**Datasets**:
1. **Drugs A, B, C, X, Y for Decision Trees**
     - **Description** This dataset is one that was collated specifically to train Decision Tree machine learning algorithms. It features a number of patient attributes and which drug was prescribed. There are 200 entries each with 6 fields.
     - **Usage**: This dataset is used to train the Decision Tree machine learning algorithm.
     - **Source Quality**: The data was sourced from Kaggle and is therefore a secondary source but features a high usability rating on the Kaggle website.
     - **Link**: [here](https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees)* 

### Data Analysis
<br>
#### Age Frequency of Data
<p>This chart shows the spread of ages in dataset. Ages are spread evenly with no large outliers.</p>
<br>

![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/fe806f98-012c-4c0a-8948-89eaac6fb491)
<br>
*Figure 1. Bar Chart showing frequency of Ages within Dataset*
<br>
#### Chart showing Drug Frequency
<p>Chart showing the frequency of drug prescription. There is high variability across numbers of prescriptions which may mean that the model may vary depending on which data in included in the training data</p>
<br>
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/5057a946-b077-4feb-9755-973dad340d55)
<br>
*Figure 2. Bar Chart showing frequency of Drugs prescribed within Dataset*
<br>
### Chart split by gender
<p>Chart showing the difference in Frequency of gender across the dataset. There is a good spread between genders in this dataset</p>
<br>

![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/62c038e2-90e6-481f-950f-8948f5fbd268)
<br>
*Figure 3. Bars chart showing split between Gender in dataset*
<br>
### Scatterplot showing 5 clusters in X vs y with a std dev of 1
<p>This chart is showing sampling 200 data points and generating five clusters (for the five classes in this project) that have a std dev of one (have low variability in the cluster). We can see how from this graph how the algorithm may split the data given these variables</p>
<br>

![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/c7e91360-aa32-4a28-bffc-159dbf26ca61)
<br>
*Figure 4. Scatterplot of dependent varible and independent variables showing 5 clusters*
<br>

### Data Preprocessing

#### Summary
As the dataset arrived cleaned and structured, initial tasks in the preprocessing stage involved Data Exploration and Data Visualization. The dataset structure was explored to confirm its suitability, and visualizations were prepared to gain insights into its contents. After confirming the dataset's quality, checks for missing values were performed. Dependent and independent variables were identified, and the data was transformed into a feature vector and a target value. These sets were then split into training and test sets for subsequent model evaluation.

#### Steps
1. **Data Cleaning**
- Dataset is checked for null values to ensure that the machine learning algorithm can be implemented
```Python
#Check for missing values
df.isnull().sum()
```
2. **Feature Vector Created**
- The dependent variable is extracted to a numpy array y
- The Feature vector numpy array is created from the independent variables excluding dependent variabl
```Python
X = df.drop(['Drug'], axis=1)
y = df['Drug'] #Dependent variable
```
3. **Categorical data encoding
- For the feature vector it consists of 3 categorical variables and 2 numerical variables
- OneHotEncoding is used to transform the categorical variables to numerical
```Python
#Import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
one = OneHotEncoder()
x_trans = make_column_transformer(
    (one, ['Sex','BP', 'Cholesterol']),
    remainder='passthrough'
)
X= x_trans.fit_transform(X)
```
- The classes are encoded using a label transformer
```Python
#Create label encoder for y
from sklearn.preprocessing import LabelEncoder
l_encoder=LabelEncoder()
y = l_encoder.fit_transform(y)
```
4. **Trained and Test sets created**
- A test set for model evalution is split at an 80/20 split
```Python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```

### Model Development

#### Summary
The aim of this project is to implement a Decision Tree algorithm which is capable of learning the relationships between patient characteristics and drug prescriptions. Once constructed the model performance will be evaluating focusing on the effects of utilising a small dataset in model training and what methods can be used to increase model reliability and stability by leveraging ensemble methodologies such as a Baggingg Classifier and Random Forest Classifier. Fine-tuning of the model will then be attempted to achieve optimum model performance.

#### Steps
1. **Decision Tree Classifier**
- A simple Decision Tree is first constructed
```Python
tree = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = tree.predict(X_test)
```
- The image below shows the class Regions that this Decision Tree defines
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/9a045cd5-9b8f-4332-8837-f0204a96ae32)
2. **Result**
- This model perform fairly well with an overall accuracy of 88%.
- The F1 metric shows model has perfect performance for classes 1 and 4.
- Class 0 has the lowest precision so fine tuning may be required
- However the size of the dataset may mean that different combinations of training data could yield different results, to check the next step is to shuffle the data and see how it effects performance
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/f9a046dd-e02a-4674-b015-b67889a7b6f9)
3. **Shuffling Data and trying again**
- The next step is to shuffle the training and test data and evaluating how it effects the model performance
```Python
# Change random seed to shuffle data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)
tree = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = tree.predict(X_test)
print(metrics.classification_report(y_pred, y_test))
```
3. **New Result**
- With a different shuffle of training and test data the model shows different performance.
- I think this shows the dangers of small datasets especially with the large disparity in the frequency of the classes.
- This set of training yields an overall accuracy of 96%, which is a good result.
- Class 0 still yields the worst performance this corresponds to drugA.
- Class 4 has had perfect performance in both sets which may indicate that its high frequency is causing overfitting</p>
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/91f9d9a1-2104-4e52-9dc1-7943bc785e70)

4. **Bagging Classifier**
- Next step was to try a Bagging Classifier
- We can see from the graph that the regions have become much less uniform by using mulitple estimators using a subset of 80% of the training points
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/5cd47929-e3fe-420f-8d6c-c781bdf19524)
- This method yielded 98% accuracy in the first test and 93% in the second test after reshuffling data
- This test has shown that this ensemble method offers a much more stable result set with less overfitting of the data
- Image below shows a heatmap representing the confusion matrix which shows only 3 incorrect predictions using this method
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/3a065676-c6e1-403b-91ac-f3375861cfc2)

5. **Random Forest**
- Next step was to try a Random Forest Classifier, this classifier allows you to set the number of estimators to be used and a random state to shuffle data
```Python
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(metrics.classification_report(y_pred, y_test))
```
- Image below showing how much more fine grained the class regions are defined across the dataset
![image](https://github.com/SHAKyMLRepo/Project3-DrugEffectivenessClassifier/assets/145592967/6aadf9da-72f6-4432-8a1a-6889b0f2be89)
- The first test yielded an accuracy score of 93% and the second test
- More finetuning of this model will be developed in the next iteration

### Final Model Evaluation
The Decision Tree model's performance was assessed on a separate test dataset. Evaluation metrics such as accuracy, precision, and recall were used to measure the model's effectiveness in drug prescription prediction.

### User Interface or Deployment
A simple text interface was developed as a proof of concept. This interface allows drug predictions based on user input parameters, showcasing how such an algorithm could be used by real healthcare professionals to expedite their drug prescription procedures.

## Benefits

- **Personalized Medicine**:
  - Contributes to the advancement of personalized medicine by providing a tool for tailoring drug prescriptions based on individual patient characteristics.

- **Efficient Healthcare Practices**:
  - Healthcare professionals can benefit from more efficient and informed decision-making in drug prescription, leading to better patient outcomes.

- **Data-Driven Healthcare**:
  - Showcases the potential of data-driven approaches in healthcare, optimizing the prescription process.

## Ethics and Regulations

- **Patient Privacy**:
  - Prioritizes patient privacy by utilizing mock data, protecting patient privacy, and ensuring compliance with healthcare regulations.

- **Transparency in Model Decisions**:
  - Efforts will be made to ensure transparency in the Decision Tree model's decisions, providing insights into the factors influencing drug prescription predictions.
