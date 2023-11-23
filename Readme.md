# Project: Drug Prescription Prediction using Decision Trees

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

<br>
*Figure 1. Bar Chart showing ratio of pictures classified as with and without masks*
<br>

<br>
*Figure 2. Sample image from dataset*
<br>

<br>
*Figure 3. Image Size Properties (width,height) for with mask images*
<br>

<br>
*Figure 4. Image Size Properties width,height) for without mask images*
<br>

### Data Preprocessing

#### Summary
As the dataset arrived cleaned and structured, initial tasks in the preprocessing stage involved Data Exploration and Data Visualization. The dataset structure was explored to confirm its suitability, and visualizations were prepared to gain insights into its contents. After confirming the dataset's quality, checks for missing values were performed. Dependent and independent variables were identified, and the data was transformed into a feature vector and a target value. These sets were then split into training and test sets for subsequent model evaluation.

#### Steps
**Step 1**
-


### Model Development
Implementation of a Decision Tree algorithm aimed to learn the relationships between patient characteristics and drug prescriptions. Fine-tuning of the model was conducted to achieve optimal performance in predicting drug choices.

### Model Evaluation
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
