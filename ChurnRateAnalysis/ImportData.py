# Below are All the imports
from typing import Set, Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import helperFunctions as hF

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif



# Importing the customer data for Churn rate analysis
churn_data = pd.read_csv('resources/customer_churn_data.csv')

# Different steps for gathering information about the data

# Displaying the info about number of columns and rows
print("Number of columns and rows in the data frame: " + str(churn_data.shape) + "\n")

# Displaying the info about different columns types
print("Following are the data type for each column in the data frame:")
print(churn_data.info())
print()

# -------------------------------------------------------------------------------------------------

# CLEANING DATA

# Making a copy of Data Frame

try:
    clean_churn_data = churn_data.copy()
    print('Data frame successfully copied')
except Exception as e:
    print(e)
    raise Exception('Copying of data frame was not successful')

# Checking for duplicate rows and dropping them if found in data frame
clean_churn_data = hF.find_drop_duplicate_rows(clean_churn_data)

# Checking for ' ' values and replacing those with NaN in different columns
clean_churn_data = hF.find_replace_invalid_values(clean_churn_data)

# Checking for NaN in each column and dropping the rows for NaN values
clean_churn_data = hF.find_drop_nan_values(clean_churn_data)

# Dropping customerID column
try:
    clean_churn_data.drop(columns=['customerID'], inplace=True)
    print('Column dropped successfully')
except Exception as e:
    print(e)
    raise Exception('Error while dropping column')

# Replacing "Bank transfer (automatic)" and "Credit card (automatic)" with
# "Bank transfer" and "Credit card" in "PaymentMethod" column
# Replacing '0' with 'No' and 'Yes' with '1' in "SeniorCitizen" column
clean_churn_data.replace({'PaymentMethod': {'Bank transfer (automatic)': 'Bank transfer',
                                            'Credit card (automatic)': 'Credit card'},
                          'SeniorCitizen': {0: 'No', 1: 'Yes'}}, inplace=True)

# Renaming the columns with initials to capital letters
clean_churn_data = clean_churn_data.rename(columns={'gender': 'Gender', 'tenure': 'Tenure'})

# Change the data type for column "TotalCharges" from object to float
clean_churn_data['TotalCharges'] = clean_churn_data['TotalCharges'].astype(float)

# Replacing the values 'Yes' and 'No' with 1 and 0 respectively in column "Churn"
clean_churn_data['ChurnRate'] = clean_churn_data['Churn']
clean_churn_data['ChurnRate'] = clean_churn_data['ChurnRate'].map({'Yes': 1, 'No': 0})

# Saving cleaned up data in a new csv file. This is just an extra step to verify clean up of data with naked eyes
clean_churn_data.to_csv('cleaned_data.csv', index=False)

# Printing the information for the cleaned up data
print(clean_churn_data.info())
print(clean_churn_data.dtypes)
print(clean_churn_data.shape)

# -------------------------------------------------------------------------------------------------

# CHURN ANALYSIS
plot_churn_data = clean_churn_data.copy()

# Following is the Univariate analysis with bar graphs
# hF.variables_distribution_with_churn(plot_churn_data)

# Following is the univariate analysis with line graphs
# line_graphs_column_list = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents']
# for column_names in line_graphs_column_list:
#     sb.relplot(data=plot_churn_data, x=column_names, y='ChurnRate', kind='line', palette="Set2").set(ylabel='Churn Rate')
# plt.show()

# Calculating unique values
hF.calc_churn_rate_per_category(plot_churn_data)

# Grouping the tenure in bins of 6 months
tenure_labels = ["{0} - {1}".format(i, i + 5) for i in range(1, 72, 6)]
plot_churn_data['TenureBins'] = pd.cut(plot_churn_data.Tenure, range(1, 74, 6), right=False, labels=tenure_labels)
print(plot_churn_data['TenureBins'])
print(plot_churn_data['TenureBins'].value_counts())

# Grouping the Monthly Charges in bins of 15 each
monthly_labels = ["{0} - {1}".format(i, i + 14) for i in range(16, 120, 15)]
plot_churn_data['MonthlyBins'] = pd.cut(plot_churn_data.MonthlyCharges, range(16, 125, 15), right=False,
                                        labels=monthly_labels)
print(plot_churn_data['MonthlyBins'])
print(plot_churn_data['MonthlyBins'].value_counts())

# Grouping the Total Charges in bins of 1500 each
total_charges_labels = ["{0} - {1}".format(i, i + 1499) for i in range(1, 9000, 1500)]
plot_churn_data['TotalChargesBins'] = pd.cut(plot_churn_data.TotalCharges, range(1, 9500, 1500), right=False,
                                             labels=total_charges_labels)
print(plot_churn_data['TotalChargesBins'])
print(plot_churn_data['TotalChargesBins'].value_counts())

# Plotting the graphs for Accounts Information category
# bins_column_list = ['TenureBins', 'MonthlyBins', 'TotalChargesBins']
# hF.bins_distribution_with_churn(plot_churn_data, bins_column_list)

# Calculating unique values
# hF.calc_churn_rate_per_category(plot_churn_data)




# dist=(plot_churn_data['TenureBins'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['TenureBins']=plot_churn_data['TenureBins'].map(wordsDict)
#
# dist=(plot_churn_data['MonthlyBins'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['MonthlyBins']=plot_churn_data['MonthlyBins'].map(wordsDict)
#
# dist=(plot_churn_data['TotalChargesBins'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['TotalChargesBins']=plot_churn_data['TotalChargesBins'].map(wordsDict)
#
# dist=(plot_churn_data['Partner'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['Partner']=plot_churn_data['Partner'].map(wordsDict)
#
#
# dist=(plot_churn_data['SeniorCitizen'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['SeniorCitizen']=plot_churn_data['SeniorCitizen'].map(wordsDict)
#
# dist=(plot_churn_data['Dependents'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['Dependents']=plot_churn_data['Dependents'].map(wordsDict)
#
#
# dist=(plot_churn_data['InternetService'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['InternetService']=plot_churn_data['InternetService'].map(wordsDict)
#
#
# dist=(plot_churn_data['OnlineSecurity'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['OnlineSecurity']=plot_churn_data['OnlineSecurity'].map(wordsDict)
#
#
# dist=(plot_churn_data['OnlineBackup'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['OnlineBackup']=plot_churn_data['OnlineBackup'].map(wordsDict)
#
# dist=(plot_churn_data['DeviceProtection'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['DeviceProtection']=plot_churn_data['DeviceProtection'].map(wordsDict)
#
#
# dist=(plot_churn_data['PaymentMethod'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['PaymentMethod']=plot_churn_data['PaymentMethod'].map(wordsDict)
#
# dist=(plot_churn_data['PaperlessBilling'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['PaperlessBilling']=plot_churn_data['PaperlessBilling'].map(wordsDict)
#
# dist=(plot_churn_data['Contract'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['Contract']=plot_churn_data['Contract'].map(wordsDict)
#
# dist=(plot_churn_data['TechSupport'])
# distset= set(dist)
# dd = list(distset)
# wordsDict={dd[i]: i for i in range(0, len(dd))}
# plot_churn_data['TechSupport']=plot_churn_data['TechSupport'].map(wordsDict)


plot_churn_data.to_csv('plot_churn_data.csv', index=False)

# plot_churn_data_dummies = pd.get_dummies(plot_churn_data, dtype=float)
#plot_churn_data_dummies = pd.get_dummies(plot_churn_data)
# print(plot_churn_data_dummies.info())
# print(plot_churn_data_dummies.dtypes)
# print(plot_churn_data_dummies.info)
# print(plot_churn_data_dummies.shape)
# print(plot_churn_data_dummies.dtypes)

# plot_churn_data_dummies.to_csv('plot_churn_data_dummies.csv', index=False)

# plot_churn_data.drop(columns=['Gender', 'Tenure', 'PhoneService', 'MultipleLines', 'StreamingTV', 'StreamingMovies', 'MonthlyCharges', 'TotalCharges'])


# Data Mapping for Machine Leaning
for i, map_column in enumerate(plot_churn_data.drop(columns=['Gender', 'Tenure', 'PhoneService', 'MultipleLines', 'StreamingTV',
                                                  'StreamingMovies', 'MonthlyCharges', 'TotalCharges', 'ChurnRate', 'Churn'])):
    dist = (plot_churn_data[map_column])
    distset= set(dist)
    dd = list(distset)
    wordsDict={dd[x]: x for x in range(0, len(dd))}
    plot_churn_data[map_column]=plot_churn_data[map_column].map(wordsDict)


# Plotting the correlation of variables
corr_plot_churn_data = plot_churn_data.drop(columns=['Gender', 'Tenure', 'PhoneService', 'MultipleLines', 'StreamingTV', 'StreamingMovies', 'MonthlyCharges', 'TotalCharges', 'Churn'])
plt.figure(figsize=(20, 8))
corr_plot_churn_data.corr()['ChurnRate'].sort_values(ascending = False).plot(kind='bar')
#plt.show()



# Feature selection and preparation for data modeling
feature_selection = plot_churn_data.drop(columns=['Gender', 'Tenure', 'PhoneService', 'MultipleLines', 'StreamingTV',
                                                  'StreamingMovies', 'MonthlyCharges', 'TotalCharges', 'ChurnRate', 'Churn'])
y = plot_churn_data['ChurnRate']
X_train, X_test, y_train, y_test = train_test_split(feature_selection, y, test_size=0.2)


# Analysis of data using model: Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
score = accuracy_score(y_test, predictions)

print("Accuracy score using the Decision Tree is: ", score)
print(classification_report(y_test, predictions, labels=[0,1]))
print(confusion_matrix(y_test, predictions))

# Analysis of data using model: Logistic Regression

lr = LogisticRegression()
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
score = accuracy_score(y_test, predictions)

print("Accuracy score using the Logistic Regression is: ", score)
print(classification_report(y_test, predictions, labels=[0,1]))

# Analysis of data using model: Random Forest Classifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
score = accuracy_score(y_test, predictions)

print("Accuracy score using the Random Forest Classifier is: ", score)
print(classification_report(y_test, predictions, labels=[0,1]))


