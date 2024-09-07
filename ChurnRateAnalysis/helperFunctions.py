import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def find_replace_invalid_values(data):
    clean_data = data
    list_of_columns = list(clean_data.columns.values)
    for column_name in list_of_columns:
        if clean_data[column_name].eq(' ').any():
            # Replacing any ' ' values with NaN in different columns
            clean_data.replace({column_name: {' ': np.nan}}, inplace=True)
            print("Values with ' ' found in column: " + column_name + " and replaced with NaN")
    return clean_data

def find_drop_nan_values(data):
    clean_data = data
    list_of_columns = list(clean_data.columns.values)
    for column_name in list_of_columns:
        if clean_data[column_name].isna().any():
            print("NaN found in column: " + column_name + " on %i places" % clean_data[column_name].isna().sum())
            clean_data.dropna(inplace=True)
            print("Rows with NaN values dropped for column: " + column_name)
    return clean_data


def find_drop_duplicate_rows(data):
    clean_data = data
    number_of_duplicates = clean_data.duplicated().sum()
    print()
    print("Checking for any duplicate rows")
    print("%i number of duplicate rows found" % number_of_duplicates)
    if number_of_duplicates > 0:
        clean_data.drop_duplicates(inplace=True)
    return clean_data


def calc_churn_rate_per_category(data):
    plot_churn_data = data
    for i, churn_variable in enumerate(
            plot_churn_data.drop(columns=['Tenure', 'MonthlyCharges', 'TotalCharges', 'Churn', 'ChurnRate'])):
        unique_value_list = np.unique(plot_churn_data["%s" % churn_variable])
        for u in unique_value_list:
            total_of_category = len(plot_churn_data.query("%s == '%s'" % (churn_variable, u)))
            print("Total number of %s users for category %s: %i" % (u, churn_variable, total_of_category))
            total_of_category_churn = len(plot_churn_data.query("%s == '%s' and Churn == 'Yes'" % (churn_variable, u)))
            calculate_churn_rate = (total_of_category_churn * 100) / total_of_category
            print("Category: %s, with value %s as Churn. The Churn Rate: %.0f%%" %
                  (churn_variable, u, calculate_churn_rate))
            print()


def variables_distribution_with_churn(data):
    plot_churn_data = data
    plt.figure(figsize=(14, 22))
    for i, univariate in enumerate(
            plot_churn_data.drop(columns=['MonthlyCharges', 'TotalCharges', 'ChurnRate', 'Tenure'])):
        ax = plt.subplot(7, 3, i + 1)
        sb.countplot(data=plot_churn_data, x=univariate, palette="Set2", hue='Churn', legend=True)
        ax.set_title(f'Distribution of {univariate}')
        ax.set_xlabel(None)
        ax.set_ylabel('Churn Count')
        plt.tight_layout(h_pad=3, w_pad=3)
    plt.show()

def bins_distribution_with_churn(data, list):
    plot_churn_data = data
    bins_column_list = list
    plt.figure(figsize=(14, 22))
    for i, column_names in enumerate(bins_column_list):
        ax = plt.subplot(2, 2, i + 1)
        sb.countplot(data=plot_churn_data, x=column_names, palette="Set1", hue='Churn', legend=True)
        ax.set_title('Distribution of %s with Churn Rate' % column_names)
        ax.set_xlabel(column_names)
        ax.set_ylabel('Churn Count')
    plt.show()

