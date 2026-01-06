#dataframe analysis code (nov 11 2025)

#last updated: nov 15 2025

import pandas as pd
import matplotlib.pyplot as plt

# load dataset
dataset = pd.read_csv("/Users/joshua.stanley/Desktop/TSA/DataScience/dataset.csv")

#print(dataset.head())


# ^ i removed all of the verbose, it was cluttering the terminal, can uncomment to reuse.

#dataset = dataset[dataset['CountryOfExploitation'] == 'USA']


#just some global variables
value_counts_list = []
singular_values_list = []

#counting frequency of values for each column; if there are only one value type per column (which is 1.0) it means it is a "true or false" column
#where there arent multiple data types to count. i organized this into a seperate series so i could graph later on the frequency of cases that had these data types,
#which tended to be means of exploitation or exploitation types.

for col in dataset.columns:
    vc = dataset[col].value_counts()
    value_counts_list.append((col, vc))
    #this is the sequence to add them to the seperate list as mentioned above.
    if len(vc) <= 1:
        
        vc.index = [col]    # type: ignore

        singular_values_list.append(vc)
        continue

    # plot normal value counts
    plt.figure(figsize=(10, 5))
    vc.plot(kind='bar')
    plt.title(f"Value Counts for '{col}'")
    plt.xlabel("Values")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.show()

# combining all of the singular value counts into one series, unlike the data above which is in multiple series for different graphs.
combined_singular_counts = pd.concat(singular_values_list)
#second graph for means of exploitation and all other single value columns.
plt.figure(figsize=(10, 5))
combined_singular_counts.plot(kind='bar')
plt.title("Means of exploitation")
plt.xlabel("Column Name")
plt.ylabel("Occurrences")
plt.tight_layout()
plt.show()

# end of code

