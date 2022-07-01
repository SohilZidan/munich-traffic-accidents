
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""## The Monatszahlen Verkehrsunfälle dataset
https://opendata.muenchen.de/

### Get the data
First download the dataset.
"""

"""Import it using pandas"""

dataset_path = "https://opendata.muenchen.de/dataset/5e73a82b-7cfb-40cc-9b30-45fe5a3fa24e/resource/40094bd6-f82d-4979-949b-26c8dc00b9a7/download/220511_monatszahlenmonatszahlen2204_verkehrsunfaelle.csv"
dataset = pd.read_csv(dataset_path, na_values = "?")

dataset.tail()

"""### Clean the data

The dataset contains a few unknown values.
"""
# print(f"number of nan: {dataset.isna().sum()}")

"""To keep this initial tutorial simple drop those rows."""

dataset = dataset.dropna()

"""Drop non important columns"""
dataset = dataset[["MONATSZAHL", "AUSPRAEGUNG", "JAHR", "MONAT", "WERT"]]

print("AUSPRAEGUNG values:", dataset["AUSPRAEGUNG"].unique())
print("MONATSZAHL:", dataset["MONATSZAHL"].unique())

"""convert MONATSZAHL and AUSPRAEGUNG into numeric """

# dataset['MONATSZAHL'] = dataset['MONATSZAHL'].map({'Alkoholunfälle': 1, 'Fluchtunfälle': 2,'Verkehrsunfälle': 3})
# dataset['AUSPRAEGUNG'] = dataset['AUSPRAEGUNG'].map({'insgesamt': 1, 'Verletzte und Getötete': 2, 'mit Personenschäden': 3})


# dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
# month_names = {
#     1: "Jan",
#     2: "Feb",
#     3: "Mar",
#     4: "Apr",
#     5: "May",
#     6: "Jun",
#     7: "Jul",
#     8: "Aug",
#     9: "Sep",
#     10: "Oct",
#     11: "Nov",
#     12: "Dec"}

dataset["MONAT"] = dataset["MONAT"].astype(np.int64)
# dataset["MONAT"] = dataset["MONAT"].map(lambda x: month_names.get(x%100, "unknown"))
dataset["MONAT"] = pd.to_datetime(dataset["MONAT"], format="%Y%m")
print(dataset.tail())



"""
visualise
"""
dataset_total = dataset[dataset.AUSPRAEGUNG == "insgesamt"]

# Plot the historical data
sns.set_theme(style="darkgrid")

""" yearly """
sns.lineplot(
    data=dataset_total,
    x="JAHR", y="WERT", hue="MONATSZAHL",
    ci=False
)
plt.savefig('imgs/yearly.png')

# 
sns.lineplot(
    data=dataset_total,
    x="MONAT", y="WERT", hue="MONATSZAHL", ci=False
)
plt.savefig('yearly.png')

# 
plt.clf()
plt.close()
sns.histplot(data=dataset, x="MONAT", y="WERT", hue=["MONATZAHL", "AUSPRAEGUNG"])
plt.savefig('hist.png')

""" monthly """
plt.clf()
plt.close()
fg = sns.relplot(
    data=dataset, x="MONAT", y="WERT", col="AUSPRAEGUNG",
    hue="MONATSZAHL",
    kind="line",
)

fg.set_xticklabels(rotation=90)
fg.tight_layout()
fg.savefig('monthly.png')
