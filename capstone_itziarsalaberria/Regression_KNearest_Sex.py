import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from IPython.display import display, HTML

import timeit


### Load in the DataFrame
df = pd.read_csv("profiles.csv")

### Augment Data
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)


df["essay_word_count"] = all_essays.apply(lambda x: len(x.split()))

sex_mappings = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mappings)

status_mapping = {"single": 0, "seeing someone": 1, "available": 2, "married": 3, "unknown": 4}
df["status_code"] = df.status.map(status_mapping) 


### Normalize Data
feature_data = df[['age', 'income', 'essay_word_count', 'status_code','height', 'sex_code']]


feature_data = feature_data.dropna()
feature_data = feature_data[feature_data['income'] != -1];

#print(feature_data.head())

x = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)

data = feature_data[['age', 'income', 'essay_word_count', 'status_code', 'height']]
target = feature_data[['sex_code']]

training_data, validation_data, training_labels, validation_labels = train_test_split(data, target, test_size = 0.2, random_state = 100)

### REGRESSION

print("\n### K-Nearest Neighbors Regression")

accuracies = []

best_k = 1;
best_accuracy = 0;

start_k = timeit.default_timer()

for k in range(1, 101):
	classifier = KNeighborsClassifier(n_neighbors=k)
	classifier.fit(training_data,training_labels.values.ravel())
	accuracy = classifier.score(validation_data, validation_labels)
	accuracies.append(classifier.score(validation_data, validation_labels))

	if(best_accuracy < accuracy):
		best_accuracy = accuracy
		best_k = k
  
	#print(str(k) + " - " + str(classifier.score(validation_data, validation_labels)))
  
k_list = range(1, 101)


print("Best K: " + str(best_k))

start = timeit.default_timer()

classifier = KNeighborsClassifier(n_neighbors = best_k)
classifier.fit(training_data, training_labels.values.ravel())

print("Score: " + str(classifier.score(validation_data, validation_labels)))

prediction = classifier.predict(validation_data)

stop = timeit.default_timer()

print("Accuracy score: " + str(accuracy_score(validation_labels, prediction)))
print("Recall score: " + str(recall_score(validation_labels, prediction)))
print("Precision score: " + str(precision_score(validation_labels, prediction)))

print("\nTime to run the model: " + str(stop-start))

print("\nTime to run the model and selecting best K: " + str(stop-start_k))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("K-Nearest Neighbors Regression Classifier Accuracy")
plt.show()