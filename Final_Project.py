

import pandas as pd
import numpy as np
import matplotlib as plt
import data_processing as dp

#sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
df = pd.read_csv("earthquake_database.csv")


#generates date from 2006 to 2016

'''
for y in range(2006, 2017):
    y_ = dp.year(y, df)
    y_.generate_CSV()
    
'''

file_names = []
for x in range(2006, 2017):
    
    file_names.append("data_per_year/earthquake_data_"+str(x)+".csv")
    
processing_data = dp.Data().concat_data(file_names)




#Splitting data for training and testing
Features = ['Latitude', 'Longitude', 'Depth', 'Magnitude']
Countries = dp.Data().getLabels(processing_data) #labels
X_general = processing_data[Features].values
X_general_scaler = preprocessing.StandardScaler().fit(X_general)
X_general = X_general_scaler.fit_transform(X_general)

x_gen_label_encoder = preprocessing.LabelEncoder()
y_general = x_gen_label_encoder.fit_transform(processing_data['Country'].values)

X_training, X_testing, Y_training, Y_testing = train_test_split(X_general, y_general)

#predicting country patterns
country_classifier = tree.DecisionTreeClassifier().fit(X_training, Y_training)
label_predictions = list(country_classifier.predict(X_testing))
print("Label prediction accuracy:")
print(metrics.r2_score(Y_testing, label_predictions))

label_predictions = x_gen_label_encoder.inverse_transform(label_predictions)
#for f in range(len(label_predictions)):
# label_predictions[f] = Countries[label_predictions[f]]

#magnitude prediction
Features = ['Depth', 'Country', 'Latitude', 'Longitude']
processing_data['Country'] = y_general

X_mag_general = processing_data[Features]

Y_mag_general = processing_data['Magnitude'].values
Y_mag_general = x_gen_label_encoder.fit_transform(Y_mag_general)

X_training, X_testing, Y_training, Y_testing = train_test_split(X_mag_general, Y_mag_general)

magnitude_classifier = KMeans().fit(X_training, Y_training) #since this data is continuous
magnitude_prediction = list(magnitude_classifier.predict(X_testing))
print("Magnitude accuracy:")
print(metrics.r2_score(Y_testing, magnitude_prediction))
#LON AND LAT PREDS
Features = ['Country', 'Magnitude', 'Depth']

X_coords_general = processing_data[Features].values
X_coords_general_scaler = preprocessing.StandardScaler().fit(X_coords_general)
X_coords_general = X_coords_general_scaler.fit_transform(X_coords_general)

Y_coords_general = processing_data[['Latitude', 'Longitude']].values

X_training, X_testing, Y_training, Y_testing = train_test_split(X_coords_general, Y_coords_general)

coords_classifier = RandomForestRegressor().fit(X_training, Y_training) #since this data is continuous
coords_prediction = coords_classifier.predict(X_testing)

coordinate_df = pd.DataFrame(coords_prediction)
print("Coordinate accuracy:")
print(metrics.r2_score(coords_prediction, Y_testing))
#DEPTH PREDICTIONS
Features = ['Country', 'Longitude', 'Latitude', 'Magnitude']

X_depth_general = processing_data[Features].values
X_depth_general_scaler = preprocessing.StandardScaler().fit(X_depth_general)
X_depth_general = X_coords_general_scaler.fit_transform(X_depth_general)

Y_depth_general = processing_data['Depth'].values

X_training, X_testing, Y_training, Y_testing = train_test_split(X_depth_general, Y_depth_general)

depth_classifier = KNeighborsRegressor().fit(X_training, Y_training) #since this data is continuous
depth_prediction = list(depth_classifier.predict(X_testing))
print("Depth accuracy:")

print(metrics.r2_score(depth_prediction, Y_testing))

prediction_results_table = {'Country/Location': label_predictions,
                            'Latitude': coordinate_df[0],
                            'Longitude': coordinate_df[1],
                            'Depth': depth_prediction,
                            'Magnitude': x_gen_label_encoder.inverse_transform(magnitude_prediction)}


prediction_results_dataframe = pd.DataFrame(prediction_results_table)
prediction_results_dataframe.to_csv('prediction_results_2011-2016.csv')