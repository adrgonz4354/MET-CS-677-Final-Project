# MET-CS-677-Final-Project
I developed an Earthquake prediction model that seeks to predict magnitudes and locations from the years 2006 through 2011 (which were the testing years) for my final project in Data Science at Boston University.

Coding Instructions:
Go into Command Prompt and do:

pip install sklearn (make sure sklearn is installed)
pip install numpy
pip install pandas
pip install global-land-mask 
pip install geopy 
pip install gmplot 

Make sure the folder “data_per_year” and the file “Data_processing.py” is in the same directory. 
“Data_processing.py” will not be ran for it is a module that will be used in “Final_project.py”.
The first script to run is “Final_project.py”, this will generate a file called “prediction_results_2011-2016.csv”. This file contains a CSV of the predictions.

The second script to run is “Map.py”, this will generate a html file which contains the plots of all the earthquakes. 



