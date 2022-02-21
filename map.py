# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:34:14 2020

@author: agonz
"""
import pandas as pd
import numpy as np
import os
import gmplot

predicted_csv = pd.read_csv("prediction_results_2011-2016.csv")

mod_lat = []
mod_lon = []
strong_lat = []
strong_lon = []
major_lat = []
major_lon = []

for lat, lon, mag in zip(predicted_csv['Latitude'], predicted_csv['Longitude'], predicted_csv['Magnitude']):
    if mag < 5.9:
        mod_lat.append(lat)
        mod_lon.append(lon)
    if mag > 5.9 and 6.9:
        strong_lat.append(lat)
        strong_lon.append(lon)
    if mag > 6.9:
        major_lat.append(lat)
        major_lon.append(lon)
current_dir = os.getcwd()

gmap  = gmplot.GoogleMapPlotter(0,0,0)

gmap.scatter(mod_lat, mod_lon, "#FFFF00", size=300, marker=False)
gmap.scatter(strong_lat, strong_lon, "#FFA500", size=300, marker=False)
gmap.scatter(major_lat, major_lon, "#FF69B4", size=300, marker=False)

gmap.draw(current_dir + "\earthquake_plot.html")
