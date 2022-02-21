
import pandas as pd
import numpy as np
import matplotlib as plt
from geopy.geocoders import Nominatim
from global_land_mask import globe

class year():
    
    def __init__(self, year, data_set):
        self.year = year
        self.data_set = data_set
        self.year_ds = self.__year_data()

    def __generate_lat_coords(self, a, b):
        latitudes = []
        if b < -179:
            for x in range(a, b, -1):
                latitudes.append(x)
        else:
            for x in range(a, -179, -1):
                latitudes.append(x)
            for y in range(179, b, -1):
                latitudes.append(y)
        return latitudes
    
    def __findOcean(self, lat, long):
        indian_ocean = self.__generate_lat_coords(-77, 138)
        pacific_ocean = self.__generate_lat_coords(138, 21)
        atlantic = self.__generate_lat_coords(21, -96)
        NS = ""
        if long > 0:
            NS = "North"
        else:
            NS = "South"
        if int(lat) > 70:
           return "Arctic Ocean"
        else:
            if int(long) in indian_ocean:
                return NS + " Indian Ocean"
            elif int(long) in pacific_ocean:
                return NS + " Pacific Ocean"
            elif int(long) in atlantic:
                return NS + " Atlantic Ocean"
            else:
                return "Ocean could not be computed"



    def __year_data(self):
      
        span = []
        x = 0 
        for data in self.data_set['Date']:
            x +=1 
            month, day, year = (int(x) for x in data.split('/'))
            if year == self.year:
                span.append(x)
                
        df = self.data_set
        year_ds = df[span[0] : span[-1]]
        
        return year_ds
    def getMagnitudes(self):
        
        Magnitude = self.year_ds['Magnitude']
        return Magnitude
    
    def getDepth(self):
        Depth = self.year_ds['Depth']
        return Depth
    def getCordinates(self):
        Latitude = list(self.year_ds['Latitude'])
        Longitude = list(self.year_ds['Longitude'])
        
        cord = {'latitude': Latitude, 'longitude':Longitude}
        cordinates = pd.DataFrame(cord)
        return cordinates
   
    def getCountries(self):
        country_names = []
        locations = self.getCordinates()

        print("Loading...")
        for lat, lon in zip(locations['latitude'], locations['longitude']):
            geolocator = Nominatim(user_agent="Final Project")
            try:
                location = geolocator.reverse(str(lat) + ", " + str(lon))
                country_names.append(location.raw['address']['country'])
            except:
                country_names.append(self.__findOcean(lat, lon))
        country_ds = {'country': country_names}
        country_df = pd.DataFrame(country_ds)

        return country_df

    def generate_CSV(self):

        countries = self.getCountries()
        coordinates = self.getCordinates()
        depth = self.getDepth()
        magnitude = self.getMagnitudes()
        csv_prep = {'Country': countries.values.tolist(),
                    'Latitude': coordinates['latitude'].values.tolist(),
                    'Longitude': coordinates['longitude'].values.tolist(),
                    'Depth': depth.values.tolist(),
                    'Magnitude': magnitude.values.tolist()}

        new_csv = pd.DataFrame(csv_prep)
        new_csv.to_csv('data_per_year/earthquake_data_'+str(self.year)+".csv")


class Data: 
    
    def __init__(self):
        
        pass
        
    def concat_data(self, csvs):
        
        data_list = []
        
        for d in csvs:
            
            data_list.append(pd.read_csv(d))
        
        conc_data = pd.concat(data_list)
        
        return conc_data
    def getLabels(self, df):

        labels = []
        df = df['Country'].values
        for k in range(len(df)):
            if not df[k] in labels:
                labels.append(df[k])

        for z in range(len(labels)):
            labels[z] = labels[z].replace("['", "")
            labels[z] = labels[z].replace("']", "")

        return labels


    
        