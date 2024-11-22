#Library
import pandas as pd
import numpy as np

#CSV retrival
file_path = "weather_data_clone.csv"
data_set = pd.read_csv(file_path, 
                       sep=',', 
                       low_memory=False)

#column picker
data_set_clone = data_set.copy()
data_set_clone = data_set_clone.drop(columns=[
    "Wind Dir Definition 10's deg",
    "Wind Spd Definition km/h",
    "Visibility Definition km",
    "Wind Chill Definition"
    ])

data_set_clone.to_csv("weather_data_clone.csv", index=False)