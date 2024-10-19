#Library
import pandas as pd;
import numpy as np;

#CSV retrival
file_path = "weather_data_clone.csv";
data_set = pd.read_csv(file_path, 
                       sep=',', 
                       low_memory=False);

#column picker
data_set_clone = data_set.copy();
label = "Stn Press Definition kPa";
column = data_set_clone[label];
column_length = len(column);
#cell checker
for i in range(column_length):
    cell = column[i];
    if cell is np.nan or \
        cell == "LegendMM":
        counter = 0;
        accumulate = 0;
        #15 days before and after a timestamp
        #accumulate the sum to average later
        for j in range (15):
            before_index = i - (j + 1) * 24;
            after_index = i + (j + 1) * 24;
            if before_index >= 0 and \
                column[before_index] is not np.nan and \
                column[before_index] != "LegendMM":
                counter += 1;
                accumulate += float(column[before_index]);
            if after_index < column_length and \
                column[after_index] is not np.nan and \
                      column[after_index] != "LegendMM":
                counter += 1;
                accumulate += float(column[after_index]);
        #update cell value direct by mean
        if counter != 0:
            data_set_clone.loc[i, label] = accumulate / counter;
        else:
            data_set_clone.loc[i, label] = 0;
        #data_set["Temp Definition °C"][i] = cell;

data_set_clone.to_csv("weather_data_clone.csv", index=False);
        
#Affected: Temp Definition °C, Dew Point Definition °C, Rel Hum Definition %, Precip. Amount Definition mm