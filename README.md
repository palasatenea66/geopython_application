## geopython_application

#A GeoPython application using Machine Learning techniques to observe fire damage, deforestation and snow cover in some places in Argentina 

This script process data from raster images (from differents years and seasons for three areas in Argentina) to observe some situations like fire damage in Patagonia, deforestation in Salta and snow cover in San Juan.


We need some Python libraries instaled in the working environment: numpy, pandas, matplotlib, gdal and sklearn.


In order to understand changes in the same area for different time, I used an unsupervised learning technique known like K-Means to classify NDVI index to identify deforested areas near Salta City for a period of twenty one years.


To observe the severity of fire damage in Cholila Lake area, near Los Alerces National Park in Patagonia, I used an interval classification on NBR index to compare forest before and after fire (some days between both satelital images).


I try to identify areas with snow cover for winter and summer, both from same year, in the La Majadita Glacier, San Juan, by K-Means classification and by interval classification.


For all of these situations, some graphics are generated for a better comprehension and I attached this files to this repo.


This project was the final examn for the postgraduate course "Python Programming", completed in the 2021 2nd semester (august to november), by the University of San Martín in Argentina.
