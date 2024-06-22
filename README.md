# fedex-logistics-carrier-delay-analysis
FedEx Delivery Data Analysis and Predictive Modeling
Project Overview
This project involves a comprehensive analysis and predictive modeling on a FedEx delivery dataset. The dataset contains records of shipments made by various carriers, with information on shipment times, delays, and delivery statuses. The goal of the project is to explore the data, perform necessary preprocessing, conduct exploratory data analysis, and build predictive models to determine delivery statuses.

Table of Contents
Project Overview
Dataset Information
Data Preprocessing
Exploratory Data Analysis (EDA)
Outlier Treatment
Clustering Analysis
Network Analysis
Text Analysis
Predictive Modeling
Naive Bayes Classifier
K-Nearest Neighbors (KNN)
Conclusion
Dataset Information
The dataset (fedex.csv) contains the following columns:

Carrier_Name
Carrier_Num
Year
Month
DayofMonth
DayOfWeek
Actual_Shipment_Time
Planned_Shipment_Time
Planned_Delivery_Time
Planned_TimeofTravel
Distance
Shipment_Delay
Delivery_Status
Source
Destination
Data Preprocessing
Dropping Unnecessary Columns:
Removed Carrier_Name and Carrier_Num columns.
Handling Missing Values:
Filled missing values using median imputation.
Checking for Duplicates:
Identified and handled duplicate entries.
Exploratory Data Analysis (EDA)
Count Plots:
Visualized the distribution of carriers, sources, destinations, and delivery statuses using count plots.
Box Plots:
Checked for outliers in numerical columns such as Year, Month, DayofMonth, DayOfWeek, Actual_Shipment_Time, Planned_Shipment_Time, Planned_Delivery_Time, and Planned_TimeofTravel.
Outlier Treatment
Applied Winsorization method to treat outliers in Planned_TimeofTravel, Distance, and Shipment_Delay.
Clustering Analysis
K-Means Clustering:
Normalized the data.
Performed K-Means clustering to group similar data points.
Determined the optimal number of clusters using the elbow method and assigned cluster labels.
Network Analysis
Graph Analysis:
Created a graph using networkx to analyze the connections between different source and destination airports.
Calculated degree centrality, closeness centrality, betweenness centrality, and eigenvector centrality.
Identified the most important airports in the network based on these centrality measures.
Text Analysis
WordCloud:
Generated word clouds for Carrier_Name, Source, and Destination to visualize the frequency of values.
Predictive Modeling
Naive Bayes Classifier
Split the data into training and testing sets.
Built a Multinomial Naive Bayes classifier.
Evaluated the model's performance using accuracy and confusion matrix.
K-Nearest Neighbors (KNN)
Built a KNN classifier.
Determined the optimal value of K using a loop and plotted the accuracy.
Evaluated the model's performance and selected K=15 as the final model.
Conclusion
Identified key source and destination airports with the highest number of shipments.
Found that the WN carrier has the highest number of shipments.
Shipment_Delay is the most correlated feature with Delivery_Status.
Achieved 99.5% accuracy with the KNN classifier (K=15) for predicting delivery status




























