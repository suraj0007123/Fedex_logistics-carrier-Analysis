import pandas as pd

### Loading the dataset for the modelbuilding
fedexdata=pd.read_csv(r"E:\DESKTOPFILES\suraj\assigments\project fed-ex dataset\FedEx\fedex.csv")

fedexdata.info() ##### To know the information about the data

fedexdata.columns # To know the columns names of the dataset

fedexdata.head()

### Droping the columns which are unnecessary
fedexdata1=fedexdata.drop(['Carrier_Name','Carrier_Num'],axis=1)

fedexdata1 

fedexdata1.columns # To check the columns names after droping 

fedexdata1.nunique() # To check the no of unique values in the dataset 

fedexdata1.info() # To Check the datatypes of each variable

fedexdata1.isnull().sum().sort_values(ascending = False) # To check the null values of each variable  in the dataset 

fedexdata1.describe() ## describe function of pandas gives us the statistical calucations of the given fedex dataset 
####  To understand the 1st moment and 2nd moment of the business calculations such as count , mean , min , max , 25th percentile, i.e Q1 ,
## 50 the percentile which is median i.e, Q2, and 75th percentile i.e, which is standard deviation , and from this we can understand the distribution of the data seeing the mean, median values and standard deviation value. 
## we can scale the difference of the different variable by seeing the minimum and maximum values 
     

#### Imputation to be done to fill the missing value. we do the median imputation to fill the NAN values
fedexdata1.fillna(fedexdata1.median(), inplace = True)
fedexdata1.isnull().sum()

##### lets check the duplicate values present in the dataset
fedexdata1[fedexdata1.duplicated()]
 
######## EXPLORATORY DATA ANALYSIS ######## 
##### Importing Libraries 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

fedexdata.columns

plt.figure(1,figsize=(24,14))
sns.countplot(fedexdata["Carrier_Name"])
### From the above countplot we can understand that WN carrier has the Highest number deliveries across the different countries 

plt.figure(figsize=(70,20))
sns.countplot(fedexdata["Source"])
plt.xticks(rotation=90, fontsize=15)
### From the above countplot we can understanad that The Highest Number of Shipment of carriers are done from ATL, ORD, DEN airports 

plt.figure(figsize=(70,20))
sns.countplot(fedexdata["Destination"])
plt.xticks(rotation=90, fontsize=15)
### From the above countplot we can understanad that The places at which the  Highest no. of deliveries  are done are AIL, ORD, DEN.

plt.figure(figsize=(10,5))
sns.countplot(fedexdata["Delivery_Status"])
plt.xticks(rotation=90, fontsize=15)
### From the above countplot we can understand that The Deliveries gets late very less no. of times and most of the time the deliveries are done on time or before time 

####### Outliers Treatment ########

fedexdata.columns

sns.boxplot(fedexdata1.Year) 

sns.boxplot(fedexdata1.Month) 

sns.boxplot(fedexdata1.DayofMonth) 

sns.boxplot(fedexdata1.DayOfWeek) 

sns.boxplot(fedexdata1.Actual_Shipment_Time) 

sns.boxplot(fedexdata1.Planned_Shipment_Time) 

sns.boxplot(fedexdata1.Planned_Delivery_Time) 

sns.boxplot(fedexdata1.Planned_TimeofTravel) ### The Outliers is present in the right side of the boxplot. lets treat the outliers by using the Winsorization method 

from feature_engine.outliers import Winsorizer

winsor=Winsorizer(capping_method='iqr',
                      tail='both',
                      fold=1.5,
                      variables=['Planned_TimeofTravel'])

fedexdata1['Planned_TimeofTravel']=winsor.fit_transform(fedexdata1[['Planned_TimeofTravel']])

from scipy import stats
import pylab

stats.probplot(fedexdata1.Planned_TimeofTravel, dist='norm', plot=pylab)
plt.xlabel('Planned_TimeofTravel')
plt.show()

##### As we can see that the data of planned time of travel variable is not normally distributed but it slightly become normal after winsorization.
#### The larger values are larger as expected from the normal distribution and smaller values ane not so small as expected from normal distribution and also the case of right skewed distribution.

fedexdata1.Distance.kurt()

sns.boxplot(fedexdata1.Distance)

stats.probplot(fedexdata1.Distance, dist='norm', plot=pylab)
plt.xlabel('Distance')
plt.show()
##### The distance data is not normally distributed. The larger values are larger as expected from the normal diatribution but smaller values are so small as expected from the normal distribution.

winsor1 = Winsorizer( capping_method = "iqr",
                    tail = "both",
                    fold = 1.5, 
                    variables = ["Distance"])
fedexdata1["Distance"] = winsor1.fit_transform(fedexdata1[["Distance"]])

sns.boxplot(fedexdata1.Distance)

stats.probplot(fedexdata1.Distance, dist='norm', plot=pylab)
plt.xlabel('Distance')
plt.show()
#### The data is not normally distributed but slightly come towards linear line after winsorization.The larger values are larger as expected from the normal diatribution but smaller values are so small as expected from the normal distribution.

plt.hist(fedexdata1.Shipment_Delay)

fedexdata1.Shipment_Delay.kurt()

sns.boxplot(fedexdata1.Shipment_Delay) ### The Outliers is present in the right side of the boxplot. lets treat the outliers by using the Winsorization method 

stats.probplot(fedexdata1.Shipment_Delay,  dist = "norm", plot = pylab)
plt.xlabel('shipment_delay')
plt.show()
##### from the Q-Q plot we can see that the data on the right side has the extreame values.The larger values are larger as expected from the normal diatribution but smaller values are so small as expected from the normal distribution.

winsor2=Winsorizer(capping_method='iqr',
                      tail='both',
                      fold=1.5,
                      variables=['Shipment_Delay'])

fedexdata1['Shipment_Delay']=winsor2.fit_transform(fedexdata1[['Shipment_Delay']])

sns.boxplot(fedexdata1.Shipment_Delay)  # To Check the Outliers after the Winsorization 

stats.probplot(fedexdata1.Shipment_Delay,  dist = "norm", plot = pylab)
plt.xlabel('shipment_delay')
plt.show()

fedexdata1.Actual_Shipment_Time.kurt()

fedexdata1.Planned_Shipment_Time.kurt()

fedexdata1.Planned_Delivery_Time.kurt()

plt.figure(figsize = (15,7), dpi = 100)
sns.boxplot(data = fedexdata1[[ 'Actual_Shipment_Time', 'Planned_Shipment_Time', 'Planned_Delivery_Time']])
## As we can see there is no outliers are available in the above three variables.

#### Bivariate Analysis ####### 

plt.figure(figsize=(16,8))
plt.scatter(x=fedexdata['Planned_TimeofTravel'], y=fedexdata['Distance'], color='pink')
plt.show()

plt.figure(figsize=(18,8))
plt.scatter(x=fedexdata['Planned_TimeofTravel'], y=fedexdata['Shipment_Delay'], color='Orange')
plt.show()
#### We can see that Shipment delay is not that much affecting by the planned time of travel and the shipment delay is more for less planned time travel 

plt.figure(figsize=(16,8))
plt.scatter(x=fedexdata['Shipment_Delay'], y=fedexdata['Distance'], color='SkyBlue')
plt.show()
### From the above graph we can see that most of the time the shipment of the carrier got delayed for the lesser distance for example :- (0-300Km) . As the Distance of the place where carrier is to delivered 
## Increases  the shipment is done fast and the product delivered fast. 

fedexdata.head()

fedexdata.shape

#### Lets apply some Unsupervised learning technique on the given dataset so, i am choosing the K-Means Clustering 
## Because the reason behind it is we have big data in size it have 15=columns or variables and 3604175=rows which is very big in size 
##### and so on that why we can't do hierarhical clustering for this size of dataset of fedex logistics 


#######  Lets apply some unsupervised learning Technique on the given dataset
####### K-Means Clustering ##### 
### import libraries ### 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

### Creating the Random Uniform Numbers 
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)

df_xy=pd.DataFrame(columns=["X","Y"]) ## Creating the Dataframe 

df_xy.X = X
df_xy.Y = Y

df_xy.plot(x="X",y="Y",kind="scatter")

model1=KMeans(n_clusters=3).fit(df_xy)
df_xy.plot(x="X", y="Y", c=model1.labels_, kind="scatter", s=10, cmap=plt.cm.rainbow)

### Kmeans on fedex dataset 
fedex=fedexdata1
fedex.head()

fedex.dtypes

fedex.shape

fedex.info()

### Taking the selected variables for the Clustering 
fedex1=fedex.iloc[:,[12,4,5,6,7,8,11]]

fedex1.head()

### Normalization Function #### 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

##### The Normalizad dataframe (considering the numerical part of the data) 
fedex1_norm = norm_func(fedex1.iloc[:,1:])
fedex1_norm.head()

##### Scree Plot or Elbow Curve ######
TWSS=[]
k=list(range(1,5))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(fedex1_norm)
    TWSS.append(kmeans.inertia_)
TWSS

### Now let's perform the Scree plot #####
plt.plot(k,TWSS,'cp-')
plt.xlabel('No_of_clusters')
plt.ylabel('total_within_SS')

#### Selecting the 2 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters = 2)
model.fit(fedex1_norm)

model.labels_ ### getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_) ## Converting numpy array into pandas series object 
fedex1['clust'] = mb ## Creating a new column and assigning it to new column
fedex1.head()


ship_delay = fedex1.iloc[:, 1 :].groupby(fedex1.clust).mean()
ship_delay.transpose()

#### From the Above Table we can see that as mean Shipment Delay time is 5.00 min and if the planned / actual shipment will do at 02:13 or 02:22 pm and the planned Delivery 
#### time will be 03:38 pm then and the average distance is high then the delivery will be on time 


### Now we will find out the most Busiest place and its Networks to the other places for deliveries by using the Network Analysis 

import networkx as nx

g = nx.Graph() #### Creating a empty graph with no. of nodes and no. of edges 

g = nx.from_pandas_edgelist(fedexdata1, source="Source", target="Destination") ## Takes Source and destination nodes from the dataframe
print(nx.info(g)) ## it gives the information about no. of edges, nodes, and average degree  

### Degree Centrality 
d = nx.degree_centrality(g)
pd.Series(d).sort_values(ascending=False).head(10)

#### We can see ATL, ORD , DFW, DEN, MSP, DTW, IAH, SLC, CVG, EWR, are the Top-10 places (Airports) which has the highest degree of centrality 

plt.figure(figsize=(30,15))
pos = nx.spring_layout(g, k=0.2)
nx.draw_networkx(g, pos, node_size=10, node_color='blue')

### Closeness Centrality 
c = nx.closeness_centrality(g)
pd.Series(c).sort_values(ascending = False).head(10)

#### Again ATL, ORD, DFW, DEN,MSP,DTW , IAH, SLC,CVG, LAS has the highest closeness centrality.

### Betweenness Centrality 
b = nx.betweenness_centrality(g)
pd.Series(b).sort_values(ascending = False).head(10)
#### ATL, DFW, SLC, MSP, ORD, DEN, ANC, IAH, DTW, LAX, are the  Top-10 airports which has the Highest betweeness centrality. Out of this ATL, ORD, DFW, DEN, MSP, DTW , IAH, SLC this airports has the highest degree of centrality and closeness centrality also.

###### Eigen vector Centrality 
e = nx.eigenvector_centrality(g)
pd.Series(e).sort_values(ascending = False).head(10)

####Again ATL, ORD, DFW, DEN, MSP, IAH, CVG, DTW are the top 10 airports which has highest eigen vector centrality.means this are the airports which having the conectivity with the other more important airports.

# Cluster cofficient
clcoeff = nx.clustering(g)
pd.Series(clcoeff).sort_values(ascending = False).head(10)

### CMI, AVL,OTZ, DAB, BRW, BMI, GPI, CLD, FSM, CHO, etc are the airports has the highest clustering coefficient.

print(clcoeff['ATL'],clcoeff['ORD'], clcoeff['DFW'], clcoeff['DEN'], clcoeff['MSP'], clcoeff['DTW'], clcoeff['IAH'], clcoeff['SLC'])
###### we can see the cluster coefficients for the airports ATL, ORD, DFW, DEN, MSP, DTW , IAH, SLC are not much bad.


# Average clustering
ac = nx.average_clustering(g) 
print(ac)

#### so from the network analysis we can see that ATL, ORD, DFW, DEN, MSP, DTW , IAH, SLC place are the top 10 most important airport from which the business happens. This airports has the more connectivity to the other airports.

####### WordCloud ##### 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

### WordCloud for carrier_Name Variable 
dict=fedexdata.Carrier_Name.value_counts()
dict


wordcloud=WordCloud(width=2000,
                    height=1000,
                    repeat=False,
                    background_color='WHITE'
                    ).generate_from_frequencies(dict)


figure=plt.figure(
    figsize=(14,6),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
#### from the wordcloud we can see that carrier WN, AA, US, UA, MQ, DL, has the highest Frequencies67

#### WordCloud for Source variable 
source=fedexdata.Source

source.count()

source.nunique()

dict1=source.value_counts()
dict1

wordcloud.generate_from_frequencies(dict1)

figure=plt.figure(
    figsize=(14,6),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

### From the wordcloud we can see that ATL, ORD, DFW, DEN, LAX, places has the highest frequency as a source. Most are the dispatches are done from the airport 

#### WordCloud for Destination variable 
dict2=fedexdata.Destination.value_counts()
dict2

wordcloud.generate_from_frequencies(dict2)

figure=plt.figure(
    figsize=(14,6),
    facecolor='k',
    edgecolor='k')
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#### As we can say that ATL, ORD, DEN, DFW, EWR, PHX airports has the high frequencies. From this airports the shipments are done most of the time and this 
### are very important for the business perspectives.

###### Let us make a classification model on the given Dataset by using Supervised learning Technique.
######## In our dataset variable delivery_status is our output or dependent variable. we have to make a model which predict whether the item got delivered at right time or not by using some independent variables.
#### Lets Build the Model by Using Naive Bayes Classifier Model 

### let us drop the unwanted column from the fedexdata 

fedexdata1.columns

fedexdata1.drop(['Year','Source','Destination'], axis=1, inplace=True)

fedexdata1.head()

fedexdata1.columns

### Normalization ###

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

fedexdata_norm = norm_func(fedexdata1)

fedexdata_norm.drop(['Planned_TimeofTravel'],axis=1, inplace=True)

fedexdata_norm.head()

target = fedexdata_norm['Delivery_Status'] 

target

fedexdata_norm.drop(['Delivery_Status'], axis=1, inplace=True)

Predictors = fedexdata_norm.iloc[:,:]

Predictors

###### splitting the data into trainand test data
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(Predictors, target, test_size = 0.3, random_state = 100 ,  shuffle=True)

train_x.shape

test_x.shape

train_x.head()

test_x.head()

train_y.head()


#### To Check the balance in the data 
import numpy as np

np.sum(train_y)/len(train_y)

test_y.head()

np.sum(test_y)/len(test_y)
### So We can see in both the train and test data the count of deliveries late is 21% and rest is deliveries on time. The data Spitted is balanced data.


# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import MultinomialNB as MB
import numpy as np

classifier_mb = MB(alpha = 2)
classifier_mb.fit(train_x , train_y)

# Evaluation on test datasets.
test_pred_m = classifier_mb.predict(test_x)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(test_pred_m, test_y)

confusion_matrix(test_pred_m, test_y)

# Evaluation on train datasets.
train_pred_m = classifier_mb.predict(train_x)

accuracy_score(train_pred_m, train_y)

confusion_matrix(train_pred_m, train_y)


######## KNN Classifier ####### 
from sklearn.neighbors import KNeighborsClassifier

knn_final = KNeighborsClassifier(n_neighbors = 15)
knn_final.fit(train_x, train_y)

pred = knn_final.predict(test_x)

# Evaluate the model
from sklearn.metrics import accuracy_score

print(accuracy_score(test_y, pred))

pd.crosstab(test_y, pred, rownames = ['Actual'], colnames= ['Predictions'])

# error on train data
pred_train = knn_final.predict(train_x)

print(accuracy_score(train_y, pred_train))

pd.crosstab(train_y, pred_train, rownames=['Actual'], colnames = ['Predictions'])

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values
for i in range(3,30,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(train_x, train_y)
    train_acc = np.mean(neigh.predict(train_x) == train_y)
    test_acc = np.mean(neigh.predict(test_x) == test_y)
    acc.append([train_acc, test_acc])
    
import matplotlib.pyplot as plt # library to do visualizations

# train accuracy plot 
plt.plot(np.arange(3,30,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,30,2),[i[1] for i in acc],"bo-")
plt.xlabel('K-value')
plt.ylabel('Accuracy')
###### From the graph we can see at k = 25 to 30 we can see the closeness in the train and the test accuracy. lets take K = 30 and check the accuracy of the model which reduce the misclassification of the model.

knn = KNeighborsClassifier(n_neighbors = 30)
knn.fit(train_x, train_y)

pred = knn.predict(test_x)
### We got the accuracy fo the model upto 99.5% by using KNN classifier model. We got the 0.5% wrong predicted values

print(accuracy_score(test_y, pred))

pd.crosstab(test_y, pred, rownames = ['Actual'], colnames= ['Predictions'])

# error on train data
pred_train = knn.predict(train_x)

print(accuracy_score(train_y, pred_train))

pd.crosstab(train_y, pred_train, rownames=['Actual'], colnames = ['Predictions'])

########  if we take k = 30 then the value of False negative is incresing and False positive decreses and which is not good foe the prediction so we take final the model by considering k = 15.

#### Conclusion
##### From the above project we found that:
#####1) 1.Places like ATL, ORD, DEN, DFW, PHX, DTW are the main sources and destinations for the maximum no.of od carriers. 
####  2.Carrier 'WN' has the highest no. of shipments among all the shipments records. 
#####  3.Shipment Delay is the most correlated feature for the Delivery_Stauts feature among all the features.
#### 4. K nearest neighbors classification method has given us the best accuracy upto 99.5% by taking K = 15 for prediction of the output feature.
