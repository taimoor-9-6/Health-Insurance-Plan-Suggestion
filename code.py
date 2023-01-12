import pandas as pd
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import Normalizer
from tabulate import tabulate


# In[81]:


#Importing first dataset
df=pd.read_csv('insurance.csv')
df


# ### Part 1: Data Preperation

# In[82]:


##EDA##
sns.set(rc={'axes.facecolor':'lightgray', 'figure.facecolor':'white'})
sns.histplot(data=df,x='smoker',color='Turquoise')


# I will add two new columns:
# - State: Each row will be randomly assigned a state, based on their region
# - Family: If children are present, this binary column will potray a 'yes', indicating that family needs coverage

# In[83]:


#printing unique states
df['region'].unique()


# In[84]:


#defining states per region
northwest=['OR','WA','ID','MT','WY']
northeast=['CT','ME','MA','NH','NJ','NY','PA','DC']
southeast=['AL','FL','GA','KY','MS','NC','SC','TN']
southwest=['AR','CO','LA','NM','ND','OK','SD','TX','UT']

#populating a new column based on state
regions=df['region']
state=[]
for i in regions:
    if i == 'southwest':
        state.append(random.choice(southwest))
    elif i == 'southeast':
        state.append(random.choice(southeast))
    elif i == 'northwest':
        state.append(random.choice(northwest))
    elif i == 'northeast':
        state.append(random.choice(northeast))
df['state']=state

#adding family column
temp=df['children']
family=[]
for i in temp:
    if i == 0:
        family.append('no')
    else:
        family.append('yes')
df['family']=family


# To meet my project goal, I do not need the 'bmi' column, as it is not a generic question

# In[85]:


#dropping bmi column
df=df.drop(columns='bmi')
df=df.drop(columns='region')
df


# In orde to predict premium plans, my second dataset uses labels categorized as:
# - brone
# - silver
# - gold
# 
# In order to create labels, for each state, I will extract all the rows and develop ranges for each label. These ranges will be defined as:
# - brone: 0-35%
# - silver: 35-70%
# - gold: 70-100%
# 
# The results, along with state and charges will be stored in another dataframe named: df_classifcation

# In[86]:


######### CALCULATING LABELS ############


# In[87]:


#creating a empty array of size -> df
labels=[None] * df.shape[0]
states=df['state'].unique()

#for each unique state the loop will run
for state in states:
    #creating a temporary list that includes all the charges present in each state
    value=df[df['state']==state]['charges'].values
    #sort that list in ascending order
    value.sort()
    length=len(value)
    #generating boundaries for each label
    lowest=round(length*.35)
    bronze=value[0:lowest]
    middle=round(length*.70)
    silver=value[lowest:middle]
    gold=value[middle:]
    #getting the id of each row 
    idx=df.index[df['state']==state]
    #the loop will run for each id present for a specific state
    for i in idx:
        #using the id, i am extracting the charges
        charge=df['charges'].values[i]
        #comparing charges to the label ranges to see which range it falls to
        #storing that label in the same index as 'df' in our pre-defined list-> labels
        if charge<=bronze[-1]:
            labels[i]='Bronze'
        elif (charge>bronze[-1]) and (charge<=silver[-1]):
            labels[i]='Silver'
        elif charge>silver[-1]:
            labels[i]='Gold'
        else:
            print('---')
            
#generating a new dataframe
df_classification=pd.DataFrame(labels,columns=['label'])
df_classification['state']=df['state']
df_classification['charges']=df['charges']
df_classification


# ### Part 2: Regression

# In[88]:


########### DATA PREPERATION ##############


# In[89]:


#splitting into train and test
x=df.drop(columns=['charges'])
y=df['charges']
y=np.array(y).reshape(-1,1)
#scaling the y column -> target variable 'charges' to get results in the range 0-1
scaler = StandardScaler()
scaler.fit(y)
y=scaler.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=30)


#label encoding the trainin variables
labelenc = preprocessing.LabelEncoder()
x['sex']= labelenc.fit_transform(x['sex'])
x['smoker']= labelenc.fit_transform(x['smoker'])
x['state']= labelenc.fit_transform(x['state'])
x['family']= labelenc.fit_transform(x['family'])

#splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=30)


# In[90]:


####### REGRESSION MODELS #######
results=[]


# Regression Models:
# - Linear Regression
# - 2nd Degree Polynomial Regression
# - Decion Tree Regression
# - Random Foresr Regression
# - SVM Regression

# In[91]:


#linear regression
reg = LinearRegression().fit(x_train, y_train)
reg.score(x_train, y_train)
preds=reg.predict(x_test)
# print(f'R2: {r2_score(y_test, preds)}')
# print("Mean Squared Error:", mean_squared_error(y_test,preds))
# print("Mean Absolute Error:", median_absolute_error(y_test,preds))
# print("RMSE:" ,np.sqrt(mean_squared_error(y_test,preds)))
results.append([mean_squared_error(y_test,preds),median_absolute_error(y_test,preds),
               np.sqrt(mean_squared_error(y_test,preds)),r2_score(y_test, preds)])


# In[92]:


#polynomial
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x_train)
x_poly_test=polynomial_features.fit_transform(x_test)
model = LinearRegression()
model.fit(x_poly, y_train)
preds=model.predict(x_poly_test)
results.append([mean_squared_error(y_test,preds),median_absolute_error(y_test,preds),
               np.sqrt(mean_squared_error(y_test,preds)),r2_score(y_test, preds)])


# In[93]:


#decision tree regressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train, y_train)
preds=regressor.predict(x_test)
results.append([mean_squared_error(y_test,preds),median_absolute_error(y_test,preds),
               np.sqrt(mean_squared_error(y_test,preds)),r2_score(y_test, preds)])


# In[94]:


#random forest regressor
regressor = RandomForestRegressor(max_depth=4, random_state=1)
regressor.fit(x_train, y_train)
preds=regressor.predict(x_test)
results.append([mean_squared_error(y_test,preds),median_absolute_error(y_test,preds),
               np.sqrt(mean_squared_error(y_test,preds)),r2_score(y_test, preds)])


# In[95]:


#svm
regressor = SVR()
regressor.fit(x_train,y_train)
preds=regressor.predict(x_test)
results.append([mean_squared_error(y_test,preds),median_absolute_error(y_test,preds),
               np.sqrt(mean_squared_error(y_test,preds)),r2_score(y_test, preds)])


# In[96]:


#Results
#create data
data = [["Linear",results[0][0],results[0][1],results[0][2],results[0][3]],
       ["2nd Degree Polunomial",results[1][0],results[1][1],results[1][2],results[1][3]],
       ["Decion Tree",results[2][0],results[2][1],results[2][2],results[2][3]],
       ["Random Forest",results[3][0],results[3][1],results[3][2],results[3][3]]]

  
#define header names
col_names = ["Model","MSE","MAE","RMSE","R2"]
  
#display table
print(tabulate(data, headers=col_names))


# ### Part 3: Classification

# After choosing the ideal model to be random forest, I ran the regression again without scaling the target variable to get predicted charges

# In[97]:


###### PREDICTING LABELS ###########

#splitting into train and test
x=df.drop(columns=['charges'])
y=df['charges']

#label encoding
labelenc = preprocessing.LabelEncoder()
x['sex']= labelenc.fit_transform(x['sex'])
x['smoker']= labelenc.fit_transform(x['smoker'])
x['state']= labelenc.fit_transform(x['state'])
x['family']= labelenc.fit_transform(x['family'])

#splitting into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.34, random_state=30)


#random forest regressor
regressor = RandomForestRegressor(max_depth=3, random_state=0)
regressor.fit(x_train, y_train)
preds=regressor.predict(x_test)


# In[98]:


##### predicting new labels
#adding the predicted charged to my result dataframe -> df_classification
idx=x_test.index
#keeping only the rows present in the testdataset
df_classification=df_classification.loc[idx]
df_classification['predicted charges']=preds


# Based on the label ranges we defined earlier, I use the predicted charges to calculate new labels

# In[100]:


#reseting index 
df_classification=df_classification.reset_index(inplace=False,drop=True)
#making an empty list of size df_classification
preds_labels=[None] * df_classification.shape[0]
#getting unique states
states=df['state'].unique()
#repeat the same process that was done earlier, this time on the predicted charges
#run for loop for each state
for state in states:
    #getting all the charges present in the complete dataframe
    value=df[df['state']==state]['charges'].values
    #sorting values in ascending order
    value.sort()
    length=len(value)
    #genrating the same ranges that were generated before
    lowest=round(length*.35)
    bronze=value[0:lowest]
    middle=round(length*.70)
    silver=value[lowest:middle]
    gold=value[middle:]
    #extracting the columns of the state that is being processed from my test dataframe
    idx=df_classification.index[df_classification['state']==state]
    #loop will run for each column
    for i in idx:
        #identify the predicted charge and assign a label
        charge=df_classification['predicted charges'].values[i]
        if charge<=bronze[-1]:
            preds_labels[i]='Bronze'
        elif (charge>bronze[-1]) and (charge<=silver[-1]):
            preds_labels[i]='Silver'
        elif charge>silver[-1]:
            preds_labels[i]='Gold'
        else:
            print(state)
            print(charge)
            print(bronze[-1])
            print(silver[0],silver[-1])
            print(gold[0])
            print('---')
df_classification['predicted labels']=preds_labels


# In[102]:


#results for estimating the classification part
#confusion matrix
y_test=df_classification['label'].values
preds=df_classification['predicted labels'].values
cm=confusion_matrix(y_test, preds,labels=["Bronze", "Silver", "Gold"])
cm_df = pd.DataFrame(cm,
                     index = ['Bronze','Silver','Gold'], 
                     columns = ['Bronze','Silver','Gold'])
#Plotting the confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm_df, annot=True,cmap="crest",fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()


# In[103]:


#further categorizing results to see the performance of each label
#for bronze
tp_bronze=cm[0][0]
fn_bronze=cm[0][1]+cm[0][2]
fp_bronze=cm[1][0]+cm[2][0]
tn_bronze=cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]
print("Bronze:", '\n TP:',tp_bronze,'|','FN:',fn_bronze,'|','FP:',fp_bronze,'|','TN:',tn_bronze)
print("Accuracy:",(tp_bronze+tn_bronze)/(tp_bronze+tn_bronze+fn_bronze+fp_bronze))
print("TPR:",tp_bronze/(tp_bronze+fn_bronze),"|" ,"TNR:",tn_bronze/(fp_bronze+tn_bronze))

#for silver
tp_silver=cm[1][1]
fn_silver=cm[1][0]+cm[1][2]
fp_silver=cm[0][1]+cm[2][1]
tn_silver=cm[0][0]+cm[0][2]+cm[2][0]+cm[2][2]
print("Silver:", '\n TP:',tp_silver,'|','FN:',fn_silver,'|','FP:',fp_silver,'|','TN:',tn_silver)
print("Accuracy:",(tp_silver+tn_silver)/(tp_silver+tn_silver+fn_silver+fp_silver))
print("TPR:",tp_silver/(tp_silver+fn_silver),"|" ,"TNR:",tn_silver/(fp_silver+tn_silver))


#for gold
tp_gold=cm[2][2]
fn_gold=cm[2][0]+cm[2][1]
fp_gold=cm[0][2]+cm[1][2]
tn_gold=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
print("Gold:", '\n TP:',tp_gold,'|','FN:',fn_gold,'|','FP:',fp_gold,'|','TN:',tn_gold)
print("Accuracy:",(tp_gold+tn_gold)/(tp_gold+tn_gold+fn_gold+fp_gold))
print("TPR:",tp_gold/(tp_gold+fn_gold),"|" ,"TNR:",tn_gold/(fp_gold+tn_gold))


# After running multple iterations, I conluded that:
# - Silver has the lowest accuracy each time
# - TPR is weaker than TNR

# ### Part 4: Suggestion
# 

# Here I will import a dataset that has the required health insurance providers and their plans and filter out the relevant columns

# In[79]:


#importing second dataset
df1=pd.read_excel('Individual_Market_Medical.xlsx',header=1)
df1.head()


# For data preperation:
# - I will filter out and keep only relevant columns:
#     - State Code
#     - FIPS County Code
#     - Metal Level (Label)
#     - Issue Name
#     - Plan Marketing Name
#  
# - Change the plans named 'Expanded Bronze' to 'bronze to match the results of my dataframe

# In[25]:


#filtering out relevant columns
df1=df1[['State Code','FIPS County Code','Metal Level','Issuer Name','Plan Marketing Name']]
df1['Metal Level'] = df1['Metal Level'].replace(['Expanded Bronze'], 'Bronze')


# In[26]:


df1


# In[104]:


#Adding the feature column to my final dataframe -> df_classification
#This acts as a database, or final testing feature of my project
x_test=x_test.reset_index(inplace=False,drop=True)
df_classification['age']=x_test['age']
df_classification['sex']=x_test['sex']
df_classification['children']=x_test['children']
df_classification['smoker']=x_test['smoker']
#removing the original charges and labels column because we only need the predicted ones
df_classification=df_classification.drop(columns=['label','charges'])
df_classification


# In[28]:


#### TESTING ####


# In[37]:


#Getting a test case
test_case=df_classification.iloc[404:405,:]
print(test_case)

if test_case.values[0][0] in df1['State Code'].unique():
    #Getting state and class label of that test case
    test_state=test_case['state'].values[0]
    test_label=test_case['labels preds'].values[0]
    #making dataframe to filter out that state and label from the plan dataframe
    result=df1[(df1['State Code']==test_state) & (df1['Metal Level']==test_label)]
    #in order to remove computation time, I will randomly allocate the testcase a random
    # county code from that state
    temp=df1[df1['State Code']==test_state]['FIPS County Code'].unique()
    test_code=random.choice(temp)
    print("County Code allocated:", test_code)
    print('--------')
    result=result[result['FIPS County Code']==test_code]
    #getting unique issuers from that dataframe
    issuer=result['Issuer Name'].unique()
    #printing out all the issuers in that area and the plans
    for i in issuer:
        print("Plans by issuer:", i)
        print(result[result['Issuer Name']==i]['Plan Marketing Name'])
        print("")
else:
    print("This state is not present in the database")


# In[1094]:


###########################################

