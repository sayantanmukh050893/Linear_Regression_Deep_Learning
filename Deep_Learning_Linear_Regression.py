#!/usr/bin/env python
# coding: utf-8

# ### First let us import the necessary packages that are required for our analysis

# In[72]:


import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


# ### We would use the keras.utils.get_file method to download the dataset from the below link provided

# In[73]:


dataset_path = keras.utils.get_file("auto-mpg.data","https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")


# ### SInce we have downloaded only the raw data we need to assign the column names separately

# In[74]:


columns = ["MPG","Cylinders","Displacement","Horsepower","Weight","Acceleration","Model Year","Origin"]


# ### Now we would read the data which we just downloaded into a dataframe

# In[96]:


auto_data = pd.read_csv(dataset_path,names=columns,na_values="?",comment="\t",sep=" ",skipinitialspace=True)


# ### Let us have a look at the first 5 instances of the data

# In[97]:


auto_data.head()


# ### Let us see the features in the dataset in details

# In[98]:


auto_data.info()


# ### From the data it is evident that the dataframe has 398 entries.
# ### The following are the independant variables in the dataset :-
# ### Horsepower, Cylinders, Displacement, Weight, Acceleration, Model Year, Origin
# ### MPG is the dependant variable which we would generalise by training our model with the independant features. 
# ### Horsepower has got 6 missing values. Let us see if we can impute the missing values

# In[99]:


auto_data.isna().sum()


# ### Instead of dropping the missing values we would impute those with the mean of the distribution. In that way we do not have to do away with 6 instances keeping in mind we have a pretty small dataset at our disposal

# In[100]:


auto_data.loc[auto_data["Horsepower"].isna(),"Horsepower"] = np.mean(auto_data["Horsepower"])


# ### Let us check the missing values again to check whether the null values have been imputed or not

# In[101]:


auto_data.isna().sum()


# ### The feature Origin is a categorical variable so we need to perform one hot encoding for it

# In[102]:


auto_data = pd.get_dummies(auto_data,columns=["Origin","Cylinders","Model Year"],prefix_sep="_")


# In[103]:


auto_data.head()


# ### From the metadeta we came to know that 1 signifies USA, 2 signifies Europe and 3 signifies Japan so we must rename the columns likewise

# In[104]:


#auto_data.columns = ["MPG","Displacement","Horsepower","Weight","Acceleration","Model Year","USA","Europe","Japan","Cylinders_3","Cylinders_4","Cylinders_5","Cylinders_6","Cylinders_8"]


# ### Let us have a quick visualization of the dataset using seaborn package

# In[105]:


sns.pairplot(auto_data[["MPG","Displacement","Weight","Acceleration","Horsepower"]],diag_kind="kde")


# ### Let us a look a the correlation matrix of the dataset which we would show in a heatmap

# In[106]:


corr = auto_data.corr()
sns.heatmap(corr)


# ### Now let us shuffle the data and make training and testing set from it

# In[107]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def scaling(dataset,columns):
    for column in columns:
        x = np.array(dataset[column])
        x = x.reshape(-1,1)
        dataset[column] = scaler.fit_transform(x)


# In[108]:


columns = ["Displacement","Horsepower","Weight","Acceleration"]
scaling(auto_data,columns)


# In[109]:


auto_data = auto_data.sample(frac=1)
y = auto_data["MPG"]
X = auto_data.drop(columns=["MPG"],axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=42)


# ### Since the data is very skewed and is spread accross a wide range for all the features we nned to normalize the data by replacing the data with the z-score of the data

# In[ ]:


#def normalize(x):
 #   return (x-np.mean(x)/(np.std(x)))


# In[ ]:


#X_train = normalize(X_train)
#X_test = normalize(X_test)


# In[110]:


len(X_train.keys())


# ### Let us now build a Sequential model using Keras

# In[111]:


def build_model():
    model = keras.Sequential([
        layers.Dense(64,activation=tf.nn.relu,input_shape=[len(X_train.keys())]),
        layers.Dense(64,activation=tf.nn.relu),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',optimizer=optimizer,metrics=["mae","mse"])
    
    return model


# In[112]:


model = build_model()


# In[113]:


model.summary()


# In[114]:


class Printdot(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs):
        if epoch % 100 == 0:
            print('')
        else:
            print(".",end='')


# In[115]:


history = model.fit(X_train,y_train,epochs=100,validation_split=0.2,verbose=0,callbacks=[Printdot()])


# In[116]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[117]:


def plot_history(history):
    
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.plot(hist['epoch'],hist['mae'],label="Train Error")
    plt.plot(hist['epoch'],hist['val_mae'],label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.legend()
    plt.ylim([0,5])
    
    
    plt.figure()
    plt.plot(hist['epoch'],hist['mse'],label="Train Error")
    plt.plot(hist['epoch'],hist['val_mse'],label="Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Sqaured Error")
    plt.legend()
    plt.ylim([0,20])
    


# In[118]:


plot_history(history)


# ### The model is highly overfit so we need to apply early stopping while training the model

# In[119]:


model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",patience=10)

history = model.fit(X_train,y_train,epochs=100,validation_split=0.2,verbose=0,callbacks=[early_stop,Printdot()])


# In[120]:


plot_history(history)


# In[121]:


loss, mae, mse = model.evaluate(X_test,y_test,verbose=0)


# In[122]:


print("Model loss : {}".format(loss))
print("Model Mean Absolute Error : {}".format(mae))
print("Model Mean Sqared Error : {}".format(mse))


# ### Now let us make predictions

# In[137]:


predictions = model.predict(X_test).flatten()

plt.figure()
plt.scatter(y_test,predictions)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.axis("equal")
plt.axis("square")
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.plot([-100,100],[-100,100],color="r",linestyle="dashed")


# ### Predictions are quite close to the baseline considering the small amount of data that we have for our analysis

# ### Let us now take a look at the errors and plot them to check whether it is normally distributed or not

# In[143]:


errors = predictions - y_test

plt.figure(figsize=(5,4))
sns.kdeplot(errors)


# ### The errors are also quite well if not perfectly normally distributed.
