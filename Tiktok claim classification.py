#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install xgboost


# In[3]:


import pandas as pd
import numpy as np

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


# In[5]:


data = pd.read_csv(r"D:\admin\Downloads\tiktok_dataset.csv")
data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.describe()


# In[10]:


data.isnull().sum()


# In[11]:


data = data.dropna(axis=0)


# In[12]:


data.duplicated().sum()


# In[13]:


data["claim_status"].value_counts(normalize=True)


# #### Feature engineering

# In[14]:


data['text_length'] = data['video_transcription_text'].str.len()
data.head()


# In[15]:


data[['claim_status', 'text_length']].groupby('claim_status').mean()


# In[16]:


sns.histplot(data=data, stat="count", multiple="dodge", x="text_length",
             kde=False, palette="pastel", hue="claim_status",
             element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for claims and opinions")
plt.show()


# #### Feature selection and transformation

# In[17]:


X = data.copy()
# Drop unnecessary columns
X = X.drop(['#', 'video_id'], axis=1)
# Encode target variable
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})
# Dummy encode remaining categorical values
X = pd.get_dummies(X,
                   columns=['verified_status', 'author_ban_status'],
                   drop_first=True)
X.head()


# ### Split the data

# In[18]:


y = X['claim_status']


# In[19]:


# Isolate features
X = X.drop(['claim_status'], axis=1)

# Display first few rows of features dataframe
X.head()


# ### Create train/validate/test sets

# In[20]:


X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[21]:


X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)


# In[22]:


X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape


# In[23]:


count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
count_vec


# In[24]:


count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
count_data


# In[25]:


# Place the numerical representation of `video_transcription_text` from training set into a dataframe
count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())

# Display first few rows
count_df.head()


# In[26]:


X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)


X_train_final.head()


# In[27]:


validation_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()
validation_count_data


# In[28]:


validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())
validation_count_df.head()


# In[29]:


X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis=1)

# Display first few rows
X_val_final.head()


# In[30]:


test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()

# Place the numerical representation of `video_transcription_text` from test set into a dataframe
test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_val` and `validation_count_df` to form the final dataframe for training data (`X_val_final`)
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']
                                      ).reset_index(drop=True), test_count_df], axis=1)
X_test_final.head()


# ### Random Forest model

# In[31]:


rf = RandomForestClassifier(random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=['accuracy', 'recall', 'f1', 'precision'], cv=5, refit='recall')


# In[32]:


get_ipython().run_cell_magic('time', '', 'rf_cv.fit(X_train_final, y_train)')


# In[33]:


rf_cv.best_score_


# In[34]:


rf_cv.best_params_


# ### XGBoost model 

# In[35]:


xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1'}

# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=['accuracy', 'recall', 'f1', 'precision'], cv=5, refit='recall')


# In[36]:


get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train_final, y_train)')


# In[37]:


xgb_cv.best_score_


# In[38]:


xgb_cv.best_params_


# #### Random forest 

# In[39]:


y_pred = rf_cv.best_estimator_.predict(X_val_final)
y_pred


# In[40]:


log_cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.show()


# In[41]:


target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# #### XGBOOST
# Now, evaluate the XGBoost model on the validation set.

# In[42]:


y_pred = xgb_cv.best_estimator_.predict(X_val_final)
y_pred


# In[43]:


log_cm = confusion_matrix(y_val, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('XGBoost - validation set');
plt.show()


# In[44]:


target_labels = ['opinion', 'claim']
print(classification_report(y_val, y_pred, target_names=target_labels))


# #### Use champion model to predict on test data.
# Both random forest and XGBoost model architectures resulted in nearly perfect models. Nonetheless, in this case random forest performed a little bit better, so it is the champion model.

# In[45]:


# Use champion model to predict on test data
y_pred = rf_cv.best_estimator_.predict(X_test_final)


# In[46]:


log_cm = confusion_matrix(y_test, y_pred)

# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=None)

# Plot confusion matrix
log_disp.plot()

# Display plot
plt.title('Random forest - test set');
plt.show()


# In[ ]:




