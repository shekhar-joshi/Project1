#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:55:04 2018

@author: shekhar
"""

import sqlite3 
import pandas as pd
 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error 
from math import sqrt
import numpy as np
# Create your connection. 


def get_feature_columns():
    return ['overall_rating', 'potential', 'preferred_foot',
            'defensive_work_rate', 'crossing',
           'finishing', 'heading_accuracy', 'short_passing',
           'dribbling',  'free_kick_accuracy', 'long_passing',
           'ball_control', 'acceleration', 'sprint_speed',
           'reactions','shot_power','stamina',
           'strength', 'long_shots', 'aggression', 'interceptions',
           'positioning','penalties', 'marking', 'standing_tackle',
           'gk_diving', 'gk_handling', 'gk_kicking',
           'gk_positioning', 'gk_reflexes']

def get_data(sql_lite_db_path,cnx):
    df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
    print("=================Columns===================")
#    print(list(df.columns))
#    dataset = df.loc[:,['overall_rating','potential', 'short_passing', 'long_passing', 'volleys','dribbling',
#       'ball_control','reactions', 'shot_power','strength', 'long_shots', 'aggression', 'interceptions',
#       'positioning', 'vision', 'penalties','overall_rating']]
    
    return df[get_feature_columns()]#dataset



def drop_na(df):
    return df[df.overall_rating.notnull()]

def get_null_counts(df):
    return (df.isnull().sum())



def create_fit_model(X_train,y_train):
    from sklearn.tree import DecisionTreeRegressor 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model

def predict(model,X_test):
    model.predict(X_test)
    
    
def get_train_test_data(df):
    from sklearn.model_selection import train_test_split
    X = df.iloc[:,1:].values
    y = dataset_1.iloc[:,0].values
    
     #Splittinvg thedataset into train_test_split
    X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=.25,random_state=0)
    return X_train,X_test,y_train,y_test

def calc_train_error(X_train, y_train, model):
    '''returns in-sample error for already fit model.'''
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    rmse = np.sqrt(mse)
    return mse
    
def calc_validation_error(X_test, y_test, model):
    '''returns out-of-sample error for already fit model.'''
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return mse

if __name__=='__main__':
    try:
        sql_lite_db_path= r'data\database.sqlite'
        cnx = sqlite3.connect(sql_lite_db_path) 
        dataset = get_data(sql_lite_db_path,cnx)
        
        # Drop the rows having overall_rating is null
        dataset_1 = drop_na(dataset)
        
        # count the number of NaN values in each column
#        print(get_null_counts(dataset_1))63789
        
        
# =============================================================================
#         Get list of non numeric columns and convert it them into numeric
# =============================================================================
        not_num_cols=dataset_1.select_dtypes(exclude = [np.number,np.int16,np.bool,np.float32] )
    
        for col in not_num_cols:
            dataset_1.loc[:,col] = dataset_1.loc[:,col].astype('category')
            dataset_1.loc[:,col] = dataset_1.loc[:,col].cat.codes
        
        
# =============================================================================
#         Get train and test data
# =============================================================================
        X_train,X_test,y_train,y_test = get_train_test_data(dataset_1)
        
# =============================================================================
#         Model creation and model fitting
# =============================================================================
        model = create_fit_model(X_train,y_train)
        
# =============================================================================
#         Predict the values 
# =============================================================================
        #y_pred = predict(model,X_test)
        mse = calc_train_error(X_train, y_train, model)
        print("Mean squared error on train data: {}".format(mse))
        mse = calc_validation_error(X_test, y_test, model)
        print("Mean squared error on test data: {}".format(mse))
        
        
#        #Calculate Accuracy of Model
#        print(model.score(X_test,y_test))
#    #    resulf_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
        
    except Exception as e:
        print(e)
        
    finally:
        print("Closing Db Connection")
cnx.close()