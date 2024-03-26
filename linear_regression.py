from typing import Annotated
import pandas as pd
import numpy as np
from threading import Thread
        

class Linear_regression:

    b=1
    w=1
    alpha=0.01
    iteration_rate=10000
    w_vector=None
    w_derivative_vector=None

    def __init__(self,**keywords):
        self.data=keywords["data"]
        
        self.response_feature=keywords["response"]

        if "b_init" in keywords:
            self.b=keywords["b_init"]

        if "w_init" in keywords:
            self.w=keywords["w_init"]

        if "alpha" in keywords:
            self.alpha=keywords["alpha"]
            
        if "iteration_rate" in keywords:
            self.iteration_rate=keywords["iteration_rate"]

        if len(keywords["predictors"])==1: # single variable prediction
            self.prediction_type="single variable"
            self.predict_feature=keywords["predictors"][0]
        else :
            self.prediction_type="multi variable"
            self.predict_features=keywords["predictors"]

    def gradient_descent(self):
        if self.prediction_type=="single variable":
            return self.single_variable_gradient_descent()
        else:
            return self.multi_variable_gradient_descent()

    def cost_calculator(self):
        costs=[]
        m=self.data.shape[0]

        for index in self.data.index:
            x=self.data[self.predict_feature][index]
            y=self.data[self.response_feature][index]

            f_x= ( self.w * x ) + self.b
            cost= ( f_x - y ) **2
            costs.append(cost)
        
        costs=np.array(costs)
        costs=np.sum(costs)
        costs= costs / ( 2 * m )
        return costs
    
    def gradient_calculator(self):
        dj_dw_array=[]
        dj_db_array=[]
        m=self.data.shape[0]
    
        for index in self.data.index :
        
            x=self.data[self.predict_feature][index]
            y=self.data[self.response_feature][index]
        
            f_x=( self.w * x ) + self.b

            dj_dw= ( f_x - y ) * x
            dj_db= f_x - y

            dj_dw_array.append(dj_dw)
            dj_db_array.append(dj_db)
        
        dj_dw=np.array(dj_dw_array)
        dj_dw=np.sum(dj_dw_array)
        dj_dw=dj_dw / m
        
        dj_db=np.array(dj_db_array)
        dj_db=np.sum(dj_db_array)
        dj_db=dj_db / m
    
        return dj_dw,dj_db
    
    def single_variable_gradient_descent(self):
    
        cost_history=[]

        print(self.alpha)
    
        for i in range(self.iteration_rate):

            cost = self.cost_calculator()
        
            cost_history.append(cost)

            dj_dw , dj_db = self.gradient_calculator()
    
            self.w= self.w - ( self.alpha * dj_dw )
    
            self.b= self.b - ( self.alpha * dj_db )

        return cost_history