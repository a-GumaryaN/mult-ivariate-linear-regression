import numpy as np
import pandas as pd
from scipy.stats import zscore

class Linear_regression:
    b=0
    w=None
    w_init=0
    alpha=1e-3
    iteration_rate=1000
    record_cost_history=False
    cost_history=[]
    landa=0

    def __init__(self,**keywords):

        predictor_features=keywords["predictors"]
        data=keywords["data"]
        self.m=data.shape[0]
        self.predictor_data=data[predictor_features]
        self.response_data=data[keywords["response"]]
    
        if "b_init" in keywords:
            self.b=keywords["b_init"]

        if "w_init" in keywords:
            self.w_init=keywords["w_init"]

        if "alpha" in keywords:
            self.alpha=keywords["alpha"]
            
        if "iteration_rate" in keywords:
            self.iteration_rate=keywords["iteration_rate"]
            
        if "w_derivation_function" in keywords:
            self.w_derivation_function=keywords["w_derivation_function"]
            
        if "b_derivation_function" in keywords:
            self.b_derivation_function=keywords["b_derivation_function"]
            
        if "record_cost_history" in keywords:
            self.record_cost_history=keywords["record_cost_history"]
            
        if "landa" in keywords:
            self.landa=keywords["landa"]

        self.w=np.full(self.predictor_data.shape[1],self.w_init)
        
        
    
    def f_x(self,data_index):

        x=self.predictor_data.iloc[data_index]

        result=np.dot(self.w,x)

        result = result + self.b

        return result

    def cost_function(self):
        f_x=[]
        for index in self.predictor_data.index:
            f_x_i=self.f_x(index)
            y_i=self.response_data[index]
            cost= (f_x_i - y_i)**2
            f_x.append(cost)

        f_x=np.array(f_x)
        f_x=np.sum(f_x)
        return f_x
    
    def w_gradient_calculator(self):
        self.w_gradients=np.full(self.predictor_data.shape[1],float(1))

        predictor_number=0

        m=self.predictor_data.shape[0]

        predictor_index=0

        for predictor in self.predictor_data.columns:

            dj_dw_array=[]

            for index in self.predictor_data.index:

                f_x_i=self.f_x(index)
                y_i=self.response_data[index]
                x_i_n=self.predictor_data[predictor][index]

                dj_dw= ( f_x_i - y_i ) * x_i_n
                dj_dw_array.append(dj_dw)
            
            dj_dw_array=np.array(dj_dw_array)
            dj_dw=np.sum(dj_dw_array)
            dj_dw=dj_dw / self.m
            dj_dw= dj_dw + ( ( self.landa / m ) * self.w[predictor_index] )
            dj_dw=dj_dw * self.alpha

            self.w_gradients[predictor_number]=dj_dw
            predictor_number=predictor_number+1
            predictor_index=predictor_index+1

    def b_gradient_calculator(self):
        dj_db_array=[]

        for index in self.predictor_data.index:
            f_x_i=self.f_x(index)
            y_i=self.response_data[index]
            dj_db= f_x_i - y_i 
            dj_db_array.append(dj_db)
        
        dj_db_array=np.array(dj_db_array)
        dj_db=np.sum(dj_db_array)
        dj_db=dj_db / self.m
        dj_db=dj_db * self.alpha
        self.b=self.b - dj_db

    def gradient_descent(self):

        self.cost_history=[]

        for k in range(self.iteration_rate):
            self.w_gradient_calculator()
            self.b_gradient_calculator()
            self.w=np.subtract(self.w,self.w_gradients)

            if self.record_cost_history:
                self.cost_history.append(self.cost_function())