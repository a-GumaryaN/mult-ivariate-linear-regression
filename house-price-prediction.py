import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os.path
from linear_regression import *
from scipy.stats import zscore
import matplotlib.pyplot as pyplot

data=pd.DataFrame()

#check existence of cleaned data file

if not os.path.isfile("./data/cleaned_data.csv"):

    print("cleaned data not exist")

    #read data

    data=pd.read_csv("./data/Housing.csv")

    # cleaning data

    # replace yes to one and no to zero in some columns

    yes_no_columns=["mainroad","hotwaterheating","airconditioning","prefarea","basement","guestroom"]

    for column in yes_no_columns:
        data[column]=data.mainroad.eq('yes').mul(1)
        data[column]=data.mainroad.eq('no').mul(0)


    # scale down price columns data to thousand dolors scale
    data.price=data.price/1000

    # delete outliers data in price
    price_mean=data.price.mean()
    upper_limit = price_mean + ( 2.5 * data.price.std() )
    lower_limit= price_mean - ( 2.5 * data.price.std() )
    data=data[data.price.apply(lambda price : price >= lower_limit and price <= upper_limit)]

    # scale price feature
    data.price=zscore(data["price"])

    # delete outliers data in area
    area_mean=data.area.mean()
    upper_limit = area_mean + ( 2.5 * data.area.std() )
    lower_limit= area_mean - ( 2.5 * data.area.std() )
    data=data[data.area.apply(lambda area : area >= lower_limit and area <= upper_limit)]

    # scale area feature
    data.area=zscore(data["area"])

    data.to_csv("./data/cleaned_data.csv")
else:
    print("cleaned data already exist")
    data=pd.read_csv("./data/cleaned_data.csv")

regg=Linear_regression(data=data[:5],predictors=["area","parking"],response="price",iteration_rate=1)
cost_his=regg.gradient_descent()

print(regg.w_vector)