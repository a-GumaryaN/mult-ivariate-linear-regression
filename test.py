import pandas as pd
import numpy as np
import os
from linear_regression import Linear_regression

data=pd.DataFrame({})

if not os.path.isfile("./data/cleaned_data.csv"):

    print("data not exist")
    print("generate data")

    A=[]
    B=[]
    C=[]
    R=[]

    import random

    for item in range(80):
        a=int(random.random()*1000)
        b=int(random.random()*10)
        c=int(random.random()*1000)
        error=random.random()
        r=a*10 + b*3 + c*0.4 + error
        A.append(a)
        B.append(b)
        C.append(c)
        R.append(r)

    data=pd.DataFrame({
        "A":A,
        "B":B,
        "C":C,
        "R":R,
    })

    data.to_csv("./data/cleaned_data.csv")

else :
    print("data already exist")
    data=pd.read_csv("./data/cleaned_data.csv")


regg=Linear_regression(alpha=1e-6,data=data,predictors=["A","B","C"],response="R",iteration_rate=2000)
regg.gradient_descent()

print(regg.w)
print(regg.b)

"""
main values:
    w=[10  3  0.4]

alpha=1e-6 with 1000 iteration rate
    w=[10.0106861   0.06288239  0.41503224]
    b=0.008914996738183939


alpha=1e-6 with 2000 iteration rate
    w=[10.01059607  0.08810191  0.41490606]
    b=0.010124895407244589

"""