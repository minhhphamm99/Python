# =============================================================================
# 1. construct the graph of Y on X
# 2. Decide the set of lines potentially matching the data
# 3. conduct the test. Make the correspndednt regressions, make a decision which regression should be selected
# 4.From the selected model, analyse what coefficients are not 0
# 5. maka a conclusion, draw the data+the regression line
# =============================================================================

import random as rd
rd.seed(123)
Y=[]
X=[]
for i in range(0,300):
    X.append(rd.uniform(-3, 3))
    Y.append(X[i]**4-X[i]**3-5*X[i]**2+X[i]+5+rd.normalvariate(0,5))
  
import pandas as pd    
OriginalData=pd.DataFrame({"Y":Y, "X":X})

import matplotlib.pyplot as plt
plt.scatter(OriginalData.X,OriginalData.Y)
plt.show()

import numpy as np
from scipy import stats
#make a starting point, iterate the process for automatic generation of M1,2,3,4... dataframe
BIC=[]
BIC_X=[]

Ydf=OriginalData[OriginalData.columns[0]]
y=Ydf.values
Xdf=pd.DataFrame({"C":np.ones(len(OriginalData.X)), "X":OriginalData.X})
for i in range(2,10):
    xp=[]
    for j in range(0, len(Xdf.X)):
        xp.append(Xdf.X[j]**i)
    ColumnName='X'+str(i)
    Xdf[ColumnName]=xp    
    #on each step conduct calculation of BIC value
    x=Xdf.values
    B=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),x.transpose()),y) #coefficients
    e=np.subtract(np.matmul(x,B),y)
    SSE=np.matmul(e.transpose(),e) # Sum of squared errors for BIC
    BIC.append(len(y)*np.log(SSE)+i*np.log(len(y)))
    BIC_X.append(i)

#check BIC plot
plt.plot(BIC_X,BIC)
plt.show()
# check the graph, find the minimal value
# it is the 4th item, correspondent to the 4th power of equation. Limit the dataframe
Xdf=Xdf[Xdf.columns[:5]]
#calculate for regression and all the rest parameeters  for the selected model

x=Xdf.values

B=np.matmul(np.matmul(np.linalg.inv(np.matmul(x.transpose(),x)),x.transpose()),y)

e=np.subtract(np.matmul(x,B),y)
MSE=np.matmul(e.transpose(),e)/(len(e)-len(Xdf.columns))
s=np.sqrt(MSE)
var_b=MSE*(np.linalg.inv(np.matmul(x.transpose(),x)).diagonal())
sd_b = np.sqrt(var_b)
ts_b=B/sd_b
p_values =np.round([2*(1-stats.t.cdf(np.abs(i),(len(y)-len(Xdf.columns)))) for i in ts_b],3) # check if there are any p_values higher that wanted threshold 

SSE=np.matmul(e.transpose(),e)
SST=np.sum((y-np.mean(y))**2)

Rsq=1-SSE/SST
# =============================================================================
# plt.plot(Xdf.X.values,np.matmul(x,B),color='r')
# =============================================================================
plt.show()

#Sorting dataframe to make a connected curve
temp=pd.DataFrame(Xdf.X.values,columns=['Value'])
temp['Target']=np.matmul(x,B)
#Sort the dataframe
temp=temp.sort_values(by=['Value'])
#plot the regression line
plt.scatter(Xdf.X.values,y,color='b')
plt.plot(temp['Value'],temp['Target'],c='r',linewidth=4)

#prediction interval
#confidence interval
ci=1.96*np.std(np.random.choice(y,100))/np.sqrt(100) 
Lowerci=temp['Target']-ci
Upperci=temp['Target']+ci  
plt.plot(temp['Value'],Lowerci,c='y',linewidth=4) 
plt.plot(temp['Value'],Upperci,c='y',linewidth=4)       




# =============================================================================
# import numpy as np
# from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt    
# from scipy import stats
# import pandas as pd
# 
# # pip install uncertainties, if needed
# try:
#     import uncertainties.unumpy as unp
#     import uncertainties as unc
# except:
#     try:
#         from pip import main as pipmain
#     except:
#         from pip._internal import main as pipmain
#     pipmain(['install','uncertainties'])
#     import uncertainties.unumpy as unp
#     import uncertainties as unc
# 
# # import data
# url = 'https://apmonitor.com/che263/uploads/Main/stats_data.txt'
# data = pd.read_csv(url)
# x = data['x'].values
# y = data['y'].values
# n = len(y)
# 
# def f(x, a, b, c):
#     return a * np.exp(b*x) + c
# 
# popt, pcov = curve_fit(f, x, y)
# 
# # retrieve parameter values
# a = popt[0]
# b = popt[1]
# c = popt[2]
# print('Optimal Values')
# print('a: ' + str(a))
# print('b: ' + str(b))
# print('c: ' + str(c))
# 
# # compute r^2
# r2 = 1.0-(sum((y-f(x,a,b,c))**2)/((n-1.0)*np.var(y,ddof=1)))
# print('R^2: ' + str(r2))
# 
# # calculate parameter confidence interval
# a,b,c = unc.correlated_values(popt, pcov)
# print('Uncertainty')
# print('a: ' + str(a))
# print('b: ' + str(b))
# print('c: ' + str(c))
# 
# # plot data
# plt.scatter(x, y, s=3, label='Data')
# 
# # calculate regression confidence interval
# px = np.linspace(14, 24, 100)
# py = a*unp.exp(b*px)+c
# nom = unp.nominal_values(py)
# std = unp.std_devs(py)
# 
# def predband(x, xd, yd, p, func, conf=0.95):
#     # x = requested points
#     # xd = x data
#     # yd = y data
#     # p = parameters
#     # func = function name
#     alpha = 1.0 - conf    # significance
#     N = xd.size          # data sample size
#     var_n = len(p)  # number of parameters
#     # Quantile of Student's t distribution for p=(1-alpha/2)
#     q = stats.t.ppf(1.0 - alpha / 2.0, N - var_n)
#     # Stdev of an individual measurement
#     se = np.sqrt(1. / (N - var_n) * \
#                  np.sum((yd - func(xd, *p)) ** 2))
#     # Auxiliary definitions
#     sx = (x - xd.mean()) ** 2
#     sxd = np.sum((xd - xd.mean()) ** 2)
#     # Predicted values (best-fit model)
#     yp = func(x, *p)
#     # Prediction band
#     dy = q * se * np.sqrt(1.0+ (1.0/N) + (sx/sxd))
#     # Upper & lower prediction bands.
#     lpb, upb = yp - dy, yp + dy
#     return lpb, upb
# 
# lpb, upb = predband(px, x, y, popt, f, conf=0.95)
# 
# # plot the regression
# plt.plot(px, nom, c='black', label='y=a exp(b x) + c')
# 
# # uncertainty lines (95% confidence)
# plt.plot(px, nom - 1.96 * std, c='orange',\
#          label='95% Confidence Region')
# plt.plot(px, nom + 1.96 * std, c='orange')
# # prediction band (95% confidence)
# plt.plot(px, lpb, 'k--',label='95% Prediction Band')
# plt.plot(px, upb, 'k--')
# plt.ylabel('y')
# plt.xlabel('x')
# plt.legend(loc='best')
# 
# # save and show figure
# plt.savefig('regression.png')
# plt.show()
# =============================================================================
