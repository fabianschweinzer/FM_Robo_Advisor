#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:23:50 2021

@author: fabianschweinzer
"""
#welcome to our fantastic RoboAdvisor, we are Group 3 from MFIN Cohort 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data
from matplotlib import style
style.use('ggplot')


#we read the sp500_data into the console, as the dataset is too big to load it into Python
#please update the path here in case you want to run that file
#to date, 20.02.2021, I haven't figured out how to link the python script to my Github repo
#I'm sorry for that, I will get better at this I swear
sp500_data = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/S&P_Database_Final')
# Welcome message for our Portfolio-Builiding Tool

#for the input questions, we first give the potential investor some intro
print("Let's build a portfolio together!")
print("Answer the following questions to get recommendations for your individualized portfolio:")

#Funds available
#we ask the question here, how many dollars can he invest 
#this is going to be used to determine the amount of shares he needs to purchase
funds_q = float(input("How many dollars do you want to invest? (Specify an amount)"))


# Ethical constraints: Weapons
#ethical question concerning weapns
weapon_yes_no = input("Do you feel comfortable investing in companies that make their money with weapons? (Yes/No) ")

#the investor is asked whethre he feels comfortable in investing in weapons
#Yes or No Question
#if he inserts a wrong input, the question is asked again
while weapon_yes_no != 'Yes' and weapon_yes_no != 'No':
    weapon_yes_no = input("Please enter a valid input: (Yes/No) ")

if weapon_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have something to do with weapons
    filter_array = sp500_data.loc[:, 'Weapons'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))


# Ethical constraints: Gambling
#same logic as with the weapons question
gambling_yes_no = input("Do you feel comfortable investing in companies that operate in the gambling sector? (Yes/No) ")

while gambling_yes_no != 'Yes' and gambling_yes_no != 'No':
    gambling_yes_no = input("Please enter a valid input: (Yes/No) ")

if gambling_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have something to do with gambling
    filter_array = sp500_data.loc[:, 'Gambling'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Ethical constraints: Tobacco
#same logic as with the questions before
tobacco_yes_no = input("Do you feel comfortable investing in companies that sell tobacco-products? (Yes/No) ")

while tobacco_yes_no != 'Yes' and tobacco_yes_no != 'No':
    tobacco_yes_no = input("Please enter a valid input: (Yes/No) ")

if tobacco_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have something to do with tobacco
    filter_array = sp500_data.loc[:, 'Tobacco'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Ethical constraints: Animal Testing
#same logic as with the questions before
animal_testing_yes_no = input("Do you feel comfortable investing in companies that test their products on animals? (Yes/No) ")

while animal_testing_yes_no != 'Yes' and animal_testing_yes_no != 'No':
    animal_testing_yes_no = input("Please enter a valid input: (Yes/No) ")

if animal_testing_yes_no == 'Yes':
    # Do nothing
    None
else:
    # Filter out all the companies that have test their products on animals
    filter_array = sp500_data.loc[:, 'Animal Testing'] == False
    sp500_data = sp500_data[filter_array]

print(len(sp500_data))

# Find out risk tolerance of investor
# Find out 25th, 50th and 75th percentile

sp500_std_25th_percentile = np.percentile(sp500_data.loc[:, "Annualized Std"], 25)
sp500_std_50th_percentile = np.percentile(sp500_data.loc[:, "Annualized Std"], 50)
sp500_std_75th_percentile = np.percentile(sp500_data.loc[:, "Annualized Std"], 75)

print(len(sp500_data))

# Find out importance of ESG for investor
#here the question is asked how important is the ESG score for the investor
esg_importance = input("How important is the ESG-criteria for you as investor? (Low/Medium/High) ")

# Find out 25th, 50th and 75th percentile
#we determine the specific percentiles for the input 
sp500_esg_25th_percentile = np.percentile(sp500_data.loc[:, "ESG Score"], 25)
sp500_esg_50th_percentile = np.percentile(sp500_data.loc[:, "ESG Score"], 50)

while esg_importance != 'Low' and esg_importance != 'Medium' and esg_importance != 'High':
    esg_importance = input("Please enter a valid input: (Low/Medium/High) ")

if esg_importance == 'Low':
    # Do nothing
    None
elif esg_importance == 'Medium':
    filter_array = sp500_data.loc[:, "ESG Score"] > sp500_std_25th_percentile
    sp500_data = sp500_data[filter_array]
else:
    filter_array = sp500_data.loc[:, "Annualized Std"] > sp500_std_50th_percentile
    sp500_data = sp500_data[filter_array]
    
#Fin out risk tolerance
#this is a key question regarding our whole syntax, as we build a lot of conditional statements around that question
#put in conditional arguments regarding the investors risk tolerance
risk_tolerance = input("Which attitude towards risk matches your character traits the most? (Low/Medium/High) ")

while risk_tolerance != 'Low' and risk_tolerance != 'Medium' and risk_tolerance != 'High':
    risk_tolerance = input("Please enter a valid input: (Low/Medium/High) ")

test = input("test")

print('#'*60)


#get the database into the framework
#we redefine the sp500_data as the given database and set the index to symbol
database = sp500_data
database.set_index('Symbol', inplace=True)

#extract target return extracted from barchart.com
#the target return is nothing else than the average consensus for the expected return of a given asset in 1 year
#we use the target return to be more forward looking in our calculations
mu_target = database.iloc[:,-2]
target_list = mu_target.index.to_list()

#initiate Cov Matrix 
#here we insert the Covariance matrix calculated in another python code
cov_matrix = pd.read_csv('/Users/fabianschweinzer/Desktop/Hult International Business School/MFIN/Python/Group Project Group 3/portfolio-modeling-main/Cov_Matrix_SP500')
cov_matrix.set_index('Unnamed: 0', inplace = True)

#in order to match the target return dataframe with the cov matrix we need to subset the cov matrix
#used a transpose technique to do the subsetting on both axis
cov_matrix = cov_matrix[cov_matrix.index.isin(mu_target.index)]
cov_matrix = cov_matrix.T
cov_matrix = cov_matrix[cov_matrix.index.isin(mu_target.index)]
#multiply the cov matrix with 252 to get the annual cov_matrix
#the target return is already annualized
cov_matrix = cov_matrix * 252


##Start using pypopft in order to optimize Portfolio according to certain constraints
##pypfopt optimization
#we use the python library "PyPortfolioOpt" for optimizing our portfolio
#we first need to import the EfficientFrontier function and other built-in methods
from pypfopt import EfficientFrontier
from pypfopt import expected_returns, objective_functions
from pypfopt.risk_models import CovarianceShrinkage 
from pypfopt import CLA, plotting

#we use the conditional statements for the risk_tolerance to get different optimization for different risk tolerances
#if the investor is very risk averse, we optimize based on the minimum volatility optimization technique
#if investor says he is only fairly risk averse ("Medium"), we optimize based on the maximum sharpe optimization technique
#if investor states he is risk friendly, we optimize based on maximizing the return given a predefined target volatility
if risk_tolerance == 'Low':
    ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
    weights = ef.min_volatility()
elif risk_tolerance == 'Medium':
    ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = ef.max_sharpe()
elif risk_tolerance == 'High':
    target_volatility = 0.3
    ef = EfficientFrontier(mu_target, cov_matrix, weight_bounds=(0,1))
    ef.add_objective(objective_functions.L2_reg, gamma=0.2)
    weights = ef.efficient_risk(target_volatility)

#clean weights for a better visualization of the weights
clean_weights = ef.clean_weights()
ef.portfolio_performance(verbose = True)

#get the ticker list from the created dictionary
tickers = pd.DataFrame.from_dict(data=clean_weights, orient='index')

#we need to slize the tickers_list so we get only tickers with non-zero weights
tickers.columns = ['Weight']
tickers = tickers[tickers['Weight']!= 0]
#sort the values in descending order
tickers.sort_values(by='Weight', ascending=False)

#create list out of sliced tickers dataframe
tickers_list = tickers.index.to_list()

#get sliced covariance matrix
#we need to slice the cov_matrix once again to the get the cov_matrix for the tickes in the optimized portfolio
cov_matrix_tickers = cov_matrix[cov_matrix.index.isin(tickers.index)]
cov_matrix_tickers = cov_matrix_tickers.T
cov_matrix_tickers = cov_matrix_tickers[cov_matrix_tickers.index.isin(tickers.index)]

#get the sliced mu_target dataframe
mu_target_tickers = mu_target[mu_target.index.isin(tickers.index)]

#for visualizing the optimized portfolio and the weights we create a piechart 
#create piechart 
piechart = pd.Series(tickers['Weight']).plot.pie(figsize=(10,10))
plt.show(piechart)

#we also create a barchart
#create barchart
barchart = pd.Series(tickers['Weight']).sort_values(ascending=True).plot.barh(figsize=(10,6))
plt.show(barchart)


#covariance heatmap
#to get a grasp how our chosen portfolio correlates with each other asset in the portfolio we state a cov heatmap
plotting.plot_covariance(cov_matrix_tickers, plot_correlation = True)

##create the Efficient frontier line and visualize it
#in order to visualize the efficient frontier from the optimized portfolio, we need to initiate a new built-in method
#we use the CLA method, because it is more robust than the default option
cla = CLA(mu_target_tickers, cov_matrix_tickers)
if risk_tolerance == 'Low':
    cla.min_volatility()
else:
    cla.max_sharpe()

#plot the efficient frontier line
ax_portfolio = plotting.plot_efficient_frontier(cla, ef_param='risk', ef_param_range=np.linspace(0.2,0.6,100), points=1000)


#initialize Discrete Allocation to get a full picture what you could buy with a given amount
#the Discrete Allocation gives you the discrete amount of shares you have to allocate given your available funds
from pypfopt import DiscreteAllocation
from datetime import datetime
from datetime import timedelta

#create list out of sliced tickers dataframe
tickers_list = tickers.index.to_list()

#create a dictionary which can be used for the Discrete Allocation as an input
weight_pf = {}
values_pf = tickers['Weight']
for ticker in tickers_list:
    weight_pf[ticker] = values_pf[ticker]
print(weight_pf)

#read in closing data from the chosen tickers in order to get the latest prices 
da = data.DataReader(tickers_list, data_source='yahoo', start=datetime(2020,1,28), end=datetime.today())['Adj Close']

#slice the dataframe to get the colum with the lates prices
latest_price = da.iloc[-1,:]

#instantiate the discrete allocation through the built-in method
alloc = DiscreteAllocation(weight_pf,latest_prices=latest_price, total_portfolio_value=funds_q)
allocation, leftover = alloc.lp_portfolio()
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))

#barchart of the Discrete Allocation
#this gives a barchart of the amount of shares the investors needs to hold in order to follow the optimized allocation given from the optimizer
barchart_alloc = pd.Series(allocation).sort_values(ascending=True).plot.barh(figsize=(10,6))
plt.show()

##############################
print('#'*60)

#calculate the portfolio return and portfolio std, to compare both
portfolio_return = mu_target_tickers.dot(tickers['Weight'])

#get the standard deviation from each ticker, which is in the optimal portfolio
sigma = database['Annualized Std'].reset_index()
sigma = sigma[sigma['Symbol'].isin(mu_target_tickers.index)]
sigma = sigma.set_index('Symbol')
sigma = sigma.iloc[:,-1]

#calculate from the cov_matrix and the portfolio weights the portfolio_std
portfolio_std = np.sqrt(np.dot(tickers['Weight'].T,np.dot(cov_matrix_tickers, tickers['Weight'])))

print('Portfolio expected annualised return is ' + str(round(portfolio_return, 3)*100) + '% with a standard deviation of ' + str(round(portfolio_std, 3)*100) +'%')#

print('#'*60)
####################################################

# Initialize Monte Carlo parameters
#monte carlo simulation to get VaR95
#we use a parametric simulation

###get historic data for the Monte Carlo simulation, as the target return is not a valid input for the simulation here
#we go back five years to retrieve historic data
s = datetime(2016,2,5)
e = datetime.today()

#get the data for our tickers from datareader
portfolio_data =  data.DataReader(tickers_list, data_source='yahoo',
                               start = s ,
                             end= e )['Adj Close']

#calculate historic returns, eean return and standard deviation
historic_returns = portfolio_data.pct_change().dropna()
historic_mean = historic_returns.mean()
historic_std = historic_returns.std()


#initiate the monte carlo simulation with defining the amount of runs, days to simulate and loss cutoff
monte_carlo_runs = 1000
days_to_simulate = [5, 30, 50] #we use 5, 30, 50 as the input for days to simulate
loss_cutoff      = [0.90, 0.95, 0.99]      # count any losses larger than 1%, 5% and 10%
df1 = [] #define the placeholder list, where we hold the VaR percentages later

#initiate total_simulations, bad_simulations
compound_returns  = historic_std.copy()
total_simulations = 0
bad_simulations   = 0

for days in days_to_simulate: #loop over the three different days input 
    for loss in loss_cutoff: #loop over the three different inputs for the loss_cutoff
        for run_counter in range(0,monte_carlo_runs):   # Loop over runs    
            for i in tickers_list:                      # loop over tickers
        
                # Loop over simulated days:
                compounded_temp = 1
        
                for simulated_day_counter in range(0,days): # loop over days
            
                    # Draw from 𝑁~(𝜇,𝜎)
                    ######################################################
                    simulated_return = np.random.normal(historic_mean[i], historic_std[i],1)
                    ######################################################
            
                    compounded_temp = compounded_temp * (simulated_return + 1)        
        
                compound_returns[i] = compounded_temp     # store compounded returns
    
            # Now see if those returns are bad by combining with weights
            portfolio_return_mc = compound_returns.dot(tickers['Weight']) # dot product
    
            if(portfolio_return_mc < loss):
                bad_simulations = bad_simulations + 1
    
            total_simulations = total_simulations + 1
        
        #store the outputs of the simulation in the created list
        VaR = round(bad_simulations/total_simulations, 3)
        df1.append(VaR)

        print("Your portfolio will lose", round((1-loss)*100,3), "%",
                 "over", days, "days", 
                 VaR * 100, "% of the time.")
        
        #in order to rebalance it again, we need to reset the bad_simulations and the total simulations after one whole run
        #otherwise, the outputs would average out
        bad_simulations = 0
        total_simulations = 0
        
#end of loops
print(df1)

print('#'*60)
#############
#Monte Carlo plot for whole portfolio returns
#here we try to plot the portfolio returns with a monte carlo simulations

#we have to define the last_return, in order to instantiate the simulation
last_return = 1 #had to define it as non-zero, otherwise we would multiply by zero later

#instantiate number of simulations and number of days to simulate
num_simulations = 1000
num_days = 252

simulation_df = pd.DataFrame() #crate a dataframe as a placeholder for the outputs
for x in range (num_simulations):
    count = 0 #instaniate the count
    daily_vol = historic_std.mean()
    
    returns_series = [] #instantiate a return series
    
    returns_mc = last_return * (1 + np.random.normal(0, daily_vol)) #compound the returns with randomized returns
    returns_series.append(returns_mc)
    
    
    for y in range (num_days):
        if count == 251:
            break
        returns_mc = returns_series[count] * (1 + np.random.normal(0, daily_vol))
        returns_series.append(returns_mc)
        count += 1
        
    simulation_df[x] = returns_series

#plot the simulation outcome    
fig=plt.figure()
fig.suptitle('Monte Carlo Simulation for Portfolio Returns')
plt.plot(simulation_df)
plt.axhline(y = last_return, color= 'r', linestyle = '-')
plt.xlabel('Day')
plt.ylabel('Returns')
plt.show()  

##############################
# Plot Count Distribution of Portfolio Returns and VaR

# setting figure size
fig, ax = plt.subplots(figsize = (13, 5)) 

#add the portfolio mean to the return dataframe, so we can use it as an input for the VaR graph
historic_returns['Portfolio'] = historic_returns.mean(axis=1)

# histogram for returns
sns.histplot(data  = historic_returns['Portfolio'], # data set
             bins  = 'fd',          # number of bins
             kde   = True,          # kernel density plot (line graph)
             alpha = 0.2,           # transparency of colors
             stat  = 'count')    #use the count here to get the count distribution


# this adds a title
plt.title(label = "Distribution of Portfolio Return")

# this adds an x-label
plt.xlabel(xlabel = 'Returns')

# this add a y-label
plt.ylabel(ylabel = 'Count')

# instantiate VaR with 95% confidence level
VaR_95 = np.percentile(historic_returns, 5)

# this adds a line to signify VaR
plt.axvline(x         = VaR_95,         # x-axis location
            color     = 'r',            # line color
            linestyle = '--')           # line style

# this adds a label to the line
plt.text(VaR_95,                         # x-axis location
         30,                             # y-axis location
         'VaR',                          # text
         horizontalalignment = 'right',  # alignment
         fontsize = 'x-large')           # fontsize

#these compile and display the plot so that it is formatted as expected
plt.tight_layout()
plt.show()

######################
#Plot a Density distribution of the Portfolio Returns
# Plot Returns

# setting figure size
fig, ax = plt.subplots(figsize = (13, 5))

# histogram for returns
sns.histplot(data  = historic_returns,      # data set
             bins  = 'fd',         # number of bin
             kde   = True,         # kernel density plot (line graph)
             alpha = 0.3,          # transparency of colors
             stat  = 'density')    #set the stat to density to get the density distribution

# this adds a title
plt.title(label = "Distribution of Portfolio Returns")

# this adds an x-label
plt.xlabel(xlabel = 'Returns')

# this add a y-label
plt.ylabel(ylabel = 'Density')

# instantiate VaR with 95% confidence level
VaR_95 = np.percentile(historic_returns, 5)

# this adds a line to signify VaR
plt.axvline(x         = VaR_95,         # x-axis location
            color     = 'r',            # line color
            linestyle = '--')           # line style

# this adds a label to the line
plt.text(VaR_95,                         # x-axis location
         1,                             # y-axis location
         'VaR_95',                          # text
         horizontalalignment = 'right',  # alignment
         fontsize = 'x-large')           # fontsize

# remove legend
ax.get_legend().remove()

# these compile and display the plot so that it is formatted as expected
plt.tight_layout()
plt.show()

########################################################
# Import the bt package so we can use the backtesting functions
import bt

#define the beginning date, so I can loop over different time periods
beginning = [datetime(2015,1,1), datetime(2017,1,1), datetime(2019,1,1)]
#instantiate the offset, over which time period the backtesting should go, here two years
offset = timedelta(weeks=104)

for b in beginning:
    # Import data
    data_bt = bt.get(tickers_list, start=b, end= b+ offset)

    # We will need the risk-free rate to get correct Sharpe Ratios 
    riskfree =  bt.get('^IRX', start=b, end = b + offset)
    # Take the average of the risk free rate over entire time period
    riskfree_rate = float(riskfree.mean()) / 100
    # Print out the risk free rate to make sure it looks good
    print('risk-free rate:', riskfree_rate)
    
    #instantiate a conditional statement to select the strategy based on the given risk tolerance of the investor
    if risk_tolerance == 'Low': 
# if the investors has a low risk tolerance, we use a Inverse Volatiliy Strategy
        s_mark = bt.Strategy('Portfolio', 
                       [bt.algos.RunMonthly(),
                       bt.algos.SelectAll(),
                       bt.algos.WeighInvVol(),
                       bt.algos.Rebalance()])
  #if the investor is fairly risk tolerant or risk friendly we use the MeanVar strategy  
    elif risk_tolerance == 'Medium':
        s_mark = bt.Strategy('Portfolio', 
                       [bt.algos.RunEveryNPeriods(30,3),
                       bt.algos.SelectAll(),
                       bt.algos.WeighMeanVar(rf=riskfree_rate),
                       bt.algos.Rebalance()])

    else: 
        s_mark = bt.Strategy('Portfolio', 
                       [bt.algos.RunEveryNPeriods(10,3), #we differentiate here for the given periods
                       bt.algos.SelectAll(),
                       bt.algos.WeighMeanVar(rf=riskfree_rate),
                       bt.algos.Rebalance()])

    #instantiate the backtesting
    b_mark = bt.Backtest(s_mark, data_bt)
    #in order to compare our strategy with a benchmark, we use the SP500 as an relative performance indicator

    # Fetch some data for the benchmark SP500
    data_sp500 = bt.get('spy,agg', start=b, end= b + offset)

    # Recreate the strategy for the SP500, we use here an Equally Weighted strategy
    b_sp500 = bt.Strategy('SP500', [bt.algos.RunOnce(),
                                     bt.algos.SelectAll(),
                                     bt.algos.WeighEqually(),
                                     bt.algos.Rebalance()])
    
    # Create a backtest named for the SP500
    sp500_test = bt.Backtest(b_sp500, data_sp500)

    #we run the backtest and get some results
    result = bt.run(b_mark, sp500_test)

    #create the run only for the b_mark
    result_1 = bt.run(b_mark) #we need this result to get the return distribution later

    #result = bt.run(b_mark, b_inv, b_random, b_best, b_sp500)
    result.set_riskfree_rate(riskfree_rate)
    result.plot()
    
    #show histogram based on the result_1 run, as we can't get the return distribution from the first run including the benchmark
    result_1.plot_histograms(bins=50, figsize=(20,10))

    # Show some performance metrics
    result.display()
    
#end of loop
#end of script









    



   
    
    
    
