
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor


def model_baseline(y_train,y_validate,target):
    ''' 
    
    '''

    baseline_mean = y_train.mean()
    baseline_median = y_train.median()

    #determining best baseline
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)

    y_train["mean"] = baseline_mean
    y_validate["mean"] = baseline_mean

    y_train["median"] = baseline_median
    y_validate["median"] = baseline_median

    ##want smallest error

    rmse_mean_train = mean_squared_error(y_train[target], y_train["mean"])**(1/2)
    rmse_mean_validate = mean_squared_error(y_validate[target], y_validate["mean"])**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_mean_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_mean_validate, 2))


    rmse_median_train = mean_squared_error(y_train[target], y_train["median"])**(1/2)
    rmse_median_validate = mean_squared_error(y_validate[target], y_validate["median"])**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_median_train, 2), 
        "\nValidate/Out-of-Sample: ", round(rmse_median_validate, 2))

    if rmse_mean_train < rmse_median_train:
        metric_df = pd.DataFrame(
            data=[{
                "model":f"baseline_mean",
                "rmse_train":rmse_mean_train,
                "rmse_validate":rmse_mean_validate,
                "r^2_validate":explained_variance_score(y_validate[target],y_validate["mean"])
            }])
    ###########
        #plot_model(y_validate,"mean",target,"Mean")
    ###########
    else:
        metric_df = pd.DataFrame(
            data=[{
                "model":f"baseline_median",
                "rmse_train":rmse_median_train,
                "rmse_validate":rmse_median_validate,
                "r^2_validate":explained_variance_score(y_validate[target],y_validate["median"])
            }])



    return y_train,y_validate,metric_df







def model_polynomial(X_train_scaled,X_validate_scaled,y_train,y_validate,target,n):
    '''
    takes in your modeling data frames (train and validate (test if moving on)) 
    (2 (X should be scaled), and 2 y), the target and the number of degrees to run the polynomial feature at
    returns the y_predictions on those dataframes and an additional dataframe of performance
    '''
    pf = PolynomialFeatures(degree=n)

    X_train_pf = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_pf = pf.transform(X_validate_scaled)


    # create the model object
    lm = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_pf, y_train[target])

    # predict train
    y_train[f'pred_pf{n}'] = lm.predict(X_train_pf)


    # evaluate: rmse
    rmse_pf_train = mean_squared_error(y_train[target], y_train[f'pred_pf{n}'])**(1/2)

    # predict validate
    y_validate[f'pred_pf{n}'] = lm.predict(X_validate_pf)

    # evaluate: rmse
    rmse_pf_validate = mean_squared_error(y_validate[target], y_validate[f'pred_pf{n}'])**(1/2)

    print(f"RMSE for Polynomial Model, degrees={n}\nTraining/In-Sample: ", rmse_pf_train, 
        "\nValidation/Out-of-Sample: ", rmse_pf_validate)
###########
    #plot_model(y_validate,f'pred_pf{n}',target,f"Model {n} degree Polynomial")
###########
    metric_df = pd.DataFrame(
        data=[{
            "model":f"Polynomial - degree {n}",
            "rmse_train":rmse_pf_train,
            "rmse_validate":rmse_pf_validate,
            "r^2_validate":explained_variance_score(y_validate[target],y_validate[f'pred_pf{n}'])
        }]
                            )
    return y_train,y_validate,metric_df


def model_linreg(X_train_scaled,X_validate_scaled,y_train,y_validate,target):
    '''
    takes in your modeling data frames (train and validate (test if moving on)) 
    (2 (X should be scaled), and 2 y), the target
    returns the y_predictions on those dataframes and an additional dataframe of performance
    for OLS
    '''
    # create the model object
    linreg = LinearRegression()

    # fit the model to our training data. 
    linreg.fit(X_train_scaled, y_train[target])

    # predict train
    y_train['pred_linreg'] = linreg.predict(X_train_scaled)

    # evaluate: rmse
    rmse_linreg_train = mean_squared_error(y_train[target], y_train.pred_linreg)**(1/2)

    # predict validate
    y_validate['pred_linreg'] = linreg.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_linreg_validate = mean_squared_error(y_validate[target], y_validate.pred_linreg)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_linreg_train, 
        "\nValidation/Out-of-Sample: ", rmse_linreg_validate)

###########
    #plot_model(y_validate,'pred_linreg',target,"Model for Linear Regression OLS")
###########
    metric_df = pd.DataFrame(
        data=[{
            "model":f"OLS Linear Regression",
            "rmse_train":rmse_linreg_train,
            "rmse_validate":rmse_linreg_validate,
            "r^2_validate":explained_variance_score(y_validate[target],y_validate['pred_linreg'])
        }])
    return y_train,y_validate,metric_df





def model_l_lars(X_train_scaled,X_validate_scaled,y_train,y_validate,target):
    '''
    takes in your modeling data frames (train and validate (test if moving on)) 
    (2 (X should be scaled), and 2 y), the target
    returns the y_predictions on those dataframes and an additional dataframe of performance
    for Lasso Lars
    '''
        # create the model object
    l_lars = LassoLars(alpha=0,normalize=False)

    # fit the model to our training data. 
    l_lars.fit(X_train_scaled, y_train[target])

    # predict train
    y_train['pred_l_lars'] = l_lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_l_lars_train = mean_squared_error(y_train[target], y_train.pred_l_lars)**(1/2)

    # predict validate
    y_validate['pred_l_lars'] = l_lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_l_lars_validate = mean_squared_error(y_validate[target], y_validate.pred_l_lars)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_l_lars_train, 
        "\nValidation/Out-of-Sample: ", rmse_l_lars_validate)

###########
    #plot_model(y_validate,'pred_l_lars',target,"Model for Lasso Lars")
###########

    metric_df = pd.DataFrame(
        data=[{
            "model":f"Lasso Lars",
            "rmse_train":rmse_l_lars_train,
            "rmse_validate":rmse_l_lars_validate,
            "r^2_validate":explained_variance_score(y_validate[target],y_validate['pred_l_lars'])
        }])
    return y_train,y_validate,metric_df


def model_tweedie_reg(X_train_scaled,X_validate_scaled,y_train,y_validate,target):
    '''
    takes in your modeling data frames (train and validate (test if moving on)) 
    (2 (X should be scaled), and 2 y), the target
    returns the y_predictions on those dataframes and an additional dataframe of performance
    for Tweedie
    '''
        # create the model object
    glm = TweedieRegressor(power=0,alpha=0)

    # fit the model to our training data. 
    glm.fit(X_train_scaled, y_train[target])

    # predict train
    y_train['pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_glm_train = mean_squared_error(y_train[target], y_train.pred_glm)**(1/2)

    # predict validate
    y_validate['pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_glm_validate = mean_squared_error(y_validate[target], y_validate.pred_glm)**(1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_glm_train, 
        "\nValidation/Out-of-Sample: ", rmse_glm_validate)
        
    ###########
    #plot_model(y_validate,'pred_glm',target,"Model for Tweedie Regression")
    ###########

    metric_df = pd.DataFrame(
        data=[{
            "model":f"Tweedie Reg",
            "rmse_train":rmse_glm_train,
            "rmse_validate":rmse_glm_validate,
            "r^2_validate":explained_variance_score(y_validate[target],y_validate['pred_glm'])
        }])
    return y_train,y_validate,metric_df


def plot_model(y_validate,pred_col, target,labeling):
    '''
    plots scatter plot model based on inputs
    '''

    # y_validate.head()
    plt.figure(figsize=(16,8))

    #plt.plot(y_validate[target].sample(n=1000, random_state=123), 
    #            y_validate[pred_col].sample(n=1000, random_state=123), 
    #            #alpha=.5, 
    #            #label='_nolegend_',
    #            color="gray" )
#
    plt.plot(y_validate[target].sample(n=1000, random_state=123), 
                y_validate[target].sample(n=1000, random_state=123), 
                #alpha=.5,
                # abel='_nolegend_', 
                color="blue")

    plt.annotate("The Ideal Line: Predicted = Actual", (10000, -10000), rotation=25)

    plt.scatter(y_validate[target].sample(n=1000, random_state=123), 
                y_validate[pred_col].sample(n=1000, random_state=123), 
                #alpha=.5, 
                color="green", 
                #s=100, 
                label=labeling)

    plt.legend()
    plt.xlabel(target)
    plt.ylabel("Predicted")
    plt.title("Where are predictions more extreme? More modest?")
    plt.show()
    return


def poly_final(X_train_scaled,X_test_scaled,y_train,y_test,target,n):
    ''' 
        takes in your modeling data frames (train and (test if moving on)) 
    (2 (X should be scaled), and 2 y), the target and the number of degrees to run the polynomial feature at
    returns the y_predictions on those dataframes and an additional dataframe of performance
    '''
    y_test = pd.DataFrame(y_test)
 
    y_test["mean"] = y_train["mean"].mean()

    ##want smallest error

    rmse_mean_train = mean_squared_error(y_train[target], y_train["mean"])**(1/2)
    rmse_mean_test = mean_squared_error(y_test[target], y_test["mean"])**(1/2)

    #print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_mean_train, 2), 
    #    "\nTest/Out-of-Sample: ", round(rmse_mean_test, 2))


    pf = PolynomialFeatures(degree=n)

    X_train_pf = pf.fit_transform(X_train_scaled)

    # transform X_test_scaled & X_test_scaled
    X_test_pf = pf.transform(X_test_scaled)


    # create the model object
    lm = LinearRegression()

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_pf, y_train[target])

    # evaluate: rmse
    rmse_pf_train = mean_squared_error(y_train[target], y_train[f'pred_pf{n}'])**(1/2)

    # predict test
    y_test[f'pred_pf{n}'] = lm.predict(X_test_pf)

    # evaluate: rmse
    rmse_pf_test = mean_squared_error(y_test[target], y_test[f'pred_pf{n}'])**(1/2)

    print(f"RMSE for Polynomial Model, degrees={n}\nTraining/In-Sample: ", rmse_pf_train, 
        "\nTest/Out-of-Sample: ", rmse_pf_test)
###########
    #plot_model(y_test,f'pred_pf{n}',target,f"Model {n} degree Polynomial")
###########
    metric_df = pd.DataFrame(
        data=[{
            "model":f"Polynomial - degree {n}",
            "rmse_train":rmse_pf_train,
            "rmse_test":rmse_pf_test,
            "r^2_test":explained_variance_score(y_test[target],y_test[f'pred_pf{n}'])
        }]
                            )
    return y_train,y_test,metric_df
    