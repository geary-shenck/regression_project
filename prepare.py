
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import acquire
from sklearn.preprocessing import MinMaxScaler

def split_function(df,target):
    ''' 
    splits a dataframe and returns train, test, and validate dataframes
    '''
    train,test = train_test_split(df,test_size= .2, random_state=123,stratify = df[target])
    train,validate = train_test_split(train,test_size= .25, random_state=123,stratify = train[target])

    print(f"prepared df shape: {df.shape}")
    print(f"train shape: {train.shape}")
    print(f"validate shape: {validate.shape}")
    print(f"test shape: {test.shape}")

    return train, test, validate

def split_continous(df):
    ''' 
    splits a dataframe and returns train, test, and validate dataframes
    '''
    train,test = train_test_split(df,test_size= .2, random_state=123)
    train,validate = train_test_split(train,test_size= .25, random_state=123)

    print(f"prepared df shape: {df.shape}")
    print(f"train shape: {train.shape}")
    print(f"validate shape: {validate.shape}")
    print(f"test shape: {test.shape}")

    return train, test, validate



#import warnings
#warnings.filterwarnings("ignore")


def prep_zillow(df0):
    ''' 
    input the dataframe from the aquire.get_zillow_sing_fam
    replaces any whitespace with nans
    renames featues
    returns the prepared dataframe
    '''
# get the data and review the data

    #df0 = acquire.get_zillow_single_fam()
    ## datatypes look good

    ## start the clean up
    ## remove cells with whitespace and replace with NaN in a new working dataframe
    ## rename to use more common terminology/legibility

    df = df0.replace(r'^\s*$', np.nan, regex=True)

    df.fips = np.where(df.fips == 6059,"Orange Cnty", 
                        np.where(df.fips == 6111,"Ventura Cnty", "Los Angeles Cnty"))

    df = df.rename(columns = {"bedroomcnt":"bedrooms",
                                "bathroomcnt":"bathrooms",
                            "calculatedfinishedsquarefeet":"area",
                            "taxvaluedollarcnt":"tax assessment",
                            "lotsizesquarefeet":"lot sqft",
                            "regionidzip":"zip",
                            "yearbuilt":"year built"
                            })

    df.zip = df[df.zip > 0]["zip"].astype(int).astype(object)

  
    return df


