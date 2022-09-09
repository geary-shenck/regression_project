import pandas as pd
import numpy as np

import prepare

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Generic splitting function for continuous target.

def split_tvt_continuous(df,target):
    """
    Takes in a df
    Returns train, validate, and test DataFrames
    """
    # Create train_validate and test datasets
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    # Create train and validate datsets
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    X_train, y_train, X_validate, y_validate, X_test, y_test = X_and_y(train,validate,test,target)

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test


def split_tvt_stratify(df, target):
    """
    takes in a dataframe, splits it into 60, 20, 20, 
    and seperates out the x variables and y (target) as new df/series
    """
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123,stratify = train[target])

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123,stratify = train[target])

    X_train, y_train, X_validate, y_validate, X_test, y_test = X_and_y(train,validate,test,target)

    print(f"train -> {train.shape}")
    print(f"validate -> {validate.shape}")
    print(f"test -> {test.shape}")

    return X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test

def X_and_y(train,validate,test,target):
    '''
    takes in 4 variables (3 df and 1 string)
    just splits into X and y groups, nothing fancy
    returns 6 variables
    '''
        # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]

    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]

    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def get_numeric_X_cols(X_train, object_cols):
    """
    RUN THIS AFTER OBJECT COLUMNS
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects.
    """
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]

    return numeric_cols


def get_object_cols(df):
    """
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names.
    """
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()

    return object_cols


def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    """
    takes in the train data sets (3) and numeric column list,
    and fits a min-max scaler to the first dataframe and transforms all 3
    returns 3 dataframes with the same scaled values.
    """
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    # scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train.
    #
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=numeric_cols).set_index(
        [X_train.index.values]
    )

    X_validate_scaled = pd.DataFrame(
        X_validate_scaled_array, columns=numeric_cols
    ).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=numeric_cols).set_index(
        [X_test.index.values]
    )

    #scaler = RobustScaler(with_centering=True,quantile_range=(10,90),unit_variance=True)
    #
    #X_train_scaled = scaler.fit_transform(X_train) ##only use fit_transform for training, after that use transform (equations are created)
    #X_validate_scaled = scaler.transform(X_validate)
    #X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_validate_scaled, X_test_scaled



def wrangle_zillow(df,outlier_list,pred):
    ''' 
    gets info from prepare
    removes outliers of numerical columns through the tukey method (k=1.5)
    drops the bathroom labeld as 1.75 (i'm not aware of that being an official measurement)
    adds column that categorizes years into decades
    splits into 60%,20%,20%
    returns the the modified dataframe, and the splits dataframe
    '''

    tukey_k = 1.5

    for col in outlier_list:
       IQR = (df[col].describe()["75%"] - df[col].describe()["25%"])
       df = df[(df[col] < (df[col].describe()["75%"] + (IQR * tukey_k))) & \
            (df[col] > (df[col].describe()["25%"] - (IQR * tukey_k)))]

    df = df[df.bathrooms != 1.75]

    df["decade built"] = (df["year built"]//10 *10).astype(int).astype(str) + "s"

    X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test = split_tvt_continuous(df,pred)

    return df,X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test


def wrangle_zillow_orig(df):
    ''' 
    gets info from prepare
    removes outliers of numerical columns through the tukey method (k=1.5)
    drops the bathroom labeld as 1.75 (i'm not aware of that being an official measurement)
    adds column that categorizes years into decades
    splits into 60%,20%,20%
    returns the the modified dataframe, and the splits dataframe
    '''

    tukey_k = 1.5

    for col in df.drop(columns=["fips","zip"]).columns:
       IQR = (df[col].describe()["75%"] - df[col].describe()["25%"])
       df = df[(df[col] < (df[col].describe()["75%"] + (IQR * tukey_k))) & \
            (df[col] > (df[col].describe()["25%"] - (IQR * tukey_k)))]

    df = df[df.bathrooms != 1.75]

    ## per IRC
    df = df[df.area > 320]

    df["decade built"] = (df["year built"]//10 *10).astype(int).astype(str) + "s"

    df["openess"] = (df["area"]/(df["bedrooms"]+df["bathrooms"]))

    df["yard ratio"] = (df["lot sqft"] / df["area"])

    ## drop n/a and review
    df = df.dropna()

    train, test, validate = prepare.split_continous(df)

    return df,train,validate,test

def wrangle_example(path):
    df = pd.read_csv(path, sep=";")

    # drop any nulls
    df = df[~df.isnull()]

    # get object column names
    object_cols = get_object_cols(df)

    # create dummy vars
    df = create_dummies(df, object_cols)

    # split data
    X_train, y_train, X_validate, y_validate, X_test, y_test, train, validate, test= split_tvt_stratify(
        df, "target"
    )

    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(
        X_train, X_validate, X_test, numeric_cols
    )

    return (
        df,
        X_train,
        X_train_scaled,
        y_train,
        X_validate_scaled,
        y_validate,
        X_test_scaled,
        y_test,
    )

def rfe(predictors_x,target_y,n_features):
    ''' 
    takes in the predictors (X) (predictors_x), the target (y) (target_y), and the number of features to select (k) 
    returns the names of the top k selected features based on the Recursive Feature Elimination class. and a ranked df
    '''

    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import RFE

    model = LinearRegression()
    rfe = RFE(model,n_features_to_select=n_features)
    rfe.fit(predictors_x,target_y)

    print(pd.DataFrame({"rfe_ranking":rfe.ranking_},index=predictors_x.columns).sort_values("rfe_ranking")[:n_features])
    X_train_transformed = pd.DataFrame(rfe.transform(predictors_x),columns=predictors_x.columns[rfe.get_support()],index=predictors_x.index)
    X_train_transformed.head(3)

    var_ranks = rfe.ranking_
    var_names = predictors_x.columns.tolist()

    rfe_ranked = pd.DataFrame({'Var': var_names, 'Rank': var_ranks}).sort_values("Rank")
    
    return rfe_ranked

def select_kbest(predictors_x,target_y,n_features):
    ''' 
    takes in the predictors (X) (X_train), the target (y) (y_train), and the number of features to select (k) 
    returns the names of the top k selected features based on the SelectKBest class.
    '''

    from sklearn.feature_selection import SelectKBest, f_regression

    kbest = SelectKBest(f_regression,k=n_features) ## k returns the best amount of features as entered
    _ = kbest.fit(predictors_x,target_y)

    kbest.pvalues_
    kbest.scores_

    kbest_resutls = pd.DataFrame(dict(p=kbest.pvalues_,f=kbest.scores_))
    kbest_resutls
    predictors_x.columns[kbest.get_support()]

    X_train_transformed = pd.DataFrame(kbest.transform(predictors_x),columns=predictors_x.columns[kbest.get_support()],index=predictors_x.index)
    X_train_transformed.head(3)
    print(X_train_transformed.columns.tolist())
    return(X_train_transformed)

#train["House sqft quantiled"] = pd.cut(train.area,4,labels=["<25% percentile area","25%~50% percentile area","50%~75% percentile area",">75% percentile area"])
#train["Yard ratio quantiled"] = pd.cut(train["yard ratio"],4,labels=["<25% percentile area","25%~50% percentile area","50%~75% percentile area",">75% percentile area"])
#temp1 = train.drop(columns=["area","year built","fips","openess","lot sqft","yard ratio"]).groupby(["zip","House sqft quantiled","Yard ratio quantiled","bedrooms","bathrooms"]).agg(["min","max"]).reset_index()
#temp1 = temp1.dropna()
#temp1["range"] = temp1["tax assessment"]["max"] - temp1["tax assessment"]["min"] 
#temp1["tax max"] = temp1["tax assessment"]["max"]
#temp1["tax min"] = temp1["tax assessment"]["min"]
#temp1.drop(columns="tax assessment",inplace=True)
#temp1