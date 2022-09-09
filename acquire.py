import env
import pandas as pd
import os

def read_googlesheet(sheet_url):
    '''
   takes in info for google sheets and exports it into a dataframe
    '''
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)
    return df

def get_zillow():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow_mod.csv"):
        df = pd.read_csv("zillow_mod.csv", index_col = 0)
    else:
        sql_query = """
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt,taxamount,fips,regionidzip,propertycountylandusecode
                        FROM properties_2017 -- `2,858,627`, "2,985,217"
                            LEFT JOIN airconditioningtype
                                USING (airconditioningtypeid)
                            LEFT JOIN architecturalstyletype
                                USING (architecturalstyletypeid)
                            LEFT JOIN buildingclasstype
                                USING (buildingclasstypeid)
                            LEFT JOIN heatingorsystemtype
                                USING (heatingorsystemtypeid)
                            LEFT JOIN propertylandusetype
                                USING (propertylandusetypeid)
                            LEFT JOIN storytype
                                USING (storytypeid)
                            LEFT JOIN typeconstructiontype
                                USING (typeconstructiontypeid)
                            WHERE properties_2017.propertylandusetypeid = 261
                    ;
                    """
        df = pd.read_sql(sql_query,env.get_db_url("zillow"))
        df.to_csv("zillow_mod.csv")
    return df


def get_zillow_all():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow.csv"):
        df = pd.read_csv("zillow.csv", index_col = 0)
    else:
        sql_query = """
                    SELECT *
                        FROM properties_2017 -- `2,858,627`, "2,985,217"
                            LEFT JOIN airconditioningtype
                                USING (airconditioningtypeid)
                            LEFT JOIN architecturalstyletype
                                USING (architecturalstyletypeid)
                            LEFT JOIN buildingclasstype
                                USING (buildingclasstypeid)
                            LEFT JOIN heatingorsystemtype
                                USING (heatingorsystemtypeid)
                            LEFT JOIN propertylandusetype
                                USING (propertylandusetypeid)
                            LEFT JOIN storytype
                                USING (storytypeid)
                            LEFT JOIN typeconstructiontype
                                USING (typeconstructiontypeid)
                            
                    ;
                    """
        df = pd.read_sql(sql_query,env.get_db_url("zillow"))
        df.to_csv("zillow.csv")
    return df

def get_zillow_single_fam_2017():
    ''' 
    checks for filename (iris_df.csv) in directory and returns that if found
    else it queries for a new one and saves it
    '''
    if os.path.isfile("zillow_single_fam_sold_2017.csv"):
        df = pd.read_csv("zillow_single_fam_sold_2017.csv", index_col = 0)
    else:
        sql_query = """
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet,taxvaluedollarcnt,yearbuilt,fips,lotsizesquarefeet,regionidzip
                        FROM properties_2017 -- `2,858,627`, "2,985,217"
                            LEFT JOIN predictions_2017
                                USING (parcelid)
							WHERE propertylandusetypeid = 261
                            AND transactiondate BETWEEN '2017-01-01' AND '2017-12-31'
                    ;
                    """
        df = pd.read_sql(sql_query,env.get_db_url("zillow"))
        df.to_csv("zillow_single_fam_sold_2017.csv")
    return df
