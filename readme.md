# Regression Depression

## Table of Contents

## I. Project Description
Determine which features are key in the development of a model that would help improve my "New Employers" model. 
 
#### 1. Objective and Goals : 
Determine drivers of Tax Assessment for single Family homes in certain FIPS to develop a Model that will predict the Tax Assessment with a RMSE beating baseline.

#### 2. Dataset : Zillow (Codeup Database) 

- ##### Description: 
	- properties_2017 - Data provided by Kaggle for home in 2017
  - predictions_2017 - Data that contains if a house was sold in 2017
  - propertylandusetype - Data that contains the description used to determine what the property type is (Single Family Homes)

#### 3. Initial Questions:
- Are the features of Tax Assessment and Year Built related?
- Does each physical location (FIPS) have a signficant difference in means?
- Does number of bathrooms have a relation with the number of bedrooms?
- Does Tax Assessment change depending on the SQFT of a house?


 
## II. DATA CONTEXT
 
#### 1. DATA DICTIONARY:
The final DataFrame used to explore the data for this project contains the following variables (columns).  The variables, along with their data types, are defined below in alphabetical order:
 

| Feature   | Definition      | Data Type |
|:-------------------------------|:--------------------------------------------------:|:---------:|
| tax assessment  | total tax assessed value of the property | float64   |
| bathrooms  | number of bathrooms and half-bathrooms in home | float64 |
| bedrooms   | number of bedrooms in the home  | float64   |
| fips | number/name of county as assigned by state_county_code | object  |
| zip  | numeric variable representing county location | object |
| area | square footage of the house | float64  |
| lot sqft  | sqft of the land the house resides on  | float64  |
| openess | engineered feature that is area of the house divided by sum of bedrooms and bathrooms | float64   |
| yard ratio | engienered feature that is the ratio of the lot sqft to the house area | float64 |
| year built | year house constructed | float64     |




## III. PROJECT PLAN - USING THE DATA SCIENCE PIPELINE:
The following outlines the process taken through the Data Science Pipeline to complete this project. 
 
Plan➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver
 
#### 1. PLAN
- Review project expectations
- Create questions related to the project
- Create questions related to the data
- Create a plan for completing the project using the data science pipeline
- Create a data dictionary to define variables and data context
 
#### 2. ACQUIRE
- Create env file with log-in credentials
- Store env file in .gitignore to ensure the security of sensitive data
- Create wrangle.py module
- Store functions needed to acquire the Zillow dataset from mySQL
- Ensure the functions to run are inside the acquire.py document
- Import functions from aquire.py module
 
#### 3. PREPARE / WRANGLE
- Create prepare functions in the prepare.py module
- Store functions needed to prepare the Zillow data such as:
   - Cleaning Function: to clean data for exploration
      - Remove whitespace, change FIPS from numerical to string to represent the County
   - Renaming the features to more understandable features
- Import functions from prepare.py module
- Create wrangle functions in the wrangle.py module
- Import functions from the wrangle module
- Remove outliers using Tukey
- Drop remaining nulls
- Add a binned version of years (decades)
- Feature engineer "openess" this is the sqft of the house divided by the bathrooms + bedrooms, giving a rough idea of how "open" a house may be
- Feature engineer "yard ratio" this is the sqft of the lot divided by the sqft of the house giving you a ratio for how much yard there may be
- Split data into train, validate, and test samples (60%, 20%, 20%)
 
#### 4. EXPLORE
- Create explore functions in the explore.py module
- Store functions needed to explore the Zillow data:
  - Discrete and Discrete data
  - Continous and Continous data
  - Contious and Discrete data
- Import functions from the explore module
- Answer key questions about hypotheses and find drivers of Tax assessment
  - Run at least two statistical tests
  - Document findings
- Create visualizations with the intent to discover variable relationships
  - Identify variables related to tax assessment
  - Identify any potential data integrity issues
- Summarize conclusions, provide clear answers, and summarize takeaways
  - Explain plan of action as deduced from work to this point
 
#### 5. MODEL & EVALUATE
- Create model functions in the modeling.py module
- Import model functions in the modeling.py module
  - Scale relavant data
  - Create and establish baseline on target
  - Create and establish predictions from different models (train and fit on scaled data, predicting target), chart a relationship for a visual reference
- Compare evaluation metrics across models
- Evaluate best performing models using relation of results from train and validate set
- Choose best performing validation model for use on test set, and test (out of sample)
- Summarize performance
- Interpret and document findings
 
#### 6. DELIVERY
- Prepare five-minute presentation using Jupyter Notebook
- Include an introduction of project and goals
- Provide an executive summary of findings, key takeaways, and recommendations
- Create walkthrough of analysis 
  - Visualize relationships
  - Document takeaways
  - Explicitly define questions asked during the initial analysis
- Provide final takeaways, recommend a course of action, and next steps
- Be prepared to answer questions following the presentation

 
 
## IV. PROJECT MODULES:
- Python Module Files - provide reproducible code for acquiring,  preparing, exploring, & modeling the data.
   - acquire.py - used to store the function needed to run an acquisition of the data from the Codeup database
   - prepare.py - used to do some initial cleaning and preparation of the file for exploration
   - wrangle.py - used to do additional cleaning, removing outliers, and splitting the data, also used to scale the data before modeling
   - explore.py - used to do exploration of the features to find the main relations for Tax assessment
   - modeling.py - used to create the models and chart the relationship of the predicted to the actual values
 
  
## V. PROJECT REPRODUCTION:
### Steps to Reproduce
 
- You will need an env.py file that contains the hostname, username, and password of the mySQL database that contains the zillow database
- Store that env file locally in the repository
- Make .gitignore and confirm .gitignore is hiding your env.py file
- Clone my repo (including the imports)
- Import python libraries:  pandas, matplotlib, seaborn, numpy, plotly, scipy, and sklearn
- Follow steps as outlined in the README.md. and final_project.ipynb
- Run final_report.ipynb to view the final product

## VI. Conclusion
### Key Takeaway
- Area, lot size, bedrooms, bathrooms, location are all very important aspects that determine Tax Assessment as well as their derivative features (Openness and Yard Ratio)
- More data would alway be beneficial. Using a version of classification to determine key characteristics (comparables and Mill rate / levies / local taxes), and then utilizing a regression model on the similar properties would give a much lower RMSE

### Next Steps
- Determine and Collect additional data that are key factors in determining Tax Assessment
- Utilize either a combination of models or a new style of model to more accurately predict Tax Assessment

### Recommendation
- Let Maggie know if she lost an email that there are methods to "find" said email to ensure data integrity rather than using an external source (assumption of match).
- Determine how local areas calculate tax rate and use a model that is structured similarly.
- Do not recommend using this model to help assist in predicting Tax Assessment

