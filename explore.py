import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


#univariate
#    continous
#        - histogram
#        - boxplot
#        - displot
#    discrete
#        - countplot
#
#
#bi-/mulit-variate
#    continous with continous
#        - scatter
#        - line
#        - pair
#        - heat map
#        - relplot
#    discrete with continous
#        - violin
#        - catplot
#        - sunburst?
#        - boxplot
#        - swarmplot
#        - striplot
#    discrete with discrete
#        - heatmap
#        -
#        -





def univariate_explore(df):
    ''' 
    takes in dataframe, and puts out a histogram of each category, binning relatively low
    '''
    plt.figure(figsize=(25, 5))
    for i, col in enumerate(df.columns.tolist()): # List of columns
        plot_number = i + 1 # i starts at 0, but plot nos should start at 1
        plt.subplot(1,len(df.columns.tolist()), plot_number) # Create subplot.
        plt.title(col) # Title with column name.
        df[col].hist(bins=10) # Display histogram for column.
        plt.grid(False) # Hide gridlines.







def plot_variable_pairs(df,num_vars):
    ''' 
    that accepts a dataframe and numerical variables as input and plots all of the 
    pairwise relationships along with the regression line for each pair.
    '''

    l=1
    plt.figure(figsize=(25, 12))
    for col1 in num_vars:
        for col2 in num_vars:
            if not num_vars.index(col2) >= num_vars.index(col1):

                x=df[col1]
                y=df[col2]

                plt.subplot(len(num_vars)-2,len(num_vars)-1,l)
                l +=1

                plt.plot(x, y, "o",color="grey")

                m,b = np.polyfit(x,y,1)
                plt.plot(x,m*x+b,label=f"regression line - f(x)={round(m,0)}x+{round(b,0)}")
                plt.legend()
                plt.title(f"{col2} value by {col1} value")
    plt.show()

    return






def plot_categorical_and_continuous_vars(df,num_vars,cat_vars):
    ''' 
    input(dataframe, list of numerical, list of categorical)
    accepts your dataframe and the name of the columns that hold 
    the continuous and categorical features and outputs 3 different plots 
    for visualizing a categorical variable and a continuous variable.
    '''

    plt.figure(figsize=(30, 20))

    i = 0
    l = 0
    for col1 in num_vars:
        i += 1
        j = 0

        for col2 in cat_vars:
            j += 1
            l += 1

            plt.subplot(len(num_vars),len(cat_vars),l)
            plot_order = df[col2].sort_values(ascending=False).unique()
            sns.boxplot(    x=col2, 
                            y=col1, 
                            data=df, 
                            order = plot_order,
                            color ="grey",
                            notch=True)
            plt.axhline(df[col1].mean(),label=f"mean line - {round(df[col1].mean(),0)}")
            plt.legend()
            plt.title(f"value of {col1} sorted by {col2}")
    plt.show()

    #plt.figure(figsize=(25, 15))
    #i = 0
    #l = 0
    #for col1 in num_vars:
    #    i += 1
    #    j = 0
#
    #    for col2 in cat_vars:
    #        j += 1
    #        l += 1
#
    #        plt.subplot(len(num_vars),len(cat_vars),l)
    #        plot_order = df[col2].sort_values(ascending=False).unique()
    #        sns.stripplot(x=col2, y=col1, data=df, order = plot_order)
    #        plt.title(f"value of {col1} sorted by {col2}")
    #plt.show()
#
    #plt.figure(figsize=(25, 15))
    #i = 0
    #l = 0
    #for col1 in num_vars:
    #    i += 1
    #    j = 0
#
    #    for col2 in cat_vars:
    #        j += 1
    #        l += 1
#
    #        plt.subplot(len(num_vars),len(cat_vars),l)
    #        plot_order = df[col2].sort_values(ascending=False).unique()
    #        sns.violinplot(x=col2, y=col1, data=df, order = plot_order)
    #        plt.title(f"value of {col1} sorted by {col2}")
    return






def heatmap_corr(train):
    ''' 
    takes in dataframe and returns a heatmap based on correlation
    '''
    plt.figure(figsize=(12, 6))
    kwargs = {'alpha':1,
            'linewidth':5, 
            'linestyle':'--',
            'linecolor':'white'}

    sns.heatmap(    train.corr(),
                    #map="YlGnBu", 
                    cmap="Spectral",
                    mask=(np.triu(np.ones_like(train.corr(),dtype=bool))),
                    annot=True,
                    vmin=-1, 
                    vmax=1, 
                    #annot=True,
                    **kwargs
                    )
    plt.title("Is there features that correlate higher than others?")
    plt.show()


def chi2_for_two(col1,col2):
    ''' 
    only pass in two series, df cols
    puts them in a crosstab, runs a chi2 test and returns results from test
    '''
    df1 = pd.crosstab(col1,col2)

    alpha = .05

    chi2, p, degf, expected = stats.chi2_contingency(df1)

    H0 = (f"{df1.index.name} is independant of {df1.columns.name}")
    H1 = (f"{df1.index.name} is not independant of being {df1.columns.name}")

    #print('Observed')
    #print(df1.values)
    #print('---\nExpected')
    #print(expected)
    print(f'---\nchi^2 = {chi2:.4f}, p = {p:.5f}, degf = {degf}')
    if p>alpha:
        print(f"due to p={p:.5f} > α={alpha} we fail to reject our null hypothesis\n({H0})")
    else:
        print(f"due to p = {p:.5f} < α = {alpha} we reject our null hypothesis\n(", '\u0336'.join(H0) + '\u0336' ,")")

    xnoise, ynoise = np.random.random(len(col1))/3, np.random.random(len(col2))/3 # The noise is in the range 0 to 0.5

    x = col1
    y = col2
    m,b = np.polyfit(x,y,1)

    plt.figure(figsize=(12, 6))

    plt.scatter(col1+xnoise,col2+ynoise,alpha=.1)
    plt.plot(x,m*x+b,label=f"regression line - f(x)={round(m,0)}x+{round(b,0)}",color="red")

    plt.ylabel(f"{col2.name}")
    plt.xlabel(f"{col1.name}")
    plt.title(f"Relation of {col1.name} to {col2.name}")
    plt.legend()
    plt.show()
    return





def cat_and_num_explore_plot(train,cat,num):
    ''' 
    takes in dataframe and string values of the categorical and numerical columns 
    to run a means TTest on and plot a visualization
    '''

    alpha = .05

    for cat_1 in train[cat].unique():
        for cat_2 in train[cat].unique():
            if not train[cat].unique().tolist().index(cat_2) >= train[cat].unique().tolist().index(cat_1):
                H0 = f"{num} of {cat}{cat_1} has identical average values to {num} of other {cat}{cat_2}"
                Ha = f"{num} of {cat} is not equal to {num} of other {cat}"
                print("-----------------------------")
                #compare variances to know how to run the test
                stat,pval = stats.levene(train[train[cat] == cat_1][num],train[train[cat] == cat_2][num])
                stat,pval
                if pval > 0.05:
                    equal_var_flag = True
                    print(f"we can accept that there are equal variance in these two groups with {round(pval,2)} certainty Flag=T",'stat=%.5f, p=%.5f' % (stat,pval))
                else:
                    equal_var_flag = False
                    print(f"we can reject that there are equal variance in these two groups with {round((1-pval),2)} certainty Flag=F",'stat=%.5f, p=%.5f' % (stat,pval))


                t, p = stats.ttest_ind( train[train[cat] == cat_1][num], train[train[cat] == cat_2][num], equal_var = equal_var_flag )

                if p > alpha:
                    print("\n We fail to reject the null hypothesis (",(H0) , ")",'t=%.5f, p=%.5f' % (t,p))
                else:
                    print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")",'t=%.5f, p=%.5f' % (t,p))


    plt.figure(figsize=(12,6))
    plt.title(f"Is the density of {num} different for {cat}")


    plt.ylabel(f"Density of {num}")
    plt.yticks([],[])

    colorlist=['c', 'm', 'y', 'k']
    linestyle_list = ['solid', 'dotted','dashed','dashdot']

    for i in enumerate(train[cat].unique()):

        sns.kdeplot(train[train[cat] == i[1]][num],
                    label=f"{i[1]}",
                    color=colorlist[i[0]])
        plt.axvline(train[train[cat] == i[1]][num].mean(),
                    color=colorlist[i[0]],
                    ls=linestyle_list[i[0]],
                    label=f"mean for {i[1]}")


    #import plotly.express as px
#
    #fig = px.histogram(train,
    #            x=num,
    #            barmode="overlay",
    #            #histfunc="avg",
    #            color=cat,
    #            title=f"Histogram showing the density of {num} for {cat}",
    #            color_discrete_map = {0:'red',1:'blue',2:'goldenrod',3:'green',4:'magenta',5:'indigo',6:'purple'})
    #fig.show()

    plt.xlabel(f"{cat}")
    plt.legend()
    plt.show()
    return



#xnoise, ynoise = np.random.random(len(train))/4, np.random.random(len(train))/4 # The noise is in the range 0 to 0.5
#
#plt.scatter(train.bathrooms+xnoise,train.bedrooms+ynoise,alpha=.5)
#plt.ylabel("Bedrooms"),plt.xlabel("Bathrooms"),plt.title("Relation of Bedrooms to Bathrooms")
#




def pearsonr_corr_explore_plot(train,num1,num2):
    ''' 
    takes in a dataframe and two series and runs a pearsonr test to determine if there's a correlation between the features
    '''
    ## putting tax value and taxes yearly into a pearsonr and then graphing it for a visual result as a result
    ## of the heat map above highlighting a good possibility of a relation


    H0 = f"That the distributions underlying the samples of {num1} and {num2} are unrelated"
    Ha = f"That the distributions underlying the samples of {num2} and {num2} are related"
    alpha = .05

    r, p = stats.pearsonr(train[num1],train[num2])

    plt.figure(figsize=(10,6))
    plt.scatter( train[num1], train[num2])
    b, a = np.polyfit(train[num1], train[num2], deg=1)
    plt.plot(train[num1], a + b * train[num1], color="k", lw=2.5,label="Regression Line")
    plt.xlabel(num1)
    plt.ylabel(num2)
    plt.xticks(np.arange(1_100_001,step=100_000), ["0","100k","200k","300k","400k","500k","600k","700k","800k","900k","1,000K","1,100K"],
       rotation=20)

    plt.title(f'Is the correlation value indicative? (r={round(r,1)})', size=16)
    plt.legend()
    plt.show()

    print('r =', r)

    if p > alpha:
        print("\n We fail to reject the null hypothesis (",(H0) , ")",'p=%.5f' % (p))
    else:
        print("\n We reject the null Hypothesis (", '\u0336'.join(H0) + '\u0336' ,")", 'p=%.5f' % (p))

    
    #xnoise, ynoise = np.random.random(len(train))/3, np.random.random(len(train))/3 # The noise is in the range 0 to 0.5
    #plt.figure(figsize=(12, 6)),plt.scatter(train.bathrooms+xnoise,train.bedrooms+ynoise,alpha=.1),plt.ylabel("Bedrooms"),plt.xlabel("Bathrooms"),plt.title("Relation of Bedrooms to Bathrooms")
    #plt.show()
    return


#import plotly.express as px
#
#fig = px.histogram(train[["zip","fips","bathrooms","bedrooms","tax assessment"]],
#                x="fips",
#                y="tax assessment",
#                barmode="group",
#                histfunc="avg",
#                facet_row="bedrooms",
#                facet_col="bathrooms",
#                facet_row_spacing=0.2,
#                color="zip")
#fig.show()