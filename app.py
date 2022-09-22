import streamlit as st
import itertools
import pandas as pd 
import numpy as np 
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt 
import plotly.express as px
import matplotlib
import seaborn as sns
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy.stats import chi2_contingency,chi2
import statsmodels.api as sm 
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from scipy.stats import anderson
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from PIL import Image
image = Image.open('cover.jpg')
matplotlib.use("Agg")

class DataFrame_Loader():
    def __init__(self):
        print("Loading DataFrame")
    def read_csv(self,data):
        self.df = pd.read_csv(data)
        return self.df

class EDA_Dataframe_Analysis():
    def __init__(self):
        print("General_EDA object created")
    def show_dtypes(self,x):
    	return x.dtypes
    def show_columns(self,x):
    	return x.columns
    def Show_Missing(self,x):
    	return x.isna().sum()
    def Show_Missing1(self,x):
	    return x.isna().sum()
    def Show_Missing2(self,x):
	    return x.isna().sum()
    def show_hist(self,x):
    	return x.hist()

    def Tabulation(self,x):
	    table = pd.DataFrame(x.dtypes,columns=['dtypes'])
	    table1 =pd.DataFrame(x.columns,columns=['Names'])
	    table = table.reset_index()
	    table= table.rename(columns={'index':'Name'})
	    table['No of Missing'] = x.isnull().sum().values    
	    table['No of Uniques'] = x.nunique().values
	    table['Percent of Missing'] = ((x.isnull().sum().values)/ (x.shape[0])) *100
	    table['First Observation'] = x.loc[0].values
	    table['Second Observation'] = x.loc[1].values
	    table['Third Observation'] = x.loc[2].values
	    for name in table['Name'].value_counts().index:
	        table.loc[table['Name'] == name, 'Entropy'] = round(stats.entropy(x[name].value_counts(normalize=True), base=2),2)
	    return table

    def Numerical_variables(self,x):
	    Num_var = [var for var in x.columns if x[var].dtypes!="object"]
	    Num_var = x[Num_var]
	    return Num_var

    def categorical_variables(self,x):
	    cat_var = [var for var in x.columns if x[var].dtypes=="object"]
	    cat_var = x[cat_var]
	    return cat_var

    def plotly(self,a,x,y):
	    fig = px.scatter(a, x=x, y=y)
	    fig.update_traces(marker=dict(size=10,
	                                  line=dict(width=2,
	                                            color='DarkSlateGrey')),
	                      selector=dict(mode='markers'))
	    fig.show()

    def Show_CountPlot(self,x):
	    fig_dims = (18, 8)
	    fig, ax = plt.subplots(figsize=fig_dims)
	    return sns.countplot(x,ax=ax)

    def plotly_histogram(self,a,x,y):
	    fig = px.histogram(a, x=x, y=y)
	    fig.update_traces(marker=dict(size=10,
	                                  line=dict(width=2,
	                                            color='DarkSlateGrey')),
	                      selector=dict(mode='markers'))
	    fig.show()

    def bar_plot(self,a,x,y):
	    fig = px.bar(a, x=x, y=y)
	    fig.update_traces(marker=dict(size=10,
	                                  line=dict(width=2,
	                                            color='DarkSlateGrey')),
	                      selector=dict(mode='markers'))
	    fig.show()


    def Show_PairPlot(self,x):
	    return sns.pairplot(x)

    def Show_HeatMap(self,x):
	    f,ax = plt.subplots(figsize=(15, 15))
	    return sns.heatmap(x.corr(),annot=True,ax=ax);

    def label(self,x):
	    from sklearn.preprocessing import LabelEncoder
	    le = LabelEncoder()
	    x=le.fit_transform(x)
	    return x

    def label1(self,x):
	    from sklearn.preprocessing import LabelEncoder
	    le = LabelEncoder()
	    x=le.fit_transform(x)
	    return x
   
    def concat(self,x,y,z,axis):
    	return pd.concat([x,y,z],axis)

    def dummy(self,x):
    	return pd.get_dummies(x)

    def qqplot(self,x):
    	return sm.qqplot(x, line ='45')

    def Anderson_test(self,a):
    	return anderson(a)

    def PCA(self,x):
	    pca =PCA(n_components=8)
	    principlecomponents = pca.fit_transform(x)
	    principledf = pd.DataFrame(data = principlecomponents)
	    return principledf

    def outlier(self,x):
	    high=0
	    q1 = x.quantile(.25)
	    q3 = x.quantile(.75)
	    iqr = q3-q1
	    low = q1-1.5*iqr
	    high += q3+1.5*iqr
	    outlier = (x.loc[(x < low) | (x > high)])
	    return(outlier)

    def check_cat_relation(self,x,y,confidence_interval):
	    cross_table = pd.crosstab(x,y,margins=True)
	    stat,p,dof,expected = chi2_contingency(cross_table)
	    print("Chi_Square Value = {0}".format(stat))
	    print("P-Value = {0}".format(p))
	    alpha = 1 - confidence_interval
	    return p,alpha
	    if p > alpha:
	        print(">> Accepting Null Hypothesis <<")
	        print("There Is No Relationship Between Two Variables")
	    else:
	        print(">> Rejecting Null Hypothesis <<")
	        print("There Is A Significance Relationship Between Two Variables")
		
	    
class Attribute_Information():

    def __init__(self):
        
        print("Attribute Information object created")
        
    def Column_information(self,data):
    
        data_info = pd.DataFrame(
                                columns=['No of observation',
                                        'No of Variables',
                                        'No of Numerical Variables',
                                        'No of Factor Variables',
                                        'No of Categorical Variables',
                                        'No of Logical Variables',
                                        'No of Date Variables',
                                        'No of zero variance variables'])


        data_info.loc[0,'No of observation'] = data.shape[0]
        data_info.loc[0,'No of Variables'] = data.shape[1]
        data_info.loc[0,'No of Numerical Variables'] = data._get_numeric_data().shape[1]
        data_info.loc[0,'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
        data_info.loc[0,'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0,'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        data_info.loc[0,'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0,'No of zero variance variables'] = data.loc[:,data.apply(pd.Series.nunique)==1].shape[1]

        data_info =data_info.transpose()
        data_info.columns=['value']
        data_info['value'] = data_info['value'].astype(int)


        return data_info

    def __get_missing_values(self,data):
        
        #Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        #Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        
        #Returning missing values
        return missing_values

        
    def __iqr(self,x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self,x):
        upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self,df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i=0
        for c in  df_num.columns:
            data_info_num.loc[c,'Negative values count']= df_num[df_num[c]<0].shape[0]
            data_info_num.loc[c,'Positive values count']= df_num[df_num[c]>0].shape[0]
            data_info_num.loc[c,'Zero count']= df_num[df_num[c]==0].shape[0]
            data_info_num.loc[c,'Unique count']= len(df_num[c].unique())
            data_info_num.loc[c,'Negative Infinity count']= df_num[df_num[c]== -np.inf].shape[0]
            data_info_num.loc[c,'Positive Infinity count']= df_num[df_num[c]== np.inf].shape[0]
            data_info_num.loc[c,'Missing Percentage']= df_num[df_num[c].isnull()].shape[0]/ df_num.shape[0]
            data_info_num.loc[c,'Count of outliers']= self.__outlier_count(df_num[c])
            i = i+1
        return data_info_num
    
    def statistical_summary(self,df):
    
        df_num = df._get_numeric_data()

        data_stat_num = pd.DataFrame()

        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))],axis=1)
            data_stat_num.columns = ['count','mean','std','min','25%','50%','75%','max','10%','90%','95%']
        except:
            pass

        return data_stat_num

class Data_Base_Modelling():

    def __init__(self):
        
        print("General_EDA object created")

    def Label_Encoding(self,x):
	    category_col =[var for var in x.columns if x[var].dtypes =="object"] 
	    labelEncoder = preprocessing.LabelEncoder()
	    mapping_dict={}
	    for col in category_col:
	        x[col] = labelEncoder.fit_transform(x[col])
	        le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
	        mapping_dict[col]=le_name_mapping
	    return mapping_dict

    def IMpupter(self,x):
	    imp_mean = IterativeImputer(random_state=0)
	    x = imp_mean.fit_transform(x)
	    x = pd.DataFrame(x)
	    return x

    def Logistic_Regression(self,x_train,y_train,x_test,y_test):
    	pipeline_dt=Pipeline([('dt_classifier',LogisticRegression())])
    	pipelines = [pipeline_dt]
    	best_accuracy=0.0
    	best_classifier=0
    	best_pipeline=""
    	pipe_dict = { 0: 'Decision Tree'}
    	for pipe in pipelines:
    		pipe.fit(x_train, y_train)
    	for i,model in enumerate(pipelines):
    		return (classification_report(y_test,model.predict(x_test)))

    def Decision_Tree(self,x_train,y_train,x_test,y_test):
    	pipeline_dt=Pipeline([('dt_classifier',DecisionTreeClassifier())])
    	pipelines = [pipeline_dt]
    	best_accuracy=0.0
    	best_classifier=0
    	best_pipeline=""
    	pipe_dict = { 0: 'Decision Tree'}
    	for pipe in pipelines:
    		pipe.fit(x_train, y_train)
    	for i,model in enumerate(pipelines):
    		return (classification_report(y_test,model.predict(x_test)))

    def RandomForest(self,x_train,y_train,x_test,y_test):
    	pipeline_dt=Pipeline([('dt_classifier',RandomForestClassifier())])
    	pipelines = [pipeline_dt]
    	best_accuracy=0.0
    	best_classifier=0
    	best_pipeline=""
    	pipe_dict = { 0: 'Decision Tree'}
    	for pipe in pipelines:
    		pipe.fit(x_train, y_train)
    	for i,model in enumerate(pipelines):
    		return (classification_report(y_test,model.predict(x_test)))

    def naive_bayes(self,x_train,y_train,x_test,y_test):
    	pipeline_dt=Pipeline([('dt_classifier',GaussianNB())])
    	pipelines = [pipeline_dt]
    	best_accuracy=0.0
    	best_classifier=0
    	best_pipeline=""
    	pipe_dict = { 0: 'Decision Tree'}
    	for pipe in pipelines:
    		pipe.fit(x_train, y_train)
    	for i,model in enumerate(pipelines):
    		return (classification_report(y_test,model.predict(x_test)))

    def XGb_classifier(self,x_train,y_train,x_test,y_test):
    	pipeline_dt=Pipeline([('dt_classifier',XGBClassifier())])
    	pipelines = [pipeline_dt]
    	best_accuracy=0.0
    	best_classifier=0
    	best_pipeline=""
    	pipe_dict = { 0: 'Decision Tree'}
    	for pipe in pipelines:
    		pipe.fit(x_train, y_train)
    	for i,model in enumerate(pipelines):
    		return (classification_report(y_test,model.predict(x_test)))

st.image(image, use_column_width=True)   
def main():
	st.title("Bizlog Machine Learining App")
	
	st.info("VALUE ADDED SERVICES  * *")
	
	activities = ["General EDA"]	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'General EDA':
		st.subheader("Exploratory Data Analysis")

		data = st.file_uploader("Upload a Dataset", type=["csv"])
		if data is not None:
			df = load.read_csv(data)
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")
			
			if st.checkbox("Show dtypes"):
				st.write(dataframe.show_dtypes(df))

			if st.checkbox("Show Columns"):
				st.write(dataframe.show_columns(df))

			if st.checkbox("Show Missing"):
				st.write(dataframe.Show_Missing1(df))

			if st.checkbox("column information"):
				st.write(info.Column_information(df))

			if st.checkbox("Aggregation Tabulation"):
				st.write(dataframe.Tabulation(df))

			if st.checkbox("Num Count Summary"):
				st.write(info.num_count_summary(df))

			if st.checkbox("Statistical Summary"):
				st.write(info.statistical_summary(df))		
                
			if st.checkbox("Show Selected Columns"):
				selected_columns = st.multiselect("Select Columns",dataframe.show_columns(df))
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Numerical Variables"):
				num_df = dataframe.Numerical_variables(df)
				numer_df=pd.DataFrame(num_df)                
				st.dataframe(numer_df)

			if st.checkbox("Categorical Variables"):
				new_df = dataframe.categorical_variables(df)
				catego_df=pd.DataFrame(new_df)                
				st.dataframe(catego_df)
               
			all_columns_names = dataframe.show_columns(df)
			all_columns_names1 = dataframe.show_columns(df)            
			selected_columns_names = st.selectbox("Select Column 1 For Cross Tabultion",all_columns_names)
			selected_columns_names1 = st.selectbox("Select Column 2 For Cross Tabultion",all_columns_names1)
			if st.button("Generate Cross Tab"):
				st.dataframe(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))

			if st.button("Generate Bar Graph PLOT"):
				st.bar_chart(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))

			st.subheader("UNIVARIATE ANALYSIS")
			
			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Column for Histogram ",all_columns_names)
			if st.checkbox("Show Histogram for Selected variable"):
				st.write(dataframe.show_hist(df[selected_columns_names]))
				st.pyplot()	

			if st.button("Generate Bar Graph for single var "):
				st.bar_chart(dataframe.show_hist(df[selected_columns_names]))	


			all_columns_names = dataframe.show_columns(df)         
			selected_columns_names = st.selectbox("Select Columns CountPlot ",all_columns_names)
			if st.checkbox("Show CountPlot for Selected variable"):
				st.write(dataframe.Show_CountPlot(df[selected_columns_names]))
				st.pyplot()

			st.subheader("BIVARIATE ANALYSIS")

			Scatter1 = dataframe.show_columns(df)
			Scatter2 = dataframe.show_columns(df)            
			Scatter11 = st.selectbox("Select Column 1 For Scatter Plot (Numerical Columns)",Scatter1)
			Scatter22 = st.selectbox("Select Column 2 For Scatter Plot (Numerical Columns)",Scatter2)
			if st.button("Generate PLOTLY Scatter PLOT"):
				st.pyplot(dataframe.plotly(df,df[Scatter11],df[Scatter22]))
                
			bar1 = dataframe.show_columns(df)
			bar2 = dataframe.show_columns(df)            
			bar11 = st.selectbox("Select Column 1 For Bar Plot ",bar1)
			bar22 = st.selectbox("Select Column 2 For Bar Plot ",bar2)
			if st.button("Generate PLOTLY histogram PLOT"):
				st.pyplot(dataframe.plotly_histogram(df,df[bar11],df[bar22]))   
             
#			if st.button("Generate Bar Graph PLOT"):
#				st.dataframe(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))


if __name__ == '__main__':
	load = DataFrame_Loader()
	dataframe = EDA_Dataframe_Analysis()
	info = Attribute_Information()
	model = Data_Base_Modelling()
	main()
