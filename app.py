from email.errors import MissingHeaderBodySeparatorDefect
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
from sklearn import preprocessing
from scipy.stats import chi2_contingency,chi2
import statsmodels.api as sm
import json
from typing import List
from pydantic import BaseModel
import datetime as dt
#### load data from db


import os
import psutil
import requests
import threading
import random
import datetime
from threading import Timer
import schedule
import time
import sys
import requests
import json

class DatabaseLoad():
    def __init__(self):
        print("Loading DatabaseLoad")
class DatabaseLoad1():
    def __init__(self):
        print("Loading DatabaseLoad")
        data = {
            "startDate": "2022-08-25 00:00:00",
            "endDate": "2022-08-31 23:59:59",
            #"startDate": st.x,
            #"endDate": st.y,

            "hubId": [
                "36317051",
                "3486275",
                "188645625",
                "176602226",
                "5067",
                "179307540",
                "bhiw1653314900933TZ986eb68f946e4a75b6614536b3dce3a9",
                "176603052",
                "127353475",
                "bika1658475073496TZ4f6f4201aa7e4835b6f33d418231ae17",
                "blr1658330763471TZbe6c98f3ae3c495c82198a53cb86c7d2",
                "4176517",
                "7524",
                "135513436",
                "122706711",
                "darb1658472775515TZ49791a45ccc84fa2bb4554890051fe2e",
                "176603459",
                "202514",
                "3484697",
                "gaya1658472638964TZ52e5a820d655441686a0bd3ad8fbea59",
                "564908",
                "179425548",
                "934742",
                "125806290",
                "176603503",
                "1lUeT6ivx3d1DXPKW74SdKyLQrM=",
                "985865",
                "185584",
                "68920142",
                "37119150",
                "176603387",
                "jodh1658474847561TZ645ae0bae2904573ae935b483458c694",
                "176601919",
                "944330",
                "176607993",
                "kozh1658233444433TZ6bfd13f3b143439eb070661bda09d4ef",
                "24632521",
                "69294334",
                "mald1658472468191TZ3007f5562b004aeeb0e8da1d706c0ec0",
                "564800",
                "176603354",
                "135515996",
                "564828",
                "15138969",
                "223075772",
                "3486014",
                "176603323",
                "210297986",
                "159586369",
                "160324798",
                "15165614",
                "987654321",
                "thir1658233306835TZ4981df839edb4090bcfe86dfa66cdf9e",
                "216807994",
                "124324640",
                "179309438",
                "565001",
                "66750808",
                "118720657"
            ],
            "retailerId": [
                "ac8f14c2-f308-4f9c-845e-5deabe90ed12",
                "abc1657626669639TZ4b32843b6b9343eb903cec4316ad1fa4",
                "153120ad-3821-40c2-b886-dbc6aac9259d",
                "ada1654867551963TZ2ba847b27e0f41af80c1db6eafb8a594",
                "d031bcce-8db3-46ad-ab69-f45bd99ce720",
                "als1654077661417TZ446c6537c3a54c4abce78975aea45bf0",
                "7a2f9929-ab5b-4e5a-a686-59e6b6b74b4b",
                "bha1659163959985TZ8503dda8d20b4988aae1cfb61342c1c1",
                "biz1640610075280TZ17be03b3e6bd4e2f96e8fb01662b84bd",
                "bat1657882078232TZced076504dad4b0aa4f8e8996ec1d82d",
                "d701c5a9-6951-42d3-ae94-6e4824cbbcc6",
                "b0d495d2-cab1-40fb-a969-68ef3b703d43",
                "856f252d-4b56-4e73-be49-ec2bf358f997",
                "0d4f43f2-1948-472c-a31b-6e81a572ed50",
                "242d0e85-d8c7-4d28-8e2e-1a5aa3ba21a2",
                "f31dc8b9-8d53-40c5-9138-920bc7c3b81e",
                "e068c5a1-9b5d-4608-b540-3b6a07e62eb5",
                "05850b41-d8a3-4cd1-b8f2-61087289b796",
                "ce1a8920-0de1-433d-b565-bfdb7986e514",
                "e07152f8-7f0d-4f01-8323-625b6b362f76",
                "4081efe1-bab3-445c-a9a5-4227456e7438",
                "cep1662022392414TZbba2b20ea15f407c9ca465fedf136ea4",
                "b6fc8ee2-73e0-4a19-ba89-3223ffc24522",
                "dwm1654082514775TZ483189080c8141fd8715fd809d65f5f1",
                "des1657689662839TZc73105a96e144e59b7c81bf84c336154",
                "266b2536-aa16-4871-bebf-c2984794a50d",
                "dig1654761250758TZf424f7783d70465389b978b53a0d9945",
                "ec962df1-d338-4b8f-bf67-39d0ccfd9031",
                "19e0b42d-3fc6-40ba-8953-5fa185d01d7d",
                "drp1657114417015TZf2b50e43392440ddbcf83fa2d5f88fc0",
                "64df412c-693f-4ceb-bf4a-5e1a83705d41",
                "51126022-cad5-4e64-88d9-0e763f9fc46a",
                "1f322e15-49ec-49d5-90f6-238d1d9dde35",
                "4c8754a7-7ef0-4525-8fb4-a77f908265ad",
                "3c07e4e3-5217-4623-819e-a281629c317c",
                "elm1654260817605TZ9cf87fd5ee234d36bc0ffe1f5029cf6f",
                "env1659171501706TZc611c07d62b9488b82cce053ba39b852",
                "70e495fa-ad51-4763-b6ed-b46039c39e10",
                "6b491b7b-5fae-4318-837f-5d249a950e89",
                "d2dd0bda-5214-4e68-89e4-3060356a09c8",
                "2bb4f6b2-2137-45f2-b379-52bc68f2c173",
                "ff065514-1484-493d-b1de-f763007ffbf4",
                "8eafe3e2-8d5d-411e-9a43-9cfb6368e242",
                "9b453cc5-ccb5-455c-9a77-fb268e2945a2",
                "gni1657090122040TZ1ca26d70549d40c897683abf26a32ae8",
                "1dc8f787-4a6d-43a8-8b3e-4941ae6b3821",
                "gow1654761004532TZbed5071e010844ee915d8d0a85851645",
                "7577e885-1515-4d55-8d83-11be11e6cd3b",
                "gvt1655297889378TZefb1706abc46407ebee308d8dd813ed9",
                "39f9a6b5-b97c-416a-b6f3-fe4b6f018ed4",
                "11ffccf3-623d-49b4-9905-fdb3786389d8",
                "1db548fb-c6f3-47ea-bd15-34f61ae63041",
                "a99541d6-d06e-4300-87db-22d010a097c9",
                "7bbbb84b-c899-4e1a-8053-456640b7eb89",
                "5fb63e87-8e50-4d7c-912a-a93457d954e6",
                "d0e7d57c-3c3b-422e-9d1b-df980f9eaf9b",
                "07a82fc2-6404-42b2-be04-df8c0f55fb71",
                "kro1656140359212TZfd7037f9f20141f883e5cd966bc5282c",
                "a295028a-c2b0-410b-b685-88371a70996a",
                "6f9d70b6-f7c4-4c84-969e-0a17227beec6",
                "lnl1661774750617TZ667035e4067c43c58bf0dcbe5437b0b2",
                "2c92c74e-d274-471d-b41b-48d70baac02e",
                "ecf7a11b-370f-409b-b548-140d2c6e2403",
                "aeb3b915-6db6-4484-9953-749399b08b31",
                "ece270a1-afa0-4920-b2bb-28e0d33ce3f7",
                "d37807bf-4e74-4d53-8b6e-9d58821a5065",
                "b5572434-76db-4cbf-8417-1a324240c855",
                "a674522b-f992-45b2-8593-68374e705f7f",
                "47f7ec45-6d7c-4148-9f2d-60a59329999f",
                "e4123fbe-1fb9-4927-ba58-26183f64faa1",
                "e772919a-1930-496c-833d-16c609ca1800",
                "de37b394-0295-4293-87ff-951ec1320d41",
                "793b040c-d11d-4fbb-9421-7c756863087f",
                "51c2bec9-2c61-4e2c-a674-1e39c73816b4",
                "51221d31-4e5a-43ee-9065-4d47fc88345a",
                "8ba26c29-658b-4d39-a996-e00ff772e2ce",
                "aa09d711-4c2d-4694-aa5f-4a6f9cafa665",
                "fe7ec7f1-ccb2-47fe-8284-65bd80ef9029",
                "b6e786ed-c43c-44c3-b976-bbd488883cb5",
                "fc2b360c-1fae-476c-b5f3-25fce6412713",
                "9fc75bce-937b-4b30-94e1-490cf5e308b4",
                "e6b6b5f3-709c-4bb7-947a-ad6e38e69928",
                "677c8cc7-aadd-4680-8cc5-c143c86ae40e",
                "2b829174-b224-4e8c-8e33-4c4cdddbb6c9",
                "6551e263-4a90-40ce-b465-77f325852d97",
                "ed13a957-0809-411a-a948-3aa1a86f57b3",
                "rlo1656322567823TZ22365cfb8e0442f2b6d85ff6118bfbc6",
                "441fa73c-ca52-4097-8380-041b8f01573c",
                "5a4c9ce9-ad8f-4118-97b3-10ca3a26d7e5",
                "res1656918296583TZ3d7a07cb2eff477ba77ee2f0146a36e5",
                "5b147bc3-8ff4-480a-b575-7aa4807b023b",
                "a24a38b0-d36b-42bd-b49f-736465e9f8fa",
                "ddd6866a-5e3b-4d66-9c95-2c643d92c421",
                "01642d1e-f9e7-41fd-a96f-a34ef291869d",
                "san1660039696545TZ28f316dadb1444dd957bed6d1e68602f",
                "0d1d87be-4e99-4525-90ce-59a31acab5f4",
                "5050",
                "413db07a-8b75-4214-b97a-7785274f418f",
                "76d80539-e818-40aa-aada-f5109b402475",
                "d7ecba8a-bb56-4116-9333-b0cb65cd8fbc",
                "8e153d77-5854-4daf-a393-17d521df4691",
                "ccd49237-7df3-475c-9f78-57376ed1bdba",
                "4e7ce710-01b2-4b10-af5a-4295a486ba32",
                "66692418-1e8a-48d3-a7f4-840de1abdee9",
                "24a36528-0749-4cd9-a6e9-cf8fec19d04b",
                "b2a39286-9feb-4b72-a26a-f84740123e1c",
                "xyz1655103163596TZd66955155f984ef3bb8c59194b0356bf",
                "d0eb6dd5-2f8f-4e39-ae3e-50631201e122",
                "04d05c7e-5a35-404e-8b08-e0ea2c6583b2",
                "493ae5bd-c70e-46ed-82a8-3a3449753450",
                "tmr1661776836357TZ05fb7f302d454af1a748623074a92eb1",
                "tr71657198111061TZace3d9fb98044b92a0a69480ecbf4612",
                "84463ac3-1a29-41b4-a607-cd639bb59f72",
                "bdc9aca4-3ce9-4c81-9fa9-81acd9f710f2",
                "b12e4456-ca40-4b99-93e4-0ea3f72eb822",
                "95d4a56d-7119-4789-99f2-fc2f911c7b33",
                "e0175f20-7588-466c-b960-8cacf25593d3",
                "47739804-bb6b-411d-97fb-f41c596ec7cb",
                "809989f3-bc9a-4069-82eb-c7bbe5c8a95c",
                "034d0ac9-be95-4685-b8aa-424608ef30c0",
                "27b8f1c3-ec71-4003-974f-45a2b01d5586"
            ],
            "searchString": "",
            "ticketStatus": "PROCESSING",
            "includeDstHub": "no"
        }
        # test token eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjp7InBsYXllcklkIjoicGxheWVyMSIsImNvZGUiOiJCSVoiLCJkZXNpZ25hdGlvbklkIjoiZmUiLCJkZXNpZ25hdGlvbiI6IkZpZWxkIEVuZ2luZWVyIiwiZGVwYXJ0bWVudElkIjoiZGVwYXJ0bWVudDEiLCJkZXBhcnRtZW50IjoiSW5mb3JtYXRpb24gVGVjaG5vbG9neSIsInVzZXJuYW1lIjoiYWRtaW4iLCJmaXJzdG5hbWUiOiJhZG1pbjEiLCJsYXN0bmFtZSI6ImFkbWluMiIsInN0YXR1cyI6ImFjdGl2ZSJ9LCJuYmYiOjE2NjI5NzI1NzIsImlhdCI6MTY2Mjk3MjU3MiwiZXhwIjoxNjYzNjYzNzcyfQ.JDUFU_RsQMaHmFTqqw5j6VCc8L5N22jvspvBA3WI4NI
        getPageCountUrl = "https://test-api.bizlog.in:8443/mis-report/ticket/mis/report/new/10/1"

        Headers = {
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyIjp7InBsYXllcklkIjoicGxheWVyMSIsImNvZGUiOiJCSVoiLCJkZXNpZ25hdGlvbklkIjoiZmUiLCJkZXNpZ25hdGlvbiI6IkZpZWxkIEVuZ2luZWVyIiwiZGVwYXJ0bWVudElkIjoiZGVwYXJ0bWVudDEiLCJkZXBhcnRtZW50IjoiSW5mb3JtYXRpb24gVGVjaG5vbG9neSIsInVzZXJuYW1lIjoiYWRtaW4iLCJmaXJzdG5hbWUiOiJhZG1pbjEiLCJsYXN0bmFtZSI6ImFkbWluMiIsInN0YXR1cyI6ImFjdGl2ZSJ9LCJuYmYiOjE2NjMwNDc1OTUsImlhdCI6MTY2MzA0NzU5NSwiZXhwIjoxNjYzNzM4Nzk1fQ.uI4NE_4D56FBB0arMrr9PkA3zuyTh6ZcUKc_zUkkAOQ",
            "Content-Type": "application/json"
        }
        result = requests.put(getPageCountUrl, json=data, headers=Headers)
        results = json.loads(result.text)
        d = json.loads(result.text)
        d = d.get('data')
        d = d.get('misReportTicket')

        lastPage = results["data"]["pageLinks"]["lastPage"]

        d1 = list()
        d2 = list()
        print(lastPage)
        x = 0
        for x in range(lastPage):
            print(x)
        #    if x != 0:
            getData = "https://test-api.bizlog.in:8443/mis-report/ticket/mis/report/new/10/" + \
            ""+str(x)
            result1 = requests.put(getData, json=data, headers=Headers)
            d1 = json.loads(result1.text)
            d1 = d1.get('data')
            d1 = d1.get('misReportTicket')
            d2 = d1 + d2
###############################################
from PIL import Image
image = Image.open('BIZLOG.PNG')
matplotlib.use("Agg")

class DataFrame_Loader():
    def __init__(self):
        print("Loading DataFrame")

    def read_csv(self,data):
        self.df = pd.read_csv(data)
        return self.df
######################################

##########################
class EDA_Dataframe_Analysis():
    def __init__(self):
        print("General_EDA object created")
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

   
    def concat(self,x,y,z,axis):
    	return pd.concat([x,y,z],axis)

	    
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

st.sidebar.image(image, use_column_width=True)   
def main():
	st.title(" ML APP")
	
	st.info("VALUE ADDED SERVICES  * *")
	
	activities = ["DATA ANALYSIS"]

	x = st.date_input('select start date',value = dt.datetime.now())
	y = st.date_input('select end date',value = dt.datetime.now())

	st.write('start date is: ', x)
	st.write(' end date is : ', y)

	
	
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice == 'DATA ANALYSIS':
		st.subheader("BIZLOG RETAILER DATA ANALYSIS ")

		if st.checkbox(" TO READ DATA FROM LOCAL FILE CLICK THIS CHECKBOX "):
			data = st.file_uploader("Upload a Dataset", type=["csv"])
			if data is not None:
				df = load.read_csv(data)
				st.dataframe(df.head())
				st.success("Data Frame Loaded successfully")
#		else:
		if st.checkbox(" TO READ DATA FROM JSON CLICK THIS CHECKBOX "):
#				with open('C:/Users/Santosh/Desktop/inputmonth.json') as json_data:
#				d = json.load(d2)
#				df = pd.DataFrame(d)
			DatabaseLoad1()
			df = pd.DataFrame.from_dict(d2)
#            DatabaseLoad1()
#				df = pd.json_normalize(d2)
#				st.write(df.describe())
			st.dataframe(df.head())
			st.success("Data Frame Loaded successfully")
		
		st.write(" To know about data and pie chart please select above option")
		if st.checkbox("Show Columns"):
			st.write(dataframe.show_columns(df))

		if st.checkbox("Show Missing"):
			st.write(dataframe.Show_Missing1(df))

		if st.checkbox("Num Count Summary"):
			st.write(info.num_count_summary(df))
                
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
       
		selected_columns_names = st.selectbox("Select Column 1 For Cross and bar Plot Tabultion",all_columns_names)
		selected_columns_names1 = st.selectbox("Select Column 2 For Cross and bar Plot Tabultion",all_columns_names1)
		if st.button("Generate Cross Tab"):
			st.dataframe(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))

		if st.button("Generate Bar Graph PLOT"):
			st.bar_chart(pd.crosstab(df[selected_columns_names],df[selected_columns_names1]))

		if st.checkbox("Pie Plot"):
			all_columns = df.columns.to_list()
			column_to_plot = st.selectbox("Select 1 Column",all_columns)
			pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()


		if st.checkbox("AREA PLOT"):
		#if type_of_plot == 'area':
			cust_data = df[selected_columns_names]
			st.area_chart(cust_data)

		#elif type_of_plot == 'bar':
		if st.checkbox("BAR PLOT"):
			cust_data = df[selected_columns_names]
			st.bar_chart(cust_data)

		#elif type_of_plot == 'line':
		if st.checkbox("LINE  PLOT"):
		#	if ( (y) > df.dropDate > str(x) ):
			cust_data = df[selected_columns_names]
			st.line_chart(cust_data)

#		if st.checkbox("additional graph"):
#			subset_data = df
#			country_name_input = st.sidebar.multiselect('selected_columns_names',
#			df.groupby('selected_columns_names/selected_columns_names1').count().reset_index()['selected_columns_names/selected_columns_names1'].tolist())
#
	st.write(df.groupby([selected_columns_names1]).count())
	

def search_callback():
	st.write("You searched for:", st.session_state.input)

search_str = st.text_input('Search', key='input', placeholder='Life of Brian', on_change=search_callback)
		
		
if __name__ == '__main__':
#   load data from database
	load = DatabaseLoad()
	load = DataFrame_Loader()
	dataframe = EDA_Dataframe_Analysis()
	info = Attribute_Information()
	model = Data_Base_Modelling()
	main()
