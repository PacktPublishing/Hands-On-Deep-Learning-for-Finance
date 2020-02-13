# download all nasdaq data and create individual csv files

import pandas as pd 
import pandas.io.sql as ps 

import sqlalchemy
import mysql.connector

engine = sqlalchemy.create_engine('mysql+mysqlconnector://sec_user:justpass@localhost:3306/Securities_Master')

# get all the symbols for which we have the data
symlist = ps.read_sql('select distinct symbolid from phist_nasdaq', con=engine)
fulldata = ps.read_sql('select * from phist_nasdaq order by symbolid asc, Date asc', con=engine)

for i in range(0,len(symlist)):
	temp = fulldata[fulldata.symbolid == symlist.symbolid[i]]
	temp.to_csv('csvdata/' + symlist.symbolid[i] + '.csv')
	fulldata.drop(fulldata[fulldata.symbolid == symlist.symbolid[i]].index, inplace=True)
	print(str(i) + ' of ' + str(len(symlist)) + ' done')
