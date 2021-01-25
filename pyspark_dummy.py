# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 19:44:57 2021

@author: Bhawna
"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
sc = SparkContext.getOrCreate();
sqlContext = SQLContext(sc)

print('Create a DataFrame by applying createDataFrame on RDD with the help of sqlContext.')
from pyspark.sql import Row
l = [('Ankit',25),('Jalfaizy',22),('saurabh',20),('Bala',26)]
rdd = sc.parallelize(l)
people = rdd.map(lambda x: Row(name=x[0], age=int(x[1])))
schemaPeople = sqlContext.createDataFrame(people)

print('type ....')
print(type(schemaPeople))
print('end')
