#########################################################################################
#											#
#			THIS IS AN UTIL FILE						#
#			CONTAINS MANY USELESS FUNCTION					#
#											#
#########################################################################################


#Import
from pyspark.mllib.regression import LabeledPoint

#Zip with index in front of data
def zip_index(rdd):
	return rdd.zipWithIndex().map(lambda x : (x[1], [x[0]]))

#do Convolution to data (used to make feature data) (Can only be used with 1 dim) (more dim come next)
def convolve(rdd, count):
	rdd1 = rdd
	for _ in range(count):
		rdd1 = rdd1.map(lambda x : (x[0]-1, x[1])).cache()
		rdd = rdd.join(rdd1).map(lambda x : (x[0], x[1][0]+x[1][1]))
	rdd =  rdd.map(lambda x : (x[0], x[1][-1], x[1][:-1])).cache()
	return rdd
	
#train test split
def train_test_split(rdd, split=0.2):
	return rdd.randomSplit(weights=[1-split, split])

#Labeled Point
def labelled_points(rdd):
	return rdd.map(lambda x : LabeledPoint(x[1], x[2]))

#MAPE
def mape(rdd):
	return rdd.map(lambda x : abs((x[0] - x[1])/x[0])).reduce(lambda x,y: x+y) / rdd.count() * 100

#normalizer
def normalization(sc, rdd):
	data_max = sc.broadcast(rdd.reduce(lambda x,y : (x[0], max(x[1], y[1])))[1][0]+10000)
	data_min = sc.broadcast(rdd.reduce(lambda x,y : (x[0], min(x[1], y[1])))[1][0]-10000)
	return data_max, data_min, rdd.map(lambda x : (x[0], ((x[1][0]-data_min.value)/(data_max.value-data_min.value),)))
	
def denormalization(sc, rdd, data_min, data_max):
	return rdd.map(lambda x : (x[0], x[1] * (data_max.value - data_min.value) + data_min.value))

#CSV
def toCSVLine(data):
  return (','.join(str(d) for d in data),)
