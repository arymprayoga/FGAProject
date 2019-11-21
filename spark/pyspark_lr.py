#import
from pyspark import SparkContext
import numpy as np
import utils
from pyspark.mllib.regression import LinearRegressionWithSGD
import sys

#Spark Context (DONT use LOCAL for production)
sc = SparkContext("local","MultipleLinearRegression")

#param
jumlah_fitur  	= int(sys.argv[1])
#jumlah_hidden 	= 6		#Not Supoorted. see: ELM.
file_input  	= sys.argv[2]
dir_output 	= sys.argv[3]

#Hidden Layer	#Not Supported. see: ELM
#hidden = sc.broadcast(np.random.uniform(-1, 1, (jumlah_fitur, jumlah_hidden)))

#Transform Function
transform = lambda x : np.dot(x[1], hidden).tolist()

#Load
data = sc.textFile(file_input).map(lambda x : float(x.split(";")[1]))

#Indexing
data = utils.zip_index(data)

#normalization
data_min, data_max, data = utils.normalization(sc, data)

#Convolve Data (Feature Selection)
data = utils.convolve(data, jumlah_fitur)


#Split train and test
train, test = utils.train_test_split(data)

#Labelling Points
train = utils.labelled_points(train)
test = utils.labelled_points(test)

#Regression (this is not least square but SGD)
lrm = LinearRegressionWithSGD()
model = lrm.train(train)

#Test
mape_train = utils.mape(train.map(lambda x: (x.label, model.predict(x.features))))
mape_test = utils.mape(test.map(lambda x: (x.label, model.predict(x.features))))

#Prediction
actual_pred = data.map(lambda x : (x[0], x[1], model.predict(x[2])))

#split
actual = actual_pred.map(lambda x : (x[0], x[1]))
prediction = actual_pred.map(lambda x : (x[0], x[2]))

#denormalization
actual = utils.denormalization(sc, actual, data_min, data_max)
prediction = utils.denormalization(sc, prediction, data_min, data_max)

#JOIN
ret = actual.join(prediction).map(lambda x: (x[0], x[1][0], x[1][1]))

#return value
ret.saveAsTextFile(dir_output+"result")
sc.parallelize([mape_train, mape_test]).saveAsTextFile(dir_output+"mape")



