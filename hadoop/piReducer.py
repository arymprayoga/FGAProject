import sys
import math
cast = lambda x : [float(x[1]), float(x[2])]
current_sum = 0
data_count = 0
calc_count = 0
calc_ape = 0 

def app_pi(current_sum, data_count):
	return float(4 * current_sum)/data_count

def ape(app_pi):
	return abs(app_pi - math.pi)/math.pi

def output():
	global calc_ape
	global calc_count
	a_pi = app_pi(current_sum, data_count)
        calc_ape += ape(a_pi)
        calc_count += 1
	print("%d,%f,%f" % (calc_count, a_pi, calc_ape/calc_count))
 
for data in sys.stdin:
	x, y = cast(data.replace("\n", "").split("\t")[1].split(','))
	current_sum += 1 if x**2 + y**2 <=1 else 0
	data_count += 1
	if data_count % 1000 == 0 :
		output()
output()
