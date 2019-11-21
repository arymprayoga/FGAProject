import random

for i in range(100000):
	x = random.uniform(-1, 1)
	y = random.uniform(-1, 1)
	print("%d,%f,%f" % (i, x, y))
