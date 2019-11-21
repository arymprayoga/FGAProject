import sys

for data in sys.stdin:
	print("1\t%s" % data.replace("\n", ""))
