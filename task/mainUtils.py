import subprocess
import string
import random
import os
import shutil

#utils
random_name = lambda N : ''.join(random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(N))

def get_merge_local(file_out, dir_input):
	with open(file_out, 'wb') as outfile:
		for filename in os.listdir(dir_input):
        		if filename == file_out or filename[0] == ".":
            			continue
        		with open(dir_input+"/"+filename, 'rb') as readfile:
            			shutil.copyfileobj(readfile, outfile)

def run_process(l):
	try:
		process = subprocess.Popen(l, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
		stdout, stderr = process.communicate()
		print(stdout)
		return 0, stdout
	except subprocess.CalledProcessError as e:
		print("Program returned exitcode %d" % process.returncode)
		print(stderr)
		print(stdout)
		return process.returncode, stderr

def parse_output(file):
	l = []
	cast = lambda x : (int(x[0]), list(map(float, x[1:])))
	for row in file:
		l.append(cast(row.replace("(", "").replace(")", "").split(",")))
	return l
