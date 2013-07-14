import os

for i in xrange(1, 2):
    cmd = "python bootstrap_ml_fit.py %d" % i
    print cmd
    os.system(cmd)
