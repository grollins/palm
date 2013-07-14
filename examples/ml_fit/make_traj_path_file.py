import os
import glob

traj_path = "../simulated_data_N5a/*.csv"
path_list = glob.glob(os.path.expanduser(traj_path))
f = open('traj_paths.txt', 'w')
for p in path_list:
    f.write("%s\n" % p)
f.close()
