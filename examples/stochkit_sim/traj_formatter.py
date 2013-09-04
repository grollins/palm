from os.path import join, basename
from glob import glob
import numpy
import pandas
from collections import Counter, defaultdict

def convert_trajectory(trajectory_table):
    '''
    Expected table format:
    time I A D B
    0.0 10.0 0.0 0.0 0.0
    0.99 9.0 1.0 0.0 0.0
    1.84 8.0 2.0 0.0 0.0
    16.7 7.0 3.0 0.0 0.0
    17.0 6.0 4.0 0.0 0.0
    '''
    new_traj_dict = defaultdict(list)
    for i in xrange(1, len(trajectory_table)):
        prev_time = trajectory_table.get_value(i-1, 'time')
        prev_I = trajectory_table.get_value(i-1, 'I')
        prev_A = trajectory_table.get_value(i-1, 'A')
        prev_D = trajectory_table.get_value(i-1, 'D')
        prev_B = trajectory_table.get_value(i-1, 'B')
        this_time = trajectory_table.get_value(i, 'time')
        this_I = trajectory_table.get_value(i, 'I')
        this_A = trajectory_table.get_value(i, 'A')
        this_D = trajectory_table.get_value(i, 'D')
        this_B = trajectory_table.get_value(i, 'B')
        delta_time = this_time - prev_time
        delta_I = this_I - prev_I
        delta_A = this_A - prev_A
        delta_D = this_D - prev_D
        delta_B = this_B - prev_B
        delta_string = "%d_%d_%d_%d" % (delta_I, delta_A, delta_D, delta_B)
        if prev_A > 0.0:
            class_label = "bright"
        elif prev_A == 0.0:
            class_label = "dark"
        else:
            print "Unexpected observation class", delta_string
        new_traj_dict['class'].append(class_label)
        new_traj_dict['dwell time'].append(delta_time)
    new_traj_table = pandas.DataFrame(new_traj_dict)
    return new_traj_table

def consolidate_trajectory(dwell_table, noisy=False):
    '''
    Expected table format:
    class dwell time
    dark   1.0
    bright 0.5
    bright 0.2
    bright 0.4
    dark   2.0
    bright 0.3
    dark   5.0
    '''
    # print dwell_table
    c = Counter()
    new_traj_dict = defaultdict(list)
    current_class = None
    current_dwell_time = 0.0
    for i in xrange(len(dwell_table)):
        this_class = dwell_table.get_value(i, 'class')
        this_dwell_time = dwell_table.get_value(i, 'dwell time')
        if current_class is None:
            current_class = this_class
            current_dwell_time += this_dwell_time
        elif current_class == this_class:
            current_dwell_time += this_dwell_time
        elif current_class != this_class:
            new_traj_dict['class'].append(current_class)
            new_traj_dict['dwell time'].append(current_dwell_time)
            c[current_class] += 1
            current_dwell_time = this_dwell_time
            current_class = this_class
        else:
            print "Unexpected branch"
    new_traj_dict['class'].append(current_class)
    new_traj_dict['dwell time'].append(current_dwell_time)
    c[current_class] += 1

    # add long dark phase signifying total bleaching.
    # if the last dwell was already a dark dwell, replace the dwell
    # time with a larger value.
    if new_traj_dict['class'][-1] == 'dark':
        new_traj_dict['dwell time'][-1] = 10000.0
    else:
        new_traj_dict['class'].append('dark')
        new_traj_dict['dwell time'].append(10000.0)
    c['dark'] += 1

    if noisy:
        print c
    else:
        pass

    new_traj_table = pandas.DataFrame(new_traj_dict)
    return new_traj_table

def convert_stochkit_trajs_to_palm_format(stochkit_traj_dir, palm_traj_dir,
                                          noisy=False):
    '''
    Expected format of input:
    time I A D B
    0.0 10.0 0.0 0.0 0.0
    0.99 9.0 1.0 0.0 0.0
    1.84 8.0 2.0 0.0 0.0
    16.7 7.0 3.0 0.0 0.0
    17.0 6.0 4.0 0.0 0.0
    '''
    file_list = glob(join(stochkit_traj_dir, "*.txt"))
    if noisy:
        print stochkit_traj_dir
        print palm_traj_dir
        print "Converting %d trajectories..." % len(file_list)
    for f in file_list:
        if noisy: print f
        base_path = basename(f)
        stochkit_traj = pandas.read_table(f, header=0)
        palm_traj = convert_trajectory(stochkit_traj)
        palm_traj = consolidate_trajectory(palm_traj, noisy)
        palm_traj_path = join(palm_traj_dir, base_path)
        palm_traj.to_csv(palm_traj_path, index=False)
        if noisy: print "Wrote %s\n" % palm_traj_path
