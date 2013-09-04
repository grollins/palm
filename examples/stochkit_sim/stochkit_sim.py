import sys
import re
from os import mkdir
from os.path import join, exists
from subprocess import Popen, PIPE, STDOUT
from shutil import move
from traj_formatter import convert_stochkit_trajs_to_palm_format

# =============
# = Constants =
# =============
STOCHKIT_HOME="/Users/grollins/src/StochKit2.0.8"
STOCHKIT_BINARY = join(STOCHKIT_HOME, 'ssa')
MODEL_XML = "blink_model.xml"
STOCHKIT_OUTPUT_DIR = join(re.sub('\.xml$', '', MODEL_XML) + "_output", "trajectories")
PALM_TRAJ_DIR = "simulated_palm_traces"

def run_stochkit(num_trajs, noisy=False):
    cmd_string = "%s -m %s -r %d -t 180 -i 3600 --keep-trajectories -f --label" % \
                 (STOCHKIT_BINARY, MODEL_XML, num_trajs)
    if noisy: print cmd_string
    process = Popen( cmd_string, shell=True, stdin=PIPE, stderr=STDOUT, stdout=PIPE)
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # non-zero return code indicates something went wrong
    if returncode:
        print stdout
        print stderr
        raise RuntimeError, "Error running stochkit"

    if noisy: print stdout
    return

def main():
    num_trajs = int(sys.argv[1])
    run_stochkit(num_trajs, noisy=True)
    if not exists(PALM_TRAJ_DIR): mkdir(PALM_TRAJ_DIR)
    convert_stochkit_trajs_to_palm_format(STOCHKIT_OUTPUT_DIR, PALM_TRAJ_DIR,
                                          noisy=True)

if __name__ == '__main__':
    main()