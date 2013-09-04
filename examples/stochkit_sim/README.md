* Install [stochkit](http://engineering.ucsb.edu/~cse/StochKit/).
* Open `stochkit_sim.py` and set `STOCHKIT_HOME` to the path of your copy of stochkit.
* Run `python stochkit_sim.py 10` to simulate palm time traces, where **10** is the number of time traces you want to generate.
* You can change the kinetic parameters or N by modifying `blink_model.xml`.
    + The kinetic parameters are specified under `<ParametersList>`.
    + N is specified near the bottom of the file under `<SpeciesList>`.
    + Set the initial value of the **I** species to the desired value of N.

Tested with stochkit v2.0.8
