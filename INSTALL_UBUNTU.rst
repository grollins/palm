Installing palm on Ubuntu
#########################

1. Install fortran compiler::

    sudo apt-get install gfortran g++

2. Install git::

    sudo apt-get install git

3. Install the Anaconda distribution of Python::

    https://store.continuum.io/cshop/anaconda/

4. Get palm from github::

    git clone https://github.com/grollins/palm.git

5. Install palm::

    cd palm
    python setup.py install

6. Test palm (optional)::

    nosetests
