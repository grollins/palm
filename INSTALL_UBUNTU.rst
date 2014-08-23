Installing palm on Ubuntu
#########################

1. Install git::

    sudo apt-get install git

2. Install the Anaconda distribution of Python::

    https://store.continuum.io/cshop/anaconda/

3. Get palm from github::

    git clone https://github.com/grollins/palm.git

4. Install palm::

    cd palm
    python setup.py install

5. Test palm (optional)::

    nosetests
