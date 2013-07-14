Installing palm on Ubuntu 12.04 64-bit
######################################

General dependencies:
---------------------

1. Install fortran compiler::

    sudo apt-get install gfortran g++

2. Install git and subversion::

    sudo apt-get install git subversion

3. Install linear algebra libraries::

    sudo apt-get install libatlas-base-dev liblapack-dev

3. Install plotting library dependencies (matplotlib)::

    sudo apt-get install libfreetype6-dev
    sudo apt-get install libpng-dev

Python dependencies:
--------------------

1. Install pip, virtualenv, and virtualenvwrapper::

    sudo apt-get update
    sudo apt-get install python-pip python-dev build-essential
    sudo pip install --upgrade pip
    sudo pip install virtualenv
    sudo pip install virtualenvwrapper
    source /usr/local/bin/virtualenvwrapper.sh

2. Create a virtualenv::

    mkvirtualenv palm_env

3. Load virtualenv::

    workon palm_env

4. Get palm from github::

    git clone https://github.com/grollins/palm.git

5. Install dependencies via pip::

    cd palm
    pip install numpy==1.7.0
    pip install -r reqs.txt

6. Download qit library via subversion::

    cd ..
    svn checkout svn://svn.code.sf.net/p/qit/code/python/trunk qit
    cd qit
    python setup.py install

7. Install palm::

    cd ..
    cd palm
    python setup.py install

8. Test palm (optional)::

    nosetests

