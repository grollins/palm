Installing palm on OS X (Lion)
##############################

The instructions are based on this page_. It may be a useful reference.

General dependencies:
---------------------

1. Install XCode and its Command Line Tools

    (get them in the App Store)

2. Install Homebrew::

    ruby -e "$(curl -fsSL https://raw.github.com/mxcl/homebrew/go)"
    brew update
    brew doctor

3. Install fortran compiler::

    brew install gfortran

4. Install git and subversion::

    brew install git
    brew install subversion

5. Install freetype (for plotting with matplotlib)::

    brew install freetype

Python dependencies:
--------------------

1. Install Python2.7

    brew install python

2. Install pip, virtualenv, and virtualenvwrapper::

    pip install virtualenv
    pip install virtualenvwrapper
    source /usr/local/share/python/virtualenvwrapper.sh

3. Create a virtualenv::

    mkvirtualenv palm_env

4. Load virtualenv::

    workon palm_env

5. Get palm from github::

    git clone https://github.com/grollins/palm.git

6. Install dependencies via pip::

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

.. _page: http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/