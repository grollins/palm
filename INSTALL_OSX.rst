Installing palm on OS X (Lion)
##############################

1. Install Homebrew::

    ruby -e "$(curl -fsSL https://raw.github.com/mxcl/homebrew/go)"
    brew update
    brew doctor

2. Install git and subversion::

    brew install git

3. Install the Anaconda distribution of Python::

    https://store.continuum.io/cshop/anaconda/

4. Get palm from github::

    git clone https://github.com/grollins/palm.git

5. Install palm::

    cd palm
    python setup.py install

6. Test palm (optional)::

    nosetests
