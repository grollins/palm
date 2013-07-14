Installing dependencies:
------------------------
1. Install python2.7 with pip, virtualenv, and virtualenvwrapper according to
the instructions on these pages:

    + OSX_
    + Linux_ (CentOS)
    + Windows_

2. Create a virtualenv::

    mkvirtualenv palm_env

3. Load virtualenv::

    workon palm_env

4. Install dependencies::

    pip install -r reqs.txt

5. Download qit library from::

    svn checkout svn://svn.code.sf.net/p/qit/code/python/trunk qit
    cd qit
    python setup.py install

Installing palm
---------------
After dependencies have been installed, run this command in the
the top directory of palm::

    python setup.py install

Tests are run by typing this command within the top directory::

    nosetests

.. _OSX: http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/
.. _Linux: http://toomuchdata.com/2012/06/25/how-to-install-python-2-7-3-on-centos-6-2/
.. _Windows: https://code.google.com/p/winpython/wiki/Installation
