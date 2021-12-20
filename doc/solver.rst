Installation
============


HiGHS
------

HiGHS is an "open source serial and parallel solvers for large-scale sparse linear programming".

Find the documentation at https://www.maths.ed.ac.uk/hall/HiGHS/.
The full list of solver options is documented at
https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set.

**Installation on Linux**

The script work for version HiGHS 1.1.1. The installation steps are

.. code:: bash

    sudo apt-get install cmake  # if not installed
    git clone git@github.com:ERGO-Code/HiGHS.git
    cd HiGHS
    git checkout 95342daa73543cc21e5b27db3e0fbf7330007541 # moves to HiGHS 1.1.1
    mkdir build
    cd build
    cmake ..
    make
    ctest

If you have problems with running the tests, make sure you are in a clean environment. If you have `conda` installations in your base environment, you can try building HiGHS from a freshly created `conda` environment.

After the build, add the paths of executables and library to your `.bashrc`:

.. code:: bash

    export PATH="${PATH}:/foo/HiGHS/build/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/foo/HiGHS/build/lib"
    source .bashrc



PIPS-IMP++
----------

**NOT IMPLEMENTED YET**
The full installation guide can be found at https://github.com/NCKempke/PIPS-IPMpp. The following commands comprize all important installation steps.

.. code:: bash

    cd where/pips/should/be/installed
    sudo apt install wget cmake libboost-all-dev  libscalapack-mpich2.1 libblas-dev liblapack-dev
    git clone https://github.com/NCKempke/PIPS-IPMpp.git
    cd PIPS-IPMpp
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=RELEASE
    make
