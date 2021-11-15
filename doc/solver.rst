Installation
============


PIPS-IMP++
----------

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
