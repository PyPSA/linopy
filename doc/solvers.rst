Installing solvers
==================


**linopy** supports the following solvers

-  `Cbc <https://projects.coin-or.org/Cbc>`__ - open source, free, fast
-  `GLPK <https://www.gnu.org/software/glpk/>`__ - open source, free, not very fast
-  `HiGHS <https://www.maths.ed.ac.uk/hall/HiGHS/>`__ - open source, free, fast
-  `Gurobi <https://www.gurobi.com/>`__  - closed source, commercial, very fast
-  `Xpress <https://www.fico.com/en/products/fico-xpress-solver>`__ - closed source, commercial, very fast
-  `Cplex <https://www.ibm.com/de-de/analytics/cplex-optimizer>`__ - closed source, commercial, very fast


Please click on the links to get further installation information. In the following we provide additional installation guides for a subset of the above listed solvers.

For installing the solvers which are available via pip, you can simply run

.. code:: bash

    pip install linopy[solvers]


and additionally for the HiGHS solver

.. code:: bash

    pip install highspy


which is not yet available on all platforms.


.. PIPS-IMP++
.. ----------

.. **NOT IMPLEMENTED YET**
.. The full installation guide can be found at https://github.com/NCKempke/PIPS-IPMpp. The following commands comprize all important installation steps.

.. .. code:: bash

..     cd where/pips/should/be/installed
..     sudo apt install wget cmake libboost-all-dev  libscalapack-mpich2.1 libblas-dev liblapack-dev
..     git clone https://github.com/NCKempke/PIPS-IPMpp.git
..     cd PIPS-IPMpp
..     mkdir build
..     cd build
..     cmake .. -DCMAKE_BUILD_TYPE=RELEASE
..     make
