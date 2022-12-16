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


Particular solver instructions
------------------------------

HiGHS
~~~~~

HiGHS is an "open source serial and parallel solvers for large-scale sparse linear programming".

Find the documentation at https://www.maths.ed.ac.uk/hall/HiGHS/.
The full list of solver options is documented at
https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set.

Install it via pip:
```
pip install highspy
```


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
