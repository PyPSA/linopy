# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd 
import dask 
import xarray as xr
import numpy as np
import os
import shutil
from tempfile import mkstemp
from dask.array import asarray
from xarray import DataArray


class Model:
    
    
    def __init__(self, solver_dir=None):
        self._xCounter = 1
        self._cCounter = 1

        self.vars = xr.Dataset()
        self.cons = xr.Dataset()
        
        self.solver_dir = solver_dir
        

    def define_variables(self, name, lower=None, upper=None, coords=None):
        assert coords is not None or (lower is not None and upper is not None)
        if coords is not None:
            # TODO: check lower and upper not array_like
            lower = -np.inf if lower is None else lower 
            upper = np.inf if upper is None else upper
        else:
            lower = DataArray(lower).chunk(1)
            upper = DataArray(upper).chunk(1)
            coords = (lower + upper).coords # lazy calculation for extracting coords
        shape = [len(c[1]) if isinstance(c, tuple) else len(c) for c in coords]
        size = np.prod(shape)
        start = self._xCounter
        var = np.arange(start, start + size).reshape(shape)
        var = xr.DataArray(var, coords=coords).chunk(1)
        self.vars = self.vars.assign({name: var})




    def to_file(self, keep_files=False):
        tmpkwargs = dict(text=True, dir=self.solver_dir)
    
        fdo, objective_fn = mkstemp('.txt', 'objectve-', **tmpkwargs)
        fdc, constraints_fn = mkstemp('.txt', 'constraints-', **tmpkwargs)
        fdb, bounds_fn = mkstemp('.txt', 'bounds-', **tmpkwargs)
        fdi, binaries_fn = mkstemp('.txt', 'binaries-', **tmpkwargs)
        fdp, problem_fn = mkstemp('.lp', 'problem-', **tmpkwargs)
    
        self.objective_f = open(objective_fn, mode='w')
        self.constraints_f = open(constraints_fn, mode='w')
        self.bounds_f = open(bounds_fn, mode='w')
        self.binaries_f = open(binaries_fn, mode='w')
    
        self.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
        self.constraints_f.write("\n\ns.t.\n\n")
        self.bounds_f.write("\nbounds\n")
        self.binaries_f.write("\nbinary\n")


        # write everything...

        self.binaries_f.write("end\n")
    
        # explicit closing with file descriptor is necessary for windows machines
        for f, fd in (('bounds_f', fdb), ('constraints_f', fdc),
                      ('objective_f', fdo), ('binaries_f', fdi)):
            getattr(self, f).close(); delattr(self, f); os.close(fd)
    
        # concat files
        with open(problem_fn, 'wb') as wfd:
            for f in [objective_fn, constraints_fn, bounds_fn, binaries_fn]:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                if not keep_files:
                    os.remove(f)
    
        logger.info(f'Total preparation time: {round(time.time()-start, 2)}s')
        return fdp, problem_fn





