.. _projects:

Projects using linopy
=====================

Linopy is used as the optimisation backend or modelling layer in a range of
open-source projects, most of them in energy system modelling. The list below
is a curated selection of established, publicly maintained projects, not a
complete census. If your project uses linopy and is missing, please open a
pull request adding it.

Energy system models built on PyPSA
-----------------------------------

`PyPSA <https://github.com/PyPSA/PyPSA>`__ uses linopy to build and solve its
optimisation problems. The linopy model is exposed as ``n.model``, so the
projects below can add custom variables and constraints on top of it.

- `PyPSA <https://github.com/PyPSA/PyPSA>`__: Python for Power System Analysis, the framework itself.
- `PyPSA-Eur <https://github.com/PyPSA/pypsa-eur>`__: sector-coupled model of the European energy system.
- `PyPSA-DE <https://github.com/PyPSA/pypsa-de>`__: German energy system model (Ariadne project).
- `PyPSA-USA <https://github.com/PyPSA/pypsa-usa>`__: model of the US energy system.
- `PyPSA-Earth <https://github.com/pypsa-meets-earth/pypsa-earth>`__: global, data-driven energy system model.
- `PyPSA-China <https://github.com/PyPSA/pypsa-china>`__: model of the Chinese energy system.
- `PyPSA-Canada <https://github.com/NRCan/pypsa-canada>`__: model of the Canadian energy system.
- `PyPSA-PL <https://github.com/instrat-pl/pypsa-pl>`__: model of the Polish energy system.
- `PyPSA-AT <https://github.com/AGGM-AG/pypsa-at>`__: model of the Austrian energy system.
- `PyPSA-GB <https://github.com/andrewlyden/PyPSA-GB>`__: model of the Great Britain power system.
- `ISPyPSA <https://github.com/Open-ISP/ISPyPSA>`__: PyPSA-based capacity expansion modelling of the Australian Integrated System Plan.
- `eTraGo <https://github.com/openego/eTraGo>`__: optimisation of flexibility options for transmission grids (open_eGo).
- `247-cfe <https://github.com/PyPSA/247-cfe>`__: 24/7 carbon-free energy procurement.
- `tza-pypsa <https://github.com/transition-zero/tza-pypsa>`__: PyPSA model of the Tanzanian power system (Transition Zero).

Energy system models and tools using linopy directly
----------------------------------------------------

- `ZEN-garden <https://github.com/ZEN-universe/ZEN-garden>`__: energy transition model of the ETH Zürich Energy Science Center.
- `flixOpt <https://github.com/flixOpt/flixopt>`__: optimisation of energy and material flow systems.
- `OSeMOSYS_linopy <https://github.com/OSeMOSYS/OSeMOSYS_linopy>`__ and `tz-osemosys <https://github.com/transition-zero/tz-osemosys>`__ (Transition Zero): OSeMOSYS implementations in Python.
- `Dispa-SET <https://github.com/energy-modelling-toolkit/Dispa-SET>`__: unit-commitment and dispatch model of the European Commission's JRC.
- `MicroGridsPy <https://github.com/MicroGridsPy/MicroGridsPy>`__: microgrid sizing and operation.
- `GemsPy <https://github.com/AntaresSimulatorTeam/GemsPy>`__: Python interpreter of the GEMS modelling language (RTE, Antares).
- `PyZEFIR <https://github.com/ncbj-idea/pyzefir>`__: long-term energy system structure model of the National Centre for Nuclear Research (Poland).
- `HAMLET <https://github.com/tum-ens/HAMLET>`__: simulation framework for local energy markets (TUM).
- `ASSUME <https://github.com/assume-framework/assume>`__: agent-based electricity market simulation (partly).
- `MGA-Compass <https://github.com/EINS-TUDa/MGA-Compass>`__: near-optimal alternatives for energy system models.
- `BATTman <https://github.com/rte-france/BATTman>`__: battery optimisation (RTE).
- `odys <https://github.com/ramirocrc/odys>`__: stochastic optimisation of energy asset operation.

Courses and tooling
-------------------

- `Data Science for Energy System Modelling <https://github.com/fneum/data-science-for-esm>`__: university course with linopy exercises.
- `Open Energy Transition solver benchmark <https://github.com/open-energy-transition/solver-benchmark>`__: benchmarks of open and commercial solvers on energy system models.
- `PyOptInterface <https://github.com/metab0t/PyOptInterface>`__, `pyoframe <https://github.com/Bravos-Power/pyoframe>`__ and `GAMSPy <https://github.com/GAMS-dev/gamspy>`__: related modelling interfaces that benchmark against linopy.
- `fluxopt <https://github.com/FBumann/fluxopt>`__: energy system optimisation built on linopy.

Other domains
-------------

- `ak-plan-optimierung <https://github.com/Die-KoMa/ak-plan-optimierung>`__: conference timetable scheduling.
- `water-path <https://github.com/fleet-path/water-path>`__: decarbonisation pathways for vessel fleets.
