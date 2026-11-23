"""XR teleoperation of a FAIRINO FR10 cobot (simulation and real robot).

The package is deliberately import-light: importing ``fr10`` pulls in nothing heavy,
so the simulator process (a different conda environment) can share ``fr10.comm`` and
``fr10.config`` without needing Pinocchio, CasADi or the XR stack.
"""

__version__ = "1.0.0"
