"""Test-session setup shared by every test module and every doctest.

On macOS, xgboost's wheel links Homebrew's libomp while torch bundles its own;
a process that loads both crashes or deadlocks unless OpenMP runs one thread.
trader never imports either, but quantlab's dataset and portfolio modules
share a process with them in a developer's environment, so the guard quantlab
uses is set here too, before any import can load an OpenMP runtime.
"""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
