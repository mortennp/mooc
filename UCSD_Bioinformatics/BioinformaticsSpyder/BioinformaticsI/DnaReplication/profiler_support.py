# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 22:09:46 2015

@author: Morten
"""
import builtins


try:
    profile = builtins.profile
except AttributeError:
    # No line profiler, provide a pass-through version
    def profile(func): return func


"""
USAGE IN SCRIPT

from profiler_support import profile

@profile
def my_func():
    return


CMD LINE

kernprof -l my_script
"""
