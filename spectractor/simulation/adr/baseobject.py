#! /usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import warnings
import numpy as np

__all__ = ["BaseObject"]

""" 
Credit: M.Rigault (m.rigault@ipnl.in2p3.fr)

Source: https://github.com/MickaelRigault/propobject
"""


class BaseObject( object ):
    """Structure to handle the
    _proprerties, _side_properties, _derived_properties
    tricks.

    A class inheriting from BaseObject could have the following
    global-class variables:

    ```
    PROPERTIES         = ["a","b"]
    SIDE_PROPERTIES    = ["t"]
    DERIVED_PROPERTIES = ["whynot"]
    ```
    if so, object created from this class or any inheriting class
    could have access to the self._properties["a"], self._properties["b"]
    self._side_properties["t"] or self._derived_properties["whynot"]
    parameters.

    BaseObject also have a basic .copy() method.
    """

    PROPERTIES         = []
    SIDE_PROPERTIES    = []
    DERIVED_PROPERTIES = []
    
    def __new__(cls,*arg,**kwargs):
        """ Upgrade of the new function to enable the
        _properties,_side_properties and _derived_properties tricks
        """
        obj = super(BaseObject,cls).__new__(cls)
        obj._properties_keys         = copy.copy([])
        obj._side_properties_keys    = copy.copy([])
        obj._derived_properties_keys = copy.copy([])
        # ---------------------------------------------
        # - load also the properties of all the Parents
        for c in obj.__class__.__mro__:
            if "PROPERTIES" in dir(c):
                obj._properties_keys += c.PROPERTIES
            if "SIDE_PROPERTIES" in dir(c):
                obj._side_properties_keys += c.SIDE_PROPERTIES
            if "DERIVED_PROPERTIES" in dir(c):
                obj._derived_properties_keys += c.DERIVED_PROPERTIES

        # -----------------
        # - avoid doublons
        obj._properties_keys         = np.unique(obj._properties_keys).tolist()
        obj._side_properties_keys    = np.unique(obj._side_properties_keys).tolist()
        obj._derived_properties_keys = np.unique(obj._derived_properties_keys).tolist()
        
        # -- keys
        if "_properties" not in dir(obj):
            obj._properties = {}
            obj._side_properties = {}
            obj._derived_properties = {}

        # -- fill empty
        for k in obj._properties_keys:
            if k in obj._properties.keys():
                warnings.warn("%s is already build.  Conflit => new key ignored"%k)
                continue
            obj._properties[k] = None
            
        for k in obj._side_properties_keys:
            if k in obj._side_properties.keys():
                warnings.warn("%s is already build.  Conflit => new key ignored"%k)
                continue
            obj._side_properties[k] = None
            
        for k in obj._derived_properties_keys:
            if k in obj._derived_properties.keys():
                warnings.warn("%s is already build.  Conflit => new key ignored"%k)
                continue
            obj._derived_properties[k] = None

            
        return obj
    
    def __init__(self):
        self.__build__()
        
    def __build__(self):
        """ """
        pass

    # - COPY: Any object inheriting from BaseObject will have a
    #         working copy method.
    def copy(self, empty=False):
        """returns an independent copy of the current object."""
        
        # Create an empty object
        newobject = self.__new__(self.__class__)
        if empty:
            return
        
        # And fill it !
        for prop in ["_properties","_side_properties",
                     "_derived_properties","_build_properties"
                     ]:
            if prop not in dir(self):
                continue
            try: # Try to deep copy because but some time it does not work (e.g. wcs) 
                newobject.__dict__[prop] = copy.deepcopy(self.__dict__[prop])
            except:
                newobject.__dict__[prop] = copy.copy(self.__dict__[prop])
                
        # This be sure things are correct
        newobject._update_()
        # and return it
        return newobject
        
                
    def _update_(self):
        """Adapte the derived properties as a function of the main ones"""
        pass

    # ================ #
    # = Properties   = #
    # ================ #
    @property
    def _fundamental_parameters(self):
        return self._properties.keys()
