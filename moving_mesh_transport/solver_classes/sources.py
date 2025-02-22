#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:38:34 2022

@author: bennett
"""
import numpy as np
import math
from numba import float64, int64, deferred_type
from numba.experimental import jitclass

from .build_problem import build
from .functions import normPn, normTn
from .functions import numba_expi as expi
from .sedov_uncollided import sedov_uncollided_solutions
from .uncollided_solutions import uncollided_solution
from .sedov_funcs import sedov_class
from .opacity import sigma_integrator
from scipy.special import expi as expi2
from numba import types, typed
import numba as nb
###############################################################################
build_type = deferred_type()
build_type.define(build.class_type.instance_type)
uncollided_solution_type = deferred_type()
uncollided_solution_type.define(uncollided_solution.class_type.instance_type)
opacity_class_type = deferred_type()
opacity_class_type.define(sigma_integrator.class_type.instance_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))
sedov_type = deferred_type()
sedov_type.define(sedov_class.class_type.instance_type)
sedovuncol_type = deferred_type()
sedovuncol_type.define(sedov_uncollided_solutions.class_type.instance_type)
data = [("S", float64[:]),
        ("source_type", int64[:]),
        ("uncollided", int64),
        ("moving", int64),
        ("M", int64),
        ("x0", float64),
        ("t", float64),
        ("xL", float64),
        ("xR", float64),
        ("argument", float64[:]),
        ('sigma_t', float64),
        ("source_vector", float64[:]),
        ("temp", float64[:]),
        ("abxx", float64),
        ("xx", float64),
        ("ix", int64),
        ("xs_quad", float64[:]),
        ("ws_quad", float64[:]),
        ("mag", float64),
        ("term1", float64),
        ("term2", float64),
        ("tfinal", float64),
        ("t0", float64),
        ("t1", float64),
        ("t2", float64), 
        ("t3", float64),
        ("tau", float64),
        ("sigma", float64),
        ('source_strength', float64),
        ('sigma_s', float64),
        ('geometry', nb.typeof(params_default)),
        ('sigma_func',nb.typeof(params_default))
        
        ]
###############################################################################
@jitclass(data)
class source_class(object):
    def __init__(self, build):
        self.S = np.zeros(build.M+1).transpose()
        self.source_type = np.array(list(build.source_type), dtype = np.int64) 
        self.uncollided = build.uncollided
        self.x0 = build.x0
        self.M = build.M
        self.xs_quad = build.xs_quad
        self.ws_quad = build.ws_quad
        self.moving = build.moving
        self.tfinal = build.tfinal
        self.t0 = build.t0
        self.sigma = build.sigma
        self.sigma_s = build.sigma_s
        self.sigma_t = build.sigma_t
        self.sigma_func = build.sigma_func
        # self.source_strength = 0.0137225 * 299.98
        self.source_strength = build.source_strength
        self.geometry = build.geometry
    
    def integrate_quad(self, t, a, b, j, func):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        self.S[j] = (b-a)/2 * np.sum(self.ws_quad * func(argument, t) * normPn(j, argument, a, b))
    
    def integrate_quad_nonconstant_opacity(self, t,mu, a, b, j, func, opacity_class, sedov, rho_interp, v_interp,type = 'scattering'):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        scattering_evaluated =  opacity_class.sigma_function(argument, t,mu, sedov, rho_interp,v_interp)
        # scattering_evaluated = np.ones(argument.size)
        self.S[j] = (b-a)/2 * np.sum(self.ws_quad * func(argument, t) * scattering_evaluated * normPn(j, argument, a, b))
    
    def integrate_quad_nonconstant_opacityTS(self, t, mu, a, b, j, sedovuncol, opacity_class, sedov, rho_interp, v_interp, type = 'scattering'):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        scattering_evaluated =  opacity_class.sigma_function(argument, t, mu, sedov, rho_interp, v_interp)
        # scattering_evaluated = np.ones(argument.size)
        func = sedovuncol.uncollided_scalar_flux(argument, t, sedov, rho_interp, rho_interp)
        self.S[j] = (b-a)/2 * np.sum(self.ws_quad * func * scattering_evaluated * normPn(j, argument, a, b))

    
    def integrate_quad_sphere(self, t, a, b, j, func):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        self.S[j] = 0.5 * (b-a) * np.sum((argument**2) * np.sqrt(1- self.xs_quad**2) * self.ws_quad * func(argument, t) * 1 * normTn(j, argument, a, b))



    def integrate_quad_not_isotropic(self, t, a, b, j, mu, func):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        self.S[j] = (b-a)/2 * np.sum(self.ws_quad * func(argument, t, mu) * normPn(j, argument, a, b))
    
    def MMS_source(self, xs, t, mu):
        temp = xs*0
        for ix in range(xs.size):
            if -t - self.x0 <= xs[ix] <= t + self.x0:
                # temp[ix] = - math.exp(-xs[ix]*xs[ix]/2)*(1 + (1+t)*xs[ix]*mu)/((1+t)**2)/2
                temp[ix] = -0.5*(1 + (1 + t)*xs[ix]*mu)/(math.exp(xs[ix]**2/2.)*(1 + t)**2)
        return temp*2.0
    
    def square_source(self, xs, t):
        temp = xs*0
        for ix in range(xs.size):
            if abs(xs[ix]) <= self.x0 and t < self.t0:
                temp[ix] = 1.0
        return temp
            
    def gaussian_source(self, xs, t):
        temp = xs*0
        for ix in range(xs.size):
            x = xs[ix]
            if t <= self.t0:
                temp[ix] = math.exp(-x*x/self.sigma**2)
        return temp
        
        
    def make_source(self, t, mu, xL, xR, uncollided_solution, opacity_class, sedov_class, rho_interp, v_interp):
        if self.geometry['slab'] == True:
            if self.uncollided == True:
                if (self.source_type[0] == 1) and  (self.moving == True):
                        self.S[0] = uncollided_solution.plane_IC_uncollided_solution_integrated(t, xL, xR)
                else:
                    if self.sigma_func['constant'] == True:
                        for j in range(self.M+1):
                            self.integrate_quad(t, xL, xR, j, uncollided_solution.uncollided_solution)
                    else:
                        self.integrate_quad_nonconstant_opacity(t, mu, xL, xR, j, uncollided_solution.uncollided_solution, opacity_class, sedov_class, rho_interp, v_interp)
                self.S = self.S * self.sigma_s
            elif self.uncollided == False:
                if self.source_type[2] == 1:
                    for j in range(self.M+1):
                        self.integrate_quad(t, xL, xR, j, self.square_source)
                elif self.source_type[5] == 1:
                    for j in range(self.M+1):
                        self.integrate_quad(t, xL, xR, j, self.gaussian_source)
        
        elif self.geometry['sphere'] == True:
            if self.uncollided == True:
                if self.source_type[1] == 1:
                    for j in range(self.M+1):
                        self.integrate_quad_sphere(t, xL, xR, j, uncollided_solution.uncollided_solution)
                    
                # if self.source_type[0] == 1:
                #     for j in range(self.M+1):
                #         if (xL <= t <= xR):
                #             t = t + 1e-10
                #             self.S[j] = math.exp(-t)/4/math.pi/t * normTn(j, np.array([t]), xL, xR)[0] 

                        

        self.S = self.S * self.source_strength

    def make_source_not_isotropic(self, t, mu, xL, xR):
            if self.source_type[4] ==1:
                for j in range(self.M+1):
                    self.integrate_quad_not_isotropic(t, xL, xR, j, mu, self.MMS_source)
   
    def make_source_TS(self, t, mu, xL, xR, uncollided_solution, opacity_class, sedov_class, rho_interp, v_interp, sedov_uncol):
        if self.uncollided == True:
            for j in range(self.M+1):
                self.integrate_quad_nonconstant_opacityTS(t, mu, xL, xR, j, sedov_uncol, opacity_class, sedov_class, rho_interp, v_interp)


        
        

            
