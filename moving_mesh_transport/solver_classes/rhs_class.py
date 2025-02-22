#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:25:35 2022

@author: bennett
"""
import numpy as np
import math

from .build_problem import build
from .matrices import G_L
from .sources import source_class
from .phi_class import scalar_flux
from .uncollided_solutions import uncollided_solution
from .sedov_funcs import sedov_class
from .sedov_uncollided import sedov_uncollided_solutions
from .numerical_flux import LU_surf
from .radiative_transfer import T_function
from .opacity import sigma_integrator
from .functions import shaper 
from .cubic_spline import cubic_spline_ob as cubic_spline

from numba.experimental import jitclass
from numba import int64, float64, deferred_type, prange
from numba import types, typed 

import numba as nb

build_type = deferred_type()
build_type.define(build.class_type.instance_type)
matrices_type = deferred_type()
matrices_type.define(G_L.class_type.instance_type)
num_flux_type = deferred_type()
num_flux_type.define(LU_surf.class_type.instance_type)
source_type = deferred_type()
source_type.define(source_class.class_type.instance_type)
flux_type = deferred_type()
flux_type.define(scalar_flux.class_type.instance_type)
uncollided_solution_type = deferred_type()
uncollided_solution_type.define(uncollided_solution.class_type.instance_type)
transfer_class_type = deferred_type()
transfer_class_type.define(T_function.class_type.instance_type)
sigma_class_type = deferred_type()
sigma_class_type.define(sigma_integrator.class_type.instance_type)
sedov_type = deferred_type()
sedov_type.define(sedov_class.class_type.instance_type)
spline_type = deferred_type()
spline_type.define(cubic_spline.class_type.instance_type)
sedovuncol_type = deferred_type()
sedovuncol_type.define(sedov_uncollided_solutions.class_type.instance_type)

params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))


data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('source_type', int64[:]),
        ('t', float64),
        ('sigma_t', float64),
        ('sigma_s', float64),
        ('IC', float64[:,:,:]),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('x0', float64),
        ("xL", float64),
        ("xR", float64),
        ("dxL", float64),
        ("dxR", float64),
        ("L", float64[:,:]),
        ("G", float64[:,:]),
        ("P", float64[:]),
        ("PV", float64[:]),
        ("S", float64[:]),
        ("LU", float64[:]),
        ("U", float64[:]),
        ("H", float64[:]),
        ("V_new", float64[:,:,:]),
        ("V", float64[:,:,:]),
        ("V_old", float64[:,:,:]),
        ('c', float64),
        ('uncollided', int64),
        ('thermal_couple', nb.typeof(params_default)),
        ('test_dimensional_rhs', int64),
        ('told', float64),
        ('division', float64),
        ('c_a', float64),
        ('sigma_a', float64),
        ('mean_free_time', float64),
        ('counter', int64),
        ('delta_tavg', float64),
        ('l', float64),
        ('times_list', float64[:]),
        ('save_derivative', int64),
        ('e_list', float64[:]),
        ('e_xs_list', float64[:]),
        ('wave_loc_list', float64[:]),
        ('sigma_func', nb.typeof(params_default)),
        ('particle_v', float64),
        ('epsilon', float64),
        ('geometry', nb.typeof(params_default)),
        ]
##############################################################################
@jitclass(data)
class rhs_class():
    def __init__(self, build):
        self.N_ang = build.N_ang 
        self.N_space = build.N_space
        self.M = build.M
        self.mus = build.mus
        self.ws = build.ws
        self.source_type = np.array(list(build.source_type), dtype = np.int64) 
        self.c = build.scattering_ratio
        self.thermal_couple = build.thermal_couple
        self.uncollided = build.uncollided
        self.test_dimensional_rhs = build.test_dimensional_rhs
        self.told = 0.0
        self.sigma_s = build.sigma_s
        self.sigma_a = build.sigma_a
        self.particle_v = build.particle_v
        self.geometry = build.geometry
        self.c_a = build.sigma_a / build.sigma_t
        self.mean_free_time = 1/build.sigma_t
        self.division = 1000
        self.counter = 0
        self.delta_tavg = 0.0
        self.l = build.l
        self.times_list = np.array([0.0])
        self.e_list = np.array([0.0])
        self.e_xs_list = np.array([0.0])
        self.wave_loc_list = np.array([0.0])
        self.save_derivative = build.save_wave_loc
        self.sigma_func = build.sigma_func
        # self.deg_freedom = shaper(self.N_ang, self.N_space, self.M + 1, self.thermal_couple)

    
    def time_step_counter(self, t, mesh):
        delta_t = abs(self.told - t)
        self.delta_tavg += delta_t / self.division
        if self.counter == self.division:
            print('t = ', t, '|', 'delta_t average= ', self.delta_tavg)
            if self.N_space <= 32:
                print(mesh.edges)
            print('--- --- --- --- --- --- --- --- --- --- --- --- --- ---')
            self.delta_tavg = 0.0
            self.counter = 0
        else:
            self.counter += 1
        self.told = t
    
    # def interpolate_sedov(self, t, sedov_class):
    #     if self.sigma_func['TaylorSedov'] == 1:
    #         density, velocity, rs = sedov_class.physical(t)
    #         rs2 = np.flip(rs)
    #         rs2[0] = 0.0
    #         density2 = np.flip(density)
    #         density2[0] = 0.0
    #         # density2[-1] = sedov_class.gpogm
    #         rs2[-1] = sedov_class.r2
    #         interpolated_density = cubic_spline(rs2, density2)
    #         interpolated_velocity = cubic_spline(rs2, np.flip(velocity))
    #     else:
    #         interpolated_density = cubic_spline(np.linspace(0,1,2), np.linspace(0,1,2))
    #         interpolated_velocity = cubic_spline(np.linspace(0,1,2), np.linspace(0,1,2))
    #     return interpolated_density, interpolated_velocity

    def interp_sedov_selfsim(self, sedov_class):
         if self.sigma_func['TaylorSedov'] == 1:
             g_fun = sedov_class.g_fun
             l_fun = sedov_class.l_fun
             f_fun = sedov_class.f_fun
             l_fun = np.flip(l_fun)
             l_fun[0] = 0.0
             l_fun[-1] = 1.0
             f_fun[-1] = 0.0
             
             g_fun = np.flip(g_fun)
             g_fun[-1] = 1.0
             
            #  g_fun[-1] = sedov_class.gpogm
            #  l_fun[-1] = 1.0 

             interpolated_g = cubic_spline(l_fun, g_fun)

             interpolated_f = cubic_spline(l_fun, f_fun)

             return interpolated_g, interpolated_f




        
    def derivative_saver(self, t,  space, transfer_class):
        if self.save_derivative == True:
            self.e_list = np.append(self.e_list, transfer_class.e_points)
            self.e_xs_list = np.append(self.e_xs_list, transfer_class.xs_points)

        if space == self.N_space - 1:
            deriv = np.copy(self.e_list)*0
            for ix in range(1,self.e_list.size-1):
                dx = self.e_xs_list[ix+1] - self.e_xs_list[ix]
                deriv[ix] = (self.e_list[ix+1] - self.e_list[ix])/dx

            max_deriv = max(np.abs(deriv))
            max_deriv_loc = np.argmin(np.abs(np.abs(self.e_list) - max_deriv))
            heat_wave_loc = self.e_xs_list[max_deriv_loc]
            self.wave_loc_list = np.append(self.wave_loc_list, abs(heat_wave_loc)) 
            self.times_list = np.append(self.times_list,t)
            # print(heat_wave_loc, 'wave x')
        
    def call(self, t, V, mesh, matrices, num_flux, source, uncollided_sol, flux, transfer_class, sigma_class, sedov_class, g_interp, v_interp, sedov_uncol):
        self.time_step_counter(t, mesh)
        
        if self.thermal_couple['none'] == 1:
            V_new = V.copy().reshape((self.N_ang, self.N_space, self.M+1))
        elif self.thermal_couple['none'] == 0:
            V_new = V.copy().reshape((self.N_ang + 1, self.N_space, self.M+1))
        V_old = V_new.copy()

        mesh.move(t)
        # mesh.edges = np.linspace(-0.15, 0.15, self.N_space+1)
        # mesh.Dedges = mesh.edges * 0
        # rho_interp, v_interp = self.interpolate_sedov(t, sedov_class)
        # g_interp = self.interp_sedov_selfsim(sedov_class)


        if self.sigma_func['TaylorSedov'] == 1:
            # sedov_class.physical(t)
            sedov_class.physical(t)
            mesh.get_shock_location_TS(sedov_class.r2_dim , sedov_class.vr2_dim,sedov_class.t_shift, (sedov_class.eblast/(sedov_class.alpha*sedov_class.rho0))**(1.0/3), t, self.sigma_a + self.sigma_s)

        
        sigma_class.sigma_moments(mesh.edges, t, sedov_class, g_interp, v_interp)
            
        flux.get_coeffs(sigma_class)
    
           

        for space in range(self.N_space):


            xR = mesh.edges[space+1]
            xL = mesh.edges[space]
            dxR = mesh.Dedges[space+1]
            dxL = mesh.Dedges[space]
            matrices.make_L(xL, xR)
            matrices.make_G(xL, xR, dxL, dxR)
            matrices.make_all_matrices(xL, xR, dxL, dxR)
            L = matrices.L
            G = matrices.G
            if (self.sigma_func['constant'] == 1) or (self.c == 0.0):
                P = flux.P
            else:
                flux.make_P(V_old[:, space, :], space, xL, xR)
            # flux.make_P(V_old[:,space,:], space, xL, xR)
            
            
                # else:

            # if self.thermal_couple['none'] == 0:
            #     transfer_class.make_H(xL, xR, V_old[self.N_ang, space, :])
            #     H = transfer_class.H
            # else: 
            H = np.zeros(self.M+1)

            self.derivative_saver(t, space, transfer_class)
  
            ######### solve thermal couple ############
            # if self.thermal_couple['none'] == 0:
            #     U = np.zeros(self.M+1).transpose()
            #     U[:] = V_old[self.N_ang,space,:]
            #     num_flux.make_LU(t, mesh, V_old[self.N_ang,:,:], space, 0.0)
            #     RU = num_flux.LU
            #     if self.test_dimensional_rhs == True:
            #         RHS_energy = np.dot(G,U) - RU + self.c_a * (2.0 * P  - H)
            #     else:
            #         RHS_energy = np.dot(G,U) - RU + self.c_a * (2.0 * P  - H) / self.l
                
            #     if self.uncollided == True:
            #         RHS_energy += self.c_a * source.S /self.l
            #     V_new[self.N_ang,space,:] = RHS_energy
                
            # elif self.thermal_couple == 1 and self.N_ang == 2:
            #     U = np.zeros(self.M+1).transpose()
            #     U[:] = V_old[self.N_ang,space,:]
            #     num_flux.make_LU(t, mesh, V_old[self.N_ang,:,:], space, 0.0)
            #     RU = num_flux.LU
            #     RHS_energy = np.dot(G,U) - RU + self.c_a * (2.0 * P  - H) / self.l
            #     if self.uncollided == True:
            #         RHS_energy += self.c_a * source.S / self.l
            #     V_new[self.N_ang ,space,:] = RHS_energy
                
            ########## Loop over angle ############
            for angle in range(self.N_ang):
          

      
                if self.sigma_func['TaylorSedov'] != 1:
                    source.make_source(t,angle, xL, xR, uncollided_sol, sigma_class, sedov_class, g_interp, v_interp)
                elif self.sigma_func['TaylorSedov'] == 1:
                    source.make_source_TS(t,angle, xL, xR, uncollided_sol, sigma_class, sedov_class, g_interp,v_interp, sedov_uncol)
                S = source.S
                # print(1)
                
                
                mul = self.mus[angle]
                # if self.source_type[4] == 1: # Make MMS source
                #     source.make_source_not_isotropic(t, mul, xL, xR)
                num_flux.make_LU(t, mesh, V_old[angle,:,:], space, mul)
                LU = num_flux.LU
                U = np.zeros(self.M+1).transpose()
                U[:] = V_old[angle,space,:]
                
                
                if self.thermal_couple['none'] == 1:
                    
                    deg_freedom = self.N_ang * self.N_space * (self.M+1)
                    if self.sigma_func['constant'] == 1:
                        if self.uncollided == False:
                            if self.test_dimensional_rhs == False:
                                if self.geometry['slab'] == True:
                                    RHS = np.dot(G,U)  - LU + mul*np.dot(L,U) - U + self.c * P + 0.5*S 
                                elif self.geometry['sphere'] == True:
                                    M = matrices.Mass
                                    J = matrices.J
                                    VV = sigma_class.VV
                                    Minv = np.linalg.inv(M)
                                    RHS = np.dot(G,U)  - LU + mul*np.dot(L,U) - np.dot(M,VV) + self.c * np.dot(M,P) + 0.5*S/4/math.pi
                                    RHS = np.dot(Minv, RHS) 

                            else:
                                epsilon = self.epsilon
                                RHS = np.dot(G,U)  - LU/ epsilon + mul*np.dot(L,U)/ epsilon - U/epsilon**2 + self.c * P/ epsilon**2 + 0.5*S 
                                # RHS = np.dot(G,U)  - LU/epsilon + mul*np.dot(L,U)/epsilon - self.sigma_s*U/epsilon**2 + self.sigma_s * P/epsilon**2 + 0.5*S 

                        elif self.uncollided == True:
                            RHS = np.dot(G,U)  - LU + mul*np.dot(L,U) - U + self.c * (P + 0.5*S)

                    elif self.sigma_func['siewert1']== 1 or self.sigma_func['siewert2']== 1: #siewert problem
                        VV = sigma_class.make_vectors(mesh.edges, V_old[angle,space,:], space, angle)
                        PV = flux.call_P_noncon(xL, xR)
                        # Q = np.zeros(PV.size) # this is for testing the steady state source problem
                        # Q[0] = math.sqrt(xR-xL)
                        RHS = np.dot(G,U)  - LU + mul*np.dot(L,U) - U + PV
                    
                    else:
                        sigma_class.make_vectors(mesh.edges, V_old[angle,space,:], space, angle)
                        VV = sigma_class.VV
                        if self.c != 0.0:
                            PV = flux.call_P_noncon(xL, xR)
                        else:
                            PV = VV * 0
                        # PV =  self.sigma_s*flux.P
                        # PV = VV*0
                        # print(np.abs(PV-flux.P))
                        # assert(np.abs(flux.cs[space,:] - sigma_class.cs[space,:]).all() <= 1e-10)
                        # if (np.abs(self.sigma_s * flux.P - PV).all() > 1e-6):
                        #     print(flux.P - PV)

                        # A = np.dot(G,U)  
                        # A -= LU 
                        # A += mul*np.dot(L,U)
                        # A -= VV

                        # A += PV 
                        # A += 0.5*self.c*S
                        # RHS = A
                        RHS = np.dot(G,U)  - LU + mul*np.dot(L,U) - VV + PV  + 0.5 * self.c * S
                        RHS = RHS 
                    V_new[angle,space,:] = RHS
                    
                # elif self.thermal_couple['none'] == 0:

                #     deg_freedom = (self.N_ang + 1) * self.N_space * (self.M+1)
                    
                #     if self.N_ang == 2:
                #         if self.uncollided == True:
                #             RHS_transport = np.dot(G,U) - LU + mul*np.dot(L,U) - U/self.l + self.c * (P/self.l + 0.5*S/self.l) + self.c_a*0.5*H/self.l
                #         elif self.uncollided == False:
                #             RHS_transport = np.dot(G,U) - LU + mul*np.dot(L,U) - U/self.l + self.c*P/self.l + 0.5*S/self.l + self.c_a*0.5*H/self.l
                #     elif self.N_ang !=2:
                #         if self.uncollided == True:
                #             RHS_transport = np.dot(G,U) - LU + mul*np.dot(L,U) - U/self.l + self.c * (P + S*0.5)/self.l + self.c_a*0.5*H/self.l
                #         elif self.uncollided == False:
                #             if self.test_dimensional_rhs == True:
                #                 RHS_transport = np.dot(G,U) - LU + 299.98*mul*np.dot(L,U) - 299.98*U + 299.98*self.c * P + 299.98 * S*0.5 + 299.98*self.c_a*0.5*H
                #             else:
                #                 RHS_transport = np.dot(G,U) - LU + mul*np.dot(L,U) - U/self.l + self.c * P /self.l + S*0.5/self.l + self.c_a*0.5*H/self.l
                #     V_new[angle,space,:] = RHS_transport 
                    
        return V_new.reshape(deg_freedom)
    
       
