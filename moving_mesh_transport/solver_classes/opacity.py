import numpy as np
import math
from .build_problem import build
from .sedov_funcs import sedov_class
from numba.experimental import jitclass
from numba import int64, float64, deferred_type, prange
from .functions import Pn, normPn
from numba import types, typed
import numba as nb
import matplotlib.pyplot as plt

build_type = deferred_type()
build_type.define(build.class_type.instance_type)
sedov_type = deferred_type()
sedov_type.define(sedov_class.class_type.instance_type)
kv_ty = (types.int64, types.unicode_type)
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))

data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('sigma_t', float64),
        ('sigma_s', float64),
        ('sigma_a', float64),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('x0', float64),
        ("xL", float64),
        ("xR", float64),
        ('sigma_func', nb.typeof(params_default)),
        ('Msigma', int64),
        ('AAA', float64[:,:,:]),
        ('xs_quad', float64[:]),
        ('ws_quad', float64[:]),
        ('edges', float64[:]),
        ('std', float64), 
        ('cs', float64[:,:,:]), 
        ('VV', float64[:]),
        ('VP', float64[:]),
        ('moving', float64),
        ('sigma_v', float64), 
        ('fake_sedov_v0', float64),
        ('csP', float64[:,:,:]),
        ('cs_moment_1', float64[:,:,:]),
        ('N_ang', int64),
        ('mus', float64[:])

        ]


@ jitclass(data)
class sigma_integrator():
    def __init__(self, build):
        self.sigma_t = build.sigma_t
        self.sigma_s = build.sigma_s
        print(self.sigma_s,'sigma_s')
        self.N_ang = build.N_ang
        self.sigma_a = self.sigma_t - self.sigma_s
        print(self.sigma_a,'sigma_a')
        self.sigma_func = build.sigma_func
        self.M = build.M
        self.Msigma = build.Msigma
        self.xs_quad = build.xs_quad
        self.ws_quad = build.ws_quad
        self.std = 2.0
        self.N_space = build.N_space
        self.edges = np.zeros(self.N_space + 1)
        self.cs = np.zeros((self.N_ang, self.N_space, self.Msigma+ 1))
        self.csP = np.zeros((self.N_ang, self.N_space, self.Msigma+ 1))
        self.cs_moment_1 = np.zeros((self.N_ang, self.N_space, self.Msigma+ 1))

        self.VV = np.zeros(self.M+1)
        self.VP = np.zeros(self.M+1)
        self.AAA = np.zeros((self.M+1, self.M + 1, self.Msigma + 1))
        self.moving = False
        
        self.mus = build.mus
        # if self.sigma_func['fake_sedov'] == True:
        #     self.moving = True
        # self.sigma_v = 0.005
        self.sigma_v = build.fake_sedov_v0
        # assert(self.sigma_v == 0.0035)

        # initialize integrals of Basis Legendre polynomials
        self.create_integral_matrices()

    def integrate_quad(self, a, b, i, j, k):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        fact = np.sqrt(2*i + 1) * np.sqrt(2*j + 1) * np.sqrt(2*k + 1) / 2
        self.AAA[i,j,k] = fact * (b-a)/2 * np.sum(self.ws_quad *  Pn(i, argument, a, b) * Pn(j, argument, a, b) * Pn(k, argument, a, b))
    
    def integrate_moments(self, a, b, j, k, l, t, mu, sedov, rho_interp, v_interp):
        argument = (b-a)/2 * self.xs_quad + (a+b)/2
        self.cs[l, k, j] = (b-a)/2 * np.sum(self.ws_quad * self.sigma_function(argument, t, mu, sedov, rho_interp, v_interp, 'total' ) * normPn(j, argument, a, b))
        if self.sigma_s >0:
            self.csP[l, k, j] = (b-a)/2 * np.sum(self.ws_quad * self.sigma_function(argument, t,mu, sedov,rho_interp, v_interp, 'scattering' ) * normPn(j, argument, a, b))
            self.cs_moment_1[l, k, j] = (b-a)/2 * np.sum(self.ws_quad * self.sigma_function(argument, t,mu, sedov,rho_interp, v_interp, 'scattering1' ) * normPn(j, argument, a, b))
        
    def both_even_or_odd(self, i, j, k):
        if i % 2 == 0:
            if (j + k) % 2 == 0:
                return True
            else:
                return False
        if i % 2 != 0:
            if (j + k) % 2 != 0:
                return True
            else:
                return False

    def create_integral_matrices(self):
        """
        creates a matrix with every integral over [-1,1] of the three normalized Legendre polynomials of order
        i, j, k. Each entry must be divided by sqrt(xR-xL) 
        """
        for i in range(self.M + 1):
            for j in range(self.M + 1):
                for k in range(self.Msigma + 1):
                    if (j + k >= i) and (self.both_even_or_odd(i, j, k)):
                        self.integrate_quad(-1, 1, i, j, k)
        # print(self.AAA)
    
    def sigma_moments(self, edges, t, sedov, rho_interp, v_interp):
        for i in range(self.N_space):
                for l in range(self.N_ang):
            # if (edges[i] != self.edges[i]) or (edges[i+1] != self.edges[i+1]) or self.moving == True :
                    for j in range(self.Msigma + 1):
                        self.integrate_moments(edges[i], edges[i+1], j, i,l, t, self.mus[l], sedov, rho_interp, v_interp)
        self.edges = edges
        
    
    def xi2(self, x, t, x0, c1, v0tilde):
        return -x - c1 - v0tilde*(t)

    def heaviside(self,x):
        if x < 0.0:
            return 0.0
        else:
            return 1.0

    def heaviside_vector(self, x):
        return_array = np.ones(x.size)
        for ix, xx in enumerate(x):
            if xx < 0:
                return_array[ix] = 0.0
        return return_array

    def sigma_function(self, x, t, mu, sedov, rho_interp, v_interp, type = 'absorption'):
        if self.sigma_func['constant'] == 1:
            return x * 0 + 1.0
        elif self.sigma_func['gaussian'] == 1:
            return np.exp(- x**2 /(2* self.std**2))  # probably shouldn't have sigma_a here
            # return x * 0 + 1.0
        elif self.sigma_func['siewert1'] == 1: # siewert with omega_0 = 1, s = 1
            return np.exp(-x - 2.5)
        elif self.sigma_func['siewert2'] == 1:
            return np.exp(-x/100000000000)
        elif self.sigma_func['fake_sedov'] == 1:
            # return np.exp(-(x- self.sigma_v * t)**2/(2*self.std**2))
            c1 = 1
            xi2x = self.xi2(x, t, 0, c1, self.sigma_v)
            rho2 = 0.1
            res = np.exp(-xi2x**2/self.std**2) * self.heaviside_vector(-xi2x - c1) + rho2*self.heaviside_vector(xi2x + c1)
            # vec_test = self.heaviside_vector(-xi2x - c1)
            # found = False
            # index = 0
            # if np.any(vec_test == 1):
            #     if np.any(x < 0):
            #         print(vec_test)
            #         print(x, t)
                

            #     if np.am y(vec_test == 0):
            #         while found == False and index < x.size:
            #             if vec_test[index] == 1:
            #                 found == True
            #                 print(x[index], 'location of shock', t, 't')
            #                 print(vec_test)
            #                 print(-self.sigma_v*t - x[index])
            #                 print("#--- --- --- --- --- --- ---#")
            #             index += 1
            return res
        elif self.sigma_func['TaylorSedov'] == 1:
            lambda1 = 3.2
            # if type == 'absorption':
                # plt.figure(79)
                # plt.ion()
            velocity = sedov.interpolate_self_similar_v(t, x, v_interp)
        # velocity is in cm/s 
            beta = velocity / (2.998e10) 
            if (beta >= 1).any():
                raise ValueError('Either Einstein is wrong or you are. ')
        # print(beta)
            gamma = 1/np.sqrt(1-beta**2)

            correction = gamma * (1 - mu * beta)
            # plt.plot(x, sedov.interpolate_self_similar(t, x, rho_interp) ** lambda1)
            # print(sedov.interpolate_self_similar(t, x, rho_interp) ** lambda1, x)
            sigma_a_res =  sedov.interpolate_self_similar(t, x, rho_interp) ** lambda1
                # return np.ones(x.size)
                
            # elif type == 'scattering':
            

            quiet_ionization = 0.00
            shocked_ionization = 1.0
            ionized_array = x * 0
            sedov.physical(t)
            # ionized_array = (abs(x) )
            ionized_array = x * 0 + quiet_ionization
            for ix, xx in enumerate(ionized_array):
                if abs(x[ix]) <= sedov.r2:
                    ionized_array[ix] = shocked_ionization
            sigma_s_res =  gamma * sedov.interpolate_self_similar(t, x, rho_interp) * ionized_array
            
            sigma_s_moment_1 = - beta * gamma * sedov.interpolate_self_similar(t, x, rho_interp) * ionized_array

            if type == 'scattering':
                return sigma_s_res/ correction **2 * self.sigma_s
            elif type == 'absorption':
                return sigma_a_res * correction * self.sigma_a
            elif type == 'total':
                return (sigma_a_res * (self.sigma_t-self.sigma_s) + sigma_s_res * self.sigma_s) * correction
            elif type == 'scattering1':
                return sigma_s_moment_1/ correction **2 * self.sigma_s
                
                # 
            else:
                assert(0)
        
    
    def make_vectors(self, edges, u, space, l):
        self.VV = u * 0
        # self.sigma_moments(edges) # take moments of the opacity
        xL = edges[space]
        xR = edges[space+1]
        dx = math.sqrt(xR-xL)
        # if self.sigma_func['constant'] == True:
        #     self.VV = u * self.sigma_t
        # else:
            # self.VV = u * self.sigma_t
        for i in range(self.M + 1):
            for j in range(self.M + 1):
                for k in range(self.Msigma + 1):
                    self.VV[i] +=  (1/ self.sigma_t) * self.cs[l, space, k] * u[j] * self.AAA[i, j, k] / dx
                    # self.VV[i] +=  (self.sigma_s / self.sigma_t) * self.csP[space, k] * u[j] * self.AAA[i, j, k] / dx




    





