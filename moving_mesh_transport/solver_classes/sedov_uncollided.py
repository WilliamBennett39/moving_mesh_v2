import numba as nb
import numpy as np
from numba.experimental import jitclass
from numba import types, typed
from numba import int64, float64, jit, njit, deferred_type
from .build_problem import build
from .cubic_spline import cubic_spline_ob as cubic_spline
from .sedov_funcs import sedov_class
from tqdm import tqdm
import math



spline_type = deferred_type()
spline_type.define(cubic_spline.class_type.instance_type)
sedov_type = deferred_type()
sedov_type.define(sedov_class.class_type.instance_type)
build_type = deferred_type()
build_type.define(build.class_type.instance_type)

data = [('x0', float64),
        ('xs_quad', float64[:]),
        ('ws_quad', float64[:]),
        ('sigma_a', float64),
        ('lambda1', float64),
        ('t0source', float64),
        ('mu_quad', float64[:]),
        ('mu_ws', float64[:]),
        ('transform', float64),
        ('relativistic', int64)
        ]

@jitclass(data)
class sedov_uncollided_solutions(object):
    def __init__(self, xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_a, t0, transform = True, lambda_1 = 3.5, relativistic = True):
        self.xs_quad = xs_quad
        self.ws_quad = ws_quad
        self.x0 = x0
        # print(self.x0 , 'x0 in uncollided')
        self.lambda1 = lambda_1
        # self.lambda1 = 1.0 
        # print(self.lambda1, 'lambda1')
        self.sigma_a = 1e-3
        self.t0source = t0
        # print(self.t0source, 't0')
        self.mu_quad = mu_quad
        self.mu_ws = mu_ws
        self.transform = transform
        self.relativistic = relativistic

    
    # def get_sedov_density(self, rho_interp, v_interp, xs, sedov_class):
    #     rho, v  = sedov_class.interpolate_solution(0.0, xs, rho_interp, v_interp)

    #     return rho
    # def get_sedov_velocity(self, rho_interp, v_interp, xs, sedov_class):
    #     rho, v  = sedov_class.interpolate_solution(0.0, xs, rho_interp, v_interp)

    #     return v

    def get_upper_integral_bounds(self, x, mu, t, sedov_class, v_interpolated):

        tp = (-self.x0 - x) / mu + t

        if tp < t:
            # tau_int = self.integrate_quad_velocity(tp, t, x, sedov_class, v_interpolated)
            tau_int = 0.0
        else:
            tau_int = 0.0
        

        return x - tau_int*0 


    def integrate_quad_velocity(self, a, b, x, sedov_class, v_interpolated):
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        func = self.xs_quad * 0
        for ix, xx in enumerate(self.xs_quad):
            func[ix] = sedov_class.interpolate_self_similar_v(argument[ix], np.array([x]), v_interpolated)[0] * np.sign(-x)
        res = (b-a)/2 * np.sum(self.ws_quad * func )
        return  res
    
    def integrate_quad_sigma(self, a, b, mu, x, t, sedov, g_interpolated, v_interpolated):
        func = self.xs_quad * 0 
        # integral_bound = sedov.find_r2_in_transformed_space(x, t, mu, self.x0)
        # print(integral_bound)
        argument = (b-a)/2*self.xs_quad + (a+b)/2
        for ix, xx in enumerate(self.xs_quad):
            func[ix] = self.transformed_sigma(argument[ix], x, t, mu, sedov, g_interpolated, v_interpolated)
        res = (b-a)/2 * np.sum(self.ws_quad * func )
        return res
        

    def sigma_func(self, x, t, mu, sedov_class, g_interpolated, v_interpolated):
        sigma = sedov_class.interpolate_self_similar(t, np.array([x]), g_interpolated)[0] ** self.lambda1
        correction = 1.0

        if self.relativistic == True:
            velocity = sedov_class.interpolate_self_similar_v(t, np.array([x]), v_interpolated)[0]
            # velocity is in cm/s 
            beta = velocity / (2.998e10) 
            if beta >= 1:
                raise ValueError('Either Einstein is wrong or you are. ')
            # print(beta)
            gamma = 1/np.sqrt(1-beta**2)
    
            correction = gamma * (1 - mu * beta)
            if np.isnan(correction):
                print('nan correction term')
                print(gamma, 'gamma')
                print(velocity, 'velocity')
                print(beta, 'beta')
                assert 0

        return sigma * correction
    
    def chi_func(self, s, x, mu, t):
        return (s-x) / mu + t
    
    def integrate_sigma(self, x, mu, t, sedov_class, g_interpolated, v_interpolated):
        lower_bound = -self.x0
        
        if self.transform == True:
            # integral_bound1, integral_bound2 = sedov_class.find_r2_in_transformed_space(x, t, mu, self.x0)
            testr2a, testr2b = sedov_class.analytic_contact_func(mu, t, self.x0, self.sigma_a)
            # print(testr2a,integral_bound1)
            v = 29.98
            sigma_t = self.sigma_a
            c1 =  (sedov_class.eblast/(sedov_class.alpha*sedov_class.rho0))**(1.0/3)* (10**-9 /v/sigma_t)**(2/3) * sigma_t
            # chi = 
            # print(c1)
            # tt = (integral_bound1-x)/mu + t
            a = x-t*mu
            xitest =   -0.3333333333333333*(c1**3 + 3*a*mu**2)/mu**3 - (2**0.3333333333333333*(-c1**6 - 6*a*c1**3*mu**2))/(3.*mu**3*(-2*c1**9 - 18*a*c1**6*mu**2 - 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333) + (-2*c1**9 - 18*a*c1**6*mu**2 - 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333/(3.*2**0.3333333333333333*mu**3)
            # print(tt,xitest)
            xbtest = -0.3333333333333333*(-c1**3 + 3*a*mu**2)/mu**3 - (2**0.3333333333333333*(-c1**6 + 6*a*c1**3*mu**2))/(3.*mu**3*(2*c1**9 - 18*a*c1**6*mu**2 + 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(-4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333) + (2*c1**9 - 18*a*c1**6*mu**2 + 27*a**2*c1**3*mu**4 + 3*math.sqrt(3)*math.sqrt(-4*a**3*c1**9*mu**6 + 27*a**4*c1**6*mu**8))**0.3333333333333333/(3.*2**0.3333333333333333*mu**3)
            # print(-c1 * tt**(2/3), integral_bound1)
            approx1 =         -((-(mu*t) + x)/mu) + (c1*((-(c1**3*mu**4*(-(mu*t) + x)**2) + math.sqrt(c1**6*mu**8*(-(mu*t) + x)**4))/c1**3)**0.3333333333333333)/(2**0.3333333333333333*mu**3)
            r2atest = (xitest-t)*mu + x
            r2btest = (xbtest-t)*mu + x
            approx_test = (approx1-t)*mu + x
            
            # print(integral_bound1)
            # print(testr2b)
            # print(r2atest-integral_bound1)
            # print(r2btest - integral_bound2)
            integral_bound1 = r2atest
            integral_bound2 = r2btest
            # print(approx_test-integral_bound1)
            
        else:
            sedov_class.physical(t)
            integral_bound1 = -sedov_class.r2 * self.sigma_a
            integral_bound1 = sedov_class.r2 * self.sigma_a
            # print(integral_bound1 -sedov_class.find_r2_in_transformed_space(x, t, mu, self.x0)[0] )

            # sedov_class.physical(t)
            # integral_bound1 = -sedov_class.r2 * self.sigma_a
            # integral_bound2 = -sedov_class.r2 * self.sigma_a
            integral_bound1, integral_bound2 = sedov_class.find_r2_in_transformed_space(x, t, mu, self.x0)
        # # upper_bound = self.get_upper_integral_bounds(x, mu, t, sedov_class, v_interpolated)
        upper_bound = x
        res = 0.0
        # print(integral_bound1, integral_bound2, x)
        if integral_bound1 > lower_bound and integral_bound1 < upper_bound:

            res += self.integrate_quad_sigma(lower_bound, integral_bound1, mu, x, t, sedov_class, g_interpolated, v_interpolated)
            if integral_bound2 < upper_bound:
                res += self.integrate_quad_sigma(integral_bound1, integral_bound2, mu, x, t, sedov_class, g_interpolated, v_interpolated)
                res += self.integrate_quad_sigma(integral_bound2, upper_bound, mu, x, t, sedov_class, g_interpolated, v_interpolated)
                
            else:
                res += self.integrate_quad_sigma(integral_bound1, upper_bound, mu, x, t, sedov_class, g_interpolated, v_interpolated)




        # if (-integral_bound < upper_bound) and (-integral_bound > -self.x0) and integral_bound != -1:
        #     res += self.integrate_quad_sigma(lower_bound, -integral_bound, mu, x, t, sedov_class, g_interpolated)
        #     if integral_bound < upper_bound:
        #         res += self.integrate_quad_sigma(-integral_bound, integral_bound, mu, x, t, sedov_class, g_interpolated)
        #         res += self.integrate_quad_sigma(integral_bound, upper_bound, mu, x, t, sedov_class, g_interpolated)
        #     else:
        #         res += self.integrate_quad_sigma(-integral_bound, upper_bound, mu, x, t, sedov_class, g_interpolated)

        else: 
            res += self.integrate_quad_sigma(lower_bound, upper_bound, mu, x, t, sedov_class, g_interpolated, v_interpolated)
        return res
    
    def transformed_sigma(self, s, x, t, mu, sedov_class, g_interpolated, v_interpolated):
        tt = self.chi_func(s, x, mu, t)
        if self.transform == True:
            return self.sigma_func(s, tt, mu, sedov_class, g_interpolated, v_interpolated) 
        else:
            # assert(0)
            return self.sigma_func(s, t, mu, sedov_class, g_interpolated, v_interpolated)
        # return self.sigma_func(s, t, sedov_class, g_interpolated)
    
    def heaviside(self, x):
        res = x * 0.0
        for ix, xx in enumerate(x):
            if xx > 0.0:
                res[ix] = 1.0
        return res
    
    
    def get_mfp(self, xs, tfinal, mu, sedov_class, g_interpolated, v_interpolated):
        heaviside_array = self.heaviside(mu - np.abs(xs + self.x0)/ (tfinal)) * self.heaviside(np.abs(-self.x0-xs) - (tfinal-self.t0source)*mu)
        mfp_array = xs * 0

        if mu > 0.0:
            for ix, xx, in enumerate(xs):
                if heaviside_array[ix] != 0.0:
                    mfp_array[ix] = self.integrate_sigma(xx, mu, tfinal, sedov_class, g_interpolated, v_interpolated)
            return mfp_array
        else:
            return xs * 0.0




    def uncollided_angular_flux(self, xs, tfinal, mu, sedov_class, g_interpolated, v_interpolated):
        heaviside_array = self.heaviside(mu - np.abs(xs + self.x0)/ (tfinal)) * self.heaviside(np.abs(-self.x0-xs) - (tfinal-self.t0source)*mu)
        mfp_array = xs * 0

        if mu > 0.0:
            for ix, xx, in enumerate(xs):
                if heaviside_array[ix] != 0.0:
                    mfp_array[ix] = self.integrate_sigma(xx, mu, tfinal, sedov_class, g_interpolated, v_interpolated)
            return np.exp(-mfp_array / mu) * heaviside_array
        else:
            return xs * 0.0
    
    def integrate_angular_flux(self, a, b, x, tfinal, sedov_class, g_interpolated, v_interpolated):
        func = self.mu_quad * 0
        argument = (b-a)/2*self.mu_quad + (a+b)/2

        for ix, xx in enumerate(self.mu_quad):
            func[ix] = self.uncollided_angular_flux(np.array([x]), tfinal, argument[ix], sedov_class, g_interpolated, v_interpolated)[0]
        res = (b-a)/2 * np.sum(self.mu_ws * func)
        res2 = (b-a)/2 * np.sum(self.mu_ws * func * argument)
        
        return res, res2

    def uncollided_scalar_flux(self, xs, tfinal, sedov_class, g_interpolated, v_interpolated):
        phi = xs * 0
        a = 0.0
        b = 1.0
        if tfinal > 0.0:
            for ix, xx in enumerate(xs):
                bb = 1.0
                if tfinal > self.t0source:
                    bb = min(1.0, abs(xx+self.x0)/ (tfinal - self.t0source))
                aa = abs(xx+self.x0) / tfinal
                if aa <= 1.0:
                    phi[ix] = self.integrate_angular_flux(aa, bb, xx, tfinal, sedov_class, g_interpolated, v_interpolated)[0]
        return phi
    
    def current(self, xs, tfinal, sedov_class, g_interpolated, v_interpolated):
        J = xs * 0
        a = 0.0
        b = 1.0
        if tfinal > 0.0:
            for ix, xx in enumerate(xs):
                bb = 1.0
                if tfinal > self.t0source:
                    bb = min(1.0, abs(xx+self.x0)/ (tfinal - self.t0source))
                aa = abs(xx+self.x0) / tfinal
                if aa <= 1.0:
                    J[ix] = self.integrate_angular_flux(aa, bb, xx, tfinal, sedov_class, g_interpolated, v_interpolated)[1]
        return J

        
