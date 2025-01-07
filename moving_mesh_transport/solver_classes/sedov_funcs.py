import numba as nb
import numpy as np
import math
from numba.experimental import jitclass
from numba import types, typed
from numba import int64, float64, jit, njit, deferred_type
from .build_problem import build
from .cubic_spline import cubic_spline_ob as cubic_spline
from .functions import newtons

build_type = deferred_type()
build_type.define(build.class_type.instance_type)

spline_type = deferred_type()
spline_type.define(cubic_spline.class_type.instance_type)

data = [('rho2', float64),
        ('u2', float64),
        ('gamma', float64),
        ('rho1', float64),
        ('gamp1', float64),
        ('gamm1', float64),
        ('gpogm', float64),
        ('r2', float64),
        ('eblast', float64),
        ('rho0', float64),
        ('omega', float64),
        ('xg2', float64),
        ('f_fun', float64[:]),
        ('g_fun', float64[:]),
        ('l_fun', float64[:]),
        ('us', float64),
        ('alpha', float64),
        ('sigma_t', float64),
        ('vr2', float64),
        ('t_shift', float64),
        ('r2_dim', float64),
        ('vr2_dim', float64),
        ('tstar', float64)
        ]

@jitclass(data)
class sedov_class(object):
    def __init__(self, g_fun, f_fun, l_fun, sigma_t, eblast =  1.8e18, tstar = 1e-12 ):
        t = 1
        self.r2 = 0.0
        self.gamma = 7.0/5.0
        self.gamm1 = self.gamma - 1.0
        self.gamp1 = self.gamma + 1.0
        self.gpogm = self.gamp1 / self.gamm1
        self.rho0 = 1.0
        self.omega = 0.0
        geometry = 1
        self.alpha = self.gpogm * 2**(geometry) /\
                (geometry*(self.gamm1*geometry + 2.0)**2)
        # print(self.alpha, 2/self.gamm1/self.gamp1)
        # self.eblast = 0.851072 * 1e18
        self.eblast = eblast
        # self.eblast = 0.0
        # self.us = (2.0/self.xg2) * self.r2 / t
        # self.u2 = 2.0 * self.us / self.gamp1
        self.xg2 = 1 + 2.0 - self.omega
        self.sigma_t = sigma_t
        self.tstar = tstar

        self.physical(t)
        # print(self.sigma_t, 'sigma_t')
        
        # self.find_r2(t)
        # self.vr2_test()


        self.g_fun = g_fun
        self.f_fun = f_fun
        self.l_fun = l_fun

    def finite_difference(self, f, x, h=1e-5, order=1):

        if order == 1:
            # First-order finite difference (forward difference)
            return (f(x + h) - f(x)) / h
        elif order == 2:
            # Second-order finite difference (central difference)
            return (f(x + h) - f(x - h)) / (2 * h)
        else:
            raise ValueError("Only first and second order finite differences are supported.")
        
    def getr2_dim(self, t):
        # res = t*0
        # for it, tt in enumerate(tt):
            self.find_r2(t * 1e-9/29.998/self.sigma_t)
            return  self.r2_dim
        # return res
        
    def vr2_test(self):
        ts = np.linspace(0,1)
        r2list = ts * 0
        dt = ts[1] - ts[0]
        dr2dt = ts*0
        vdimlist = ts * 0
        for it, t in enumerate(ts):
            self.find_r2(t * 1e-9/29.998/self.sigma_t)
            
            r2list[it] = self.r2_dim
            vdimlist[it] = self.vr2_dim
            dr2dt[it] = self.finite_difference(self.getr2_dim, ts)


        
        print(dr2dt,'dr2dt')
        print(vdimlist, 'vdim')
        # assert(0)
        


        
    def find_r2(self, tt):
        
        t = tt + self.t_shift
        self.r2 = (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) * t**(2.0/self.xg2)
        # if t < 0:
        #     assert(0)
        # print('###')
        # print((self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) * self.sigma_t/ (self.sigma_t * 29.98/1e-9)**(2/3) )
        # print('###')
        if math.isnan(self.r2):
            print(tt)
            print(t)
            print((self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2))
            
            assert(0)

        self.vr2 = (1/ self.sigma_t/29.98) * 1e-9 * 2/self.xg2 * (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) *\
                (tt + self.t_shift)**(2/self.xg2 -1) 
        # self.vr2 = (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) * ((2.0/self.xg2 * t) * (t**(2.0/self.xg2-1))) *1e-9/29.998/self.sigma_t 
        
        self.rho1 = self.rho0 * self.r2**(-self.omega)
        self.rho2 = self.gpogm * self.rho1
        self.r2_dim = self.r2 * self.sigma_t # this is actually r2 nondim

        self.us = (2.0/self.xg2) * self.r2 / t
        self.u2 = 2.0 * self.us / self.gamp1
        # self.u2_dim = 
        
        self.vr2_dim = self.vr2 / self.sigma_t


    def physical(self, tt):
        '''Returns physical variables from single values of Sedov functions'''
        self.t_shift = (self.tstar /29.98)/self.sigma_t *1e-9
        # self.t_shift = self.tstar
        t_sh = tt / 29.98 / self.sigma_t # convert mean free times to ns
        t = t_sh * 1e-9 # convert ns to seconds
        self.find_r2(t)
        # print(self.r2)
        density = self.rho2 * self.g_fun
        velocity = self.u2 * self.f_fun
        rs = self.l_fun * self.r2_dim
        return density, velocity, rs
    
    # def splice_blast(self, x, interpolated_sol, elseval):
    #     res = x * 0 
    #     for ix, xx in enumerate(x):
    #         if xx < self.r2:
    #             res[ix] = interpolated_sol.eval_spline(np.array([xx]))[0]
    #         else:
    #             res[ix] = elseval
    #     return res


    # def evaluate_sedov(self, x, interpolated_rho, interpolated_v):
    #     rho = x * 0
    #     v = x * 0 

    #     v = self.splice_blast(x, interpolated_v, 0.0)
    #     rho = self.splice_blast(x, interpolated_rho, self.rho0)

    #     return rho, v

    # def interpolate_solution(self, t, xs, interpolated_rho, interpolated_v):
       
    #     # density, velocity, rs = self.physical(tt + t_shift)
    #     # interpolated_density = cubic_spline(rs, density)
        
    #     # interpolated_velocity = cubic_spline(rs, velocity)
        

    #     # if (xs > self.r2).all():
    #     #     print(self.r2)
    #     #     return self.rho0 * np.ones(xs.size), np.zeros(xs.size)
    #     # else:
    #         # interpolated_density = cubic_spline(rs2, np.flip(density))
    #         # interpolated_velocity = cubic_spline(rs2, np.flip(velocity))
            
    #         # get mirrored Taylor-Sedov blast solution
    #         res_rho = xs * 0 
    #         res_v = xs * 0 
    #         res = self.evaluate_sedov(np.abs(xs), interpolated_rho, interpolated_v)
    #         res_rho = res[0]
    #         res_v = res[1]
            
            
    #         return res_rho, res_v
    
    def interpolate_self_similar(self, t, xs_ndm, interpolated_g):
        self.physical(t)
        xs = xs_ndm / self.sigma_t
        res = xs * 0
        if self.r2 == 0.0:
            assert(0)
        for ix, xx in enumerate(xs):
            if abs(xx) <= self.r2:
                res[ix] = self.rho2 * interpolated_g.eval_spline(np.array([abs(xx) / self.r2]))[0] 
                # print(self.rho2 * interpolated_g.eval_spline(np.array([abs(xx) / self.r2]))[0] )
                # res[ix] = self.rho0
            else:
                res[ix] = self.rho0
        return res
   
    def interpolate_self_similar_v(self, t, xs_ndm, interpolated_v):
        self.physical(t)
        xs = xs_ndm / self.sigma_t
        res = xs * 0
        if self.r2 == 0.0:
            assert(0)
        for ix, xx in enumerate(xs):
            if abs(xx) <= self.r2:
                res[ix] = self.u2 * interpolated_v.eval_spline(np.array([abs(xx) / self.r2]))[0] * np.sign(xx)
            else:
                res[ix] = 0.0
        return res
        

    # def interior_interpolate(self, t, xs):
    #     density, velocity, rs = self.physical(t)
    #     rs2 = np.flip(rs)
    #     rs2[0] = 0.0
    #     density2 = np.flip(density)
    #     density2[0] = 0.0
    #     density2[-1] = self.gpogm
    #     rs2[-1] = self.r2

    #     interpolated_density = cubic_spline(rs2, density2)
    #     # interpolated_density = cubic_spline(xs, np.cos(xs))
    #     # interpolated_velocity = cubic_spline(xs, np.cos(xs))
    #     interpolated_velocity = cubic_spline(rs2, np.flip(velocity))
    #     res_rho, res_v = self.interpolate_solution(t, xs, interpolated_density, interpolated_velocity)
    #     return res_rho, res_v
    
    def find_contact_time(self, x0, t0 = 0):
       t_hits = np.zeros(2)
       print('searching for wave contact')
       contact_time = self.bisection(self.contact_func, 0.0, x0, x0)
       t_hits[0] = contact_time
       contact_time2 = self.bisection(self.contact_func2, contact_time, 2*x0, x0)
    #    contact_time3 = self.bisection(self.contact_func3, 0, 2*x0, x0)
    #    print(contact_time3, 'negative wave contact')
    #    self.physical(contact_time3)
    #    print(self.r2_dim, 'r2 at negative wave contact')
       t_hits[1] = contact_time2
       self.physical(contact_time)
       print(self.r2_dim, 'r2 at t_hit1')
       self.physical(contact_time2)
       print(self.r2_dim, 'r2 at t_hit2')
       return t_hits - 1e-5
    
    def chi_func(self, s, x, mu, t):
        return (s-x) / mu + t
    
    def r2_func(self, t):
        self.physical(t)
        return self.r2
   
    def integral_bounds_func(self, s, x, t, mu):
        r2 = self.r2_func(self.chi_func(s, x, mu, t)) * self.sigma_t
        return  s - r2
    
    def integral_bounds_func2(self, s, x, t, mu):

        r2 = self.r2_func(self.chi_func(s, x, mu, t)) * self.sigma_t
        return -r2 - s
    
    def find_r2_in_transformed_space(self, x, t, mu, x0):
        a = 0
        b = x0
        shock_point1 = self.bisection2(self.integral_bounds_func, a, b, x, t, mu)
        a = -x0
        b = 0
        shock_point2 = self.bisection2(self.integral_bounds_func2, a, b, x, t, mu)

        c1 = (self.eblast/(self.alpha*self.rho0))**(1.0/self.xg2) * self.sigma_t * (1e-9/self.sigma_t / 29.98)**(2/3)
        r2 = self.r2_func(t) * self.sigma_t
        guesspoint1 =         (3*c1*(-x + t*mu)*((-x + t*mu)/mu)**0.6666666666666666)/(-3*x + 3*t*mu + 2*c1*((-x + t*mu)/mu)**0.6666666666666666)
        # print(c1 * (t+self.t_shift)**(2/3), 'c1t^2/3')
        # print(r2, 'r2dim')
        tp = t - (x+x0)/mu
        guesspoint1 =  -0.3333333333333333*c1**3/mu**2 - (2**0.3333333333333333*(-(c1**6/mu**4) + (6*c1**3*(x0 + tp*mu))/mu**2))/(3.*(-27*c1**3*tp**2 - (2*c1**9)/mu**6 + (18*c1**6*x0)/mu**4 + (18*c1**6*tp)/mu**3 - (27*c1**3*x0**2)/mu**2 - (54*c1**3*tp*x0)/mu + math.sqrt((-27*c1**3*tp**2 - (2*c1**9)/mu**6 + (18*c1**6*x0)/mu**4 + (18*c1**6*tp)/mu**3 - (27*c1**3*x0**2)/mu**2 - (54*c1**3*tp*x0)/mu)**2 + 4*(-(c1**6/mu**4) + (6*c1**3*(x0 + tp*mu))/mu**2)**3))**0.3333333333333333) + (-27*c1**3*tp**2 - (2*c1**9)/mu**6 + (18*c1**6*x0)/mu**4 + (18*c1**6*tp)/mu**3 - (27*c1**3*x0**2)/mu**2 - (54*c1**3*tp*x0)/mu + math.sqrt((-27*c1**3*tp**2 - (2*c1**9)/mu**6 + (18*c1**6*x0)/mu**4 + (18*c1**6*tp)/mu**3 - (27*c1**3*x0**2)/mu**2 - (54*c1**3*tp*x0)/mu)**2 + 4*(-(c1**6/mu**4) + (6*c1**3*(x0 + tp*mu))/mu**2)**3))**0.3333333333333333/(3.*2**0.3333333333333333)
        # guesspoint1 = -0.3333333333333333*c2**3/mu**2 - (2**0.3333333333333333*(-(c2**6/mu**4) + (6*c2**3*x0)/mu**2))/(3.*((-2*c2**9)/mu**6 + (18*c2**6*x0)/mu**4 - (27*c2**3*x0**2)/mu**2 + (3*math.sqrt(3)*math.sqrt(-4*c2**9*x0**3 + 27*c2**6*x0**4*mu**2))/mu**3)**0.3333333333333333) + ((-2*c2**9)/mu**6 + (18*c2**6*x0)/mu**4 - (27*c2**3*x0**2)/mu**2 + (3*math.sqrt(3)*math.sqrt(-4*c2**9*x0**3 + 27*c2**6*x0**4*mu**2))/mu**3)**0.3333333333333333/(3.*2**0.3333333333333333)
        # print(guesspoint1-shock_point1, 'difference1')
        # print(c1, 'c1')
        # print(guesspoint1-shock_point2, 'difference2')
        # guesspoint1 =         (c1*((-(c1**3*x**2) + 2*c1**3*t*x*mu - c1**3*t**2*mu**2 + mu**2*math.sqrt((c1**6*(x - t*mu)**4)/mu**4))/(c1**3*mu**2))**0.3333333333333333)/2**0.3333333333333333
        # print(guesspoint1, 'guess')
        
        return shock_point2, shock_point1
    
    def contact_func(self, t, x0, tshift = 0.0):
        self.physical(t)
        return t - x0 + self.r2_dim
    
    def contact_func2(self, t, x0, tshift = 0.0):
        self.physical(t)
        return t - x0 - self.r2_dim
    
    def contact_func3(self, t, x0, tshift = 0.0):
        self.physical(t-0.1)
        return t - 0.1 - x0 - self.r2_dim


    def bisection(self, f, a, b, x0, tol=1e-14):
        if np.sign(f(a, x0)) == np.sign(f(b, x0)):
            print(x0, 'x0')
            print(a, b, 'a, b')
            print(f(a, x0), 'f(a)')
            print(f(b, x0), 'f(b)')
            assert 0
        while b-a > tol:
            m = a + (b-a)/2
            fm = f(m, x0)
            if np.sign(f(a, x0)) != np.sign(fm):
                b = m
            else:
                a = m
        return m
    
    def bisection2(self, f, a, b, x,t, mu, tol=1e-8):
            if np.sign(f(a, x, t, mu)) == np.sign(f(b, x, t, mu)):
                return x
            else:
                while b-a > tol:
                    m = a + (b-a)/2
                    fm = f(m, x, t, mu)
                    if np.sign(f(a, x, t, mu)) != np.sign(fm):
                        b = m
                    else:
                        a = m
                return m
    def analytic_contact_func(self, mu, t, x0, sigma_t):
        # x0 = self.x0
        v = 29.98
        # c1 = (self.eblast/(self.alpha*self.rho0))**(1.0/3) * (v * sigma_t * 1e-9)**(-2/3) * sigma_t
        tau = t * 10**-9 /v/sigma_t
        c1 =  (self.eblast/(self.alpha*self.rho0))**(1.0/3)* (10**-9 /v/sigma_t)**(2/3) * sigma_t

        eta = (c1/mu + (2**0.3333333333333333*c1**2)/(mu*(2*c1**3 + 27*mu**2*x0 + 3*math.sqrt(3)*math.sqrt(4*c1**3*mu**2*x0 + 27*mu**4*x0**2))**0.3333333333333333) + (2*c1**3 + 27*mu**2*x0 + 3*math.sqrt(3)*math.sqrt(4*c1**3*mu**2*x0 + 27*mu**4*x0**2))**0.3333333333333333/(2**0.3333333333333333*mu))/3.
        tt = eta ** (1/3)
        # r2a = -x0 + mu * tt
        # r2a = -c1 * tau**(2/3) * sigma_t
        r2b = -c1 * tau **2/3 * sigma_t
        t1p = ((mu*t-x0)/mu)**2
        t1 = t1p**1/3
        # print(mu,'mu')
        # print(((mu*t-x0)/mu))

        r2a = - 3 * c1*(mu*t-x0)*t1 /(2*c1 * t1 +3 * mu * t - 3 * x0)



        # r2a =  (2*2**0.3333333333333333*c1**6 + mu**4*((4*c1**9)/mu**6 + (36*c1**6*(mu*t - x0))/mu**4 +  6*math.sqrt(3)*math.sqrt((c1**6*(4*c1**3 + 27*mu**2*(mu*t - x0))*(mu*t - x0)**3)/mu**6) + (54*c1**3*(-(mu*t) + x0)**2)/mu**2)**0.6666666666666666 + 2*c1**3*mu**2*(6*2**0.3333333333333333*mu*t - 6*2**0.3333333333333333*x0 + ((2*c1**9)/mu**6 + (18*c1**6*(mu*t - x0))/mu**4 +  3*math.sqrt(3)*math.sqrt((c1**6*(4*c1**3 + 27*mu**2*(mu*t - x0))*(mu*t - x0)**3)/mu**6) + (27*c1**3*(-(mu*t) + x0)**2)/mu**2)**0.3333333333333333))/ (6.*mu**4*((2*c1**9)/mu**6 + (18*c1**6*(mu*t - x0))/mu**4 + 3*math.sqrt(3)*math.sqrt((c1**6*(4*c1**3 + 27*mu**2*(mu*t - x0))*(mu*t - x0)**3)/mu**6) + (27*c1**3*(-(mu*t) + x0)**2)/mu**2)**0.3333333333333333)
        return r2a, r2b