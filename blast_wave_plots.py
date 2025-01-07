import matplotlib.pyplot as plt
from moving_mesh_transport.plots.plot_functions.show import show
from moving_mesh_transport.solver_classes.functions import get_sedov_funcs
from moving_mesh_transport.solver_classes.sedov_funcs import sedov_class
from moving_mesh_transport.solver_classes.cubic_spline import cubic_spline_ob as cubic_spline

from moving_mesh_transport.solver_classes.sedov_uncollided import sedov_uncollided_solutions
from moving_mesh_transport.loading_and_saving.load_solution import load_sol
from moving_mesh_transport.solver_classes.functions import quadrature
import numpy as np
import math
from scipy.special import erf
import scipy.integrate as integrate
from tqdm import tqdm
import quadpy
import h5py
import sys
import quadpy
sys.path.append('/Users/bennett/Documents/Github/exactpack/')


def opts0(*args, **kwargs): 
       return {'limit':50, 'epsabs':1.5e-12, 'epsrel':1.5e-12}

def toy_blast_psi(mu, tfinal, x, v0, t0source, x0):
        c1 = 1.0
        x0 = -5 
        if mu!= 0:
            t0 =  (x0-x)/mu + tfinal # time the particle is emitted
        else: 
             t0 = np.inf
        x02 = 0.0
        sqrt_pi = math.sqrt(math.pi)
        kappa = 2
        rho0 = 0.1
        # beta = c1 * (v0-1) - v0 * (x0/mu + t0)
        
        # b2 =  v0 * (-x0/mu - t0 + c1) / (1+v0/mu)
        b2 = ((v0*x0) - t0*v0*mu)/(v0 + mu)
        b1 = max(x, b2)
        # b2 = 0

        b4 = x0
        # b3 = min(x,0)

        b3 =  min(x, b2)

        # print(b1, b2, b3, b4, 'bs', x, 'x', t0, 't0')

        # t1 = lambda s: -0.5*(mu*sqrt_pi*kappa*erf((beta - (s*(mu + v0))/mu)/kappa))/(mu + v0)

        t1 = lambda s: (sqrt_pi*kappa*mu*erf((v0*(s - x0) + (c1 + s + t0*v0)*mu)/(kappa*mu + 1e-12)))/(1e-12 + 2.*(v0 + mu))
        t2 = lambda s: rho0 * s

        mfp = t1(b1) - t1(b2) + t2(b3) - t2(b4)
        if mu == 0:
             return 0.0
        else:
            if mfp/mu >40:
                mfp = 40 * mu
                # print(np.exp(-mfp/mu))
                return 0.0
                # mfp = rho0 * x - rho0 * (-x0)
                # print(mfp, x, 'mfp')
            if np.isnan(np.exp(-mfp / mu) * np.heaviside(mu - abs(x - x0)/ (tfinal), 0) * np.heaviside(abs(x0-x) - (tfinal-t0source)*mu,0)):
                    print(np.exp(-mfp / mu))
                    print(mu)
                    assert(0)

            if mu > 0:
                return np.exp(-mfp / mu) * np.heaviside(mu - abs(x - x0)/ (tfinal), 0) * np.heaviside(abs(x0-x) - (tfinal-t0source)*mu,0)
            else:
                    return 0.0
        
def toy_blast_phi(tfinal, x, v0, t0, x0):
    aa = 0.0
    bb = 1.0
    if tfinal > t0:
        bb = min(1.0, abs(x-x0)/ (tfinal - t0))
    aa = abs(x-x0) / tfinal
    if aa <= 1.0:     
        res = integrate.nquad(toy_blast_psi, [[aa, bb]], args = (tfinal, x, v0, t0, x0), opts = [opts0])
        return res[0]
    else:
         return 0.0

def toy_blast_phi_vector(tfinal, xs, v0, t0, x0):
     res = xs*0
     for ix, x in enumerate(xs):
        aa = 0.0
        bb = 1.0
        if tfinal > t0:
            bb = min(1.0, abs(x-x0)/ (tfinal - t0))
        aa = abs(x-x0) / tfinal
        if aa <= 1.0:     
            res[ix] = integrate.nquad(toy_blast_psi, [[aa, bb]], args = (tfinal, x, v0, t0, x0), opts = [opts0])[0]
     return res
     
def RMSE(xs, phi, v0, tfinal, t0, x0):
    benchmark = xs*0
    for ix, xx in enumerate(xs):
        benchmark[ix] = toy_blast_phi(tfinal, xx, v0, t0, x0)
    
    res = math.sqrt(np.mean((phi-benchmark)**2))
    return res, benchmark

def RMSETS(benchmark, phi):
    
    res = math.sqrt(np.mean((phi-benchmark)**2))
    return res


def error_toy_blast_wave_absorbing(N_space=32, M=6, v0 = 0.0035, x0 = -5.0, t0 = 15.0):
    plt.ion()
    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = 0.0)
    tlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])
    RMSE_list = tlist*0
    counter1 = 0
    for it, t in enumerate(tqdm(tlist)):
        # plt.figure(1)
        loader.call_sol(t, M, int(-x0), N_space, 'rad', True, True)
        x = loader.xs
        phi = loader.phi
        psi = loader.psi
        res = RMSE(x, phi, v0, t, t0, x0)
        RMSE_list[it] = res[0]
  
        counter1 += 1
        if counter1 == 1:
            plt.figure(23)
            plt.plot(x, phi, 'k--')
            plt.plot(x, res[1], 's', mfc = 'none', label = f't={t}')
            counter1 = 0

            plt.figure(24)
            # plt.plot(x,  'k--')
            plt.plot(x, np.abs(phi-res[1]), '-', mfc = 'none',  label = f't={t}')
            counter1 = 0
        plt.legend()
        plt.show()



    plt.figure(5)
    plt.loglog(tlist, RMSE_list, '-o')
    plt.xlabel('evaluation time', fontsize = 16)
    plt.ylabel("RMSE", fontsize = 16)
    show('blast_wave_absorbing_error')
    plt.show()


def exit_distributions(v0 = 0.000035, x0 = -5.0, t0 = 15.0, tf = 50):
    tlist = np.linspace(0.001, tf, 100)
    left = tlist * 0
    right = tlist * 0
    for it, tt in enumerate(tlist):
          left[it] = toy_blast_phi(tt, x0, v0, t0, x0)
          right[it] = toy_blast_phi(tt, -x0, v0, t0, x0)

    plt.figure(1)
    plt.title('left exit dist')
    plt.plot(tlist, left, '-o')
    plt.show()

    plt.figure(2)
    plt.title('right exit dist')
    plt.plot(tlist, right, '-o')

    plt.show()

    xs = np.linspace(-x0, x0)
    res = xs * 0

    for ix, xx in enumerate(xs):
         res[ix] = toy_blast_psi(0.5, 1.0, xx, v0, t0, x0)
    plt.figure(3)
    plt.plot(xs, res)
    plt.show()



def plot_analytic_solutions(x0 = -5, v0 =0.0035, t0 = 15, tf = 50):
    tlist = np.array([1, 2, 3, 5, 7, 15.0])
    plt.ion()
    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    for it, tt in enumerate(tlist):
         xs = np.linspace(-x0, min(x0, tt-x0), 500)
         sol = toy_blast_phi_vector(tt, xs+1e-8, v0, t0, x0)
         plt.plot(xs, sol, 'k-')
    # plt.text(-4.96,0.2,r'$t=0$')
    plt.text(-4.21,0.16,r'$t=1$')
    plt.text(-3.25,0.09,r'$2$')
    plt.text(-2.34,0.09,r'$3$')
    plt.text(-.70,0.092,r'$5$')
    plt.text(0.93,0.047,r'$7$')
    plt.text(0.4,0.2,r'$15$')
    # plt.text(-3.56,0.226,r'$t=17$')
    # plt.text(-2.97,0.157,r'$t=18$')
    # plt.text(-1.46,0.137,r'$t=19$')
    # plt.text(-1.18,0.067,r'$t=20$')
    # plt.text(1.8,0.014,r'$t=22$')
    plt.show()
    show('uncollided_solutions_before')
    tlist = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 30.0])
    plt.ion()
    ax = plt.figure(11)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    for it, tt in enumerate(tlist):
        xs = np.linspace(-x0 , min(x0, tt-x0), 500)
        sol = toy_blast_phi_vector(tt, xs+1e-8 , v0, t0, x0)
        plt.plot(xs, sol, 'k-')
    plt.text(-4.96,1.004,r'$t=15$')
    plt.text(-4.25,0.532,r'$16$')
    plt.text(-3.4,0.424,r'$17$')
    plt.text(-2.85,0.31,r'$18$')
    plt.text(-1.5,0.277,r'$19$')
    plt.text(-1.18,0.161,r'$20$')
    plt.text(-.6,0.04,r'$22$')


    ax.show()
    show('uncollided_solutions_after')



    # tlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])
    tlist = np.linspace(0.001, tf, 500)
    left = tlist * 0
    right = tlist * 0
    for it, tt in enumerate(tlist):
          left[it] = toy_blast_phi(tt, x0, v0, t0, x0)
          right[it] = toy_blast_phi(tt, -x0, v0, t0, x0)
    plt.figure(2)
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    plt.plot(tlist, left, 'k-', )
    plt.show()
    show('left_exit_uncollided')
    plt.figure(3)
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    plt.plot(tlist, right, 'k-')
    # plt.xlabel('t', fontsize = 16)
    # plt.ylabel(r'$\phi$', fontsize = 16)
    plt.show()
    show('right_exit_uncollided')
    

def toy_blast_scattering_profiles(N_space = 16, cc = 0.125, uncollided = True):
    M = 6
    x0 = -5
    fulltlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])

    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    tlist1 = np.array([1, 2, 3, 5, 7, 15.0])
    tlist2 = np.array([15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 30.0])
    tlist = np.concatenate((tlist1, tlist2[1:]))
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    plt.ion()
    for it, tt in enumerate(tlist1):
        loader.call_sol(tt, M, int(-x0), N_space, 'rad', uncollided, True)
        xs = loader.xs
        phi = loader.phi
        output_phi = loader.exit_phi
        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #     right_exit[it] = output_phi[1]
 
        plt.figure(3)
        string1 = 'before'
    #   plt.text(-4.96,0.2,r'$t=0$')
        plt.text(-4.21,0.16,r'$t=1$')
        plt.text(-3.25,0.09,r'$2$')
        plt.text(-2.34,0.09,r'$3$')
        plt.text(-.70,0.092,r'$5$')
        plt.text(0.93,0.047,r'$7$')
        plt.text(0.4,0.2,r'$15$')
        plt.plot(xs, phi, 'k-')     
        plt.ylabel(r'$\phi$', fontsize = 16)
        plt.xlabel('x', fontsize = 16)
         
        show(f'c={cc}_solutions_' + string1)
    
    for it, tt in enumerate(tlist2):
            loader.call_sol(tt, M, int(-x0), N_space, 'rad', uncollided, True)
            xs = loader.xs
            phi = loader.phi
            output_phi = loader.exit_phi
            plt.figure(4)
            string1 = 'after'
            plt.text(-4.96,1.023,r'$t=15$')
            plt.text(-4.25,0.532,r'$16$')
            plt.text(-3.4,0.424,r'$17$')
            plt.text(-2.85,0.31,r'$18$')
            plt.text(-1.86,0.277,r'$19$')
            plt.text(-2.12,0.161,r'$20$')
            plt.text(-.6,.156,r'$22$')
            plt.text(-.6,.04,r'$30$')
            plt.plot(xs, phi, 'k-')
                
            plt.ylabel(r'$\phi$', fontsize = 16)
            plt.xlabel('x', fontsize = 16)
            
            show(f'c={cc}_solutions_' + string1)



    plt.figure(1)
    plt.plot(fulltlist, output_phi[:,0], 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'left_exit_dist_c={cc}')

    plt.figure(2)       
    plt.plot(fulltlist, output_phi[:,1], 'k-')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'right_exit_dist_c={cc}')

    
    plt.show()

# exit_distributions()
# error_toy_blast_wave_absorbing(N_space = 16)
# plot_analytic_solutions()


def error_TS_blast_wave_absorbing(N_space=16, M=6, x0 = 0.5, t0 = 15.0, sigma_t = 0.000005):
    plt.ion()
    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = 0.0)
    tlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])
    RMSE_list = tlist*0
    counter1 = 0
    for it, t in enumerate(tqdm(tlist)):
        # plt.figure(1)
        loader.call_sol(t, M, x0, N_space, 'rad', False, True)
        x = loader.xs
        phi = loader.phi
        psi = loader.psi
        # res = RMSE(x, phi, v0, t, t0, x0)
        # RMSE_list[it] = res[0]
  
        counter1 += 1
        if counter1 == 1:
            plt.figure(23)
            plt.plot(x, phi, 'k--')
            # plt.plot(x, res[1], 's', mfc = 'none', label = f't={t}')
            counter1 = 0

            # plt.figure(24)
            # plt.plot(x,  'k--')
            # plt.plot(x, np.abs(phi-res[1]), '-', mfc = 'none',  label = f't={t}')
            counter1 = 0
        plt.legend()
        plt.show()



    plt.figure(5)
    plt.loglog(tlist, RMSE_list, '-o')
    plt.xlabel('evaluation time', fontsize = 16)
    plt.ylabel("RMSE", fontsize = 16)
    show('blast_wave_absorbing_TSe_error')
    plt.show()


# def TS_blast_scattering_profiles(N_space = 16, cc = 0.0, uncollided = False):
#     M = 6
#     x0 = 1.0
#     fulltlist = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 4.75, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0,13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 35.0, 40.0, 45.0,  50.0])

#     loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
#     tlist1 = np.array([0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 15.0])
#     tlist2 = np.array([15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0])
#     tlist = np.concatenate((tlist1, tlist2[1:]))
#     tlist = np.array([1.0])
#     left_exit = np.zeros(tlist.size)
#     right_exit = np.zeros(tlist.size)
#     counter = 0

#     g_interp, v_interp, sedov = TS_bench_prime()

#     plt.ion()
#     for it, tt in enumerate(tqdm(tlist)):
#         #  loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, False)
#         #  xs = loader.xs
#         #  phi = loader.phi
#         #  output_phi = loader.exit_phi
#          xs = np.linspace(-x0, x0, 50)
#          phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov)

#         #  eval_times = loader.eval_array
#         #  if eval_times[counter] = tlist[counter]:         
#         #     left_exit[it] = output_phi[counter, 0]
#         #     right_exit[it] = output_phi[1]
#          if tt <= tlist1[-1]:
#               plt.figure(3)
#               string1 = 'before'
#             #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
#             #   plt.text(-4.96,0.2,r'$t=0$')
#             #   plt.text(-4.21,0.16,r'$t=1$')
#             #   plt.text(-3.25,0.09,r'$t=2$')
#             #   plt.text(-2.34,0.09,r'$t=3$')
#             #   plt.text(-.70,0.092,r'$t=5$')
#             #   plt.text(0.93,0.047,r'$t=7$')
#             #   plt.text(0.4,0.2,r'$t=15$')
#          else:
#             plt.figure(4)
#             string1 = 'after'
#             # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
#             # plt.text(-4.96,0.704,r'$t=15$')
#             # plt.text(-4.25,0.532,r'$t=16$')
#             # plt.text(-3.4,0.424,r'$t=17$')
#             # plt.text(-2.85,0.31,r'$t=18$')
#             # plt.text(-1.5,0.277,r'$t=19$')
#             # plt.text(-1.18,0.161,r'$t=20$')
#             # plt.text(-.6,0.04,r'$t=22$')
#         #  plt.plot(xs, phi, 'k-')
#          plt.plot(xs, phi_bench, 'o', mfc = 'none')
#          plt.ylabel(r'$\phi$', fontsize = 16)
#          plt.xlabel('x', fontsize = 16)
         
#          show(f'c={cc}_solutions_TS' + string1)



#     plt.figure(1)
#     # plt.plot(fulltlist, output_phi[:,0], 'k-')
#     plt.xlabel('t', fontsize = 16)
#     plt.ylabel(r'$\phi$', fontsize = 16)
#     show(f'left_exit_dist_c={cc}')

#     plt.figure(2)       
#     # plt.plot(fulltlist, output_phi[:,1], 'k-')  
#     plt.xlabel('t', fontsize = 16)
#     plt.ylabel(r'$\phi$', fontsize = 16)
#     show(f'right_exit_dist_c={cc}')

    
    plt.show()





def TS_blast_absorbing_profiles(tf = 5.5, N_space = 16, cc = 0.0, uncollided = False, transform = True, moving = True, sigma_t = 1e-3, x0 = 0.15, beta = 2.7, eblast = 1e20, plotcurrent = False, plotbad = False, relativistic = False, tstar = 1e-12):
    M = 6
    npts = 250
    # x0 = 0.15
    t0 = 0.5
    e1 = 10**19
    e2 = 10**20
    # fulltlist = np.array([0.05,0.5, 0.6, 0.7,  1.0, 1.5, 2.0, 2.5, 2.5, 2.6,  3.0, 3.5,  4.0, 4.5,  5.0])
    # fulltlist = np.array([0.0005, 0.005, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07,0.08,0.09, 0.1,0.12,0.13,0.14,0.15,0.16, 0.17, 0.18, 0.19, 0.2, 0.21,0.22,0.23,0.24, 0.25,0.26,0.27,0.28,0.29, 0.3,0.31,0.32,0.33,0.34, 0.35,0.36,0.37,0.38,0.39, 0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49, 0.5, 0.55, 0.6, 0.65,  0.7,0.75,0.8,0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    fulltlist = np.linspace(0.00001, tf, 150)
    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    # tlist1 = np.array([0.05, 0.4, 0.7, 1.0, 2.5])
    # tlist1 = np.array([0.0005, 0.005, 0.01,0.02,0.03,0.04, 0.05, 0.06,0.07,0.08,0.09, 0.1, 0.12,0.13,0.14,0.15,0.16, 0.17, 0.18, 0.19, 0.2, 0.21,0.22,0.23,0.24, 0.25,0.26,0.27,0.28,0.29, 0.3,0.31,0.32,0.33,0.34, 0.35,0.36,0.37,0.38,0.39, 0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49, 0.5])
    tlist1 = np.array([0.01, 0.1, 0.3, 0.4, 0.49999])
    # tlist1 = np.array([0.0005, 0.005, 0.05, 0.1, 0.1125,0.13,0.14, 0.15])
    # tlist2 = np.array([0.0005, 0.005, 0.05, 0.1, 0.1125,0.13,0.14, 0.15])
    # tlist1 = np.array([1.0])
    # tlist2 = np.array([2.5, 2.6, 3.0, 3.5,  4.0,  5.0])
    tlist2 = np.array([0.5, 0.7,0.9, 1.5, 2.0, 3.0])

    plot_list = np.array([0.01, 0.1, 0.3, 0.49999, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5 ])

    tlist = np.concatenate((tlist1, tlist2))
    # tlist = np.array([1.0])
    # tlist = tlist1
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    RMSE_list = tlist*0
    string2 = ''
    if plotbad == True:
         string2 += 'bad'
    if plotcurrent == True:
         string2 += 'current'
    # moving = False

    g_interp, v_interp, sedov = TS_bench_prime(sigma_t, eblast, tstar)
    f, (a1, a2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    # f2, (b1, b2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    plt.ion()
    for it, tt in enumerate(tqdm(tlist1)):
        #  loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, moving)
        #  xs = loader.xs
         xs = np.linspace(-x0, x0, npts)

        #  phi = loader.phi
        #  N_ang = loader.psi.size 
        #  output_phi = loader.exit_phi
        #  xs = np.linspace(-x0, x0, 200)
    
         phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov, beta = beta, transform=transform, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = True)
         
         current = TS_current(tt, xs, g_interp, v_interp, sedov, beta = beta ,transform=transform, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = True)
         current_bad = TS_current(tt, xs, g_interp, v_interp, sedov, beta = beta ,transform=transform, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)
         current_nottransformed = TS_current(tt, xs, g_interp, v_interp, sedov, beta = beta ,transform=False, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)

        #  if plotbad == True:
         phi_bench_bad = TS_bench(tt, xs, g_interp, v_interp, sedov,beta = beta, transform=True, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False )
         phi_bench_nottransformed = TS_bench(tt, xs, g_interp, v_interp, sedov,beta = beta, transform=False, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False )
              
        

        #  RMSE_list[it] = RMSETS(phi_bench, phi) #/ np.average(phi_bench)

        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #
        #  plt.figure(1101)
        #  if tt > 0.15:
        #         plt.plot(xs, np.abs(phi-phi_bench), label = f't = {tt}')
        #         plt.legend()
        #         plt.show()
         string1 = 'before'
        
            #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
            #   plt.text(-4.96,0.2,r'$t=0$')
            #   plt.text(-4.21,0.16,r'$t=1$')
            #   plt.text(-3.25,0.09,r'$t=2$')
            #   plt.text(-2.34,0.09,r'$t=3$')
            #   plt.text(-.70,0.092,r'$t=5$')
            #   plt.text(0.93,0.047,r'$t=7$')
            #   plt.text(0.4,0.2,r'$t=15$')
  

            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            # plt.text(-4.96,0.704,r'$t=15$')
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
         xrho = np.linspace(-x0, x0, 555000)
         density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
         density_final = sedov.interpolate_self_similar(tlist1[-1], xrho, g_interp)
        #  plt.text(-.489,0.404,r'$t=0.05$')
        #  plt.text(-.299,0.154,r'$0.4$')
        #  plt.text(-.107,0.161,r'$0.7$')
        #  plt.text(0.335,0.065,r'$1.0$')
        #  plt.text(.156,0.191,r'$2.5$')
         plt.ion()
         plt.show()
        #  plt.figure(61)
         if tt in plot_list:
            if plotcurrent == False:
            # a2.plot(xs, phi, 'b-o', mfc = 'none')
                a2.plot(xs/sigma_t, phi_bench, 'k-', mfc = 'none')
                # if plotbad == True:
                # if eblast > e2:
                a2.plot(xs/sigma_t, phi_bench_bad, 'k:', mfc = 'none')
                if eblast < e1:
                    a2.plot(xs/sigma_t, phi_bench_nottransformed, 'k:', mfc = 'none')
                a2.set_ylabel(r'$\phi$', fontsize = 16)
                a1.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
                # a2.set_xlabel('x [cm]')
                a2.set_xlabel('x [cm]', fontsize = 16)
                a1.plot(xrho/sigma_t, density_initial, 'k--')
                a1.plot(xrho/sigma_t, density_final, 'k-')
                sedov.physical(plot_list[-1])
                rf = sedov.r2 
                # a1.set_xlim(-rf*1.1, rf * 1.1)
                if rf > x0/sigma_t:
                     print(rf, x0/sigma_t)
                     raise ValueError('blast has moved outside of the domain')
                # plt.xlabel('x [cm]', fontsize = 16)
                plt.legend()
                # show(f'blast_plots/c={cc}_solutions_TS_epsilon0={eblast}' + string2 + string1 )
                
                
                


            else:
                a2.plot(xs/sigma_t, current, 'k-', mfc = 'none')
                # if eblast > e2:
                a2.plot(xs/sigma_t, current_bad, 'k:', mfc = 'none')
                if eblast < e1:
                    a2.plot(xs/sigma_t, current_nottransformed, 'k:', mfc = 'none')
                a2.set_ylabel(r'$J$', fontsize = 16)
                a1.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
                # a2.set_xlabel('x [cm]')
                a2.set_xlabel('x [cm]', fontsize = 16)
                a1.plot(xrho/sigma_t, density_initial, 'k--')
                a1.plot(xrho/sigma_t, density_final, 'k-')
                sedov.physical(plot_list[-1])
                rf = sedov.r2 
                # a1.set_xlim(-rf*1.1, rf * 1.1)

            # show(f'blast_plots/c={cc}_solutions_TS_epsilon0={eblast}' + string2 + string1 )

    f, (a3, a4) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    # f2, (b3, b4) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})

    for it2, tt in enumerate(tqdm(tlist2)):
            # loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, moving)
            # xs = loader.xs
            xs = np.linspace(-x0, x0, npts)
            # phi = loader.phi
            # output_phi = loader.exit_phi
            # N_ang = loader.psi[:,0].size 
            #  xs = np.linspace(-x0, x0, 200)
            phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov, beta = beta, transform = transform, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = True)
            current = TS_current(tt, xs, g_interp, v_interp, sedov, beta= beta, transform=transform, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = True)
            current_bad = TS_current(tt, xs, g_interp, v_interp, sedov, beta= beta, transform=transform, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)
            current_nottransformed = TS_current(tt, xs, g_interp, v_interp, sedov, beta= beta, transform=False, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)
            # if plotbad == True:
            phi_bench_bad = TS_bench(tt, xs, g_interp, v_interp, sedov, beta = beta,transform=True, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False)

            phi_bench_nottransformed = TS_bench(tt, xs, g_interp, v_interp, sedov,beta = beta, transform=False, x0 = x0, t0 = t0, sigma_t = sigma_t, relativistic = False )
              
            # RMSE_list[it + it2] = RMSETS(phi_bench, phi) #/ np.average(phi_bench)
            # plt.figure(1101)
            # if tt > 0.15:
            #     plt.plot(xs, np.abs(phi-phi_bench), label = f't = {tt}')
            #     plt.legend()
            #     plt.show()


            #  eval_times = loader.eval_array
            #  if eval_times[counter] = tlist[counter]:         
            #     left_exit[it] = output_phi[counter, 0]
            #     right_exit[it] = output_phi[1]
            
                #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
                #   plt.text(-4.96,0.2,r'$t=0$')
                #   plt.text(-4.21,0.16,r'$t=1$')
                #   plt.text(-3.25,0.09,r'$t=2$')
                #   plt.text(-2.34,0.09,r'$t=3$')
                #   plt.text(-.70,0.092,r'$t=5$')
                #   plt.text(0.93,0.047,r'$t=7$')
            #     #   plt.text(0.4,0.2,r'$t=15$')
            # plt.text(-.439,0.9,r'$t=2.5$')
            # plt.text(-.402,0.38,r'$2.6$')
            # plt.text(-.16,0.16,r'$3.0$')
            # plt.text(0.122,0.108,r'$3.5$')
            # plt.text(0.335,0.048,r'$4.0$')
           
            string1 = 'after'
            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
            xrho = np.linspace(-x0, x0, 55000)
            density_initial = sedov.interpolate_self_similar(tlist1[-1], xrho, g_interp)
            density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
            
            plt.ion()
            plt.show()
            # plt.figure(60)
            if tt in plot_list:
                if plotcurrent == False:
                # a4.plot(xs, phi, 'b-o', mfc = 'none')
                    a4.plot(xs/sigma_t, phi_bench, 'k-', mfc = 'none')
                    a4.set_ylabel(r'$\phi$', fontsize = 16)
                    a3.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
                    a4.set_xlabel('x [cm]', fontsize = 16)
                    a3.plot(xrho/sigma_t, density_initial, 'k--')
                    a3.plot(xrho/sigma_t, density_final, 'k-')
                    sedov.physical(plot_list[-1])
                    rf = sedov.r2
                    # a3.set_xlim(-rf*1.1, rf * 1.1)
                    # if plotbad == True:
                    # if eblast > e2:
                    a4.plot(xs/sigma_t, phi_bench_bad, 'k:', mfc = 'none')
                    if eblast < e1:
                        a4.plot(xs/sigma_t, phi_bench_nottransformed, 'k:', mfc = 'none')

                    # plt.xlabel('x [cm]', fontsize = 16)
                    
                else:

                    a4.plot(xs/sigma_t, current, 'k-', mfc = 'none')
                    # if eblast > e2:
                    a4.plot(xs/sigma_t, current_bad, 'k:', mfc = 'none')
                    if eblast < e1:
                        a4.plot(xs/sigma_t, current_nottransformed, 'k:', mfc = 'none')
                    a4.set_ylabel(r'$J$', fontsize = 16)
                    a3.set_ylabel(r'$\rho$ [g/cc]', fontsize = 16)
                    a4.set_xlabel('x [cm]', fontsize = 16)
                    a3.plot(xrho/sigma_t, density_initial, 'k--')
                    a3.plot(xrho/sigma_t, density_final, 'k-')
                    sedov.physical(plot_list[-1])
                    rf = sedov.r2 
                    # a3.set_xlim(-rf*1.1, rf * 1.1)

            # show(f'blast_plots/current_c={cc}_solutions_TS_epsilon0={eblast}' +string2 + string1)


        





    plt.figure(32)
    # plt.plot(fulltlist, output_phi[:,0], 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    # show(f'left_exit_dist_c={cc}')
    tlistdense = np.linspace(fulltlist[0], fulltlist[-1], 120)
    rise_time = 2 * x0 
    fall_time = t0 + 2*x0
    tlistdense = np.append(tlistdense, rise_time)
    tlistdense = np.append(tlistdense, fall_time)
    tlistdense = np.sort(tlistdense)
    t_bench_right = tlistdense * 0
    t_bench_right_bad = tlistdense * 0
    t_bench_right_nottransformed = tlistdense * 0
    current_bad = tlistdense*0
    current = tlistdense * 0
    current_nottransformed = tlistdense * 0
    for it, tt in enumerate(tlistdense):
         t_bench_right[it] = TS_bench(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic=True, x0 = x0)
         t_bench_right_nottransformed[it] = TS_bench(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic=False, transform=False, x0 = x0)
         t_bench_right_bad[it] = TS_bench(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic = False, x0 = x0)
         current[it] = TS_current(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic=True, x0 = x0)
         current_bad[it] = TS_current(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic= False, x0 = x0)

         current_nottransformed[it] = TS_current(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic=False, transform=False, x0 = x0)

    plt.figure(33)       
    plt.plot(tlistdense/sigma_t/29.98, t_bench_right, 'k-')  
    plt.plot(tlistdense/sigma_t/29.98, t_bench_right_bad, 'k:', mfc = 'none') 
    if eblast < e1:
        plt.plot(tlistdense/sigma_t/29.98, t_bench_right_nottransformed, 'k:', mfc = 'none')  
    plt.xlabel('t [ns]', fontsize = 16)

    # plt.text( 2*x0/sigma_t/29.98, max(t_bench_right) *1.05, r'$\mathcal{E}_0 = $' + f'{eblast}' + ' erg', fontsize = 'medium')  
    plt.ylabel(r'$\phi^+$', fontsize = 16)
    show(f'blast_plots/right_exit_dist_c={cc}_epsilon0={eblast}')


    # plt.figure(66)       
    # plt.plot(tlistdense/sigma_t/29.98, t_bench_right - t_bench_right_bad, 'k-')  
    # plt.xlabel('t [ns]', fontsize = 16)
    # # plt.text( 2*x0/sigma_t/29.98, max(t_bench_right), r'$\mathcal{E}_0 = $' + f'{eblast}' + ' erg', fontsize = 'medium')  
    # plt.ylabel(r'$\phi^+$', fontsize = 16)
    # print('##################')
    # print(t_bench_right - t_bench_right_bad)
    # print('relativistic difference')
    # print('##################')
    # show(f'right_exit_dist_c={cc}_difference')

    plt.figure(34)       
    plt.plot(tlistdense/sigma_t/29.98, current, 'k-')
    plt.plot(tlistdense/sigma_t/29.98, current_bad, 'k:', mfc = 'none')
    if eblast < e1:
        plt.plot(tlistdense/sigma_t/29.98, current_nottransformed, 'k:', mfc = 'none') 

    # plt.text(2*x0/sigma_t/29.98, max(current) * 1.05,r'$\mathcal{E}_0 = $ ' + f'{eblast}' + ' erg', fontsize = 'medium')  

    plt.xlabel('t [ns]', fontsize = 16)
    plt.ylabel(r'$J^+$', fontsize = 16)
    plt.legend()
    show(f'blast_plots/right_current_c={cc}_epsilon0={eblast}')


    
    # plt.figure(5)
    # plt.loglog(tlist[:-1], RMSE_list[:-1], '-o', label = f'{N_space} spatial cells, ' + r'$S_{%.0f}$' % int(N_ang))
    # plt.xlabel('evaluation time', fontsize = 16)
    # plt.ylabel(r"$\mathrm{RMSE/avg}(\phi)$", fontsize = 16)
    # plt.legend()
    # show('blast_wave_absorbing_TSe_error')
    # plt.show()
    # print(RMSE_list, 'RMSE')
    

    plt.figure(783)
    xrho = np.linspace(-x0, x0, 55000)
    x = 0.05
    density_initial = sedov.interpolate_self_similar(0.05, xrho, g_interp)
    density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
    plt.plot(xrho/sigma_t, density_initial, 'k--')
    plt.plot(xrho/sigma_t, density_final, 'k-')
    plt.xlabel('x [cm]', fontsize= 16)
    plt.ylabel(r'$\rho$ [g/cc]', fontsize=16)
    # plt.xlim(-0.05/sigma_t, 0.05/sigma_t)
    # show('blast_plots/density_profiles_TS_epsilon0={eblast}')
    plt.show()

    plt.figure(773)
    xrho = np.linspace(-x0, x0, 55000)
    x = 0.05
    v_initial = sedov.interpolate_self_similar_v(0.05, xrho, v_interp)
    v_final = sedov.interpolate_self_similar_v(tlist[-1], xrho, v_interp)
    plt.plot(xrho/sigma_t, v_initial/2.998e10 , 'k--')
    plt.plot(xrho/sigma_t, v_final/2.998e10 , 'k-')
    plt.xlabel('x [cm]', fontsize= 16)
    plt.ylabel(r'$\beta$', fontsize=16)
    # plt.xlim(-0.05/sigma_t, 0.05/sigma_t)
    # show(f'blast_plots/velocity_profiles_TS_epsilon0={eblast}')
    plt.show()

    # plt.figure(784)
    # tt = 0.5
    # xrho = np.linspace(-0.05, 0.05, 55000)
    # density_initial = sedov.interpolate_self_similar(0.05, xrho, g_interp)
    # density_final = sedov.interpolate_self_similar(tt, xrho, g_interp)
    # # plt.plot(xrho/sigma_t, density_initial, 'k--')
    # plt.plot(xrho/sigma_t, density_final, 'k-')
    # plt.xlabel('x [cm]', fontsize= 16)
    # plt.ylabel(r'$\rho$ [g/cc]', fontsize=16)
    # plt.xlim(-0.05/sigma_t, 0.05/sigma_t)
    # show('density_profile_TS_1')
    # plt.show()

    # plt.figure(785)
    # xrho = np.linspace(-0.05, 0.05, 55000)
    # # density_initial = sedov.interpolate_self_similar(0.05, xrho, g_interp)
    # t = 0.7
    # x = 0.03
    # mu = 0.75
    # density_static = sedov.interpolate_self_similar(t, xrho, g_interp)
    # density_shifted = density_static * 0 
    # for ix, xx in enumerate(density_shifted):
    #      s = xrho[ix]
    #      chi = (s - x) /mu + t
    #      density_shifted[ix] = sedov.interpolate_self_similar(chi, np.array([s]), g_interp)[0]
    # plt.plot(xrho/sigma_t, density_static, 'k--')
    # plt.plot(xrho/sigma_t, density_shifted, 'k-')
    # plt.xlabel('x [cm]', fontsize= 16)
    # plt.ylabel(r'$\rho$ [g/cc]', fontsize=16)
    # plt.xlim(-10,10)
    # show('density_profiles_chi')
    # plt.show()

def TS_x0sensitivity(mu = 1.0, x01 = 0.1, x02 = 5):
     tt = 30
     sigma_t = 0.001
     npts = 100
     t0 = 100000
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
     x0list = np.linspace(x01, x02, npts)
     right_detector = x0list * 0
     
     v = 29.98
     t = tt
     gamma = 7/5
     epsilon0 = 0.851072 * 1e18
     ns = 1e-9
     rho0 = 1
     inflection_point =     (-27*t*mu + 27*t*gamma**2*mu - (8*math.sqrt(2)*math.sqrt(ns**4*epsilon0**2*sigma_t**2 - 4*ns**4*gamma**2*epsilon0**2*sigma_t**2 + 6*ns**4*gamma**4*epsilon0**2*sigma_t**2 - 4*ns**4*gamma**6*epsilon0**2*sigma_t**2 + ns**4*gamma**8*epsilon0**2*sigma_t**2))/(v**2*mu**2*rho0))/(27.*(-1 + gamma**2))
     print('###')
     print(inflection_point, 'inflection')
     print('###')
     for ix, xx in enumerate(x0list):
          right_detector[ix] = TS_bench(tt, np.array([xx]), g_interp, v_interp, sedov, t0 = t0)[0]
     plt.plot(x0list,right_detector, 'k-' )
     plt.show()



 
def plot_TS_rightdetector(t1 = 0.0, tf = 1.5, sigma_t = 0.001, t0 = 0.5, x0 = 0.15, beta = 3.5):
    # x0 = 0.15
    plt.ion()
    plt.figure(32)

    # plt.plot(fulltlist, output_phi[:,0], 'k-')
    g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    # show(f'left_exit_dist_c={0.0}')
    tlistdense = np.linspace(t1, tf, 50)
    tlistdense = np.append(tlistdense, 2*x0)
    tlistdense = np.sort(tlistdense)
    t_bench_right = tlistdense * 0
    t_experiment = tlistdense * 0 
    integral = integrate_sedov_selfsim(beta = beta, sigma_t=sigma_t, x0 = x0)
    for it, tt in enumerate(tqdm(tlistdense)):
         t_bench_right[it] = TS_bench(tt, np.array([x0]), g_interp, v_interp, sedov, t0 = t0, beta = beta)
        #  if tt > 2 * x0:

         t_experiment[it] = integrate_density(tt, integral, beta = beta)
        
    plt.figure(33)       
    plt.plot(tlistdense/sigma_t/29.98, t_bench_right, 'k-')  
    plt.plot(tlistdense/sigma_t/29.98, t_experiment, 'bo', mfc = 'none')  
    plt.xlabel('t [ns]', fontsize = 16)
    plt.ylabel(r'$\phi^+$', fontsize = 16)


    plt.figure(34)       
    plt.plot(tlistdense, t_bench_right -t_experiment , 'k-')  
    # plt.plot(tlistdense, , 'bo', mfc = 'none')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi^+$', fontsize = 16)


    show(f'f2_right_exit_dist_c={0.0}')

def plot_TS_middle(t=1.0, t0 = 0.5, x0 = 0.15):
    
    plt.ion()
    sigma_t = 0.01
    plt.figure(32)

    # plt.plot(fulltlist, output_phi[:,0], 'k-')
    g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    xs = np.linspace(-x0, x0, 150)
    # show(f'left_exit_dist_c={0.0}')
    # tlistdense = np.linspace(t1, tf, 600)
    # t_bench_right = tlistdense * 0
    # for it, tt in enumerate(tlistdense):
    t_bench = TS_bench(t, xs, g_interp, v_interp, sedov, t0 = t0)
        
    plt.figure(33)       
    plt.plot(xs, t_bench, 'k-')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi^+$', fontsize = 16)

def TS_psi(t, xs, mu, interp_g_fun, interp_v_fun, sedov, sigma_t = 1e-3, t0 = 0.5, x0 = 0.15, transform = True):
    xs_quad, ws_quad = quadrature(20, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(64)
    mu_quad = res1.points
    mu_ws = res1.weights
    sedov_uncol = sedov_uncollided_solutions(xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_t, t0, transform)
    psi = sedov_uncol.uncollided_angular_flux(xs, t, mu, sedov_class, interp_g_fun, interp_v_fun)[0]
    return psi
     
def TS_bench(t, xs, interp_g_fun, interp_v_fun, sedov, beta = 3.5, sigma_t = 1e-3, t0 = 0.5, x0 = 0.15, transform = True, relativistic = False):

    xs_quad, ws_quad = quadrature(20, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(64)
    mu_quad = res1.points
    mu_ws = res1.weights
    
    
    sedov_uncol = sedov_uncollided_solutions(xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_t, t0, transform, lambda_1 = beta, relativistic= relativistic)

    phi = sedov_uncol.uncollided_scalar_flux(xs, t, sedov, interp_g_fun, interp_v_fun)
    
    return phi

def TS_current(t, xs, interp_g_fun, interp_v_fun, sedov, sigma_t = 1e-3, t0 = 0.5, x0 = 0.15, beta = 3.5, transform = True, relativistic = False):

    xs_quad, ws_quad = quadrature(20, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(64)
    mu_quad = res1.points
    mu_ws = res1.weights
    
    
    sedov_uncol = sedov_uncollided_solutions(xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_t, t0, transform, lambda_1 = beta, relativistic= relativistic)

    J = sedov_uncol.current(xs, t, sedov, interp_g_fun, interp_v_fun)
    
    return J


def TS_bench_prime(sigma_t = 1e-3, eblast = 1e20, tstar = 1e-12):
    f_fun, g_fun, l_fun = get_sedov_funcs()
    sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t, eblast = eblast, tstar = tstar)
    foundzero = False
    iterator = 0
    while foundzero == False:
        if g_fun[iterator] == 0.0:
            iterator += 1
        else:
            foundzero = True
        if iterator >= g_fun.size:
                assert(0)
    f_fun = f_fun[iterator:]
    g_fun = g_fun[iterator:]
    l_fun = l_fun[iterator:]
    
    interp_g_fun, interp_v_fun = interp_sedov_selfsim(g_fun, l_fun, f_fun, sedov)

    # plt.figure(24)
    # plt.ion()
    # plt.plot(l_fun, interp_g_fun.eval_spline(l_fun))
    # plt.show()
    # plt.figure(25)

    # plt.figure(24)
    # plt.ion()
    # plt.plot(l_fun, interp_v_fun.eval_spline(l_fun))
    # plt.show()
    # plt.figure(25)

    # rs = np.linspace(-0.5, 0.5,100)
    # plt.ion()
    # plt.plot(rs, sedov.interpolate_self_similar(1.0, rs, interp_g_fun))
    # plt.show()
    



    return interp_g_fun, interp_v_fun, sedov
    

def interp_sedov_selfsim(g_fun, l_fun, f_fun, sedov_class):

             l_fun = np.flip(l_fun)
             l_fun[0] = 0.0
             g_fun = np.flip(g_fun)
             f_fun[-1] = 0.0
             l_fun[-1] = 1.0
            #  g_fun[-1] = sedov_class.gpogm
            #  l_fun[-1] = 1.0
             g_fun[-1] = 1.0

             interpolated_g = cubic_spline(l_fun, g_fun)
             interpolated_v = cubic_spline(l_fun, np.flip(f_fun))
            #  print('g interpolated')
             return interpolated_g, interpolated_v


def TS_blast_scattering_profiles(N_space = 8, cc = 0.35, uncollided = False, t0 = 0.05):
    M = 6
    x0 = 0.15
    ylim1 = 1.0
    ylim2 = 0.14
    ylim3 = 0.26
    fulltlist = np.array([0.0005, 0.005, 0.01,.015, 0.02,0.025, 0.03,0.035, 0.04,0.045, 0.05,0.055, 0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095, 0.1,0.12,0.13,0.14,0.15,0.155,0.16, 0.17, 0.18, 0.19, 0.2,0.205, 0.21, 0.215, 0.22, 0.225,0.23,0.235,0.24,0.245, 0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295, 0.3,0.31,0.315,0.32,0.325,0.33,0.34,0.345, 0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.395, 0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49, 0.5, 0.55, 0.6, 0.65,  0.7,0.75,0.8,0.9, 1.0,1.1,1.2,1.3,1.4, 1.5])
    loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    # tlist1 = np.array([0.05, 0.4, 0.7, 1.0, 2.5])
    # tlist1 = np.array([0.0005, 0.005, 0.05, 0.1, 0.1125,0.13,0.14,0.141,0.142, 0.15, 0.151, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21,0.22,0.23,0.24, 0.25, 0.3,0.31,0.32,0.33,0.34, 0.35, 0.4,0.45, 0.5])
    tlist1 = np.array([0.0005, 0.005, 0.01,.015, 0.02,0.025, 0.03,0.035, 0.04,0.045, 0.05,0.055, 0.06,0.065,0.07,0.075,0.08,0.085,0.09,0.095, 0.1])    # tlist1 = np.array([1.0])
    # tlist2 = np.array([2.5, 2.6, 3.0, 3.5,  4.0,  5.0])
    tlist2 = np.array([0.12,0.13,0.14,0.15,0.155,0.16, 0.17, 0.18, 0.19, 0.2,0.205, 0.21, 0.215, 0.22, 0.225,0.23,0.235,0.24,0.245, 0.25,0.255,0.26,0.265,0.27,0.275,0.28,0.285,0.29,0.295, 0.3,0.31,0.315,0.32,0.325,0.33,0.34,0.345, 0.35,0.355,0.36,0.365,0.37,0.375,0.38,0.385,0.39,0.395, 0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49, 0.5, 0.55, 0.6, 0.65,  0.7,0.75,0.8,0.9, 1.0,1.1,1.2,1.3,1.4, 1.5])
    tlist = np.concatenate((tlist1, tlist2[1:]))
    plot_list = np.array([0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 5.5])
    # tlist = np.array([1.0])
    # tlist = tlist1
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    RMSE_list = tlist*0
    phi_left = fulltlist*0

    g_interp, v_interp, sedov = TS_bench_prime()
    # f, (a1, a2) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    fig1 = plt.figure(1)
    gs = fig1.add_gridspec(15,15)
    a1= fig1.add_subplot(gs[0:2,: ])
    a2 = fig1.add_subplot(gs[4:15, 4:-4])
    a3 = fig1.add_subplot(gs[4:, -2:])
    a4 = fig1.add_subplot(gs[4:, 0:2])
    plt.ion()
    phi_right = fulltlist * 0
    for it, tt in enumerate(tqdm(tlist1)):
         loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)
         xs = loader.xs
         phi = loader.phi
         output_phi = loader.exit_phi
        #  print(loader.exit_phi)
        #  assert(0)
        #  phi_right[it] = loader.exit_phi[1][it]
        #  phi_left[it] = loader.exit_phi[0][it]
        #  mus = loader.mus


        #  xs = np.linspace(-x0, x0, 200)
        #  phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov)
        #  RMSE_list[it] = RMSETS(phi_bench, phi)

        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #
         string1 = 'before'
        
            #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
            #   plt.text(-4.96,0.2,r'$t=0$')
            #   plt.text(-4.21,0.16,r'$t=1$')
            #   plt.text(-3.25,0.09,r'$t=2$')
            #   plt.text(-2.34,0.09,r'$t=3$')
            #   plt.text(-.70,0.092,r'$t=5$')
            #   plt.text(0.93,0.047,r'$t=7$')
            #   plt.text(0.4,0.2,r'$t=15$')
  

            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            # plt.text(-4.96,0.704,r'$t=15$')
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
         xrho = np.linspace(-x0, x0, 500)
         density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
         density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
        #  plt.text(-.489,0.404,r'$t=0.05$')
        #  plt.text(-.299,0.154,r'$0.4$')
        #  plt.text(-.107,0.161,r'$0.7$')
        #  plt.text(0.335,0.065,r'$1.0$')
        #  plt.text(.156,0.291,r'$2.5$')
         plt.ion()
         plt.show()

        #  a2.plot(xs, phi, 'k-', mfc = 'none')
        # #  a2.plot(xs, phi_bench, 'k-', mfc = 'none')
        #  a2.set_ylabel(r'$\phi$', fontsize = 16)
        #  a1.set_ylabel(r'$\rho$', fontsize = 16)
        #  a1.plot(xrho, density_initial, 'k--')
        #  a1.plot(xrho, density_final, 'k-')

        #  plt.xlabel('x', fontsize = 16)
         if tt in plot_list:
            a2.plot(xs, phi, 'k-', mfc = 'none')
            a2.set_ylabel(r'$\phi$', fontsize = 16)
            a2.set_xlabel('x', fontsize = 16)
            a1.set_ylabel(r'$\rho$', fontsize = 16)
            a1.set_xlim(-0.05, 0.05)
            a1.set_xlabel('x', fontsize = 16)
            a1.plot(xrho, density_initial, 'k--')
            a1.plot(xrho, density_final, 'k-')
            a2.set_ylim(0, ylim1)
            tstopargmin = np.argmin(np.abs(fulltlist-t0))
            a3.plot(fulltlist[0:tstopargmin], loader.exit_phi[0:tstopargmin,1], 'k-')
            a4.plot(fulltlist[0:tstopargmin], loader.exit_phi[0:tstopargmin,0], 'k-')
            a3.set_ylim(0, ylim3)
            a3.set_ylabel(r'$\phi^+(x=x_0)$', fontsize = 16)
            a4.set_ylabel(r'$\phi^-(x=-x_0)$', fontsize = 16)
            a3.set_xlabel('t', fontsize = 16)
            a4.set_xlabel('t', fontsize = 16)
            a3.yaxis.set_label_position("right")
            a3.yaxis.tick_right()
            a3.set_xlim(0.0, 1.5)
            a4.set_xlim(0.0, 1.5)
            a4.set_ylim(0, ylim1)
            a2.set_xlim(-.15, .15)
            # a1.set_ylim(0,5.8)
            # a4.set_ylim(0, 0.14)
            
            show(f'c={cc}_solutions_TS_t0={t0}' + string1)
        #  assert(0)

    # f, (a3, a4) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4]})
    fig2 = plt.figure(2)
    gs = fig2.add_gridspec(15,15)
    a1= fig2.add_subplot(gs[0:2,: ])
    a2 = fig2.add_subplot(gs[4:15, 4:-4])
    a3 = fig2.add_subplot(gs[4:, -2:])
    a4 = fig2.add_subplot(gs[4:, 0:2])
    for it2, tt in enumerate(tqdm(tlist2)):
            loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)
            xs = loader.xs
            phi = loader.phi
            output_phi = loader.exit_phi
            # phi_right[it2 + it] = phi[-1]
            #  xs = np.linspace(-x0, x0, 200)
            # phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov)
            # RMSE_list[it + it2] = RMSETS(phi_bench, phi)

            #  eval_times = loader.eval_array
            #  if eval_times[counter] = tlist[counter]:         
            #     left_exit[it] = output_phi[counter, 0]
            #     right_exit[it] = output_phi[1]
            
                #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
                #   plt.text(-4.96,0.2,r'$t=0$')
                #   plt.text(-4.21,0.16,r'$t=1$')
                #   plt.text(-3.25,0.09,r'$t=2$')
                #   plt.text(-2.34,0.09,r'$t=3$')
                #   plt.text(-.70,0.092,r'$t=5$')
                #   plt.text(0.93,0.047,r'$t=7$')
                #   plt.text(0.4,0.2,r'$t=15$')
            # plt.text(-.439,0.9,r'$t=2.5$')
            # plt.text(-.402,0.38,r'$2.6$')
            # plt.text(-.16,0.16,r'$3.0$')
            # plt.text(0.122,0.108,r'$3.5$')
            # plt.text(0.335,0.048,r'$4.0$')
           
            string1 = 'after'
            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
            xrho = np.linspace(-x0, x0, 55000)
            density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
            density_final = sedov.interpolate_self_similar(tlist[-1], xrho, g_interp)
            
            plt.ion()
            # plt.show()

            # a4.plot(xs, phi, 'k-', mfc = 'none')
            # # a4.plot(xs, phi_bench, 'k-', mfc = 'none')
            # a4.set_ylabel(r'$\phi$', fontsize = 16)
            # a3.set_ylabel(r'$\rho$', fontsize = 16)
            # a3.plot(xrho, density_initial, 'k--')
            # a3.plot(xrho, density_final, 'k-')

            # plt.xlabel('x', fontsize = 16)
            if tt in plot_list:
                a2.plot(xs, phi, 'k-', mfc = 'none')
                a2.set_ylabel(r'$\phi$', fontsize = 16)
                a2.set_xlabel('x', fontsize = 16)
                a1.set_ylabel(r'$\rho$', fontsize = 16)
                # a1.set_xlim(-0.05, 0.05)
                a1.set_xlabel('x', fontsize = 16)
                a1.plot(xrho, density_initial, 'k--')
                a1.plot(xrho, density_final, 'k-')
                a2.set_ylim(0, ylim1)
                a3.plot(fulltlist[0:], loader.exit_phi[:,1], 'k-')
                a4.plot(fulltlist[0:], loader.exit_phi[:,0], 'k-')
                a3.set_ylim(0, ylim3)
                
                a3.set_ylabel(r'$\phi^+(x=x_0)$', fontsize = 16)
                a4.set_ylabel(r'$\phi^-(x=-x_0)$', fontsize = 16)
                a3.set_xlabel('t', fontsize = 16)
                a4.set_xlabel('t', fontsize = 16)
                a4.set_xlim(0.0, 1.5)
                a3.yaxis.set_label_position("right")
                a3.yaxis.tick_right()
                a3.set_xlim(0.0, 1.5)
                a4.set_ylim(0, ylim2)
                # a1.set_ylim(0,5.8)
                # a4.set_ylim(0, 0.14)
            
                show(f'c={cc}_solutions_TS_t0={t0}' + string1)






    plt.figure(111)
    plt.plot(fulltlist[0:], loader.exit_phi[:,0], 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi^-(x=-x_0)$', fontsize = 16)
    show(f'left_exit_dist_c={cc}_t0={t0}_TS')
    plt.show()

    plt.figure(222)       
    plt.plot(fulltlist, output_phi[:,1], 'k-')  
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi$', fontsize = 16)
    show(f'right_exit_dist_c={cc}_t0={t0}_TS')

    plt.figure(333)
    plt.plot(fulltlist, np.gradient(loader.exit_phi[:,0], fulltlist), 'k--')
    plt.xlim(0.01, fulltlist[-1])
    plt.ylim(-0.003, 0.003)
    plt.show()


    plt.figure(444)
    # quiescent fiducial
    f = h5py.File('moving_mesh_transport/local_run_data/blast_fiducial.h5', 'r+')
    outputpsi = f['phiout'][:,:,:]
    outputphi = np.zeros(len(outputpsi[:,0,0]))
    N_ang = len(outputpsi[0,:,0])
    print(len(outputpsi[0,:,0]))
    ws = quadpy.c1.gauss_lobatto(len(outputpsi[0,:,0])).weights
    for it in range(len(outputpsi[:,0,0])):
         outputphi[it] = np.sum(np.multiply(ws[0:int(N_ang/2)], outputpsi[it,0:int(N_ang/2),0]))
         
         
    print(outputphi)
    print(np.shape(outputphi))
    outuputt = f['times'][:]
    f.close()   
    
    plt.plot(fulltlist[0:], loader.exit_phi[:,0] - outputphi, 'k-')
    plt.xlabel('t', fontsize = 16)
    plt.ylabel(r'$\phi^-(x=-x_0)-\phi_q(x=-x_0)$', fontsize = 16)
    show(f'left_exit_dist_c={cc}_t0={t0}_TS_minusfiducial')
    plt.show()

    plt.figure(445)
    plt.plot(fulltlist[0:], np.gradient(loader.exit_phi[:,0] - outputphi), 'k-')
    plt.xlabel('t', fontsize = 16)

    plt.ylabel(r'$\frac{d}{dt}\left(\phi^-(x=-x_0)-\phi_q(x=-x_0)\right)$', fontsize = 16)
    
    show(f'left_exit_dist_c={cc}_t0={t0}_TS_minusfiducial_gradient')
    





def TS_video():
    M = 6
    x0 = 0.15
    t0 = 0.5
    fulltlist = np.array([0.05,0.5, 0.6, 0.7,  1.0, 1.5, 2.0, 2.5, 2.5, 2.6,  3.0, 3.5,  4.0, 4.5,  5.0])

    # loader = load_sol('transport', 'plane_IC', 'transport', s2 = False, file_name = 'run_data.hdf5', c = cc)
    # tlist1 = np.array([0.05, 0.4, 0.7, 1.0, 2.5])
    # tlist1 = np.array([0.05,0.5, 0.6, 0.7,  1.0, 1.5, 2.0, 2.5])

    # tlist1 = np.array([1.0])
    # tlist2 = np.array([2.5, 2.6, 3.0, 3.5,  4.0,  5.0])
    # tlist2 = np.array([2.5, 2.5, 2.6,  3.0, 3.5,  4.0, 4.5,  5.0])
    # tlist = np.concatenate((tlist1, tlist2))
    nt = 900
    tlist = np.linspace(0.0000001, 1.5, nt)
    dt = (tlist[-1]-tlist[0])/nt
    # tlist = np.array([1.0])
    # tlist = tlist1
    left_exit = np.zeros(tlist.size)
    right_exit = np.zeros(tlist.size)
    counter = 0
    RMSE_list = tlist*0
    sigma_t = 1e-3
    g_interp, v_interp, sedov = TS_bench_prime(sigma_t = sigma_t)
    # f, (a1, a2, a3) = plt.subplots(2,1, gridspec_kw={'height_ratios': [1,  4,1]})
    fig = plt.figure()
    gs = fig.add_gridspec(3,3)
    a1= fig.add_subplot(gs[0,: ])
    a2 = fig.add_subplot(gs[1:3, :-1])
    a3 = fig.add_subplot(gs[1:, -1])
    
    xs = np.linspace(-x0, x0, 120)
    
    plt.ion()
    t_bench_right = tlist*0
    for it, tt in enumerate(tqdm(tlist)):
        #  loader.call_sol(tt, M, x0, N_space, 'rad', uncollided, True)

        #  xs = np.linspace(-x0, x0, 200)
         phi_bench = TS_bench(tt, xs, g_interp, v_interp, sedov, x0 = x0, t0 = t0)

         t_bench_right[it] = TS_bench(tt, np.array([x0]), g_interp, v_interp, sedov, x0 = x0, t0 = t0, sigma_t= sigma_t)


        #  eval_times = loader.eval_array
        #  if eval_times[counter] = tlist[counter]:         
        #     left_exit[it] = output_phi[counter, 0]
        #
        #  string1 = 'before'
        
            #   plt.text(0.0, 0.1, r'$t\uparrow$', fontsize = 32)
            #   plt.text(-4.96,0.2,r'$t=0$')
            #   plt.text(-4.21,0.16,r'$t=1$')
            #   plt.text(-3.25,0.09,r'$t=2$')
            #   plt.text(-2.34,0.09,r'$t=3$')
            #   plt.text(-.70,0.092,r'$t=5$')
            #   plt.text(0.93,0.047,r'$t=7$')
            #   plt.text(0.4,0.2,r'$t=15$')
  

            # plt.text(0.0, 0.1, r'$t\downarrow$', fontsize = 32)
            # plt.text(-4.96,0.704,r'$t=15$')
            # plt.text(-4.25,0.532,r'$t=16$')
            # plt.text(-3.4,0.424,r'$t=17$')
            # plt.text(-2.85,0.31,r'$t=18$')
            # plt.text(-1.5,0.277,r'$t=19$')
            # plt.text(-1.18,0.161,r'$t=20$')
            # plt.text(-.6,0.04,r'$t=22$')
         
         xrho = np.linspace(-x0, x0, 55000)
         density_initial = sedov.interpolate_self_similar(tlist[0], xrho, g_interp)
         density_final = sedov.interpolate_self_similar(tlist[it], xrho, g_interp)
        #  plt.text(-.489,0.404,r'$t=0.05$')
        #  plt.text(-.299,0.154,r'$0.4$')
        #  plt.text(-.107,0.161,r'$0.7$')
        #  plt.text(0.335,0.065,r'$1.0$')
        #  plt.text(.156,0.191,r'$2.5$')
         
        #  plt.show()
         

        #  a2.plot(xs, phi, 'b-o', mfc = 'none')
         a2.plot(xs, phi_bench, 'k-', mfc = 'none')
         a2.set_ylabel(r'$\phi$', fontsize = 16)
         a2.set_xlabel('x', fontsize = 16)
         a1.set_ylabel(r'$\rho$', fontsize = 16)
        #  a1.set_xlim(-0.05, 0.05)
         a1.set_xlabel('x', fontsize = 16)
         a1.plot(xrho, density_initial, 'k--')
         a1.plot(xrho, density_final, 'k-')
         a2.set_ylim(0, 1.05)
         a3.plot(tlist[0:it], t_bench_right[0:it], 'k-')
         a3.set_ylim(0, 0.5)
         a3.set_ylabel(r'$\phi(x=x_0)$', fontsize = 16)
         a3.set_xlabel('t', fontsize = 16)
         a3.yaxis.set_label_position("right")
         a3.yaxis.tick_right()
         a3.set_xlim(0.0, 2)
         a1.set_ylim(0,5.8)


 
         plt.pause(dt)
        #  show(f'blast_plot_vid/t={tt}_solutions_TS' )
         plt.savefig(f'blast_plot_vid/solutions_TS_{it}.jpg', bbox_inches="tight" )
         a2.clear()
         a1.clear()
         a3.clear()
         
         
         
         


def plot_r2_vs_t(sigma_t = 1e-3):
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t = sigma_t)
     tlist = np.linspace(0.0, 1.5)
     rlist = tlist * 0 
     tshift = (0.0000000001 /29.98)*10/sigma_t
     rlistguess = tlist * 0
     conversion = 1/29.98/sigma_t * 1e-9
     for it, tt in enumerate(tlist):
          sedov.physical(tt)
          rlist[it] = sedov.r2_dim
          t = tt/ 29.98/sigma_t * 1e-9
          rlistguess[it] = (sedov.eblast/(sedov.alpha*sedov.rho0))**(1.0/3) * (t + tshift)**(2.0/3) * sigma_t
     c1 = (sedov.eblast/(sedov.alpha*sedov.rho0))**(1.0/3)
     ts = 0.15
     start_index = np.argmin(np.abs(tlist-ts))
     r2v = (rlist[-1]-rlist[start_index]) / (tlist[-1]-tlist[start_index])
     th = tlist[int(tlist.size/2)]
     r2th = rlist[int(tlist.size/2)]
     tf = tlist[-1]
     r2f = rlist[-1]
     x0 = rlist[start_index]
     

    #  r2va = (r2half - rlist[0] - r2v * thalf) * 2 / (thalf**2)
     r2va =  (2*(r2th*tf - r2f*th + r2f*ts - r2th*ts - tf*x0 + th*x0))/((tf - th)*(tf - ts)*(-th + ts))
     r2v =  -((r2th*tf**2 - r2f*th**2 - 2*r2th*tf*ts + 2*r2f*th*ts - r2f*ts**2 + r2th*ts**2 - tf**2*x0 + th**2*x0 + 2*tf*ts*x0 - 2*th*ts*x0)/((tf - th)*(tf - ts)*(-th + ts)))
    #  tstar = 0.2
     plt.figure(1)
     plt.plot(tlist, rlist, 'k-')
     plt.plot(tlist, c1*(tlist * conversion+tshift) ** (2/3) * sigma_t, 'o', mfc = 'none')
     plt.plot(tlist, rlistguess, 'x')
    #  plt.plot(tlist, rlist[0] + r2v * tlist, '-^')
     plt.plot(tlist, rlist[start_index] + r2v * (tlist-ts) + 0.5*r2va * (tlist-ts) **2, '-s')
     plt.xlabel('t')
     plt.ylabel('r2')
     plt.show()
    #  plt.figure(2)
    #  dt = tlist[1]-tlist[0]
    #  dt_guess = (2/3)*sigma_t * c1  * (tlist * conversion + tshift)**(-1/3) * conversion
    #  plt.plot(tlist, np.gradient(rlist, dt))
    #  plt.plot(tlist, dt_guess , 'o', mfc = 'none')
    #  plt.show()
    #  print(dt_guess/np.gradient(rlist, dt ))
     plt.figure(1)
    #  plt.plot(tlist, -rlist, 'k-')
    #  plt.plot(tlist, c1*(tlist * conversion+tshift) ** (2/3) * sigma_t, 'o', mfc = 'none')
    #  plt.plot(tlist, rlistguess, 'x')
    #  plt.plot(tlist, rlist[0] + r2v * tlist, '-^')
     plt.plot(tlist, -rlist -(-rlist[start_index] - r2v * (tlist-ts) - 0.5*r2va * (tlist-ts) **2), '-s')
     plt.xlabel('t')
     plt.ylabel('r2')
     plt.show()
     print(r2v, r2va)



def sparsify(old_x, old_y, n):
     new_x = []
     new_y = []
     counter = 0
     for ix, xx in enumerate(old_x):
          if counter == n:
               new_x.append(xx)
               new_y.append(old_y[ix])
               counter == 0
          counter += 1

def integrate_sedov_selfsim(beta = 3.5, sigma_t=0.001, x0=0.15):
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
     def integrand(x):
        return g_interp.eval_spline(np.array([x]))**beta
     integral = integrate.quad(integrand, 0,1)[0]
     return integral


def dbeta_sedov_selfsim(beta = 3.5, sigma_t=0.001, x0=0.15):
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
     def integrand(x):
        return beta * g_interp.eval_spline(np.array([x]))**(beta-1)
     integral = integrate.quad(integrand, 0,1)[0]
     return integral
     
def integrate_density(t,integral, sigma_t = 0.001, x0 = 0.15, beta = 2.5):
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
     xs = np.linspace(0,1)
    #  plt.plot(xs, g_interp.eval_spline(xs))
    #  plt.show()

     
     
     
    #  
    #  integral = 0.060887790433472026
     f_fun, g_fun, l_fun = get_sedov_funcs()
     sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t)
     def lambda1(mu):
          return new_bench(x0, t, mu, sedov, integral, beta = beta)[0]
     def lambda2(mu):
          return new_bench(x0, t, mu, sedov, integral, beta = beta)[1]
     
     def psi(mu):
          arg = - 1.0 / (mu+1e-16)  * lambda2(mu)
        #   print(arg)
          if mu - 2*x0/t >= 0:
            return math.exp(arg) 
            
          else:
               return 0.0
     
     
      
     aa = abs(x0+x0) / t
     integral2 = integrate.quad(psi, aa,1)[0]

    #  print(integral2)
     return integral2


def J_approx(t,integral, sigma_t = 0.001, x0 = 0.15, beta = 2.5):
     g_interp, v_interp, sedov = TS_bench_prime(sigma_t)
     xs = np.linspace(0,1)
    #  plt.plot(xs, g_interp.eval_spline(xs))
    #  plt.show()

     
     
     
    #  
    #  integral = 0.060887790433472026
     f_fun, g_fun, l_fun = get_sedov_funcs()
     sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t)
     def lambda1(mu):
          return new_bench(x0, t, mu, sedov, integral, beta = beta)[0]
     def lambda2(mu):
          return new_bench(x0, t, mu, sedov, integral, beta = beta)[1]
     
     def integrand(mu):
          arg = - 1.0 / (mu+1e-16)  * lambda2(mu)
        #   print(arg)
          if mu - 2*x0/t >= 0:
            return math.exp(arg) * mu
            
          else:
               return 0.0
     
     
      
     aa = abs(x0+x0) / t
     integral2 = integrate.quad(integrand, aa,1)[0]

    #  print(integral2)
     return integral2

def new_bench(x, t, mu, sedov, integral, x0 = .15, sigma_t = 0.001, beta = 3.5):
     rho0 = 1.0
     rho2 = 6.0
    #  beta = 3.5
     integral_val = 0.0
     integral_val2 = 0.0
     
    #  print(t, 't')
     if mu -  (x+ x0) / t > 0:
        gamma = 7/5
        x2, x3 = sedov.find_r2_in_transformed_space(x, t, mu, x0)
        # print(sedov.find_r2_in_transformed_space(.15, .3/.5, .5, .15)[0] -0.00767215 )
        # print(x2, x3, 'x2, x3')
        integral_val += rho0 ** beta * (2 * x0 + x2 - x3)
        # print(integral_val)
        integral_val +=  rho2 ** beta * integral * abs(x2)
        # integral_val +=  rho2 ** beta * integral * x3
        integral_val +=  rho2 ** beta * integral * x3
        # c1 = 0.016564618101813887
        # c1 = 0.007688614641443944
        # # c1 = integrate_sedov_selfsim(beta = beta)
        # c1 = integral
        # # t1 = 12 * c1**2 * (t-x/mu)**(1/3) * (x-mu*t) / (4 * c1**2 * (t-x/mu)**(1/3)+9*mu*(x-mu*t))
        # t1 =  (18*c1*(t - x/mu)**0.6666666666666666*mu*(x - t*mu))/(4*c1**2*(t - x/mu)**0.3333333333333333 + 9*mu*(x - t*mu))
        # t2 = integral * ((gamma + 1) /(gamma -1))**beta - 1
        # integral_val2 = rho0**beta * (2* x0 + t1 * t2 )
        # integral_val2 = rho0**beta *( (x3-x2) * t1 + 2*x0) 
        gammp1 = (gamma+1)/(gamma-1)
        v = 29.98
        # integral_val2 = rho0 ** beta *(2*x0 + x2-x3)
        # integral_val2 += (x3 -x2) * rho0**beta * (gammp1)**beta*integral
        # diff =  (18*c1*(t - x/mu)**0.6666666666666666*mu*(x - t*mu))/(4*c1**2*(t - x/mu)**0.3333333333333333 + 9*mu*(x - t*mu))
        # print(diff, x3-x2, 'true vs estimated')
        # epsilon0 = 0.851072 * 1e18
        epsilon0 = 0.851072 * 1e18
        ns = 1e-9
        omega = integral
        sigma = sigma_t
        diff2 =  (-9*v*(2*t - (2*x)/mu)**0.6666666666666666*mu*(-x + t*mu)*(((-1 + gamma**2)*epsilon0)/rho0)**0.3333333333333333*(ns/(v*sigma))**0.6666666666666666*sigma)/ (9*v*mu*(x - t*mu) + 2*ns*(2*t - (2*x)/mu)**0.3333333333333333*(((-1 + gamma**2)*epsilon0)/rho0)**0.6666666666666666*(ns/(v*sigma))**0.3333333333333333*sigma)
        integral_val2 = rho0**beta * (2*x0  + diff2 * ((gammp1)**beta*integral-1)  )
        # x2est = (3*c1*(-x + t*mu)*((-x + t*mu)/mu)**0.6666666666666666)/(-3*x + 3*t*mu + 2*c1*((-x + t*mu)/mu)**0.6666666666666666)
        # print(x2est, x3, 'x diff')

        # integral_val = rho0 ** beta * ()
        # integral_val2 = rho0**beta * (2* x0 + (abs(x2) + x3) * t2 )
        integral_val3 = (24.937436335987897*np.exp((0.029707062964723748*t)/mu**0.3333333333333333))/t
        # integral_val2 =         rho0**beta*(2*x0 - (9*v*(2*t - (2*x)/mu)**0.6666666666666666*mu*(-x + t*mu)*(((-1 + gamma**2)*epsilon0)/rho0)**0.3333333333333333*(ns/(v*sigma))**0.6666666666666666*sigma*(-1 + ((1 + gamma)/(-1 + gamma))**beta*omega))/(9*v*mu*(x - t*mu) + 2*ns*(2*t - (2*x)/mu)**0.3333333333333333*(((-1 + gamma**2)*epsilon0)/rho0)**0.6666666666666666*(ns/(v*sigma))**0.3333333333333333*sigma))
        integral_val2 = lambdaapprox(rho0, beta, x0, t, x0, mu,epsilon0, sigma_t, omega )
        return integral_val, integral_val2, integral_val3
     
     else:
          return 0.0,0.0,0.0

     



def chi_x_vs_t():
    sigma_t = 1e-3
    g_interp, v_interp, sedov = TS_bench_prime()
    f_fun, g_fun, l_fun = get_sedov_funcs()
    sedov = sedov_class(g_fun, f_fun, l_fun, sigma_t)
    tpts = 1000
    x0 = 0.15
    mumu = 1
    xx = 0.05
    tt = 0.0

    xlist = np.linspace(-x0, xx, tpts)
    r2_position = np.zeros(tpts)
    tlist = np.linspace(0.001, 1.5, tpts)
    for it, tt in enumerate(tlist):
         sedov.physical(tt)
         r2_position[it] = sedov.r2_dim
    
    def chi(x, s, t, mu):
         return (s-x)/mu + t
    chilist = np.zeros(tpts)
    for it in range(tpts):
         ss = xlist[it]
         chilist[it] = chi(xx,ss, tt, mumu )

    plt.figure(1)
    plt.plot(r2_position, tlist, 'k--' )
    plt.plot(-r2_position, tlist, 'k--' )
    plt.plot(xlist, chilist, 'k-' )
    plt.ylabel('t')
    plt.xlabel('x')
    plt.show()
         

def lambdaapprox(rho0, beta, x0, t, x, mu,epsilon0, sigma, omega ):
     gamma = 7/5
     ns = 1e-9
     v = 29.98
     return rho0**beta*(2*x0 - (9*v*(2*t - (2*x)/mu)**0.6666666666666666*mu*(-x + t*mu)*(((-1 + gamma**2)*epsilon0)/rho0)**0.3333333333333333*(ns/(v*sigma))**0.6666666666666666*sigma*(-1 + ((1 + gamma)/(-1 + gamma))**beta*omega))/(9*v*mu*(x - t*mu) + 2*ns*(2*t - (2*x)/mu)**0.3333333333333333*(((-1 + gamma**2)*epsilon0)/rho0)**0.6666666666666666*(ns/(v*sigma))**0.3333333333333333*sigma))



def dJ_epsilon(x0, t, epsilon0, omega,  sigma_t=0.001, beta = 3.5, rho0 = 1.0):
    #  epsilon0 = 0.851072 * 1e18
    #  omega = integrate_sedov_selfsim(beta = beta, sigma_t=sigma_t, x0=x0)

     integrald = lambda mu: dlambda_depsilon(rho0, beta, x0, t, mu, omega, 1/sigma_t, epsilon0 )
     mfp = lambda mu : lambdaapprox(rho0, beta, x0, t, x0, mu,epsilon0, sigma_t, omega )

     def integrand(mu):
      if mu-2*x0/t >0:
           return -integrald(mu) * np.exp(-mfp(mu)/mu) * mu
      else:
           return 0.0

     dJdepsilon = integrate.quad(integrand, 0, 1)[0] 

     return dJdepsilon


def dJ_beta(x0, t, epsilon0, omega,  sigma_t=0.001, beta = 3.5, rho0 = 1.0):
    #  epsilon0 = 0.851072 * 1e18
    #  omega = integrate_sedov_selfsim(beta = beta, sigma_t=sigma_t, x0=x0)
    xs_quad, ws_quad = quadrature(20, 'gauss_legendre')
    # mu_quad, mu_ws = quadrature(200, 'gauss_legendre')
    res1 = quadpy.c1.gauss_legendre(64)
    mu_quad = res1.points
    mu_ws = res1.weights
    g_interp, v_interp, sedov = TS_bench_prime()
    sedov_uncol = sedov_uncollided_solutions(xs_quad, ws_quad, mu_quad, mu_ws, x0, sigma_t, 8000, True, lambda_1 = beta)
    #  integrald = lambda mu: dlambda_dbeta(rho0, beta, omega, epsilon0, sigma_t, mu,x0, t, dbeta_sedov_selfsim(beta =beta, sigma_t=sigma_t, x0=x0))
    integrald = lambda mu: dmfp_dbeta(sedov_uncol, sedov, g_interp, v_interp, beta, mu, x0, t)
    # mfp = lambda mu : lambdaapprox(rho0, beta, x0, t, x0, mu,epsilon0, sigma_t, omega )
    mfp = lambda mu: sedov_uncol.get_mfp( np.array([x0]), t, mu, sedov, g_interp, v_interp)[0]

    def integrand(mu):
      if mu-2*x0/t >0:
           return -integrald(mu) * np.exp(-mfp(mu)/mu) 
        # return np.exp(-mfp(mu)/mu)  * mu 
      else:
           return 0.0
    aa = abs(x0+x0) / t
    dJbeta = integrate.quad(integrand, aa, 1)[0] 
    # dJbeta = J_approx(t,integral, sigma_t = sigma_t, x0 = x0, beta = ee)

  

    return dJbeta

def plot_djdepsilon( sigma_t = 0.001, beta = 3.5, rho0 = 1.0):
     x0 = 0.15
     t = 0.6
     omega = integrate_sedov_selfsim(beta = beta, sigma_t=sigma_t, x0=x0)
     epsilon0 =  0.851072 * 1e18
     epsilon_list = np.linspace( 1e7,  1e9, 10)
     sensitivities = 0*epsilon_list
     for ie, ee in enumerate(epsilon_list):
        sensitivities[ie] = dJ_epsilon(x0, t, ee, omega,  sigma_t=sigma_t, beta = beta, rho0 = rho0)

     plt.ion()
     plt.plot(epsilon_list, sensitivities)
     plt.ylim(-1,1)
     plt.show()


def plot_dbeta( t = 0.6, sigma_t = 0.001, beta = 3.5, rho0 = 1.0):
     x0 = 0.15
    #  t = 0.6
     t0 = t + 0.1
     g_interp, v_interp, sedov = TS_bench_prime()
     
     epsilon0 =  0.851072 * 1e18
    #  epsilon_list = np.linspace( 1e7,  1e9, 10)
     beta_list = np.linspace(0.01, 6, 100)
     sensitivities = 0*beta_list
     current_list = 0 * beta_list
     current_list2 = 0 * beta_list
     dx = 1e-2
     dmfp_dbetalist = beta_list*0
     for ie, ee in enumerate(beta_list):
        omega = integrate_sedov_selfsim(beta = ee, sigma_t=sigma_t, x0=x0)
        # f1 = dJ_beta(x0, t, epsilon0, omega,  sigma_t=0.001, beta = ee + dx, rho0 = 1.0)
        # f2 = dJ_beta(x0, t, epsilon0, omega,  sigma_t=0.001, beta = ee, rho0 = 1.0)
        # dmfp_dbetalist[ie] = dmfp_dbeta(sedov_uncol, sedov, g_interp, v_interp, beta, mu, x0, t)

        sensitivities[ie] = dJ_beta(x0, t, epsilon0, omega,  sigma_t=0.001, beta = ee, rho0 = 1.0)
        # sensitivities[ie] = deriv(f1, f2, dx)
        integral = integrate_sedov_selfsim(beta = ee, sigma_t=sigma_t, x0=x0)
        current_list[ie] = TS_current(t, np.array([x0]), g_interp, v_interp, sedov, sigma_t = 1e-3, t0 = t0, x0 = 0.15, beta = ee, transform = True)[0]
        current_list2[ie] = J_approx(t,integral, sigma_t = sigma_t, x0 = x0, beta = ee)

     plt.ion()
     plt.figure(-10)
     plt.plot(beta_list, sensitivities)
    #  plt.ylim(-1,1)
     plt.show()


     plt.figure(-9)
     plt.plot(beta_list, current_list, 'k-', label =r'$J^+$')
    #  plt.plot(beta_list, current_list2, 'o', mfc = 'none')
     plt.plot(beta_list, sensitivities, 'k--',  label = r'$\frac{\partial J^+}{\partial \beta}$')
    #  plt.plot(beta_list, np.gradient(current_list, beta_list), 'k--', label = r'$\frac{\partial J^+}{\partial \beta}$')
     plt.ylabel(r'$J^+$')
     plt.xlabel(r'$\beta$')
     plt.legend()

    #  plt.plot(beta_list, np.gradient(beta_list, beta_list), 'b-')
     show('beta_sensitivity')
     plt.show()
     


     


     
     

def dlambda_depsilon(rho0, beta, x0, t, mu, omega, g, epsilon0 ):
        v = 29.98
        return rho0**beta*((3.0411009977867e6*g*(-1. + 6.**beta*omega)*t*(mu*t - 1.*x0)*(2.*mu - (2.*x0)/t))/(rho0*(g*t*v*(-4.624147639369632e12*mu*t + 4.624147639369632e12*x0) + 1.*(epsilon0/rho0)**0.6666666666666666*(g/v)**0.3333333333333333*(2.*mu - (2.*x0)/t)**0.3333333333333333)**2) - (1.52055049889335e6*(-1. + 6.**beta*omega)*t*(g/v)**0.6666666666666666*v*(mu*t - 1.*x0)*(2.*mu - (2.*x0)/t)**0.6666666666666666)/((epsilon0/rho0)**0.6666666666666666*rho0*(g*t*v*(-4.624147639369632e12*mu*t + 4.624147639369632e12*x0) + 1.*(epsilon0/rho0)**0.6666666666666666*(g/v)**0.3333333333333333*(2.*mu - (2.*x0)/t)**0.3333333333333333)))



def dlambda_dbeta(rho0, beta, omega, epsilon0, sigma, mu,x0, t, domega):
    ns = 1e-9
    v = 29.98
    x = x0
    return rho0**beta*(2*x0 - (18*3**0.3333333333333333*(-1 + 6**beta*omega)*(epsilon0/rho0)**0.3333333333333333*t*v*(ns/(v*sigma))**0.6666666666666666*sigma*((-2*x)/t + 2*mu)**0.6666666666666666*(-x + t*mu))/(5**0.6666666666666666*((8*3**0.6666666666666666*ns*(epsilon0/rho0)**0.6666666666666666*(ns/(v*sigma))**0.3333333333333333*sigma*((-2*x)/t + 2*mu)**0.3333333333333333)/(5.*5**0.3333333333333333) + 9*t*v*(x - t*mu))))*math.log(rho0) - (18*3**0.3333333333333333*(epsilon0/rho0)**0.3333333333333333*rho0**beta*t*v*(ns/(v*sigma))**0.6666666666666666*sigma*((-2*x)/t + 2*mu)**0.6666666666666666*(-x + t*mu)*(6**beta*omega*math.log(6) + 6**beta*domega))/(5**0.6666666666666666*((8*3**0.6666666666666666*ns*(epsilon0/rho0)**0.6666666666666666*(ns/(v*sigma))**0.3333333333333333*sigma*((-2*x)/t + 2*mu)**0.3333333333333333)/(5.*5**0.3333333333333333) + 9*t*v*(x - t*mu)))

def deriv(f1, f2, dx):
     return (f1 - f2) / dx

def dmfp_dbeta(sedov_uncol, sedov_class, g_interp, v_interp, beta, mu, x0, t):
     dx = 1e-3
     xs = np.array([x0])
     sedov_uncol.lambda1 = beta + dx
     f1 = sedov_uncol.get_mfp(xs, t, mu, sedov_class, g_interp, v_interp)[0]

     sedov_uncol.lambda1 = beta 
     f2 = sedov_uncol.get_mfp(xs, t, mu, sedov_class, g_interp, v_interp)[0]
     return deriv(f1, f2, dx)
    #  return np.gradient(np.array([f2, f1]), np.array([dx,dx]))[0]



def analytic_paper_plots():
    plot_list = np.array([0.005, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 5.5])
    xt = 1/29.98/0.001
    # low energy 
    tlist1 = np.round(np.array([0.01, 0.1, 0.3, 0.5]) * xt, 1)
    tlist2 = np.round(np.array([0.5, 0.7,0.9, 1.5]) * xt, 1)
    TS_blast_absorbing_profiles(beta = 3.2, eblast = 1.8e14, tstar = 150.0, tf = 2.0)
    plt.figure(1)
    plt.text(9.6, .542,f't = {tlist1[-1]} [ns]' )
    plt.text(22.3, .223, f'{tlist1[-2]}' )
    plt.text(-65, .26 , f'{tlist1[-3]}')
    plt.text(-136, .198, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/low_energy_scalar_flux_before.pdf')
    plt.figure(2)
    plt.text(0.0, .559,f't = {tlist1[-1]} [ns]' )
    plt.text(-66.2, .280, f'{tlist1[-2]}' )
    plt.text(0.0, .110 , f'{tlist1[-3]}')
    plt.text(82.8,  0.032, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/low_energy_scalar_flux_after.pdf')
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    TS_blast_absorbing_profiles(beta = 3.2, eblast = 1.8e14, tstar = 150.0, tf = 2.0, plotcurrent= True)
    plt.figure(1)
    plt.text(9.6, .342,f't = {tlist1[-1]} [ns]' )
    plt.text(22.3, .183, f'{tlist1[-2]}' )
    plt.text(-75, .26 , f'{tlist1[-3]}')
    plt.text(-136, .198, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/low_energy_current_before.pdf')
    plt.figure(2)
    plt.text(0.0, .359,f't = {tlist1[-1]} [ns]' )
    plt.text(-66.2, .110, f'{tlist1[-2]}' )
    plt.text(0.0, .06 , f'{tlist1[-3]}')
    plt.text(82.8,  0.0052, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/low_energy_current_after.pdf')
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    # # mid energy
    TS_blast_absorbing_profiles(beta = 3.2, eblast = 1.8e18, tstar =0.0 , tf = 2.0, plotcurrent=False)
    tlist1 = np.round(np.array([0.01, 0.1, 0.3, 0.5]) * xt, 1)
    tlist2 = np.round(np.array([0.5, 0.7,0.9, 1.5]) * xt, 1)
    plt.figure(1)
    plt.text(6.2, .465,f't = {tlist1[-1]} [ns]' )
    plt.text(-42, .534, f'{tlist1[-2]}' )
    plt.text(-76, .254 , f'{tlist1[-3]}')
    plt.text(-141, .256, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/mid_energy_scalar_flux_before.pdf')
    
    plt.figure(2)
    plt.text(-42.8, .68,f't = {tlist2[0]} [ns]' )
    plt.text(-61, .3,f'{tlist2[1]}' )
    plt.text(-16.3, .11,f'{tlist2[2]}' )
    plt.text(-58.5, .012,f'{tlist2[3]}' )
    plt.savefig('blast_plots/mid_energy_scalar_flux_after.pdf')
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()

    TS_blast_absorbing_profiles(beta = 3.2, eblast = 1.8e18, tstar = 0, plotcurrent = True, tf = 2.0)
    plt.figure(1)
    plt.text(6.2, .353,f't = {tlist1[-1]} [ns]' )
    plt.text(90.5, .150, f'{tlist1[-2]}' )
    plt.text(-76, .254 , f'{tlist1[-3]}')
    plt.text(-141, .256, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/mid_energy_current_before.pdf')
    plt.figure(2)
    plt.text(-42.8, .411,f't = {tlist2[0]} [ns]' )
    plt.text(-61, .13,f'{tlist2[1]}' )
    plt.text(-16.3, .037,f'{tlist2[2]}' )
    plt.text(113.8, .007,f'{tlist2[3]}' )
    plt.savefig('blast_plots/mid_energy_current_after.pdf')

    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()


    # high energy
    TS_blast_absorbing_profiles(beta = 2.4, eblast = 1.75e22, tstar = 0,tf = 3.5,  x0 = 0.5, plotcurrent= False)
    tlist1 = np.round(np.array([0.01, 0.1, 0.3, 0.5]) * xt, 1)
    tlist2 = np.round(np.array([0.5, 0.7,0.9, 1.5]) * xt, 1)
    plt.figure(1)
    plt.text(-201, .329,f't = {tlist1[-1]} [ns]' )
    plt.text(-250, .186, f'{tlist1[-2]}' )
    plt.text(-414, .17 , f'{tlist1[-3]}')
    plt.text(-480, .184, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/high_energy_scalar_flux_before.pdf')
    plt.figure(2)
    plt.text(-412, .772,f't = {tlist2[0]} [ns]' )
    plt.text(-214, .419,f'{tlist2[1]}' )
    plt.text(-312, .21,f'{tlist2[2]}' )
    plt.text(-274, .026,f'{tlist2[3]}' )
    plt.savefig('blast_plots/high_energy_scalar_flux_after.pdf')
    # plt.text(-312, .177,f'{tlist2[2]}' )
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close()
    plt.close() 
    TS_blast_absorbing_profiles(beta = 2.4, eblast = 1.75e22, tstar = 0,tf = 3.5,  x0 = 0.5, plotcurrent= True)
    plt.figure(1)
    plt.text(-182, .236,f't = {tlist1[-1]} [ns]' )
    plt.text(-263, .143, f'{tlist1[-2]}' )
    plt.text(-415, .125 , f'{tlist1[-3]}')
    plt.text(-480, .084, f'{tlist1[-4]}' )
    plt.savefig('blast_plots/high_energy_current_before.pdf')
    plt.figure(2)
    plt.text(-442, .450,f't = {tlist2[0]} [ns]' )
    plt.text(-218, .290,f'{tlist2[1]}' )
    plt.text(-322, .091,f'{tlist2[2]}' )
    plt.text(-253, .01,f'{tlist2[3]}' )
    plt.savefig('blast_plots/high_energy_current_after.pdf')


def sensitivity_to_e0(x0 = 0.5, tf = 3.5):
     epts = 3000
     tpts = 200
     e1 = 1.8e19
     e2 = 6e22
     eolist = np.linspace(e1, e2, epts)
     max_list = eolist * 0
     max_list_nonrel = eolist * 0
     max_list_middle = eolist * 0
     max_list_middle_nonrel = eolist * 0
    #  tmax = 5.5
    #  x0 = 0.5
     t0 = 1000
     beta = 2.4
     tlist = np.linspace(0, tf, 100 )
     for ie, ee in tqdm(enumerate(eolist)):
      
        g_interp, v_interp, sedov = TS_bench_prime(0.001, ee, 1e-12)
        t_bench_right = tlist * 0
        t_bench_right_nonrel = tlist * 0
        t_bench_middle_nonrel = tlist * 0
        t_bench_middle = tlist * 0
        for it, tt in enumerate(tlist):
            sedov.physical(tt)
            if sedov.r2 * 1e-3 <= x0:
                t_bench_right[it] = TS_current(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic=True, x0 = x0, t0 = t0)
                t_bench_middle[it] = TS_current(tt, np.array([0.0]), g_interp, v_interp, sedov, beta = beta, relativistic=True, x0 = x0, t0 = t0)
                t_bench_middle_nonrel[it] = TS_current(tt, np.array([0.0]), g_interp, v_interp, sedov, beta = beta, relativistic=False, x0 = x0, t0 = t0)
                t_bench_right_nonrel[it] = TS_current(tt, np.array([x0]), g_interp, v_interp, sedov, beta = beta, relativistic=False, x0 = x0, t0 = t0)
        max_list[ie] = np.max(t_bench_right)
        max_list_nonrel[ie] = np.max(t_bench_right_nonrel)
        max_list_middle_nonrel[ie] = np.max(t_bench_middle_nonrel)
        max_list_middle[ie] = np.max(t_bench_middle)
        plt.ion()
        plt.figure(2)
        plt.plot(tlist, t_bench_right, 'k-')

     plt.figure(1)
     plt.plot(eolist, max_list, 'k-')
     plt.plot(eolist, max_list_nonrel, 'k--')

     plt.ylabel(r'$J_\mathrm{max}$', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
     plt.savefig('blast_plots/e0_vs_max.pdf')
     plt.show()

     plt.figure(11)
     plt.semilogx(eolist, max_list, 'k-')
     plt.semilogx(eolist, max_list_nonrel, 'k--')

     plt.ylabel(r'$J_\mathrm{max}$', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
     plt.savefig('blast_plots/e0_vs_max_log.pdf')
     plt.show()


     plt.figure(7)
     plt.plot(eolist, max_list_middle, 'k-')
     plt.plot(eolist, max_list_middle_nonrel, 'k--')

     plt.ylabel(r'$J_\mathrm{max}$', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
     plt.ticklabel_format(axis = 'x',style =  'sci')
     plt.savefig('blast_plots/e0_vs_max_middle.pdf')
     plt.show()

     plt.figure(77)
     plt.semilogx(eolist, max_list_middle, 'k-')
     plt.semilogx(eolist, max_list_middle_nonrel, 'k--')

     plt.ylabel(r'$J_\mathrm{max}$', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
    #  plt.ticklabel_format(axis = 'x',style =  'sci')
     plt.savefig('blast_plots/e0_vs_max_middle_log.pdf')
     plt.show()


     plt.figure(4)
     plt.plot(eolist, max_list-max_list_nonrel, 'k-')

     plt.ylabel(r'$J_\mathrm{max} - J^\mathrm{nonrelativistic}_\mathrm{max} $', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
     plt.ticklabel_format(axis = 'x', style = 'sci')
     plt.savefig('blast_plots/e0_vs_max_difference.pdf')
     plt.show()
     
     sensitivity_right = np.gradient(max_list, eolist[1]-eolist[0] )
     sensitivity_middle = np.gradient(max_list_middle, eolist[1]-eolist[0] )
     sensitivity_right_nonrel = np.gradient(max_list_nonrel, eolist[1]-eolist[0] )
     sensitivity_middle_nonrel = np.gradient(max_list_middle_nonrel, eolist[1]-eolist[0] )
     plt.figure(3)
     plt.plot(eolist, sensitivity_right, 'k-')
     plt.plot(eolist, sensitivity_right_nonrel, 'k--')
     
     plt.ylabel(r'$\frac{\partial J_\mathrm{max}}{\partial \mathcal{E}_0} $', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
     plt.ticklabel_format(axis = 'x', style = 'sci')
    #  plt.ylim(-1.5e-22, 0)
     plt.savefig('blast_plots/e0_vs_sensitivity_right.pdf')
     plt.show()


     plt.figure(33)
     plt.semilogx(eolist, sensitivity_right, 'k-')
     plt.semilogx(eolist, sensitivity_right_nonrel, 'k--')
     
     plt.ylabel(r'$\frac{\partial J_\mathrm{max}}{\partial \mathcal{E}_0} $', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
    #  plt.ticklabel_format(axis = 'x', style = 'sci')
    #  plt.ylim(-1.5e-22, 0)
     plt.savefig('blast_plots/e0_vs_sensitivity_right_log.pdf')
     plt.show()

     plt.figure(8)
     plt.plot(eolist, sensitivity_middle, 'k-')
     plt.plot(eolist, sensitivity_middle_nonrel, 'k--')
     
     plt.ylabel(r'$\frac{\partial J_\mathrm{max}}{\partial \mathcal{E}_0} $', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
     plt.ticklabel_format(axis = 'x', style = 'sci')
    #  plt.ylim(-1.5e-22, 0)
     plt.savefig('blast_plots/e0_vs_sensitivity_middle.pdf')
     
     plt.show()


     plt.figure(88)
     plt.semilogx(eolist, sensitivity_middle, 'k-')
     plt.semilogx(eolist, sensitivity_middle_nonrel, 'k--')
    
     plt.ylabel(r'$\frac{\partial J_\mathrm{max}}{\partial \mathcal{E}_0} $', fontsize = 16)
     plt.xlabel(r'$\mathcal{E}_0$ [erg]', fontsize = 16)
    #  plt.ticklabel_format(axis = 'x', style = 'sci')
#  p lt.ylim(-1.5e-22, 0)
     plt.savefig('blast_plots/e0_vs_sensitivity_middle_log.pdf')
    
     plt.show()

          