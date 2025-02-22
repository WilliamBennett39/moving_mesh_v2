# imports functions to run package from terminal 

import sys
import matplotlib.pyplot as plt
sys.path.append('/Users/bennett/Documents/Github/transport_benchmarks/')
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaPerformanceWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
                      
from benchmarks import integrate_greens as intg
from moving_mesh_transport.plots import plotting_script as plotter
from moving_mesh_transport import solver
import matplotlib.pyplot as plt

from moving_mesh_transport.solver_classes.functions import *

# from moving_mesh_transport.plots.plot_square_s_times import main as plot_square_s_times
# from moving_mesh_transport.solution_plotter import plot_thin_nonlinear_problems as plot_thin
# from moving_mesh_transport.solution_plotter import plot_thin_nonlinear_problems_s2 as plot_thin_s2
# from moving_mesh_transport.solution_plotter import plot_thick_nonlinear_problems as plot_thick
# from moving_mesh_transport.solution_plotter import plot_thick_nonlinear_problems_s2 as plot_thick_s2
# from moving_mesh_transport.solution_plotter import plot_thick_suolson_problems as plot_sut
# from moving_mesh_transport.solution_plotter import plot_su_olson as plot_su
# from moving_mesh_transport.solution_plotter import plot_su_olson_gaussian as plot_sug
# from moving_mesh_transport.solution_plotter import plot_coeffs_nov28_crc as pca_28
# from moving_mesh_transport.solution_plotter import plot_coeffs_nov23_crc as pca_23
# from moving_mesh_transport.solution_plotter import plot_coeffs_nov31_crc as pca_31
# from moving_mesh_transport.solution_plotter import plot_coeffs_all_local as pca_loc
# from moving_mesh_transport.table_script import make_all_tables as mat
from moving_mesh_transport.solver_classes.functions import test_square_sol
from moving_mesh_transport.solver_classes.functions import test_s2_sol
#from moving_mesh_transport.tests.test_functions import test_interpolate_point_source
# from moving_mesh_transport.mesh_tester import test_square_mesh as test_mesh
# from moving_mesh_transport.solution_plotter import make_tables_su_olson as tab_sus

# from moving_mesh_transport.solver_classes.functions import test_s2_sol
from moving_mesh_transport.loading_and_saving.load_solution import load_sol as load




from moving_mesh_transport.solver_functions.run_functions import run



run = run()
run.load()

loader = load()

sigma_t = 0.001
eblast = 1.8e18
tstar = 1e-12
beta = 3.2
M = 1
def RMSE(l1, l2):
    return math.sqrt(np.mean((l1-l2)**2))

from blast_wave_plots import TS_bench, TS_bench_prime
g_interp, v_interp, sedov = TS_bench_prime(sigma_t, eblast, tstar)

run.boundary_source()
plt.close()
plt.close()
plt.close()
plt.close()

def test_sedov(N_spaces):
    run.parameters['all']['Ms'] = [M]
    run.parameters['all']['N_spaces'] = [N_spaces]
    run.mesh_parameters['Msigma'] = M
    run.boundary_source()

    times = run.eval_array
    xs = run.xs
    phi = run.phi

    RMSE_list = times * 0
    for it in range(times.size):
        phi_bench = TS_bench(times[it], xs[it], g_interp, v_interp, sedov, beta = beta, transform=True, x0 = run.x0, t0 = run.t0, sigma_t = sigma_t, relativistic = True)
        RMSE_list[it] = RMSE(phi[it], phi_bench)
    # plt.close()
    # plt.close()
    # plt.close()
    # plt.close()
    plt.figure(101)
    plt.loglog(times, RMSE_list, '-o', label = f'{run.N_space} cells')
    plt.legend()
    plt.savefig(f'blast_plots/absorbing_ts_test_e0={eblast}_{run.N_space}_cells.pdf')
    plt.show()

    # plt.figure(2)
    # plt.plot(xs[-1], phi[-1], 'b-o', mfc = 'none')
    # plt.plot(xs[-1], phi_bench, 'k-', mfc = 'none')


N_spaces_list = [10, 20, 30, 40, 50]

for N in N_spaces_list:
    test_sedov(N)




