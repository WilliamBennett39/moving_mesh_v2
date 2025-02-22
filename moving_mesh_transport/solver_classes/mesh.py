import numpy as np
from numba import int64, float64
import numba
from numba.experimental import jitclass
import math
from .functions import problem_identifier 
from .mesh_functions import set_func, _interp1d
# import quadpy
import numpy.polynomial as nply
from scipy.special import roots_legendre
from .mesh_functions import boundary_source_init_func_outside
import numba as nb
params_default = nb.typed.Dict.empty(key_type=nb.typeof('par_1'),value_type=nb.typeof(1))


#################################################################################################
data = [('N_ang', int64), 
        ('N_space', int64),
        ('M', int64),
        ('tfinal', float64),
        ('mus', float64[:]),
        ('ws', float64[:]),
        ('x0', float64),
        ("moving", int64),
        ("move_type", int64[:]),
        ("edges", float64[:]),
        ("edges0", float64[:]),
        ("Dedges", float64[:]),
        ("N_space", int64),
        ('middlebin', int64),
        ('sidebin', int64),
        ('speed', float64),
        ('geometry', nb.typeof(params_default)),
        ('opacity', nb.typeof(params_default)),
        ('Dedges_const', float64[:]),
        ('source_type', int64[:]),
        ('thick', int64), 
        ('move_func', int64),
        ('debugging', int64),
        ('wave_loc_array', float64[:,:,:]),
        ('delta_t', float64),
        ('tactual', float64),
        ('told', float64),
        ('index_old', int64),
        ('right_speed', float64),
        ('left_speed', float64),
        ('test_dimensional_rhs', int64),
        ('move_factor', float64),
        ('T_wave_speed', float64),
        ('pad', float64),
        ('follower_speed', float64),
        ('leader_speed', float64),
        ('span_speed', float64),
        ('thick_quad', float64[:]),
        ('middlebin', int64),
        ('sidebin', int64),
        ('leader_pad', float64),
        ('packet_leader_speed', float64),
        ('thick_quad_edge', float64[:]),
        ('t0', float64),
        ('edges0_2', float64[:]),
        ('c1s', float64[:]),
        ('finite_domain', int64),
        ('domain_width', float64),
        ('mesh_stopped', int64),
        ('vnaught', float64),
        ('boundary_on', int64[:]),
        ('vv0', float64),
        ('t0', float64),
        ('eval_array', float64[:]),
        ('saved_edges', float64[:,:]),
        ('itt', int64),
        ('location_of_shock', float64),
        ('shock_vel', float64),
        ('has_shock_hit', int64),
        ('t_hit', float64[:]),
        ('re_meshed', int64),
        ('r2', float64),
        ('r2v', float64),
        ('thits', float64[:]),
        ('sedovfinalr', float64),
        ('a', float64),
        ('r2_old', float64),
        ('r2v', float64),
        ('r2va', float64),
        ('r2_naught', float64),
        ('blast_edge1', int64),
        ('blast_edge2', int64)

        # ('problem_type', int64)
        ]
#################################################################################################


# Really need a better mesh function 
@jitclass(data)
class mesh_class(object):
    def __init__(self, N_space, x0, tfinal, moving, move_type, source_type, edge_v,
     thick, move_factor, wave_loc_array, pad, leader_pad, thick_quad, thick_quad_edge,
     finite_domain, domain_width, fake_sedov_v, boundary_on, t0, eval_array, geometry, opacity, r2 = 0.0, sedovfinalr = 0.0):

        
        self.debugging = True
        self.test_dimensional_rhs = False

        self.pad = pad
        self.tfinal = tfinal
        self.N_space = N_space
        self.x0 = x0
        self.moving = moving
        self.move_type = np.array(list(move_type), dtype = np.int64)
        self.edges = np.zeros(N_space+1)
        self.edges0 = np.zeros(N_space+1)
        self.Dedges = np.zeros(N_space+1)
        self.N_space = N_space
        self.speed = edge_v
        self.geometry = geometry
        self.opacity = opacity
        self.t_hit = np.zeros(2)
        self.r2 = r2
        self.r2_old = r2
        self.sedovfinalr = sedovfinalr
        self.a = 0.0
        self.move_factor = move_factor
        if self.test_dimensional_rhs == True:
            self.speed = 299.98

        print('mesh edge velocity: ', edge_v)
        self.source_type = np.array(list(source_type), dtype = np.int64)

        if self.move_type[0] == True:
            self.move_func = 0 # simple linear
        elif self.move_type[1] == True:
            self.move_func = 1 # thick square source move
            # print('thick square source edge estimation mesh')
        elif self.move_type[2] == True:
            self.move_func = 2 # sqrt t static
        elif self.move_type[3] == True:
            self.move_func = 3
        
        # self.problem_type = problem_identifier(self.source_typem, self.x0)
        self.thick = thick
        # self.wave_loc_array = wave_loc_array
        # self.smooth_wave_loc_array
        # for count, element in enumerate(self.wave_loc_array[0, 3, :]):
        #     if element < self.x0:
        #         self.wave_loc_array[0, 3, count] = self.x0 + 1e-8
        self.thick_quad = thick_quad
        self.thick_quad_edge = thick_quad_edge

        # print(self.wave_loc_array[0,2,-1], 'wave final location')

        self.sidebin = int(self.N_space/4)
        self.middlebin = int(self.N_space/2)
        self.re_meshed = False
        self.tactual = -1.
        self.told = 0.0
        self.index_old = 0
        self.T_wave_speed = 0.0
        self.follower_speed = 0.0
        self.leader_speed = 0.0
        self.span_speed = 0.0
        # self.drdt = 0.0
        self.leader_pad = leader_pad
        self.t0 = t0
        print(self.t0, 't0')
        self.finite_domain = finite_domain
        if self.finite_domain == True:
            print('finite domain')
        self.domain_width = domain_width
        
        self.boundary_on = np.array(list(boundary_on), dtype = np.int64)

        self.mesh_stopped = False
        self.vnaught = fake_sedov_v
        print(fake_sedov_v, 'v0')
        if fake_sedov_v != 0 and np.all(self.source_type == 0):
            self.speed = fake_sedov_v
            print('speed is ', self.speed)
        self.eval_array = eval_array
        self.saved_edges = np.zeros((self.eval_array.size, self.N_space+1))

        self.itt = 0
        self.initialize_mesh()
    
    # def check_tangle(self):
        # if any ((self.edges[1:] - self.edges[:-1]) <=0):
        #     assert 0
    def check_crossed(self, stop = False):
        res = 0
        for it in range(self.N_space):
            if self.edges[it+1] <= self.edges[it]:
                print('crossed')
                print(self.told, 'time')
                print(self.edges)
                print(self.r2, 'r2')
                if stop == True:
                    raise ValueError('The mesh is tangled in at least one place')
                else:
                    res = 1
                    return res
        return res
          

    # def correct_edge_questionmark(self, blast_edge):
    #     if blast_edge == self.N_space:
    #         blast_edge -=1

    #     if abs(self.edges[blast_edge+1]-self.r2) <  abs(self.edges[blast_edge]-self.r2):
    #         blast_edge += 1
    #     elif abs(self.edges[blast_edge+1]-self.r2) <  abs(self.edges[blast_edge]-self.r2)
    

    def get_TS_acceleration(self,t, tstar, oneorboth, edgesold, Dedgesold):
        
        # blast_edge1 = np.argmin(np.abs(self.edges[0:self.N_space/2]+ self.r2))
            if oneorboth == 0:
                # blast_edge1= int(self.N_space/3) - 1
                self.edges[self.blast_edge1] = -self.r2_naught - self.r2v * (t-tstar) - 0.5 * self.r2va * (t-tstar)**2
                self.Dedges[self.blast_edge1] = -self.r2v - self.r2va * (t-tstar)
            # print(self.edges[blast_edge1]+self.r2v)
            # if oneorboth == 0:
            #     print(self.edges[blast_edge1]+self.r2)
            

            # print(self.edges[blast_edge1])
            # print(self.edges)
            
                
            if oneorboth == 1:
                # blast_edge2 = np.argmin(np.abs(self.edges[self.N_space/2:] - self.r2))
                # blast_edge2= int(2*self.N_space/3) +1
                # blast_edge1 = int(self.N_space/3) 
                self.edges[self.blast_edge1] = -self.r2_naught - self.r2v * (t-tstar) - 0.5 * self.r2va * (t-tstar)**2
                self.edges[self.blast_edge2] = self.r2_naught + self.r2v * (t-tstar) + 0.5 * self.r2va * (t-tstar)**2
                self.Dedges[self.blast_edge1] = -self.r2v - self.r2va * (t-tstar)
                self.Dedges[self.blast_edge2] = self.r2v + self.r2va * (t-tstar)
                # print(self.edges[blast_edge1]+self.r2)
                # print(self.edges[blast_edge2]-self.r2)

        
            # print(blast_edge1)
            dt = abs(t-self.told)
            # tnew = t+dt-tstar
            # edges_old = self.edges
            # Dedges_old = self.Dedges

            # x01 = self.edges0[blast_edge1]
            # bump1 = 2 * (-self.r2 - x01 - self.Dedges_const[blast_edge1]*tnew) / tnew ** 2
            # bump1 = 0.0
            # if oneorboth == 1:
            #     x02 = self.edges0[blast_edge2]
            #     bump2 = 2 * (self.r2 - x02 - self.Dedges_const[blast_edge2]*tnew) / tnew ** 2
            #     bump2 = 0.0
            # self.edges[blast_edge1] = self.edges0[blast_edge1] + self.Dedges_const[blast_edge1] * (t-tstar) + 0.5 * bump1 * (t-tstar)**2
            # self.Dedges[blast_edge1] = self.Dedges_const[blast_edge1] + bump1 * (t-tstar) 
            # if oneorboth == 1:
            #     self.edges[blast_edge2] = self.edges0[blast_edge2] + self.Dedges_const[blast_edge2] * (t-tstar) + 0.5 * bump2 * (t-tstar)**2
            #     self.Dedges[blast_edge2] = self.Dedges_const[blast_edge2] + bump2 * (t-tstar) 
            # res = self.check_crossed()
            # if res == 1:
            #     # self.edges[blast_edge1] = self.edges0[blast_edge1] + self.Dedges_const[blast_edge1] * (t-tstar)
            #     # self.Dedges[blast_edge1] = self.Dedges_const[blast_edge1]
            #     self.edges = edges_old
            #     self.Dedges = Dedges_old
            #     # if oneorboth == 1:
            #     #     self.edges[blast_edge2] = self.edges0[blast_edge2] + self.Dedges_const[blast_edge2] * (t-tstar) 
            #     #     self.Dedges[blast_edge2] = self.Dedges_const[blast_edge2] 






            # # second attempt 
            # x0old = edgesold[blast_edge1]
            # # Dedgesold = Dedgesold[blast_edge1]
            # dx = x0old - self.r2
            # if abs(t - self.told) <= 1e-14:
            #     self.Dedges[blast_edge1] = self.Dedges_const[blast_edge1]
            #     self.edges[blast_edge1] = self.edges0[blast_edge1] + self.Dedges[blast_edge1] * (t-tstar)
            # else:
            #     self.Dedges[blast_edge1] = dx/ (t-self.told)
            #     self.edges[blast_edge1] =  x0old + self.Dedges[blast_edge1] * (t-self.told)
            # if oneorboth == 1:
            #     x0old = edgesold[blast_edge2]
            #     # Dedgeold = Dedgesold[blast_edge2]
            #     dx = x0old - self.r2
            #     if abs(t - self.told) <= 1e-14:
            #         self.Dedges[blast_edge2] = self.Dedges_const[blast_edge2]
            #         self.edges[blast_edge2] = self.edges0[blast_edge2] + self.Dedges[blast_edge2] * (t-tstar)
            #     else:
            #         self.Dedges[blast_edge2] = dx/ (t-self.told)
            #         self.edges[blast_edge2] = x0old + self.Dedges[blast_edge2] * (t-self.told)

            # third attempt
           
            # dx = -self.r2 - x0old
            # self.edges[blast_edge1] = -self.r2
            # if dt > 0.0:
            #     self.Dedges[blast_edge1] = dx / dt
            #     print(self.Dedges[blast_edge1])
            # if t-tstar > 1e-6:
            #     if oneorboth == 0:
            #         x0old = edgesold[blast_edge1]
            #         Dedgeold = Dedgesold[blast_edge1]
            #         a = 2 * (-x0old - self.r2 - Dedgeold * (t-tstar)) / (t-tstar)**2
            #         aprime = (-2*Dedgeold)/(t - tstar)**2 - (4*(-self.r2 - (t - tstar)*Dedgeold + x0old))/(t - tstar)**3
            #         self.edges[blast_edge1] = x0old + Dedgeold * (t-tstar) + 0.5 * a * (t-tstar)**2
            #         self.Dedges[blast_edge1] = Dedgeold  + a * (t-tstar) + 0*0.5 * (t-tstar)**2 * aprime 
            #     # print(a)
            #     # print(t, 't')
            #     # print(tstar, 'tstar')
            #     # print(t-tstar)
            #     # print((x0old+ self.r2), self.edges[blast_edge1]+self.r2)
            #     elif oneorboth == 1:
            #         x0old = edgesold[blast_edge2]
            #         Dedgeold = Dedgesold[blast_edge2]
            #         a = -2 * (-x0old + self.r2 - Dedgeold * (t-tstar)) / (t-tstar)**2
            #         aprime = (-2*Dedgeold)/(t - tstar)**2 - (4*(-self.r2 - (t - tstar)*Dedgeold + x0old))/(t - tstar)**3
            #         self.edges[blast_edge2] = x0old + Dedgeold * (t-tstar) + 0.5 * a * (t-tstar)**2
            #         self.Dedges[blast_edge2] = Dedgeold  + a * (t-tstar) + 0*0.5 * (t-tstar)**2 * aprime 

            #         x0old = edgesold[blast_edge1]
            #         Dedgeold = Dedgesold[blast_edge1]
            #         a = -2 * (-x0old - self.r2 - Dedgeold * (t-tstar)) / (t-tstar)**2
            #         aprime = (-2*Dedgeold)/(t - tstar)**2 - (4*(-self.r2 - (t - tstar)*Dedgeold + x0old))/(t - tstar)**3
            #         self.edges[blast_edge1] = x0old + Dedgeold * (t-tstar) + 0.5 * a * (t-tstar)**2
            #         self.Dedges[blast_edge1] = Dedgeold  + a * (t-tstar) + 0*0.5 * (t-tstar)**2 * aprime 
            # if t != self.told:    
            #     r2_velocity = (self.r2 - self.r2_old)/ (t-self.told)
            #     # print(r2_velocity)
            #     self.edges[blast_edge1] = -self.r2
            #     self.Dedges[blast_edge1] = -r2_velocity
            #     if oneorboth == 1:
            #         self.edges[blast_edge2] = self.r2
            #         self.Dedges[blast_edge2] = r2_velocity

            # self.r2_old = self.r2
            # c1 = self.r2 / t**(2/3)
            # drdt = 2 * c1 * t **(1/3) /3
            # print(c1)
            # self.told = t
            # self.edges[blast_edge1] = -self.r2
            # self.Dedges[blast_edge1] = - self.drdt
            # if oneorboth == 1:
            #     self.edges[blast_edge2] = self.r2
            #     self.Dedges[blast_edge2] = self.drdt
            # if t > self.told:
            #     dx =  edgesold[blast_edge1] + self.r2 
            #     dt = t-self.told
            #     v0_old = Dedgesold[blast_edge1]
            #     x_new = edgesold[blast_edge1] + v0_old * (t - self.told) 
            #     x_new2 = self.edges0[blast_edge1] + self.Dedges[blast_edge1]*(t-tstar)
            #     if abs(x_new - x_new2) >= 1e-5:
            #         print(abs(x_new - x_new2))
            #         print(t, self.told, 't, told')
            #         assert 0

            # if t > self.told:
            #     self.told = t
        
        

        # self.edges = edges_old
        # self.Dedges = 0 * self.edges

            self.check_crossed(True)




    def move(self, t):
        # print(self.edges)pr
        """
        Called each time the rhs moves the mesh. Changes edges and Dedges
        """
        # print(self.edges)
        # if self.moving == True:
        """
        This mode moves all of the edges at a constant speed,
        linearly increasing from 0 to the wavespeed
        """
        # self.check_tangle()
        if self.moving == True:
            # if self.source_type[1] == 1 or self.source_type[2] == 1:
                # if t > 10.0:
                #     self.Dedges = self.edges/self.edges[-1] * self.speed

            if self.source_type[0] == 1 or self.source_type[0]==2:

                self.edges = self.edges0 + self.Dedges_const*t



            elif self.source_type[1] == 1:
                if self.finite_domain == True and t == self.tfinal:

                # print(self.edges0[-1] + t * self.speed, "###### final edge ######")
                
                # if self.edges0[-1] + t * self.speed > 5 and self.finite_domain == True:
                    self.edges = np.linspace(-self.domain_width/2, self.domain_width/2, self.N_space+1)
                    

                elif (self.finite_domain == True) and (self.edges[-1] >= self.domain_width/2):
                    self.edges = self.edges
                    self.Dedges = self.Dedges_const*0
                else:
                    # print(self.edges0, 'edges0')
                    self.edges = self.edges0 + self.Dedges_const*t
                    self.Dedges = self.Dedges_const


            elif self.source_type[3] == 1 or self.source_type[5] == 1:
                if self.move_func == 2:
                    final_edge = math.sqrt(self.tfinal)*self.move_factor + self.x0
                    velocity = (final_edge - self.x0) / self.tfinal

                    self.edges = self.edges0 + self.Dedges_const * t * velocity
                    self.Dedges = self.Dedges_const *velocity

            elif self.source_type[2] == 1 or self.source_type[1] == 1 or self.source_type[0]!=0:

                # self.finite_domain = True # what is the deal with this?
                if (self.finite_domain == True) and (self.edges[-1] >= self.domain_width/2):
                        self.edges = self.edges
                        self.Dedges = self.Dedges_const*0
                    # if t == self.tfinal:
                    #     self.edges = np.linspace(-5, 5, self.N_space+1)
                    #     self.Dedges = self.Dedges_const*0
                    # else:
                else:
                    if self.move_func == 0:
                            if t >= self.t0:
                                self.move_middle_edges(t)
                                tnew1 = t - self.t0 

                            ### uncomment this to go back to the old mesh

                            # self.edges = self.edges0_2 + self.Dedges_const * (t-self.t0)
                            # self.Dedges = self.Dedges_const


                            ### uncomment this for constant vel. 

                            # self.edges = self.edges0_2 + self.Dedges * (t-self.t0)

                            ### uncomment this for acceleration case

                                self.edges = 0.5 * self.c1s * (tnew1) ** 2 + self.Dedges_const * tnew1 + self.edges0_2
                                self.Dedges = self.c1s * tnew1 + self.Dedges_const


                            elif (t < self.t0):
                            
                                self.edges = self.edges0 + self.Dedges*t


                            # self.Dedges = self.Dedges_const
                            

                    # else:

                    #         self.edges = self.edges0 + self.Dedges*t


                    elif self.move_func == 1: 
                            """
                            This mode has the wavefront tracking edges moving at a constant speed
                            and interior edges tracking the diffusive wave
                            """
                            # self.thick_square_moving_func(t)
                            self.thick_square_moving_func_2(t)
                    

                    elif self.move_func == 2:
                        self.square_source_static_func_sqrt_t(t)

                    
                    else:
                            print("no move function selected")
                            assert(0)
            
            elif np.all(self.source_type == 0):
                # print(self.Dedges)
                # self.edges = self.edges0 + self.Dedges*t
                if self.opacity['fake_sedov'] == 1:
                    self.move_boundary_source(t)
                elif self.opacity['TaylorSedov'] == 1:
                    self.move_boundary_source_TS(t)
            

            # if self.debugging == True:
            #     for itest in range(self.edges.size()):
            #         if self.edges[itest] != np.sort(self.edges)[itest]:
            #             print("crossed edges")
            #             assert(0)
        # print(abs(t-self.eval_array[self.itt]))
        # if abs(t-self.eval_array[self.itt]) <= 1e-4:
        #     self.saved_edges[self.itt] = self.edges
        #     self.itt += 1
                    
    def move_boundary_source_TS(self, t):


        # self.edges = np.linspace(-0.15, 0.15, self.N_space+1)
        # self.Dedges = self.edges * 0
        oldedges = self.edges
        oldDedges = self.Dedges

        if self.move_func == 0:

            self.edges = self.edges0 + self.Dedges_const * t
            self.Dedges = self.Dedges_const

            self.edges0[0] = -self.x0
            self.Dedges[0] = 0.0

        elif self.move_func == 1:
            tstar = self.thits[0]
            if math.isnan(tstar):
                assert(0)
            # if(abs(self.Dedges_const[-1]-1.0) <= 1e-8):
            #     print(self.Dedges_const[-1])
            #     assert(0)
            # self.edges = self.edges0
            # self.Dedges = self.edges * 0
            
            self.edges[int(self.N_space/3):] = self.edges0[int(self.N_space/3):] + self.Dedges_const[int(self.N_space/3):] * (t-tstar) 
            self.Dedges[int(self.N_space/3):] = self.Dedges_const[int(self.N_space/3):]

            # self.r2v = 0.0

            self.edges[int(self.N_space/3)-1] = self.edges0[int(self.N_space/3)-1] + self.r2v * (t-tstar) 
            # self.edges[0] = -self.x0
      
            # self.edges[int(self.N_space/3):] = oldr2edge + 0.5 * a  * (t-tstar) ** 2
            

            # self.edges[int(self.N_space/3)-1] = -self.r2
            self.Dedges[int(self.N_space/3)-1] = self.r2v
            self.Dedges_const[int(self.N_space/3)-1] = self.r2v
            # print(self.Dedges[int(self.N_space/3)-1])

            # 
            self.edges[0] = -self.x0
            self.Dedges[0] = 0.0
            self.get_TS_acceleration(t, tstar, 0, oldedges, oldDedges)
            # print(min(np.abs(np.abs(self.edges) - self.r2)))



        elif self.move_func == 2:
            tstar = self.thits[1]
            if math.isnan(tstar):
                assert(0)
            # self.edges = self.edges0
            # self.Dedges = self.edges * 0

            # self.edges[1:int(self.N_space/3)] = self.edges0[1:int(self.N_space/3)] - self.r2v * (t-tstar) 
            # self.Dedges[1:int(self.N_space/3)] = -self.r2v
            # self.edges[int(2*self.N_space/3)-1:-1] = self.edges0[int(2*self.N_space/3)-1:-1] + self.r2v * (t-tstar) 
            # self.Dedges[int(2*self.N_space/3)-1:-1] = self.r2v
            # self.edges[1:int(self.N_space/2)] = self.edges0[1:int(self.N_space/2)] - self.r2v * (t-tstar) 
            # self.edges[int(self.N_space/2)+1:-1] = self.edges0[int(self.N_space/2)+1:-1] + self.r2v * (t-tstar) 
            # self.Dedges[int(self.N_space/2)+1:-1] = self.r2v 
            # self.Dedges[1:int(self.N_space/2)] = - self.r2v
            # print(self.edges)
            # self.edges[0] = -self.x0
            for ix in range(self.N_space):
                if self.edges[ix+1] <= self.edges[ix]:
                    print('crossed')
                    print(self.r2v)
                    assert(0)
            self.edges = self.edges0 + self.Dedges_const * (t-tstar) #* 0 
            # self.Dedges = self.Dedges_const * 0
            self.get_TS_acceleration(t, tstar, 1, oldedges, oldDedges)
            self.edges[0] = -self.x0
            self.Dedges[0] = 0.0

            blast_edge2= int(2*self.N_space/3) +1
            # print(self.edges[blast_edge2]-self.r2)

            # print(np.abs(self.edges) - self.r2)
            # print(min(np.abs(np.abs(self.edges) - self.r2)))
            


    def smooth_wave_loc_array(self):
        for ix in range(0,self.wave_loc_array[0,3,:].size-1):
            if self.wave_loc_array[0,3,ix] < self.wave_loc_array[0,3,ix +1]:
                self.wave_loc_array[0,3,ix] = self.wave_loc_array[0,3,ix +1]

    def move_middle_edges(self,t):
        middlebin = int(self.N_space/2)
        sidebin = int(middlebin/2)
        if self.Dedges[sidebin] == 0:
            self.edges0_2 = self.edges0 + self.Dedges_const * self.t0
            # final_pos = self.edges0[-1] + self.Dedges[-1] * self.tfinal
            final_pos = self.pad
            # final_pos = self.x0 + self.pad
            final_array = np.linspace(-final_pos, final_pos, self.N_space + 1)
            # print(final_array, 'final array')

            # constant velocity
            # new_Dedges = (final_array - self.edges0_2) / (self.tfinal-self.t0)
            # self.Dedges = new_Dedges
            # self.Dedges[sidebin:sidebin+middlebin+1] = 0    




            #### constant acceleration ###

            # print(self.Dedges_const, 'const edges')
            # print(self.tfinal, 'tfinal')
            # print(self.edges0_2, 'second edges0')
            tnew = self.tfinal - self.t0
            # print(self.t0, 't0 in move middle' )
            self.c1s = 2 * (self.Dedges_const * (self.t0) - self.tfinal * self.Dedges_const - self.edges0_2 + final_array) / ((self.t0-self.tfinal)**2)       

       
    def move_boundary_source(self, t):
      
        self.get_shock_location(t, self.vnaught)
        # if self.has_shock_hit == False:
        if self.move_func == 0:
            self.edges = self.edges0 + self.Dedges_const * t
            self.Dedges = self.Dedges_const
            # print(-self.edges[1:]+self.edges[:-1])
            # print(t)
        elif self.move_func == 1:
            self.edges = self.edges0 + self.Dedges_const * (t-self.t_hit)
            self.Dedges = self.Dedges_const

        # print('#', self.Dedges, 'Dedges #')
        # print('#', self.edges, 'edges #')
        # else:
        #     if self.re_meshed == False:
        #         self.re_mesh_boundary_source(t)

            # tnew1 = t - self.t_hit
            # self.edges = 0.5 * self.c1s * (tnew1) ** 2 + self.Dedges_const * tnew1 + self.edges0_2
            # self.Dedges = self.c1s * tnew1 + self.Dedges_const
            # self.edges = self.edges0_2 + self.c1s * (t-self.t_hit)
            # self.Dedges = self.c1s
            # print(self.Dedges)
            # print(self.edges, 'edges')

    def re_mesh_boundary_source(self, t):
        self.get_shock_location(self.tfinal, self.vnaught)
        N_left_of_shock = int(self.N_space /2)
        self.edges0_2 = self.edges0 + self.Dedges_const * self.t_hit

        N_right_of_shock = int(self.N_space  - N_left_of_shock)
        left_edges = np.linspace(-self.x0, self.location_of_shock, N_left_of_shock+1)
        right_edges = np.linspace(self.location_of_shock, self.x0, N_right_of_shock+1)
        final_array = np.concatenate((left_edges[:-1], right_edges))
        assert(final_array.size == self.N_space + 1)
        print(final_array, 'final_edges')
        print(self.edges, 'current edges')
        print('-----------------------')
        # tnew = self.tfinal - t
            # print(self.t0, 't0 in move middle' )
        self.c1s = 2 * (self.Dedges_const * (self.t_hit) - self.tfinal * self.Dedges_const - self.edges0_2 + final_array) / ((self.tfinal-self.t_hit)**2)   
        self.re_meshed = True
        # print(self.c1s, 'cs')
        # self.c1s = (self.edges0_2 - final_array) / (self.tfinal-self.t_hit)
        # self.get_shock_location(t, self.vnaught)





    def thick_wave_loc_and_deriv_finder(self, t):
        
        interpolated_wave_locations = _interp1d(np.ones(self.wave_loc_array[0,3,:].size)*t, self.wave_loc_array[0,0,:], self.wave_loc_array[0,3,:], np.zeros(self.wave_loc_array[0,3,:].size))
        # interpolated_wave_locations = np.interp(t, self.wave_loc_array[0,0,:], self.wave_loc_array[0,3,:] )

        # derivative = (interpolated_wave_locations[1] - interpolated_wave_locations[0])/delta_t_2

        edges = np.copy(self.edges)
        if t == 0 or  interpolated_wave_locations[0] < self.x0:
            edges = self.edges0
        else:
            edges[-1] = interpolated_wave_locations[0] + self.leader_pad
            if edges[-1] < self.edges0[-1]:
                edges[-1] = self.edges0[-1]
            edges[-2] = interpolated_wave_locations[0] + self.pad
            if edges[-2] < self.edges0[-2]:
                edges[-2] = self.edges0[-2]
            edges[-3] = interpolated_wave_locations[0] 
            if edges[-3] < self.edges0[-3]:
                edges[-3] = self.edges0[-3]
            edges[-4] = interpolated_wave_locations[0]  - self.pad
            if edges[-4] < self.edges0[-4]:
                edges[-4] = self.edges0[-4]

        
            edges[self.sidebin + self.middlebin: self.N_space-3] = np.linspace(self.x0, edges[-4], self.sidebin - 2)[:-1] 
            edges[0:self.sidebin] =  - np.flip(np.copy(edges[self.sidebin+self.middlebin+1:]))
        return edges

    def thick_square_moving_func_2(self, t):
        delta_t = 1e-7
        self.edges = self.thick_wave_loc_and_deriv_finder(t)
        edges_new = self.thick_wave_loc_and_deriv_finder(t + delta_t)

        self.Dedges = (edges_new - self.edges) / delta_t

        if self.edges[-3] < self.edges[-4]:
            print('crossed')

    def recalculate_wavespeed(self, t):
        sidebin = int(self.N_space/4)
        T_index = -2
        index = np.searchsorted(self.wave_loc_array[0,0,:], t)
        # pad = self.edges[int(self.N_space/2)+1] - self.edges[int(self.N_space/2)
        # print(abs(self.edges[-2]-self.wave_loc_array[0,3,index+1]), 'T wave from mesh edge')
        if self.debugging == True:
            if index >0:
                if not (self.wave_loc_array[0,0,index-1] <= t <= self.wave_loc_array[0,0,index+1]):
                    print('indexing error')
        if index != self.index_old:
            # print('calculating wavespeeds') s
            T_wave_location = self.wave_loc_array[0,3,index+1]
            # print(self.pad, 'pad')
            # print(index, 'index')
            # print(T_wave_location, 'T wave loc')
            self.delta_t = self.wave_loc_array[0,0,index+1] - t
            self.right_speed = (self.wave_loc_array[0,2,index+1]  - self.edges[-1])/self.delta_t
            self.T_wave_speed = (T_wave_location - self.edges[-2])/self.delta_t
            # print(T_wave_location, 't edge is moving to')
            self.leader_speed = (T_wave_location + self.leader_pad - self.edges[-1])/self.delta_t
            self.packet_leader_speed = (T_wave_location + self.pad - self.edges[-2])/self.delta_t
            # print(T_wave_location + self.pad, 'leader edge is moving to')
            self.follower_speed = (T_wave_location - self.pad - self.edges[-3])/self.delta_t

            last_follower_edge_loc = self.edges[-3] + self.Dedges_const[-3] * self.follower_speed * self.delta_t
            dx_span = (last_follower_edge_loc - self.x0) / (sidebin/2)  
            self.span_speed = (last_follower_edge_loc - dx_span - self.edges[-int(sidebin-2)])/self.delta_t
        
        self.index_old = index
        # print(self.edges)

        # print(self.delta_t, 'delta t')
        # print(self.leader_speed, 'leader')
        # print(self.T_wave_speed, "T speed")
        # print(self.follower_speed, 'follower')
        # # print(self.span_speed, 'span')
        # print(self.T_wave_speed, 't wave s')
        # print(self.leader_speed, 'leader s')
        # print(index, 'index')
        if self.T_wave_speed > self.leader_speed:
            print("speed problem")
            print(self.pad, 'pad')

      
    
        # if abs(self.edges[T_index] + self.Dedges_const[T_index] * self.T_wave_speed * self.delta_t - (self.edges[T_index -1] + self.Dedges_const[T_index-1] * self.T_wave_speed * self.delta_t)) <= 1e-12:
        #     #catching up
        #     print('catching up')
        #     self.T_wave_speed = (self.edges[(T_index)-1] - 0.0005 - self.edges[(T_index)])/self.delta_t

        
        if self.debugging == True:
            if abs(t - self.wave_loc_array[0,0,index+1]) < 1e-5:
                print('checking location')
                print(self.wave_loc_array[0,3,index+1] - self.edges[-2], 'T wave difference')
                print(self.wave_loc_array[0,2,index+1]  - self.edges[-1], 'right edge difference')
        

        if self.right_speed < 0.0:
            self.right_speed = 0.0
        if self.T_wave_speed < 0.0:
            # print('negative t speed')
            self.T_wave_speed = 0.0
        if self.follower_speed < 0.0:
            self.follower_speed = 0.0
        if self.leader_speed < 0.0:
            self.leader_speed = 0.0
        if self.span_speed < 0.0:
            self.span_speed = 0.0
        
        
        # print(self.edges[-2], "|", self.wave_loc_array[0,3,index+1])


    

            

    def thick_gaussian_static_init_func(self):
        # if abs(self.wave_loc_array[0, 2, -1]) > 5:
        if self.move_func == 1:
            right_edge = self.wave_loc_array[0,3,-1] + self.pad
        elif self.move_func == 0:
            right_edge = self.x0
        print(self.move_func, 'move_func')
        print(right_edge, 'right edge')
        # else:
            # right_edge = self.x0 + self.tfinal
        
        # if right_edge < self.x0:
            # right_edge = self.x0 + self.tfinal

        self.edges = np.linspace(-right_edge, right_edge, self.N_space + 1)
        self.Dedges = self.edges * 0


    def simple_thick_square_init_func(self):
        # does not accomodate moving mesh edges
        
        # wave_edge = self.wave_loc_array[0,2,index+1]
        wave_edge = self.wave_loc_array[0,2,-1] + self.pad

        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-8
        left = np.linspace(-wave_edge, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, wave_edge, sidebin + 1)
        middle = np.linspace(-self.x0, self.x0, middlebin + 1)
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed * 0 
 

    def square_source_static_func_sqrt_t(self, t):
        # only to be used to estimate the wavespeed
        move_factor = self.move_factor

        if t > 1e-10:
            sqrt_t = math.sqrt(t)
        else:
            sqrt_t = math.sqrt(1e-10)

        # move the interior edges
        self.Dedges = self.Dedges_const * move_factor * 0.5 / sqrt_t
        self.edges = self.edges0 + self.Dedges_const * move_factor * sqrt_t

    

        # move the wavefront edges
        # Commented code below moves the exterior edges at constant speed. Problematic because other edges pass them
        # self.Dedges[0] = self.Dedges_const[0]
        # self.Dedges[-1] = self.Dedges_const[-1]
        # self.edges[0] = self.edges0[0] + self.Dedges[0]*t
        # self.edges[-1] = self.edges0[-1] + self.Dedges[-1]*t
        # self.Dedges[0] = self.Dedges_const[0] * move_factor * 0.5 / sqrt_t
        # self.edges[0] = self.edges0[0] + self.Dedges_const[0] * move_factor * sqrt_t
        # print(self.edges0[0], 'x0')
        # print(self.Dedges_const[0]*move_factor*sqrt_t, 'f')
        # self.Dedges[-1] = self.Dedges_const[-1] * move_factor * 0.5 / sqrt_t
        # self.edges[-1] = self.edges0[-1] + self.Dedges_const[-1] * move_factor * sqrt_t

    ####### Initialization functions ########


    def simple_moving_init_func(self):
            if self.geometry['slab'] == True:
                self.edges = np.linspace(-self.x0, self.x0, self.N_space+1)
            elif self.geometry['sphere'] == True:
                self.edges = np.linspace(0, self.x0, self.N_space+1)
                self.edges0 = self.edges
            self.Dedges = self.edges/self.edges[-1] * self.speed
            self.Dedges_const = self.Dedges
            if self.source_type[0] == 2:
                self.edges += 0.0001

    def shell_source(self):
        dx = 1e-5
        N_inside = int(self.N_space/2 + 1)
        edges_inside = np.linspace(0, self.x0, N_inside+1)
        N_outside = int(self.N_space + 1 - N_inside )
        edges_outside = np.linspace(self.x0, self.x0 + dx, N_outside)
        self.edges = np.concatenate((edges_inside, edges_outside[1:]))
        self.edges0 = self.edges
        assert(self.edges.size == self.N_space + 1)
        self.Dedges = np.zeros(self.N_space + 1)

        self.Dedges[N_inside + 1:] = (self.edges[N_inside + 1:] - self.x0)/(self.edges[-1] - self.x0) * self.speed
        self.Dedges_const = self.Dedges
        print(self.Dedges_const, 'dedges')


        

    # def thick_square_moving_func(self, t):
    #     middlebin = int(self.N_space/2)
    #     sidebin = int(self.N_space/4)
    #     self.recalculate_wavespeed(t)
    #     little_delta_t = t-self.told

    #     # self.Dedges[0:sidebin/2] = self.Dedges_const[0:sidebin/2] * self.right_speed
    #     # self.Dedges[0:int(sidebin/4 + 1)] = self.Dedges_const[0:int(sidebin/4 + 1)] * self.leader_speed
    #     # self.Dedges[int(sidebin/4 + 1)] = self.Dedges_const[int(sidebin/4 + 1)] * self.T_wave_speed
    #     # self.Dedges[int(sidebin/4 + 2):int(sidebin/2 + 1)] = self.Dedges_const[int(sidebin/4 + 2):int(sidebin/2 + 1)] * self.follower_speed
    #     # self.Dedges[int(sidebin/2 + 2):sidebin] = self.Dedges_const[int(sidebin/2 + 2):sidebin] * self.span_speed

    #     self.Dedges[0] = self.Dedges_const[0] * self.leader_speed
    #     self.Dedges[1] = self.Dedges_const[1] * self.T_wave_speed
    #     self.Dedges[2] = self.Dedges_const[2] * self.follower_speed
    #     self.Dedges[3:sidebin] = self.Dedges_const[3:sidebin] * self.span_speed

    #     self.Dedges[middlebin+sidebin + 1:] =  - np.flip(np.copy(self.Dedges[0:sidebin]))

    #     # self.Dedges[1:-1] =  self.Dedges_const[1:-1] * self.T_wave_speed 
    #     # self.Dedges[0] =  self.Dedges_const[0] * self.right_speed 
    #     # self.Dedges[-1] =  self.Dedges_const[-1] * self.right_speed 


    #     # self.Dedges = self.Dedges_const * self.right_speed
    #     # self.edges = self.edges + self.Dedges * delta_t
    #     # self.told = t
    #     # print(self.edges[-1]-self.edges[-2], 'thin zone')

    #     self.edges = self.edges + self.Dedges * little_delta_t
    #     self.told = t

    def thin_square_init_func(self):
        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-12
        left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)

        
        middle = np.linspace(-self.x0, self.x0, middlebin + 1)

        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed
        self.edges0 = self.edges


    def thin_square_init_func_legendre(self):
        print('calling mesh with legendre spacing')
        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-3
        dx2 = 0.0000
        # left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        # right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)
        left_old = self.thick_quad_edge
        right_old = self.thick_quad_edge
        right = (right_old*(self.x0-self.x0-dx)-self.x0-dx-self.x0)/-2 
        left = (left_old*(-self.x0-dx+self.x0)+self.x0+dx+self.x0)/-2 

        # if self.N_space == 32 and self.move_func == 2:
        #     middle = np.array([-0.99057548, -0.95067552, -0.88023915, -0.781514  , -0.65767116,
        #                 -0.51269054, -0.35123176, -0.17848418,  0.        ,  0.17848418,
        #                 0.35123176,  0.51269054,  0.65767116,  0.781514  ,  0.88023915,
        #                 0.95067552,  0.99057548]) 
        # else:
        # if self.move_func == 2:
        middle = self.x0 * self.thick_quad

            # left = roots_legendre(siebin+1)[0]
            # right = roots_legendre(siebin+1)[0]
            # right =(right*(self.x0-self.x0-dx)-self.x0-dx-self.x0)/-2
            # left =(left*(-self.x0-dx+self.x0)+self.x0+dx+self.x0)/-2
            # print(left, right)
        left += dx
        middle[:len(middle)/2] += dx
        middle[len(middle)/2+1:] -= dx
        right -= dx
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 

        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 - dx )/(self.edges[-1] - self.x0 + dx)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0 + dx)/(self.edges[-1] - self.x0 + dx)
        self.Dedges = self.Dedges * self.speed 
        self.Dedges_const = np.copy(self.Dedges)
        self.edges[0] -= dx2
        self.edges[-1] += dx2
        self.edges0 = self.edges
        print(self.edges, 'edges0')


    def simple_thick_square_init_func_2(self):
        if self.N_space == 2:
            print("don't run this problem with 2 spaces")
            assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        dx = 1e-14
        left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)
        middle = np.linspace(-self.x0, self.x0, middlebin + 1)
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)
        self.Dedges = self.Dedges * self.speed * 0
    
    def thick_square_moving_init_func(self):
        # if self.N_space ==2 or self.N_space == 4 or self.N_space == 8:
        #     print(f"don't run this problem with {self.N_space} spaces")
        #     assert(0)
        middlebin = int(self.N_space/2)   # edges inside the source - static
        sidebin = int(middlebin/2) # edges outside the source - moving
        # dx_min = 1e-4
        # wave_source_separation = self.wave_loc_array[0,3,:] - self.x0
        # wave_sep_div = wave_source_separation / (sidebin-1)
        # index = 0
        # if wave_sep_div[index] < dx_min:
        #     index +=1
        # else:
        #     first = index
        # dx = self.wave_loc_array[0,3,index]
        dx = 1e-3
        left = np.linspace(-self.x0-dx, -self.x0, sidebin + 1)
        right = np.linspace(self.x0, self.x0 + dx, sidebin + 1)
        # middle = np.linspace(-self.x0, self.x0, middlebin + 1)
        middle = 0.5 * self.thick_quad
        self.edges = np.concatenate((left[:-1], middle[:-1], right[:])) # put them all together 
        
        # initialize derivatives
        # self.Dedges[0:sidebin] = (self.edges[0:sidebin] + self.x0 )/(self.edges[-1] - self.x0)
        # self.Dedges[sidebin:sidebin+middlebin] = 0       
        # self.Dedges[middlebin+sidebin + 1:] = (self.edges[middlebin+sidebin + 1:] - self.x0)/(self.edges[-1] - self.x0)

        self.Dedges[0] = -1.0
        self.Dedges[1] = -1.0
        self.Dedges[2] = -1.0
        self.Dedges[3:sidebin] = -(self.edges[3:sidebin] + self.x0) / (self.edges[3] + self.x0)

        # self.Dedges[int(sidebin/2+2):sidebin] = (self.edges[int(sidebin/2+2):sidebin] + self.x0 )/(self.edges[-int(sidebin/2 + 2)] - self.x0)
        self.Dedges[sidebin:sidebin+middlebin] = 0       
        self.Dedges[middlebin+sidebin + 1:] =  - np.flip(np.copy(self.Dedges[0:sidebin]))
        self.Dedges = self.Dedges * self.speed



    def thick_square_init_func(self):
        print("initializing thick square source")

        dx = 1e-5

        half = int(self.N_space/2)
        self.edges = np.zeros(self.N_space+1)
        self.Dedges = np.zeros(self.N_space+1) 
        
        self.edges[half] = 0 # place center edge
        self.Dedges[half]= 0

        # self.edges[0] = -self.x0 - 2*dx# place wavefront tracking edges
        # self.edges[-1] = self.x0 + 2*dx

        # self.Dedges[0] = -1 * self.speed
        # self.Dedges[-1] = 1 * self.speed

        number_of_interior_edges = int(self.N_space/2 - 1)
        # print(number_of_interior_edges, "interior")

        # don't use N=4 
        if number_of_interior_edges == 1: # deal with N=4 case 
            self.edges[number_of_interior_edges] = -self.x0
            self.edges[number_of_interior_edges + half] = self.x0
            self.Dedges[number_of_interior_edges] = -1.0 * self.speed
            self.Dedges[number_of_interior_edges + half] = 1.0 * self.speed 
        
        else:                               # set interior edges to track the wavefront

            # set one edge to travel back towards zero and one to be fixed at the source width
            # self.set_func([half-2, half+2], [-self.x0, self.x0], [0,0])
            # self.set_func([half-1, half+1], [-self.x0+dx, self.x0-dx], [self.speed, -self.speed])
            # # set edges to track the wave
            # left_xs = np.linspace(-self.x0-2*dx, -self.x0-dx, half-2)
            # right_xs = np.linspace(self.x0+dx, self.x0+2*dx, half-2)
            # speeds = np.linspace(half-2, 1, half-2)
            # speeds = speeds/(half-2) * self.speed
            # indices_left = np.linspace(0, half-2-1, half-2)
            # indices_right = np.linspace(half+3, self.N_space, half-2)

            # self.set_func(indices_left, left_xs, -speeds)
            # self.set_func(indices_right, right_xs, np.flip(speeds))


            indices_left = np.linspace(0, half-1, half)
            indices_right = np.linspace(half+1, self.N_space, half)
            xs_left = np.zeros(half)
            xs_right = np.zeros(half)
            speeds = np.zeros(half)
            xs_left[int(half/2)] = -self.x0
            # xs_right[int(half/2)] = self.x0
            speeds[int(half/2)] = 0.0 
            xs_left[0:int(half/2)] = np.linspace(-self.x0-2*dx, -self.x0-dx,int(half/2))
            xs_left[int(half/2)+1:] = np.linspace(-self.x0+dx, -self.x0+2*dx, int(half/2)-1)
            # xs_right[0:int(half/2)] = np.linspace(self.x0-2*dx, self.x0-dx, int(half/2))
            # xs_right[int(half/2)+1:] = np.linspace(self.x0+dx, self.x0+2*dx, int(half/2)-1)
            xs_right = -np.flip(xs_left)
            speeds[0:int(half/2)] = np.linspace(int(half/2), 1, int(half/2))/int(half/2)
            speeds[int(half/2)+1:] = -np.linspace(1,int(half/2), int(half/2) -1)/ int(half/2)
            speeds = speeds * self.speed
            # print("#   #   #   #   #   #   #   #   #   #   #   ")
            # print(speeds, "speeds")
            # print("#   #   #   #   #   #   #   #   #   #   #   ")
            self.set_func(indices_left, xs_left, -speeds)
            self.set_func(indices_right, xs_right, np.flip(speeds))

            # self.edges[0:half-1] = np.linspace(-self.x0-dx, -self.x0 + dx, number_of_interior_edges + 1)
            # self.edges[half+2:] = np.linspace(self.x0 - dx, self.x0 + dx, number_of_interior_edges + 1)
            # self.edges[half-1] = -self.x0

            # # self.Dedges[1:half-1] = - self.edges[1:half-1]/self.edges[1] * self.speed
            # # self.Dedges[half+2:-1] = self.edges[half+2:-1]/self.edges[-2] * self.speed 
            # self.Dedges[0:half] = -np.linspace(1,-1, number_of_interior_edges + 1)* self.speed   
            # self.Dedges[half+1:] = np.linspace(-1,1, number_of_interior_edges + 1)* self.speed   


            self.delta_t = self.wave_loc_array[0,0,1] - self.wave_loc_array[0,0,0]

            # print(self.delta_t, 'delta_t')



    # def boundary_source_init_func(self, v0):
    #     mid = int(self.N_space/2)
    #     self.edges = np.linspace(-self.x0, self.x0, self.N_space+1)
    #     self.Dedges = np.copy(self.edges)*0
    #     if self.moving == False:
    #         v0 = 0
    #     # self.Dedges[mid] = - self.fake_sedov_v0
    #     ### First attempt -- Even spacing
    #     final_shock_point = - self.tfinal * v0
    #     final_edges_left_of_shock = np.linspace(-self.x0, final_shock_point, int(self.N_space/2+1))
    #     final_edges_right_of_shock = np.linspace(final_shock_point, self.x0, int(self.N_space/2+1))
    #     final_edges = np.concatenate((final_edges_left_of_shock[:-1], final_edges_right_of_shock))
    #     self.Dedges = (final_edges - self.edges) / self.tfinal
    #     self.edges0 = self.edges

        ### Second -- squared  spacing

        # xsi = ((np.linspace(0,1, mid + 1))**2)
        # final_shock_point = - self.tfinal * v0
        # initial_edges_right = ((xsi*2-1)*(-self.x0)-self.x0)/(-2)
        # initial_edges_left = ((xsi*2-1)*(self.x0)+self.x0)/(-2)
        # self.edges = np.concatenate((np.flip(initial_edges_left), initial_edges_right[1:]))
        # print(self.edges)
        # final_shock_point = - self.tfinal * v0
        # # final_edges_left_of_shock = np.linspace(-self.x0, final_shock_point, int(self.N_space/2+1))
        # # final_edges_right_of_shock = np.linspace(final_shock_point, self.x0, int(self.N_space/2+1))
        # final_edges_right_of_shock = ((xsi*2-1)*(final_shock_point-self.x0)-final_shock_point-self.x0)/(-2)
        # final_edges_left_of_shock = np.flip(((xsi*2-1)*(final_shock_point+self.x0)+ self.x0-final_shock_point)/(-2))
        # print(final_edges_left_of_shock)
        # final_edges = np.concatenate((final_edges_left_of_shock[:-1], final_edges_right_of_shock))
        # print(final_edges, 'final edges')
        # self.Dedges = (final_edges - self.edges) / self.tfinal
        # self.edges0 = self.edges
            
    def boundary_source_init_func(self, v0):
        self.get_shock_location(0.0, v0)
        dx = 1e-4
        N_left_of_shock = int(self.N_space /2)
        N_right_of_shock = int(self.N_space  - N_left_of_shock) 
        left = np.linspace(-self.x0, -self.x0 + dx, N_left_of_shock)
        # print(self.location_of_shock, "shock initial")
        right = np.linspace(self.location_of_shock, self.x0, N_right_of_shock+1)
        self.edges = np.concatenate((left, right))
        # print(self.edges, 'edges')
        assert(self.edges.size == self.N_space +1)
        self.Dedges[0: N_left_of_shock] = (self.edges[0: N_left_of_shock] + self.x0)/dx 
        # print(self.Dedges, 'dedges')
        self.Dedges[N_left_of_shock:] = v0 * (self.edges[N_left_of_shock:] - self.x0) /self.x0

        # assert(0)
        self.Dedges_const = self.Dedges
        self.edges0 = self.edges
        print(self.Dedges_const, 'dedges')
        print(self.edges0, 'edges0')

    def boundary_source_init_func_2(self, v0):
        self.get_shock_location(0.0, v0)
        self.get_shock_location(self.t_hit, v0)

        print("##", self.location_of_shock, 'shock location now ##')
        N_left_of_shock = int(self.N_space /2)
        N_right_of_shock = int(self.N_space  - N_left_of_shock) 
        left = np.linspace(-self.x0, self.location_of_shock, N_left_of_shock + 1)
        right = np.linspace(self.location_of_shock, self.x0, N_right_of_shock+1)
        self.edges = np.concatenate((left[:-1], right))
        assert(self.edges.size == self.N_space +1)
        self.get_shock_location(self.tfinal, v0)
        final_edges_left = np.linspace(-self.x0, self.location_of_shock, N_left_of_shock + 1)
        final_edges_right = np.linspace(self.location_of_shock, self.x0, N_right_of_shock+1)
        final_edges = np.concatenate((final_edges_left[:-1], final_edges_right))
        assert(final_edges.size == self.N_space + 1)
        self.Dedges = (final_edges - self.edges) / (self.tfinal - self.t_hit)
        # assert(0)
        self.Dedges_const = self.Dedges
        self.edges0 = self.edges

    def TS_init_func1(self):
        dx = 1e-5
        self.edges = np.linspace(-self.x0, -self.x0 + dx, self.N_space + 1)
        self.Dedges = (self.edges + self.x0) / dx
        self.edges0 = self.edges
        print(self.edges0, 'edges0')
        self.Dedges_const = self.Dedges
        print(self.Dedges_const, 'dedges ')


        # self.edges = np.linspace(-0.15, 0.15, self.N_space+1)
        # self.Dedges = self.edges * 0
    
    def TS_init_func2(self):
        dx = 1e-4
        print(-self.r2, 'shock location')
        self.edges[0:int(self.N_space/3)] = np.linspace(-self.x0, -self.r2, int(self.N_space/3))
        self.edges[int(self.N_space/3):] = np.linspace(-self.r2 + dx, -self.r2 + 2*dx, int(self.N_space + 1 - int(self.N_space/3)))
        self.edges0 = np.copy(self.edges)
        self.Dedges = np.zeros(self.N_space+1)
        self.Dedges[int(self.N_space/3):] = (self.edges[int(self.N_space/3):] + self.r2 - dx ) / dx
        self.Dedges[int(self.N_space/3) - 1] = -self.r2v
        self.Dedges_const = self.Dedges
        print(self.edges0, 'edges0')
        print(self.Dedges_const, 'dedges ')
        self.blast_edge1 = int(self.N_space/3) -1

        # self.edges = np.linspace(-0.15, 0.15, self.N_space+1)
        # self.Dedges = self.edges * 0

    def TS_init_func3(self):
        #  self.get_shock_location(0.0, v0)
        # self.get_shock_location(self.t_hit, v0)
        
        N_third = int(self.N_space/3) + 1
        self.edges[:N_third] = np.linspace(-self.x0, -self.r2, N_third)
        
        self.edges[N_third:2*N_third] = np.linspace(-self.r2, self.r2, N_third + 1)[1:]
        self.blast_edge1 = N_third -1
        N_left = len(self.edges[2*N_third:])
        # assert(0)
        self.edges[2*N_third:] = np.linspace(self.r2, self.x0, N_left+1)[1:]
        self.blast_edge2 =  2 * N_third - 1
        final_shock_loc = self.sedovfinalr
        edges_final = np.zeros(self.N_space+1)
        edges_final[:N_third] = np.linspace(-self.x0, -final_shock_loc, N_third)
        edges_final[N_third:2*N_third] = np.linspace(-final_shock_loc, final_shock_loc, N_third + 1)[1:]
        edges_final[2*N_third:] = np.linspace(final_shock_loc, self.x0,  N_left + 1)[1:]
        print(edges_final, 'final edges, third remesh #####')
        self.edges0 = self.edges
        self.Dedges = (edges_final - self.edges) / (self.tfinal - self.t_hit[1])
        self.Dedges_const = self.Dedges
        print(self.Dedges)

        


        # print(self.r2, 'r2 in mesh 3')
        # self.edges = np.zeros(self.N_space+1)
        # self.edges[0] = -self.x0
        # self.edges[-1] = self.x0
        # self.edges[1:int(self.N_space/2)] = np.linspace(-self.x0, -self.r2, int(self.N_space/2))
        # self.edges[int(self.N_space/2):] = np.linspace(self.r2, self.x0, int(self.N_space/2))

        # self.edges[int(self.N_space/3)-1] = -self.r2
        # self.edges[int(2*self.N_space/3)-1] = self.r2

        # self.edges[0:int(self.N_space/3)-1] = np.linspace(-self.x0, -self.r2, int(self.N_space/3))[:-1]

        # midd_index = int(2*self.N_space/3-1) -(int(self.N_space/3))

        # self.edges[int(self.N_space/3):int(2*self.N_space/3)-1] = np.linspace(-self.r2, self.r2, midd_index +2)[1:-1]

        # left_index = self.N_space  - int(2*self.N_space/3)

        # self.edges[int(2*self.N_space/3):] = np.linspace(self.r2, self.x0, left_index+2)[1:]


        # self.Dedges = np.zeros(self.N_space +1)
        # self.Dedges[int(self.N_space/3) - 1] = -self.r2v
        # self.Dedges[int(2*self.N_space/3) - 1] = self.r2v
        # self.Dedges_const = self.Dedges
        # self.edges0 = self.edges

        print(self.edges0, 'edges0')
        print(self.Dedges_const, 'dedges ')

        # self.edges = np.linspace(-0.15, 0.15, self.N_space+1)
        # self.Dedges = self.edges * 0
    
        
    def initialize_mesh(self):

        """
        Initializes initial mesh edges and initial edge derivatives. This function determines
        how the mesh will move
        """
        print('initializing mesh')
        # if self.problem_type in ['plane_IC']:
        if self.source_type[0] == 1 or self.source_type[0] == 2:
            self.simple_moving_init_func()

        if self.thick == False:     # thick and thin sources have different moving functions

            # if self.problem_type in ['gaussian_IC', 'gaussian_source']:
            if self.source_type[3] == 1 or self.source_type[5] == 1:
                self.simple_moving_init_func()
                self.edges0 = self.edges
                # self.square_source_static_func_sqrt_t()
            # elif self.problem_type in ['square_IC', 'square_source']:
            elif self.source_type[1] == 1 or self.source_type[2] == 1:
                if self.domain_width == 2*self.x0:
                    self.simple_moving_init_func()
                else:
                    print('calling thin square init')
                    self.thin_square_init_func_legendre()
                    # self.simple_moving_init_func()


            
            elif np.all(self.source_type == 0):
                if self.opacity['fake_sedov'] == 1:

                    if self.move_type[0] == 1:
                        print('function 1')
                        self.boundary_source_init_func(self.vnaught)
                    else:
                        print('function 2')
                        self.boundary_source_init_func_2(self.vnaught)
                elif self.opacity['TaylorSedov'] == 1:
                    print('TS mesh')
                    print(self.move_func)
                    if self.moving == True:
                        if self.move_func == 0:
                            print('function 1')
                            self.TS_init_func1()
                        elif self.move_func == 1:
                            print('function 2')
                            self.TS_init_func2()
                        elif self.move_func == 2:
                            print('function 3')
                            self.TS_init_func3()
                        else:
                            self.edges = np.linspace(-self.x0, self.x0, self.N_space+1)
                            self.Dedges = self.edges * 0
                            self.edges0 = self.edges
                            self.Dedges_const = self.Dedges
                            print(self.edges, 'edges0')
                    else:
                        self.edges = np.linspace(-self.x0, self.x0, self.N_space+1)
                        self.Dedges = self.edges * 0
                        self.edges0 = self.edges
                        self.Dedges_const = self.Dedges

                    

                # boundary_source_init_func_outside(self.vnaught, self.N_space, self.x0, self.tfinal) 
                print('calling boundary source func')
            # if self.


        elif self.thick == True:
            # if self.problem_type in ['gaussian_IC', 'gaussian_source']:
            if self.source_type[3] == 1 or self.source_type[5] == 1:
                if self.moving == True:
                    self.simple_moving_init_func()
                elif self.moving == False:
                    self.thick_gaussian_static_init_func()

            elif self.source_type[1] == 1 or self.source_type[2] == 1 or self.source_type[0]!= 0:
                if self.move_func == 0:
                    self.simple_moving_init_func()

                if self.moving == False:

                    if self.move_func == 1:

                        self.simple_thick_square_init_func()
                    elif self.move_func == 2:

                        if self.source_type[0] != 0:

                            self.simple_moving_init_func()
                        else:
                            self.thin_square_init_func()
                elif self.moving == True:
                    if self.move_func == 1:
                        self.thick_square_moving_init_func()
                    elif self.move_func == 2:
                            self.thin_square_init_func()

        # self.edges0 = self.edges
        self.Dedges_const = self.Dedges



        if self.moving == False:

            self.tactual = 0.0
            # static mesh -- puts the edges at the final positions that the moving mesh would occupy
            # sets derivatives to 0
            self.moving = True
            if self.thick == True:
                self.delta_t = self.tfinal 
            self.move(self.tfinal)
            self.Dedges = self.Dedges*0
            self.moving = False
            # self.edges[-1] = self.x0 + self.tfinal * self.speed
            # self.edges[0] = -self.x0 + -self.tfinal * self.speed

            for it in range(self.eval_array.size):
                self.saved_edges[it] = self.edges
            print(self.edges[-1], "final edges -- last edge")

    def get_shock_location(self, t, v0):
        # just toy problem for now
        self.location_of_shock = - v0 * t
        self.shock_vel = v0
        self.has_shock_hit = 0
        dx = 1e-4
        tol = 1e-3
        self.t_hit[0] = (self.x0 - dx - tol) / (1 + v0) 
        if t >= (self.x0-tol-dx) / (1 + v0):
            self.has_shock_hit = True
            # print(self.location_of_shock, 'shock location')
            # print('shock contact')
        else:
            self.has_shock_hit = False
            self.re_meshed = False
    
    def get_shock_location_TS(self, r2, vr2, tshift, c1, t, sigma_t = 1e-3):
        self.r2 = r2
        conversion = 1/29.98/sigma_t * 1e-9
        # self.drdt = (2/3) * sigma_t * c1  * (t * conversion + tshift)**(-1/3) * conversion
        # print(self.drdt)
        # self.r2 = 0.08
        # self.r2v = vr2 
        # self.vr2_guess = self.r2 - self.r2_old
        # self.r2_old = 
        
    def save_thits(self, thits):
        self.thits = thits

    def save_edges(self, t):
        if t > 0:
            timestep_approx = t - self.told 
        it = np.argmin(np.abs(t-self.eval_array))    
        # if abs(t - self.eval_array[self.itt]) <= timestep_approx:
        self.saved_edges[it] = self.edges

        
        self.told = t

        # if self.itt == self.eval_array.size:
        #     print("#### #### ### ### ###")
        #     print('edge_array_filled')
        # if t == self.tfinal and self.itt != self.eval_array.size-1:
            # print('not filled')
            # assert(0)
        
        
    

    def linear_interpolate_sedov_blast(self, r1, r2, t1, t2):
        # self.r2v = (r2-r1) / (t2-t1)
        # self.final_shock_loc = rf
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')
        print('#################')

        print(self.r2v)

    def quadratic_interpolate_sedov(self, r1, r2, r3, t2, t3, ts):
        tf = t3
        th = t2
        r2th = r2
        r2tf = r3
        x0 = r1
        r2f = r3
        self.r2va =  (2*(r2th*tf - r2f*th + r2f*ts - r2th*ts - tf*x0 + th*x0))/((tf - th)*(tf - ts)*(-th + ts))
        self.r2v =  -((r2th*tf**2 - r2f*th**2 - 2*r2th*tf*ts + 2*r2f*th*ts - r2f*ts**2 + r2th*ts**2 - tf**2*x0 + th**2*x0 + 2*tf*ts*x0 - 2*th*ts*x0)/((tf - th)*(tf - ts)*(-th + ts)))

    def initialize_r2(self, r2naught):
        self.r2_naught = r2naught   
