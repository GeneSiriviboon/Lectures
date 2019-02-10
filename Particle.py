#!/usr/bin/python
from scipy.integrate import odeint
import matplotlib.pyplot as plt # for plotting
import numpy as np
import copy

class Particle (object):

    """Class that describes particle"""
    m = 1.0

    def __init__(self, xv0=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  tf = 10.0, dt = 0.001):
        self.xv = xv0
        self.t = 0.0
        self.tf = tf
        self.dt = dt
        npoints = int(tf/dt) # always starting at t = 0.0
        self.tarray = np.linspace(0.0, tf,npoints, endpoint = True) # include final timepoint
        self.xv0 = xv0 # NumPy array with initial position and velocity

        print("A new particle has been init'd")

    def F(self, xv, t):
        # The force on a free particle is 0
        return array([0.0, 0.0, 0.0])

    def Euler_step(self):
        """
        Take a single time step using Euler method
        """
        a = self.F(self.xv, self.t) / self.m
        self.xv[0:3] += self.xv[3:6] * self.dt
        self.xv[3:6] += a * self.dt
        #print(self.xv)
        self.t += self.dt

    # def RK4_step(self):
    #     """
    #     Take a single time step using RK4 midpoint methon
    #     """
    #     a1 = self.F(self.xv[0:3], self.v[3:6], self.t) / self.m
    #     k1 = np.array([self.v, a1])*self.dt
    #
    #     a2 = self.F(self.x+k1[0]/2, self.v+k1[1]/2, self.t+self.dt/2) / self.m
    #     k2 = np.array([self.v+k1[1]/2 ,a2])*self.dt
    #
    #     a3 = self.F(self.x+k2[0]/2, self.v+k2[1]/2, self.t+self.dt/2) / self.m
    #     k3 = np.array([self.v+k2[1]/2, a3])*self.dt
    #
    #     a4 = self.F(self.x+k3[0], self.v+k3[1], self.t+self.dt) / self.m
    #     k4 = np.array([self.v+k3[1], a4])*self.dt
    #
    #     self.xv[0:3] += (k1[0]+ k4[0])/6 + (k2[0] + k3[0])/3
    #     self.xv[3:6] += (k1[1]+ k4[1])/6 + (k2[1] + k3[1])/3
    #
    #     self.t += self.dt

    def Euler_trajectory(self):
        """
        Loop over all time steps to construct a trajectory with Euler method
        Will reinitialize euler trajectory everytime this method is called
        """

        x_euler = []
        v_euler = []

        while(self.t < self.tf-self.dt/2):
            x = copy.deepcopy(self.xv[0:3])
            v = copy.deepcopy(self.xv[3:6])
            v_euler.append(v)
            x_euler.append(x)
            self.Euler_step()


        self.x_euler = np.array(x_euler)
        self.v_euler = np.array(v_euler)

     # def RK4_trajectory(self):
     #    """
     #    Loop over all time steps to construct a trajectory with RK4 method
     #    Will reinitialize euler trajectory everytime this method is called
     #    """
     #
     #    x_RK4 = []
     #    v_RK4 = []
     #
     #    while(self.t < self.tf - self.dt/2):
     #        x_RK4.append(self.x)
     #        v_RK4.append(self.v)
     #        self.RK4_step()
     #
     #    self.x_RK4 = np.array(x_RK4)
     #    self.v_RK4 = np.array(v_RK4)

    # def scipy_trajectory(self):
    #     """calculate trajectory using SciPy ode integrator"""
    #
    #     self.xv = odeint(self.derivative, self.xv0, self.tarray)
    #
    # def derivative(self, xv, t):
    #     """right hand side of the differential equation
    #         Required for odeint """
    #
    #     x =xv[0]
    #     v =xv[1]
    #     a = self.F(x, v, t) / self.m
    #     return np.ravel(np.array([v, a]))

    def results(self):
        """
        Print out results in a nice format
        """

        print('\n\t Position and Velocity at Final Time:')
        print('Euler:')
        print('t = {} x = {} v = {}'.format(self.t, self.x , self.v))
        #
        # if hasattr(self, 'xv'):
        #     print('SciPy ODE Integrator:')
        #     print('t = {} x = {} v = {}'.format(self.tarray[-1], self.xv[-1, 0], self.xv[-1,1]))

    def plot(self):
        """
        Make nice plots of our results
        """

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        # if hasattr(self,'xv'):
        #     ax1.plot(self.tarray, self.xv[:, 0], "k", label = 'odeint')
        #     ax2.plot(self.xv[:, 0], self.xv[:, 1], "k", label = 'odeint')
        if hasattr(self,'x_euler'):
            ax1.plot(self.tarray, self.x_euler[:,0], "b", label = 'x')
            ax2.plot(self.x_euler[:,0], self.v_euler[:,0], "b", label = 'x')

        if hasattr(self,'x_euler'):
            ax1.plot(self.tarray, self.x_euler[:,1], "r", label = 'y')
            ax2.plot(self.x_euler[:,1], self.v_euler[:,1], "r", label = 'y')

        if hasattr(self,'x_euler'):
            ax1.plot(self.tarray, self.x_euler[:,2], "black", label = 'z')
            ax2.plot(self.x_euler[:,2], self.v_euler[:,2], "black", label = 'z')

        # if hasattr(self,'x_RK4'):
        #     ax1.plot(self.tarray, self.x_RK4, "r", label = 'RK4')
        #     ax2.plot(self.x_RK4, self.v_RK4, "r", label = 'RK4')

        ax1.set_title('Trajectory')
        ax1.set_xlabel("t")
        ax1.set_ylabel("x")

        ax2.set_title('Phase space')
        ax2.set_xlabel("v")
        ax2.set_ylabel("x")

        ax1.legend()
        ax2.legend()

class FallingParticle (Particle):

    """Subclass of Particle Class that describes a falling particle"""
    g = np.array([0.0, 0.0, 10.0])

    def __init__(self,m = 1.0, xv0 = np.array([0.0,0,0,0,0,0]), tf = 10.0,  dt = 0.1):
        self.m = m
        super().__init__(xv0,tf,dt)   # call initialization method of the super (parent) class

    def F(self, xv, t):
            return  -self.g * self.m

class ElectricCharge (Particle):

    def __init__(self,m = 1.0, xv0 = np.array([0.0,0,0,0,0,0]), tf = 10.0,  dt = 0.1, E = np.array([0,0,0]), B= np.array([0,0,0]), e = 0.0):
        self.m = m
        super().__init__(xv0,tf,dt)
        self.E = E
        self.B = B
        self.e = e

    def F(self, xv, t):
        Ex,Ey,Ez =self.E[0],self.E[1],self.E[2]
        Bx,By,Bz =self.B[0],self.B[1],self.B[2]
        vx, vy, vz = self.xv[3],self.xv[4],self.xv[5]
        return self.e*(np.array([Ex+vy*Bz-vz*By, Ey+vz*Bx-vx*Bz, Ez+vx*By-vy*Bx]))
