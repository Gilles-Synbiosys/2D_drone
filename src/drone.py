import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import RK45

# Define the drone class
class Drone:
    g = 9.81 # m/s^2

    # Define the constructor
    def __init__(self, mass, inertia, com):
        """
        mass: mass of the drone in kg
        inertia: moment of inertia of the drone in kg*m^2
        com: distance from the center of mass to the center of rotation in m
        
        """
        self.mass = mass
        self.inertia = inertia
        self.com = com
        self.stVec = np.zeros((1, 6))
        self.t = np.zeros(1)
        self.cmd = np.zeros((1,2)) # [F, theta]
        self.cmdTmp = np.zeros(2)
        self.eq = None # equations of motion
        self.wpt = np.zeros(2) # waypoint
        self.gainDict = {'Kp_f': 0.001,
                         'Kd_f': 0.001, 
                         'Ki_f': 1, 
                         'Kp_d': 20.0, 
                         'Kd_d': 40.0, 
                         'Ki_d': 1}

    def __str__(self):
        """
        Returns a string representation of the drone
        """
        return "Drone with mass {} kg, inertia {} kg*m^2, and center of mass at {}".format(self.mass, self.inertia, self.com)

    def setPhysics(self, g):
        """
        Sets the physics of the drone
        args:
            g: acceleration due to gravity in m/s^2
        """
        self.g = g

    def setConditions(self, x0, y0, theta0, xdot0, ydot0, thetadot0):
        """
        Sets the initial conditions of the drone
        
        args:
            x0: initial x position in m
            y0: initial y position in m
            theta0: initial angle in radians
            xdot0: initial x velocity in m/s
            ydot0: initial y velocity in m/s
            thetadot0: initial angular velocity in rad/s
        """
        self.stVec[-1, :] = [x0, y0, theta0, xdot0, ydot0, thetadot0]

    def setWaypoint(self, x, y):
        """
        Sets the current waypoint of the drone
        args:
            x: x position in m
            y: y position in m
        """
        self.wpt = np.array([x, y])


    def control(self, t, y):
        """
        Sets the control inputs for the drone
        args:
            t: time in s
            y: state vector
        """
        # -0.10*(y[0]-self.wpt[0])
        self.cmdTmp[1] =  - self.gainDict['Kp_d']*y[2] - self.gainDict['Kd_d']*y[5] - 1.0*(y[3]-20.0)# - np.arctan2(self.wpt[1]-y[1], self.wpt[0]-y[0])*self.gainDict['Kp_d'] #+ y[5]*self.gainDict['Kd_f'] + self.gainDict['Ki_f']*np.trapz(self.wpt[1]-y[1], self.wpt[0]-y[0])
        #- self.gainDict['Kp_d']*y[2] - self.gainDict['Kd_d']*y[5]
        # limit the angle
        self.cmdTmp[1] = np.max([self.cmdTmp[1], -np.pi*20/180])
        self.cmdTmp[1] = np.min([self.cmdTmp[1], np.pi*20/180])
        
        self.cmdTmp[0] = self.mass*self.g/np.cos(y[2]+self.cmdTmp[1]) - 1.0*(y[4] +2.0)# + dist*self.gainDict['Kp_f'] - vel*self.gainDict['Kd_f'] #+ self.gainDict['Ki_d']*np.trapz(self.wpt[0]-y[0])
        
        # Limit the control inputs
        self.cmdTmp[0] = np.max([self.cmdTmp[0], 0])
        self.cmdTmp[0] = np.min([self.cmdTmp[0], 1000.0])


    def eqGenerator(self):
        """
        Generates the equations of motion for the drone
        
        Args:
            None
        Returns:
            None
        """
        def eq(t, y):
            """
            Returns the equations of motion for the drone
            args:
                t: time in s
                y: state vector
            returns:
                ydot: derivative of the state vector
            """

            # y = [x, y, theta, xdot, ydot, thetadot]
            # ydot = [xdot, ydot, thetadot, xddot, yddot, thetaddot]
            #self.pid(t, y)
            self.control(t, y)
            ydot = np.zeros(6)
            ydot[0] = y[3]
            ydot[1] = y[4]
            ydot[2] = y[5]
            ydot[3] = self.cmdTmp[0]*np.sin(y[2]+self.cmdTmp[1])/self.mass
            ydot[4] = self.cmdTmp[0]*np.cos(y[2]+self.cmdTmp[1])/self.mass - self.g
            ydot[5] = self.cmdTmp[0]*np.sin(self.cmdTmp[1])*self.com/self.inertia
            return ydot
        self.eq = eq
    
    def updateState(self, t, y):
        """
        Updates the state vector and time vector
        args:
            t: time in s
            y: state vector
        """
        self.stVec = np.vstack((self.stVec, y))
        self.cmd = np.vstack((self.cmd, self.cmdTmp))
        self.t = np.append(self.t, t)
    
    def plot(self):
        """
        Plots the state vector
        args:
            None
        """
        fig, axs = plt.subplots(4, 2)
        # Make the figure large
        fig.set_size_inches(18.5, 10.5)

        # Separate the plots
        fig.tight_layout(pad=3.0)

        axs[0, 0].plot(self.t, self.stVec[:, 0])
        axs[0, 0].set_title('x')
        axs[0, 0].set_ylabel('Position (m)')
        axs[0, 0].set_xlabel('Time (s)')

        axs[0, 1].plot(self.t, self.stVec[:, 1])
        axs[0, 1].set_title('y (m)')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Position (m)')

        axs[1, 0].plot(self.t, self.stVec[:, 2]*180/np.pi)
        axs[1, 0].set_title('theta (deg)')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Angle (deg)')

        axs[1, 1].plot(self.t, self.stVec[:, 3])
        axs[1, 1].set_title('xdot (m/s)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity (m/s)')

        axs[2, 0].plot(self.t, self.stVec[:, 4])
        axs[2, 0].set_title('ydot (m/s)')
        axs[2, 0].set_xlabel('Time (s)')
        axs[2, 0].set_ylabel('Velocity (m/s)')

        axs[2, 1].plot(self.t, self.stVec[:, 5]*180/np.pi)
        axs[2, 1].set_title('thetadot (deg/s)')
        axs[2, 1].set_xlabel('Time (s)')
        axs[2, 1].set_ylabel('Angular Velocity (deg/s)')

        axs[3, 0].plot(self.t, self.cmd[:, 0])
        axs[3, 0].set_title('F (N)')
        axs[3, 0].set_xlabel('Time (s)')
        axs[3, 0].set_ylabel('Force (N)')

        axs[3, 1].plot(self.t, self.cmd[:, 1]*180/np.pi)
        axs[3, 1].set_title('delta (deg)')  
        axs[3, 1].set_xlabel('Time (s)')
        axs[3, 1].set_ylabel('Angle (deg)')

        plt.show()

    def plotControl(self):
        """
        Plots the control inputs
        args:
            None
        """
        fig, axs = plt.subplots(2, 1)
        # Make the figure large
        fig.set_size_inches(18.5, 10.5)

        # Separate the plots
        fig.tight_layout(pad=3.0)

        axs[0].plot(self.t, self.cmd[:, 0])
        axs[0].set_title('F (N)')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Force (N)')

        axs[1].plot(self.t, self.cmd[:, 1]*180/np.pi)
        axs[1].set_title('delta (deg)')  
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Angle (deg)')

        plt.show()


    def plot2D(self):
        """
        Plots the drone in 2D
        args:
            None
        """
        fig = plt.figure()
        ax = plt.axes(xlim=(-0.1, 5), ylim=(-0.5, 0.5))
        ax.set_aspect('equal')
        x_top = self.com*np.sin(self.stVec[:, 2])
        y_top = self.com*np.cos(self.stVec[:, 2])
        x_bot = -self.com*np.sin(self.stVec[:, 2])
        y_bot = -self.com*np.cos(self.stVec[:, 2])
        x_force = self.cmd[:,0]*np.sin(self.stVec[:, 2]+self.cmd[:,1])*0.05
        y_force = self.cmd[:,0]*np.cos(self.stVec[:, 2]+self.cmd[:,1])*0.05
        
        for i in range(len(self.t),10):
            plt.plot([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_bot[i]], 
                     [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_bot[i]], color='red')
            plt.plot([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_top[i]+x_force[i]], 
                     [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_top[i]+y_force[i]], color='blue')
        plt.show()

    def animate(self):
        """
        Animates the drone  in 2D
        args:
            None
        """

        fig = plt.figure()
        ax = plt.axes(xlim=(-5.0, 5.0), ylim=(-5.0, 5.0))
        ax.set_aspect('equal')
        x_top = self.com*np.sin(self.stVec[:, 2])
        y_top = self.com*np.cos(self.stVec[:, 2])
        x_bot = -self.com*np.sin(self.stVec[:, 2])
        y_bot = -self.com*np.cos(self.stVec[:, 2])
        x_force = self.cmd[:,0]*np.sin(self.stVec[:, 2]+self.cmd[:,1])*0.02
        y_force = self.cmd[:,0]*np.cos(self.stVec[:, 2]+self.cmd[:,1])*0.02

        line, = ax.plot([], [], lw=2, color='red')
        force, = ax.plot([], [], lw=2, color='blue')

        def init():
            line.set_data([], [])
            force.set_data([], [])
            return line, force
        def animate(i):
            line.set_data([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_bot[i]], 
                          [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_bot[i]])
            force.set_data([self.stVec[i, 0]+x_top[i], self.stVec[i, 0]+x_top[i]+x_force[i]], 
                           [self.stVec[i, 1]+y_top[i], self.stVec[i, 1]+y_top[i]+y_force[i]])
            return line, force
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(self.t), interval=20, blit=True)
        plt.show()


    def solve(self, t0, tf, dt):
        """
        Solves the equations of motion for the drone
        args:
            t0: initial time in s
            tf: final time in s
            dt: time step in s
        """
        self.eqGenerator()
        r = RK45(self.eq, t0, self.stVec[-1, :], tf, max_step=dt)
        while r.status == 'running':
            r.step()
            self.updateState(r.t, r.y)
        self.plot()

if __name__ == "__main__":
    drone = Drone(0.5, 0.1, 0.1)
    drone.setPhysics(9.81)
    drone.eqGenerator()
    drone.setConditions(0, 0, np.pi*5.0/180.0, 0.01, 0.0, 0)
    # print(drone.eq(0, np.zeros(6)))
    drone.setWaypoint(1.0, 1.0)
    print('Starting simulation')
    drone.solve(0, 100, 0.01)
    #drone.plot2D()
    drone.animate()
    

    
