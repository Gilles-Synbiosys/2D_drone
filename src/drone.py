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
        self.cmd = np.zeros(2) # [F, theta]
        self.eq = None # equations of motion
        self.wpt = np.zeros(2) # waypoint
        self.gainDict = {'Kp_f': 0.1,
                         'Kd_f': 1, 
                         'Ki_f': 1, 
                         'Kp_d': 0.0001, 
                         'Kd_d': 1, 
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

        self.cmd[1] = -y[2] + np.arctan2(self.wpt[1]-y[1], self.wpt[0]-y[0])*self.gainDict['Kp_d'] #+ y[5]*self.gainDict['Kd_f'] + self.gainDict['Ki_f']*np.trapz(self.wpt[1]-y[1], self.wpt[0]-y[0])
        self.cmd[0] = self.mass*self.g/np.cos(y[2]+self.cmd[1]) + ((self.wpt[0]-y[0])**2 + (self.wpt[1]-y[1])**2)**.5*self.gainDict['Kp_f'] #+ (self.wpt[0]-y[0])*self.gainDict['Kd_d'] + self.gainDict['Ki_d']*np.trapz(self.wpt[0]-y[0])
        
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
            self.control(t, y)
            ydot = np.zeros(6)
            ydot[0] = y[3]
            ydot[1] = y[4]
            ydot[2] = y[5]
            ydot[3] = self.cmd[0]*np.sin(y[2]+self.cmd[1])/self.mass
            ydot[4] = self.cmd[0]*np.cos(y[2]+self.cmd[1])/self.mass - self.g
            ydot[5] = self.cmd[0]*np.sin(self.cmd[1])*self.com/self.inertia
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
        self.t = np.append(self.t, t)
    
    def plot(self):
        """
        Plots the state vector
        args:
            None
        """
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(self.t, self.stVec[:, 0])
        axs[0, 0].set_title('x')
        axs[0, 1].plot(self.t, self.stVec[:, 1])
        axs[0, 1].set_title('y')
        axs[1, 0].plot(self.t, self.stVec[:, 2])
        axs[1, 0].set_title('theta')
        axs[1, 1].plot(self.t, self.stVec[:, 3])
        axs[1, 1].set_title('xdot')
        axs[2, 0].plot(self.t, self.stVec[:, 4])
        axs[2, 0].set_title('ydot')
        axs[2, 1].plot(self.t, self.stVec[:, 5])
        axs[2, 1].set_title('thetadot')
        plt.show()

    def animate(self):
        """
        Animates the drone  in 2D
        args:
            None
        """
        fig = plt.figure()
        ax = plt.axes(xlim=(-2, 2), ylim=(-2, 2))
        line, = ax.plot([], [], lw=2)
        def init():
            line.set_data([], [])
            return line,
        def animate(i):
            x = self.stVec[i, 0]
            y = self.stVec[i, 1]
            line.set_data(x, y)
            return line,
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(self.t), interval=20, blit=True)
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
    drone.setConditions(0, 0, np.pi*20.0/180.0, -1.0, 2.0, 0)
    print(drone.eq(0, np.zeros(6)))
    drone.setWaypoint(1, 1)
    drone.solve(0, 100, 0.01)
    

    
