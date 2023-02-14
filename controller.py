# System libraries
from time import sleep

# Standard libraries
import urx
import numpy as np
import pickle

# Local project files
#from constraint import *

class Controller():
    def __init__(self, c_set = 'constraints_set.pkl'):
        # Load all constraints things
        #self.load_constraints(c_set)
        self.tcp_to_obj = None # Pose of object in TCP frame

        # Init robot etc
        self.init_robot()
        #self.loop()

        self.tcp_to_obj = None # Pose of object in TCP frame

    def load_constraints(self, c_file):
        with open(c_file) as f:
            self.c_set = pickle.load(f)

    def init_robot(self):
        try:
            self.rob = urx.Robot(host="192.168.29.102", use_rt=True)
        except:
            print("Error opening robot connection")
            self.stop()


    def get_robot_data(self):
        # Get the TCP pose and forces from robot
        f = np.array(self.rob.get_tcp_force(wait=True))
        x_tcp = np.array(self.rob.getl(wait=True))
        print(f'Got robot data: \n  tcp {x_tcp}    force {f}')
        return x_tcp, f

    def get_object_data(self):
        # Get the object coordinates
        x_tcp, f = self.get_robot_data()
        return x_obj, f

    def speedl(self):
        vel_null = [0.00, 0.0, 0.0, 0.0, 0.0, 0.0]
        vels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        accs = 0.01

        print('Setting robot speed')
        self.rob.speedl_tool(velocities=vels, acc=accs, min_time=10.0)
        sleep(5.0)
        print('Sending zero vel')
        self.rob.speedl_tool(velocities=vel_null, acc=accs, min_time=0.1) # Abrupt stop even with 1.0 for min_time
        print('Ending robot speed')

    def loop(self):
        # Control loop, runs til kill
        while(True):
            x_tcp, f_tcp = self.get_robot_data()
            sleep(0.1)
            #constraint_mode = self.c_set.id_constraint(x_tcp, f_tcp)


    def stop(self):
        try:
            self.rob.close()
        except:
            pass

    def def_grip2object_pose(self):
        pass
        #1) grasp object -Save TCP
        #2) scan object -> Save Object pose / TCP pose

if __name__ == '__main__':
    print("starting controller!!!")
    try:
        controller = Controller()
        # controller.speedl()
        controller.loop()
    finally:
        controller.stop()
