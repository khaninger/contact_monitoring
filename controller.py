# System libraries
import os
from time import sleep

# Standard libraries
# import urx
import numpy as np
import pickle

# Local project files
from camera import cvf, camera2, helper_functions
from camera.skripts import main
#from camera.robot2 import Robot
from urx import Robot
from . import kp2pose
#from constraint import *

class Controller():
    def __init__(self, c_set = 'constraints_set.pkl'):
        # Load all constraints things
        #self.load_constraints(c_set)

        self.cam = None
        #self.init_camera()
        self.init_robot(cam = self.cam)
        self.tcp_to_obj = None # Pose of object in TCP frame

    def load_constraints(self, c_file):
        with open(c_file) as f:
            self.c_set = pickle.load(f)

    def init_camera(self):
        self.cam = camera2.Camera()
        self.cam.streaming()
        self.obj_kp = main.object_data(camera=self.cam, robot=self.rob)
        self.obj_name = "plug"

    def init_robot(self, cam = None):
        try:
            #self.rob = Robot(cam = cam)
            self.rob = Robot(host="192.168.29.102", use_rt=True)
        except:
            print("Error opening robot connection")
            self.stop()

    def get_robot_data(self):
        # Get the TCP pose and forces from robot
        f = np.array(self.rob.get_tcp_force(wait=True))
        x_tcp = np.array(self.rob.getl(wait=True))
        #print(f'Got robot data: \n  tcp   {x_tcp} \n  force {f}')
        return x_tcp, f

    def get_object_data(self):
        # Get the object coordinates
        x_tcp, f = self.get_robot_data()
        return x_obj, f

    def speedl(self):
        vel_null = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        vels = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        vels2 = [0.0, 0.0, -0.1, 0.0, 0.0, 0.0]
        accs = 0.01

        print('Setting robot speed')
        self.rob.speedl_tool(velocities=vels, acc=accs, min_time=10.0)
        sleep(5.0)
        print('Setting robot speed2')
        self.rob.speedl_tool(velocities=vels2, acc=accs, min_time=10.0)
        sleep(3.0)
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

    def def_grip2object_pose(self, set=True):
        if set:
            num_kp = 1
            pose_set = False
            while not pose_set:
                self.obj_kp.cam.streaming()
                T_cam2base, c_frame, d_frame = self.obj_kp.data_point()
                KeyPoint_list, heatmap = self.obj_kp.kpd.get_kp_coord(c_frame)
                keypoint_array, n_kp = self.obj_kp.Keypoint2WorldPoint(KeyPoint_list, d_frame, T_cam2base)
                if n_kp == num_kp:
                    print("Keypoint(s) detected, move robot into grip pose")
                    self.obj_kp.cam.streaming()
                    T_tcp2base = self.obj_kp.robot.rob.get_pose().array
                    T_kp2base = kp2pose.init_T(keypoint_array[0])
                    self.T_kp2tcp = helper_fuctions.inv_T(T_tcp2base)@T_kp2base
                    pose_set = True
                else:
                    print(f"So sorry, detected {n_kp} of {num_kp} but all need to be visible\nTry again please")
            plug().save(specifier='T', T=self.T_kp2tcp, index=1)
        else:
            dictionary = plug().load_data(path=os.getcwd() + "/data/" + "plug_constraint_0.pickle")
            self.T_kp2tcp = dictionary['T']

    def T_conpose2base(self, T_tcp2base):
        return T_tcp2base@self.T_kp2tcp

if __name__ == '__main__':
    print("starting controller!!!")
    try:
        controller = Controller()
        controller.speedl()
        # controller.loop()
    finally:
        controller.stop()
