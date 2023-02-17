# System libraries
import os
from time import sleep

# Standard libraries
import numpy as np
import pickle
#from urx import Robot
import rospy
#TODO add messages

# Local project files
from camera import cvf, camera2, helper_functions
from camera.skripts import main
#from camera.robot2 import Robot
from . import kp2pose

from .constraint import *
from .rotation_helpers import xyz_to_rotation

class Controller():
    def __init__(self, constraint): #c_set = 'constraints_set.pkl'):
        # Load all constraints things
        #self.load_constraints(c_set)
        self.constraint = constraint

        self.cam = None
        #self.init_camera()
        self.init_robot(cam = self.cam)
        self.tcp_to_obj = None # Pose of object in TCP frame

        self.f_tcp = None
        self.T_tcp = None

    def init_camera(self):
        self.cam = camera2.Camera()
        self.cam.streaming()
        self.obj_kp = main.object_data(camera=self.cam, robot=self.rob)
        self.obj_name = "plug"

    def init_ros(self):
        self.force_sub = rospy.Subscriber('tcp_force', JointState,
                                          self.force_callback, queue_size=1)
        self.tcp_sub =   rospy.Subscriber('x_tcp', JointState,
                                          self.tcp_callback, queue_size=1)

    def tcp_callback(self, msg):
        try:
            self.T_tcp[:3,-1] = np.array(msg.position)
            self.T_tcp[:3,:3] = quat_to_rotation(np.array(msg.quaternion))
        except:
            print("Error loading ROS message in joint_callback")
                
    def force_callback(self, msg):
        try:
            self.f_tcp = msg.effort[:6]
        except:
            print("Error loading ROS message in force_callback")

    def shutdown(self):
        #TODO add sending all zeros on the TCP cmd interface
        print("Shutting down controller")
    
    def loop(self):
        # Control loop, runs til kill
        while(True):
            x_tcp, f_tcp = self.get_robot_data()
            R_tcp = xyz_to_rotation(x_tcp[3:])
            p = np.expand_dims(np.array(x_tcp[:3]),0)
            T_tcp = np.hstack((np.vstack((R_tcp, np.zeros((1,3)))),
                              np.vstack((p.T,np.array(1)))))
            sim = self.constraint.get_similarity(T_tcp, f_tcp)
            print(f'similarity: {sim}')
            #print(f'recieved {f_tcp}')
            #sleep(0.1)
            #constraint_mode = self.c_set.id_constraint(x_tcp, f_tcp)

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

def start_node():
    rospy.init_node('contact_observer')
    controller = Controller()
    rospy.on_shutdown(controller.shutdown)
    rospy.spin()
    
if __name__ == '__main__':
    start_node()

    
