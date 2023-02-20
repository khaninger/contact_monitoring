# System libraries
import os
import argparse
from time import sleep

# Standard libraries
import numpy as np
import pickle
#from urx import Robot
import rospy
from geometry_msgs.msg import WrenchStamped
import tf

# Local project files
#from camera import cvf, camera2, helper_functions
#from camera.skripts import main
#from camera.robot2 import Robot
from . import kp2pose

from .constraint import *
from .rotation_helpers import xyz_to_rotation

class Controller():
    def __init__(self, constraint_set = None):
        self.cam = None
        #self.init_camera()
        #self.init_robot(cam = self.cam)
        self.init_ros()
        self.tcp_to_obj = None # Pose of object in TCP frame

        self.cset = constraint_set
        self.f_tcp = None
        self.T_tcp = np.eye(4)

    def init_camera(self):
        self.cam = camera2.Camera()
        self.cam.streaming()
        self.obj_kp = main.object_data(camera=self.cam, robot=self.rob)
        self.obj_name = "plug"

    def init_ros(self):
        self.listener = tf.TransformListener()
        self.force_sub = rospy.Subscriber('wrench', WrenchStamped,
                                          self.force_callback, queue_size=1)

    def tcp_process(self):
        (trans,rot) = self.listener.lookupTransform('base', 'tool0_controller', rospy.Time(0))
        try:
            #print(trans)
            self.T_tcp[:3,-1] = np.array(trans)
            self.T_tcp[:3,:3] = quat_to_rotation(np.vstack((rot[3], rot[0], rot[1], rot[2])))
        except:
            print("Error loading ROS message in joint_callback")

    def force_callback(self, msg):
        try:
            self.f_tcp = np.vstack((msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                                    msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z))
        except:
            print("Error loading ROS message in force_callback")
        self.detect_contact()
        
    def shutdown(self):
        #TODO add sending all zeros on the TCP cmd interface
        print("Shutting down controller")

    def detect_contact(self):
        self.tcp_process()
        if not self.cset:
            print("No cset object, skipping similarity eval")
            return

        sim = self.cset.id_constraint(self.T_tcp, self.f_tcp)


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

    def T_compose2base(self, T_tcp2base):
        return T_tcp2base@self.T_kp2tcp

def start_node(constraint_set):
    rospy.init_node('contact_observer')
    controller = Controller(constraint_set = constraint_set)
    rospy.on_shutdown(controller.shutdown)
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cset", default="contact_monitoring/data/cable_constraint.pickle", help="path to saved constraint set")
    args = parser.parse_args()

    cset = ConstraintSet(file_path = args.cset) if args.cset else None
    start_node(constraint_set = cset)
    
