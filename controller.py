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
from .rotation_helpers import xyz_to_rotation, invert_TransMat

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
        self.T_object2base = np.eye(4)
        self.T_object2tcp = pickle.load(open("/home/ipk410/converging/contact_monitoring/data/constraint_T.pickle", "rb"))["plug"]
        print("self.T_object2tcp")
        print(self.T_object2tcp)
        import time
        time.sleep(5)

    def objectpose_process(self):
        self.T_tcp2base = self.T_tcp


        self.T_object2base = self.T_tcp2base @ self.T_object2tcp


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
        self.objectpose_process()
        # The transformation of force frame is from the MS Thesis of Bo Ho
        # SIX-DEGREE-OF-FREEDOM ACTIVE REAL-TIME FORCE CONTROL OF MANIPULATOR, pg. 63
        self.R_object2tcp = self.T_object2tcp[:3, :3]
        self.R_tcp2object = self.R_object2tcp.T
        self.pos_tcp2base = self.T_tcp2base[:3,-1]
        self.pos_object2base = self.T_object2base[:3,-1]
        self.T_tcp2object = invert_TransMat(self.T_object2tcp)
        self.pos_tcp2object = self.T_tcp2object[:3,-1]
        self.cross_product = np.array([[0,-self.pos_tcp2object[-1],self.pos_tcp2object[1]],
                                       [self.pos_tcp2object[-1],0,-self.pos_tcp2object[0]],
                                       [-self.pos_tcp2object[1],self.pos_tcp2object[0],0]])
        self.T_wrench_tcp2object = np.hstack([np.vstack([self.R_tcp2object,self.cross_product@self.R_tcp2object]),np.vstack([np.zeros((3,3)),self.R_tcp2object])])
        self.f_object = self.T_wrench_tcp2object @ self.f_tcp

        if not self.cset:
            print("No cset object, skipping similarity eval")
            return

        sim = self.cset.id_constraint(self.T_object2base, self.f_object)


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
    
