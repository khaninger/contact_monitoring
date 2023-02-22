# System libraries
import os
import argparse
from time import sleep

# Standard libraries
import numpy as np
import pickle

import rospy
from geometry_msgs.msg import WrenchStamped, Twist
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers, LoadController, SwitchController
import tf


from . import kp2pose
from .constraint import *
from .rotation_helpers import xyz_to_rotation, invert_TransMat


p = {'vel_max': np.array([*[0.05]*3, *[0.01]*3]),
    }

class Controller():
    def __init__(self, constraint_set = None, online_control = False):
        self.cset = constraint_set
        self.f_tcp = None
        self.T_tcp = np.eye(4)
        self.T_object2base = np.eye(4)
        self.T_object2tcp = pickle.load(open("/home/ipk410/converging/contact_monitoring/data/constraint_T.pickle", "rb"))["plug"]
        #print("self.T_object2tcp")
        #print(self.T_object2tcp)

        self.init_ros(online_control)

    def init_ros(self, online_control = False):
        self.listener = tf.TransformListener()
        self.force_sub = rospy.Subscriber('wrench', WrenchStamped,
                                          self.force_callback, queue_size=1)
        self.vel_pub = rospy.Publisher('twist_controller/command', Twist, queue_size=1)
        self.sim_pub = rospy.Publisher('contact_mode', JointState, queue_size=1)

        rospy.on_shutdown(self.shutdown)

        if online_control:
            if not self.cset:
                raise Exception('Trying to control without a constraint set?')
            list_controllers = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
            load_controller  = rospy.ServiceProxy('controller_manager/load_controller', LoadController)
            self.switch_controller = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)
            avail_controllers = list_controllers()
            for ctrl in avail_controllers.controller:
                if ctrl.name == 'twist_controller':
                    if ctrl.state == 'initialized':
                        self.build_and_send_twist() # Send some zeros, just to be sure there's no naughty messages waiting
                        print('**** Controller initialized ****')

            if self.switch_controller(['twist_controller'],['scaled_pos_joint'],1,True,5):
                print("****** Controller started ******")
            else:
                raise Exception('Error initializing the controller')

    def tcp_process(self):
        (trans,rot) = self.listener.lookupTransform('base', 'tool0_controller', rospy.Time(0))
        try:
            self.T_tcp[:3,-1] = np.array(trans)
            self.T_tcp[:3,:3] = quat_to_rotation(np.vstack((rot[3], rot[0], rot[1], rot[2])))
        except:
            print("Error loading ROS message in joint_callback")

    def objectpose_process(self):
        self.T_tcp2base = self.T_tcp
        self.T_object2base = self.T_tcp2base @ self.T_object2tcp

    def force_callback(self, msg):
        try:
            self.f_tcp = np.vstack((msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                                    msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z))
        except:
            print("Error loading ROS message in force_callback")
        sim = self.detect_contact()
        self.control()

    def interpolate(self, T_f, time = 4):
        #T_err = T_f@invert_TransMat(self.T_tcp)
        lin_vel = (T_f[:3,3]-self.T_tcp[:3,3])/time
        rot_vel = np.zeros(3)
        return np.hstack((lin_vel, rot_vel))

    def control(self):
        # TODO something clever with the constraint jacobians?
        vel_cmd = self.interpolate(np.eye(4))
        vel_cmd[:3] *= p['vel_max'][0]/np.linalg.norm(vel_cmd[:3]) # respect the speed limit
        self.build_and_send_twist(vel_cmd)

    def build_and_send_twist(self, vel_cmd = np.zeros(6), kill_pub = False):
        # Builds a twist message out of the 6x1 vel_cmd, and sends this on vel_pub

        # Limit the velocity command
        vel_cmd = np.clip(vel_cmd, -p['vel_max'], p['vel_max'])

        # Build twist message
        msg = Twist(vel_cmd[:3], vel_cmd[3:])

        if not rospy.is_shutdown() and hasattr(self,'vel_pub'):
            self.vel_pub.publish(msg)
            if kill_pub: del self.vel_pub # This is needed because callbacks from force might be queued up
            return True

    def shutdown(self):
        # Gets executed when the node is shutdown
        res = self.build_and_send_twist(vel_cmd = np.zeros(6), kill_pub = True)
        if res:
            print("Sent zero velocity command")
        else:
            print("** Failed to send zero velocity command! Hit that Estop **")
        if self.switch_controller([],['twist_controller'],1,True,5):
            print("Killed the twist_controller")
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
        name = sim.keys()
        sim_vals = [sim[n] for n in name]
        sim_msg = JointState(name = name, position = sim_vals)
        sim_msg.header.stamp = rospy.Time.now()
        self.sim_pub.publish(sim_msg)



def start_node(constraint_set, online_control):
    rospy.init_node('contact_observer')
    controller = Controller(constraint_set = constraint_set, online_control = online_control )
    rospy.sleep(1e-1)
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cset", default="contact_monitoring/data/cable_constraint.pickle", help="path to saved constraint set")
    parser.add_argument("--ctrl", default=False, action='store_true',
                        help="Connect for the online control")
    args = parser.parse_args()
    
    cset = ConstraintSet(file_path = args.cset) if args.cset else None

    start_node(cset, args.ctrl)

