# System libraries
import os
import argparse
from time import sleep

# Standard libraries
import numpy as np
import pickle

import rospy
from geometry_msgs.msg import WrenchStamped, PoseStamped, Twist
from sensor_msgs.msg import JointState
from controller_manager_msgs.srv import ListControllers, LoadController, SwitchController
from ur_msgs.srv import SetSpeedSliderFraction
import tf


from . import kp2pose
from .constraint import *
from .rotation_helpers import xyz_to_rotation, invert_TransMat


p = {'vel_max': np.array([*[0.02]*3, *[0.001]*3]), # maximum velocity for lin, angular motion
     'acc_max': 0.1,       # maximum linear acceleration
     'hz':500,              # sample rate of controller
     'stop_distance': 0.01, # meters from goal pose to stop, linear
     'ctrl_type': 'cartesian_compliance_controller',  # twist_controller, cartesian_compliance_controller
     'speed_slider': 15.0/100.0, # Initial speed override value, as fraction
     }

class Controller():
    def __init__(self, task, online_control = False):
        self.cset = ConstraintSet(file_path = "contact_monitoring/data/"+task+"_constraint.pickle")
        self.T_object2tcp = pickle.load(open("contact_monitoring/data/constraint_T.pickle", "rb"))[task]

        self.f_tcp = None
        self.T_tcp = np.eye(4)
        self.T_object2base = np.eye(4)

        self.vel_cmd_prev = np.zeros(6)
        self.active_constraint_prev = None

        self.online_control = online_control
        self.init_ros()

    def init_ros(self):
        self.listener = tf.TransformListener()
        self.force_sub = rospy.Subscriber('wrench', WrenchStamped,
                                          self.force_callback, queue_size=1)
        self.vel_pub = rospy.Publisher('twist_controller/command', Twist, queue_size=1)
        self.wrench_pub = rospy.Publisher('target_wrench', WrenchStamped, queue_size=1)
        self.pose_pub = rospy.Publisher('target_frame', PoseStamped, queue_size=1)
        self.sim_pub = rospy.Publisher('contact_mode', JointState, queue_size=1)

        rospy.on_shutdown(self.shutdown)

        if self.online_control:
            if not self.cset:
                raise Exception('Trying to control without a constraint set?')

            set_spd_slider = rospy.ServiceProxy('ur_hardware_interface/set_speed_slider', SetSpeedSliderFraction)
            list_controllers = rospy.ServiceProxy('controller_manager/list_controllers', ListControllers)
            load_controller  = rospy.ServiceProxy('controller_manager/load_controller', LoadController)
            self.switch_controller = rospy.ServiceProxy('controller_manager/switch_controller', SwitchController)

            set_spd_slider(p['speed_slider'])

            avail_controllers = list_controllers()
            for ctrl in avail_controllers.controller:
                if ctrl.name == p['ctrl_type']:
                    if ctrl.state == 'initialized':
                        self.build_and_send_twist() # Send some zeros, just to be sure there's no naughty messages waiting
                        self.build_and_send_wrench()
                        print('**** Controller initialized ****')

            if self.switch_controller([p['ctrl_type']],['scaled_pos_joint'],1,True,5):
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
        active_constraint = self.detect_contact()
        if self.online_control: self.control(active_constraint)

    def interpolate(self, T_f):
        #T_err = T_f@invert_TransMat(self.T_tcp)
        err = T_f[:3,3]-self.T_object2base[:3,3]
        err_norm = np.linalg.norm(err)
        if err_norm < p['stop_distance']:
            lin_vel = np.zeros(3)
        else:
            lin_vel = err/err_norm
        rot_vel = np.zeros(3)
        return np.hstack((lin_vel, rot_vel))

    def control(self, active_constraint):
        #print(f'Active constraint {active_constraint}') # final pose {active.params["T_final"]}')

        #vel_cmd = self.interpolate(active_constraint.params['T_final'])#self.interpolate(np.eye(4))
        #vel_cmd[:3] *= p['vel_max'][0] # respect the speed limit
        #vel_cmd = np.clip(vel_cmd,
        #                  self.vel_cmd_prev-p['acc_max']/p['hz'],
        #                  self.vel_cmd_prev+p['acc_max']/p['hz'])
        #self.build_and_send_twist(vel_cmd)

        #print(active_constraint.params['T_final'])

        tcp_cmd = active_constraint.params['T_final']@invert_TransMat(self.T_object2tcp)
        #tcp_cmd = self.T_tcp
        self.build_and_send_pose(tcp_cmd)
        self.build_and_send_wrench([0, 0, 0, 0, 0, 0])

    def build_and_send_pose(self, T_cmd):
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'base'

        msg.pose.position.x = T_cmd[0,3]
        msg.pose.position.y = T_cmd[1,3]
        msg.pose.position.z = T_cmd[2,3]

        quat = rotation_to_quat(T_cmd[:3, :3])
        msg.pose.orientation.w = quat[0]
        msg.pose.orientation.x = quat[1]
        msg.pose.orientation.y = quat[2]
        msg.pose.orientation.z = quat[3]

        if not rospy.is_shutdown() and hasattr(self, 'pose_pub'):
            self.pose_pub.publish(msg)
            return True

    def build_and_send_wrench(self, wrench_cmd = np.zeros(6)):
        msg = WrenchStamped()
        msg.header.stamp = rospy.Time.now()
        msg.wrench.force.x = wrench_cmd[0]
        msg.wrench.force.y = wrench_cmd[1]
        msg.wrench.force.z = wrench_cmd[2]
        msg.wrench.torque.x = wrench_cmd[3]
        msg.wrench.torque.y = wrench_cmd[4]
        msg.wrench.torque.z = wrench_cmd[5]

        if not rospy.is_shutdown() and hasattr(self, 'wrench_pub'):
            self.wrench_pub.publish(msg)
            return True

    def build_and_send_twist(self, vel_cmd = np.zeros(6)):
        # Builds a twist message out of the 6x1 vel_cmd, and sends this on vel_pub

        # Limit the velocity command
        vel_cmd = np.clip(vel_cmd, -p['vel_max'], p['vel_max'])

        self.vel_cmd_prev = vel_cmd

        # Build twist message
        msg = Twist()
        msg.linear.x = vel_cmd[0]
        msg.linear.y = vel_cmd[1]
        msg.linear.z = vel_cmd[2]
        msg.angular.x = vel_cmd[3]
        msg.angular.y = vel_cmd[4]
        msg.angular.z = vel_cmd[5]

        if not rospy.is_shutdown() and hasattr(self,'vel_pub'):
            self.vel_pub.publish(msg)
            return True

    def detect_contact(self):
        self.tcp_process()
        self.objectpose_process()
        # The transformation of force frame is from the MS Thesis of Bo Ho
        # SIX-DEGREE-OF-FREEDOM ACTIVE REAL-TIME FORCE CONTROL OF MANIPULATOR, pg. 63

        self.f_base = transform_force(self.T_tcp, self.f_tcp)

        sim, active_constraint = self.cset.id_constraint(self.T_object2base, self.f_base)

        if active_constraint is not self.active_constraint_prev:
            print(f'Changing constraint to {active_constraint}')
            self.active_constraint_prev = active_constraint

        name = sim.keys()
        sim_vals = [sim[n] for n in name]
        sim_msg = JointState(name = name, position = sim_vals)
        sim_msg.header.stamp = rospy.Time.now()
        if not rospy.is_shutdown() and hasattr(self, 'sim_pub'):
            self.sim_pub.publish(sim_msg)
        return active_constraint

    def shutdown(self):
        # Gets executed when the node is shutdown
        res = self.build_and_send_twist(vel_cmd = np.zeros(6))
        res = self.build_and_send_wrench(wrench_cmd = np.zeros(6))
        del self.vel_pub # This is needed because callbacks from force might be queued up
        del self.sim_pub
        del self.wrench_pub
        if hasattr(self, 'twist_pub'): del self.twist_pub
        if hasattr(self, 'pose_pub'):  del self.pose_pub
        if res:
            print("Sent zero velocity command")
        else:
            print("** Failed to send zero velocity command! Hit that Estop **")
        if self.switch_controller([],[p['ctrl_type']],1,True,5):
            print("Killed the twist_controller")
        print("Shutting down controller")


def start_node(task, online_control):
    rospy.init_node('contact_observer')
    Controller(task, online_control = online_control )
    rospy.sleep(2e-1)
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='rake', help="cable or rake")
    parser.add_argument("--ctrl", default=False, action='store_true',
                        help="Connect for the online control")
    args = parser.parse_args()

    start_node(args.task, args.ctrl)

