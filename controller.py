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

     'traj_dist_threshold': 0.05, # meters from next point to advance the trajectory
     'traj_max_wait': 50,         # num time steps to keep same goal
     'ctrl_type': 'cartesian_compliance_controller',  # twist_controller, cartesian_compliance_controller
     'speed_slider': 15.0/100.0, # Initial speed override value, as fraction
     'contact_magnitude_active': 4, # 12, # Force to apply in the contact direction, 4 for rake
     'contact_magnitude_free': 6,
     }

class Controller():
    def __init__(self, task, online_control = False):
        self.cset = ConstraintSet(file_path = "contact_monitoring/data/"+task+"_constraint.pickle")
        self.T_object2tcp = pickle.load(open("contact_monitoring/data/constraint_T.pickle", "rb"))[task]
        print(f'Object to tcp {self.T_object2tcp}')
        self.f_tcp = None
        self.T_tcp = np.eye(4)
        self.T_object2base = np.eye(4)

        self.active_constraint_prev = None
        self.change_in_contact = True
        self.traj_pt_prev = None
        self.time_at_prev = 0   # num of timesteps with previous goal

        self.online_control = online_control
        self.init_ros()

    def init_ros(self):
        self.listener = tf.TransformListener()
        self.force_sub = rospy.Subscriber('wrench', WrenchStamped,
                                          self.force_callback, queue_size=1)
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

        self.T_object2base = self.T_tcp @ self.T_object2tcp # T_tcp is T_tcp2base

    def force_callback(self, msg):
        try:
            self.f_tcp = np.vstack((msg.wrench.force.x,  msg.wrench.force.y,  msg.wrench.force.z,
                                    msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z))
        except:
            print("Error loading ROS message in force_callback")
        self.tcp_process()
        active_constraint, name = self.detect_contact()
        if self.online_control: self.control(active_constraint, name)

    def get_closest_pt_index(self, points):
        # IN: a list of points on the trajectory
        dist = [dist_T(self.T_object2base,pt) for pt in points]
        index_closest = np.argmin(np.array(dist))
        return index_closest

    def get_next_pt(self, points):
        index_next_pt = self.traj_pt_prev
        timed_out = '          '
        if self.change_in_contact:
            index_next_pt = self.get_closest_pt_index(points)
        else:
            if dist_T(points[self.traj_pt_prev], self.T_object2base) < p['traj_dist_threshold']:
                index_next_pt = self.traj_pt_prev + 1
            elif self.time_at_prev > p['traj_max_wait']:
                index_next_pt = self.traj_pt_prev + 1
                timed_out = ' timed out'
            else:
                self.time_at_prev += 1

        index_next_pt = min(index_next_pt, len(points)-1)
        if index_next_pt != self.traj_pt_prev:
            print(f'Current trajectory point:  {index_next_pt}/{len(points)}, {timed_out}', end="\r", flush=True)
            self.traj_pt_prev = index_next_pt
            self.time_at_prev = 0
        if index_next_pt > 90:
            traj_vel = points[index_next_pt][:3,3]-points[index_next_pt-90][:3,3]
        else:
            traj_vel = np.zeros(3)
        return points[index_next_pt], traj_vel

    def control(self, active_constraint, constraint_name):
        #tcp_cmd = active_constraint.params['T_final']@invert_TransMat(self.T_object2tcp)
        #traj = [*active_constraint.params['T_traj'], *self.cset.get_next(active_constraint), *self.cset.get_next(self.cset.get_next(active_constraint))]
        traj = self.cset.constraints["hinge"].params['T_traj']





        #next_pt, traj_vel = self.get_next_pt(active_constraint.params['T_traj'])
        next_pt, traj_vel = self.get_next_pt(traj)
        tcp_cmd = next_pt@invert_TransMat(self.T_object2tcp)
        self.build_and_send_pose(self.T_tcp)
        #self.build_and_send_pose(tcp_cmd)


        #force stuff
        wrench_cmd_base = active_constraint.calc_constraint_wrench(self.T_object2base, p['contact_magnitude_active'])
        next_constraint = self.cset.get_next(active_constraint)
        #if constraint_name == 'free_space':
        #    wrench_cmd_base += next_constraint.calc_constraint_wrench(self.T_object2base, p['contact_magnitude_free'])
        #wrench_cmd_base[:3] += 150*traj_vel
        if next_constraint:
            wrench_cmd_base = next_constraint.calc_constraint_wrench(self.T_object2base, p['contact_magnitude_free'])
            #pass

        wrench_cmd_tcp = transform_force(invert_TransMat(self.T_tcp), wrench_cmd_base)
        #print(f'base {wrench_cmd_base} tcp {wrench_cmd_tcp}')
        wrench_cmd_tcp[3:] = 0
        self.build_and_send_wrench(wrench_cmd_tcp)
        #print("DEBUG")
        #print(f"constraint name: {constraint_name}")
        #print(wrench_cmd_tcp[:3])

        #self.build_and_send_wrench(np.array([0, 0, 10, 0, 0 ,0]))


    def detect_contact(self):
        # The transformation of force frame is from the MS Thesis of Bo Ho
        # SIX-DEGREE-OF-FREEDOM ACTIVE REAL-TIME FORCE CONTROL OF MANIPULATOR, pg. 63

        # Building a transformation with the orientation of TCP and position of object
        T_obj_adjust = np.eye(4)
        T_obj_adjust[:3,:3] = self.T_tcp[:3,:3]
        T_obj_adjust[3,:3] = self.T_object2base[3,:3]

        # Force at origin of object, with the orientation of base
        self.f_constraint = transform_force(T_obj_adjust, self.f_tcp)
        f_base = transform_force(self.T_tcp, self.f_tcp)

        #print(f'Real  forces: {f_base.T}')
        #print(f'Const forces: {self.f_constraint.T}')

        sim, active_constraint, name = self.cset.id_constraint(self.T_object2base, self.f_constraint)

        if active_constraint is not self.active_constraint_prev:
            #print(f'Changing constraint to {active_constraint}')
            self.change_in_contact = True
            self.active_constraint_prev = active_constraint
        else:
            self.change_in_contact = False

        self.build_and_send_sim(sim)
        return active_constraint, name

    def build_and_send_sim(self, sim): # IN: sim is a dictionary of simililarity scores
        name = sim.keys()
        sim_vals = [sim[n] for n in name]
        sim_msg = JointState(name = name, position = sim_vals)
        sim_msg.header.stamp = rospy.Time.now()
        if not rospy.is_shutdown() and hasattr(self, 'sim_pub'):
            self.sim_pub.publish(sim_msg)

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

    def shutdown(self):
        # Gets executed when the node is shutdown
        res = self.build_and_send_wrench(wrench_cmd = np.zeros(6))
        del self.sim_pub # This is needed because callbacks from force might be queued up
        if hasattr(self, 'wrench_pub'): del self.wrench_pub
        if hasattr(self, 'pose_pub'):   del self.pose_pub

        print("Sent zero  command") if res else  print("** Failed to send zero velocity command! Hit that Estop **")
        if self.online_control and self.switch_controller([],[p['ctrl_type']],1,True,5):
            print("**** Killed the controller ****")
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

