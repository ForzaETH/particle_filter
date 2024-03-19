#!/usr/bin/env python3
import rospy
from dynamic_reconfigure.server import Server
from particle_filter.cfg import particle_filter2Config

def callback(config, level):
    return config

if __name__ == "__main__":
    rospy.init_node("dynamic_synPF_tuner_node", anonymous=False)
    print('SynPF Dynamic Server Launched...')
    srv = Server(particle_filter2Config, callback)
    rospy.spin()

