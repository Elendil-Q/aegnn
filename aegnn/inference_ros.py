#! /usr/bin/env python
# coding :utf-8

"""
put the model into the ROS
"""

import sys

sys.path.append('/home/elendil/anaconda3/envs/aegnn/lib/python3.8/site-packages')
sys.path.append('/home/elendil/PKG/DV_ROS/devel/lib/python3/dist-packages')
import os
import aegnn
import numpy as np
import rospy
import torch
import torch_geometric
from torch_geometric.transforms import FixedPoints
from torch_geometric.data import Data
from torch_geometric.nn.pool import radius_graph

from typing import List
from dv_ros_msgs.msg import EventArray, Event


class Classifier:
    def __init__(self, dt=0.02, radius=3, buffer_max_num=50000, event_batch_size=10000):
        """
        init ros
        """
        rospy.init_node('class_node')
        self.dt = dt
        # subs
        # events:List[Event]
        self.event_suber = rospy.Subscriber("/capture_node/events", EventArray, queue_size=1,
                                            callback=self.call_back_events)

        """
        events buffer
        """
        self.buffer_max_num = buffer_max_num
        self.current_event_num = 0
        self.events = None

        """
        graph generation
        """
        self.event_batch_size = event_batch_size
        self.edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
        # sample rates
        self.sample_K = 10

        """
        model init
        """
        edge_attr = torch_geometric.transforms.Cartesian(cat=False, max_value=10.0)
        if torch.cuda.is_available():
            device = torch.device('cuda', 0)
        else:
            device = torch.device('cpu')
        self.device = device
        input_shape = torch.tensor([346, 260, 3])
        model = aegnn.models.networks.GraphRes('ncars', input_shape, 2, pooling_size=(10, 10))
        model.to(device)
        self.model = aegnn.asyncronous.make_model_asynchronous(model, radius, [346, 260], edge_attr)

    def call_back_events(self, event_array: EventArray):
        new_events = event_array.events
        total_num = len(new_events) + self.current_event_num
        if total_num > self.buffer_max_num:
            pop_num = total_num - self.buffer_max_num
            del self.events[0:pop_num]
        self.events.extend(new_events)
        self.current_event_num = len(self.events)

    def build_graph(self, events: List[Event], radius=3.0, max_neighbors=32):
        """
        build a new graph with incoming events and sample it in an uniform way.
        Args:
            events: 事件数据
            radius: 邻域节点判定距离
            max_neighbors: 最多邻居数
        Returns: graph: Data
        """
        """build graph"""
        x = []
        pos = []
        for event in events:
            pos.append([event.x, event.y, event.ts])
            x.append([event.polarity])
        x = torch.tensor(x)
        pos = torch.tensor(pos)
        graph = Data(x=x, pos=pos)

        """sample events"""
        num_sample = int(len(events) / self.sample_K)
        sampler = FixedPoints(num=num_sample, allow_duplicates=False, replace=False)
        graph = sampler(graph)

        """create new graph"""
        graph.edge_index = radius_graph(graph.pos, r=radius, max_num_neighbors=max_neighbors)
        return graph

    def build_new_graph(self):
        """
        从事件缓存中读取最近的一个batch
        Returns:

        """
        if self.current_event_num < self.event_batch_size:
            rospy.logwarn('Events are insufficient to build a graph!')
            return None
        else:
            events = self.events[:self.event_batch_size]
            del events[:self.event_batch_size]
            graph = self.build_graph(events)
            return graph

    def recognition(self):
        graph = self.build_new_graph()
        _ = self.model(graph.to(self.device))
        rospy.loginfo("runing!")
        del graph

    def timer_F(self):
        self.timer = rospy.Timer(rospy.Duration(self.dt), self.recognition)


if __name__ == "__main__":
    recognition = Classifier()
    recognition.timer_F()
    rospy.spin()
