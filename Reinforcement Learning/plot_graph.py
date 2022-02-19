#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 23:00:48 2021

@author: hussein
"""

import matplotlib.pyplot as plt
import csv

x=[]
y=[]

with open('plot_reward.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))


plt.plot(x,y, marker='o')

plt.title('Rewards function of Episodes')

plt.xlabel('Number of Episode')
plt.ylabel('Reward')

plt.show()
