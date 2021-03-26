#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:57:49 2021

@author: hussein
"""

import csv

with open('robot1.csv', 'w', newline='') as f:
    fieldnames = ['column1', 'column2', 'column3']
    thewriter = csv.DictWriter(f, fieldnames=fieldnames)
    
    thewriter.writeheader()
    for i in range(1,5):
        thewriter.writerow({'column1' : 'one', 'column2' : 'two', 'column3' : 'three'})
    
    