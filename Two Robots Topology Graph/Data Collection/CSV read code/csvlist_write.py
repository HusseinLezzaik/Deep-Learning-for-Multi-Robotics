#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hussein
"""

import csv

with open('robot1.csv', 'w', newline='') as f :
    thewriter = csv.writer(f)
    
    thewriter.writerow(['X', 'Y', 'U'])
    
    
    for i in range(1,100):
        thewriter.writerow(['one','two', 'three'])
    
    
    
    
    
    
    
    
with open('robot1.csv', 'w', newline='') as f :
            thewriter = csv.writer(f)
    
            thewriter.writerow(['X', 'Y', 'U'])
            print(self.Y1)
            for i in range(1,100):
                thewriter.writerow(['one',self.Y1, 'three'])
            #print(self.X1)
            #thewriter.writerow([self.X1,'two', 'three'])