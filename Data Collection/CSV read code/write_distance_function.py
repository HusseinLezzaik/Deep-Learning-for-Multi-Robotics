#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hussein
"""

if distance > 0.2:
            
                
            with open('robot1.csv', 'a', newline='') as f:
                fieldnames = ['Data_X', 'Data_Y', 'Label_X', 'Label_Y']
                thewriter = csv.DictWriter(f, fieldnames=fieldnames)
                
                if self.i1 == 0:
                    thewriter.writeheader()
                    self.i1 = 1
                
                thewriter.writerow({'Data_X' : self.X1, 'Data_Y' : self.Y1, 'Label_X' : self.U1[0], 'Label_Y' : self.U1[1]})
                
            with open('robot2.csv', 'a', newline='') as f:
                fieldnames = ['Data_X', 'Data_Y', 'Label_X', 'Label_Y']
                thewriter = csv.DictWriter(f, fieldnames=fieldnames)
            
                if self.i2 == 0:
                    thewriter.writeheader()
                    self.i2 = 1
            
                thewriter.writerow({'Data_X' : self.X2, 'Data_Y' : self.Y2, 'Label_X' : self.U2[0], 'Label_Y' : self.U2[1]})