#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:35:24 2019

@author: keyur
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
A = np.zeros([10,6])
a = []
with open('output_1.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(len(a)):
    if (j)%11 == 1:
        A[0,0] += float(a[j])
    elif j%11 == 2:
        A[1,0] +=float(a[j])
    elif j%11 == 3:
        A[2,0] +=float(a[j])
    elif j%11 == 4:
        A[3,0] +=float(a[j])        
    elif j%11 == 5:
        A[4,0] +=float(a[j])
    elif j%11 == 6:
        A[5,0] +=float(a[j])
    elif j%11 == 7:
        A[6,0] +=float(a[j])
    elif j%11 == 8:
        A[7,0] +=float(a[j])
    elif j%11 == 9:
        A[8,0] +=float(a[j])
    elif j%11 == 10:
        A[9,0] +=float(a[j])

a = []
with open('output_2.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(len(a)):
    if (j)%11 == 1:
        A[0,1] += float(a[j])
    elif j%11 == 2:
        A[1,1] +=float(a[j])
    elif j%11 == 3:
        A[2,1] +=float(a[j])
    elif j%11 == 4:
        A[3,1] +=float(a[j])        
    elif j%11 == 5:
        A[4,1] +=float(a[j])
    elif j%11 == 6:
        A[5,1] +=float(a[j])
    elif j%11 == 7:
        A[6,1] +=float(a[j])
    elif j%11 == 8:
        A[7,1] +=float(a[j])
    elif j%11 == 9:
        A[8,1] +=float(a[j])
    elif j%11 == 10:
        A[9,1] +=float(a[j])                                              

a = []
with open('output_4.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(len(a)):
    if (j)%11 == 1:
        A[0,2] += float(a[j])
    elif j%11 == 2:
        A[1,2] +=float(a[j])
    elif j%11 == 3:
        A[2,2] +=float(a[j])
    elif j%11 == 4:
        A[3,2] +=float(a[j])        
    elif j%11 == 5:
        A[4,2] +=float(a[j])
    elif j%11 == 6:
        A[5,2] +=float(a[j])
    elif j%11 == 7:
        A[6,2] +=float(a[j])
    elif j%11 == 8:
        A[7,2] +=float(a[j])
    elif j%11 == 9:
        A[8,2] +=float(a[j])
    elif j%11 == 10:
        A[9,2] +=float(a[j])        

a = []
with open('output_8.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(len(a)):
    if (j)%11 == 1:
        A[0,3] += float(a[j])
    elif j%11 == 2:
        A[1,3] +=float(a[j])
    elif j%11 == 3:
        A[2,3] +=float(a[j])
    elif j%11 == 4:
        A[3,3] +=float(a[j])        
    elif j%11 == 5:
        A[4,3] +=float(a[j])
    elif j%11 == 6:
        A[5,3] +=float(a[j])
    elif j%11 == 7:
        A[6,3] +=float(a[j])
    elif j%11 == 8:
        A[7,3] +=float(a[j])
    elif j%11 == 9:
        A[8,3] +=float(a[j])
    elif j%11 == 10:
        A[9,3] +=float(a[j])                       

a = []
with open('output_16.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(len(a)):
    if (j)%11 == 1:
        A[0,4] += float(a[j])
    elif j%11 == 2:
        A[1,4] +=float(a[j])
    elif j%11 == 3:
        A[2,4] +=float(a[j])
    elif j%11 == 4:
        A[3,4] +=float(a[j])        
    elif j%11 == 5:
        A[4,4] +=float(a[j])
    elif j%11 == 6:
        A[5,4] +=float(a[j])
    elif j%11 == 7:
        A[6,4] +=float(a[j])
    elif j%11 == 8:
        A[7,4] +=float(a[j])
    elif j%11 == 9:
        A[8,4] +=float(a[j])
    elif j%11 == 10:
        A[9,4] +=float(a[j])                       

a = []
with open('output_32.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(len(a)):
    if (j)%11 == 1:
        A[0,5] += float(a[j])
    elif j%11 == 2:
        A[1,5] +=float(a[j])
    elif j%11 == 3:
        A[2,5] +=float(a[j])
    elif j%11 == 4:
        A[3,5] +=float(a[j])        
    elif j%11 == 5:
        A[4,5] +=float(a[j])
    elif j%11 == 6:
        A[5,5] +=float(a[j])
    elif j%11 == 7:
        A[6,5] +=float(a[j])
    elif j%11 == 8:
        A[7,5] +=float(a[j])
    elif j%11 == 9:
        A[8,5] +=float(a[j])
    elif j%11 == 10:
        A[9,5] +=float(a[j])                                              
""" 
a = []
with open('output_64.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(11,len(a)):
    if (j)%11 == 1:
        A[0,6] += float(a[j])
    elif j%11 == 2:
        A[1,6] +=float(a[j])
    elif j%11 == 3:
        A[2,6] +=float(a[j])
    elif j%11 == 4:
        A[3,6] +=float(a[j])        
    elif j%11 == 5:
        A[4,6] +=float(a[j])
    elif j%11 == 6:
        A[5,6] +=float(a[j])
    elif j%11 == 7:
        A[6,6] +=float(a[j])
    elif j%11 == 8:
        A[7,6] +=float(a[j])
    elif j%11 == 9:
        A[8,6] +=float(a[j])
    elif j%11 == 10:
        A[9,6] +=float(a[j])                                              
 
a = []
with open('output_128.csv','r') as csvfile:
    data=csv.reader(filter(lambda row: row[0]!='#', csvfile))
    for i in data:
        a.append((i[0]))    

for j in range(11,len(a)):
    if (j)%11 == 1:
        A[0,7] += float(a[j])
    elif j%11 == 2:
        A[1,7] +=float(a[j])
    elif j%11 == 3:
        A[2,7] +=float(a[j])
    elif j%11 == 4:
        A[3,7] +=float(a[j])        
    elif j%11 == 5:
        A[4,7] +=float(a[j])
    elif j%11 == 6:
        A[5,7] +=float(a[j])
    elif j%11 == 7:
        A[6,7] +=float(a[j])
    elif j%11 == 8:
        A[7,7] +=float(a[j])
    elif j%11 == 9:
        A[8,7] +=float(a[j])
    elif j%11 == 10:
        A[9,7] +=float(a[j])                                              
""" 
A = A/20
#A = np.log10(A)
#B = np.array([22,18,15,13,11,10,9,8,7,6])
B = np.array([22**3,18**3,15**3,13**3,11**3,10**3,9**3,8**3,7**3,6**3])
#B = np.log10(B)
plt.semilogy(B,A[:,0],'r',linewidth=2,label='Analytic (1 thread)')
plt.semilogy(B,A[:,1],'g',linewidth=2,label='Analytic (2 threads)')
plt.semilogy(B,A[:,2],'b',linewidth=2,label='Analytic (4 threads)')
plt.semilogy(B,A[:,3],'k',linewidth=2,label='Analytic (8 threads)')
plt.semilogy(B,A[:,4],'y',linewidth=2,label='Analytic (16 threads)')
plt.semilogy(B,A[:,5],'c',linewidth=2,label='Analytic (32 threads)')
#plt.semilogy(B,A[:,6],'m',linewidth=2,label='Analytic (64 threads)')
#plt.semilogy(B,A[:,7],linewidth=2,label='Analytic (128 threads)')
plt.xlabel('Number of tiles',fontsize=10)
plt.ylabel('Execution time in seconds',fontsize=10)
#plt.title('Performance of Analytic Method for Linear Elastic Regularizer (one thread vs many threads)')
plt.legend(loc='best')
#plt.ylim(0, 2.5)
#plt.show()
plt.savefig('le_openmp2.eps', format='eps', dpi=100,bbox_inches = "tight",pad_inches=0.2)
       
C = np.array([1,2,4,8,16,32])
#error1=[-1*error_pmat[0,:],error_pmat[0,:]]
#error2=[-1*error_pmat[5,:],error_pmat[5,:]]
#error1=1
#error2=1
plt.figure()
#locs,labels = plt.xticks()
#tick_values = 2**np.arange(0,8)
#print(tick_values)
#plt.xticks(tick_values,[("%.0f" % x) for x in tick_values])
plt.loglog(C,A[0,:],'r',label='Number of regions = 22*22*22',linewidth=2,basex=2,basey=10)
plt.loglog(C,A[5,:],'g',label='Number of regions = 10*10*10',linewidth=2,basex=2,basey=10)
plt.loglog(C,A[7,:],'b',label='Number of regions = 8*8*8',linewidth=2,basex=2,basey=10)
plt.xlabel('Number of threads',fontsize=10)
plt.ylabel('Execution time in seconds',fontsize=10)
#plt.ylim(0, 2.5)
plt.legend(loc='best')
#plt.show()
plt.savefig('le_openmp_21.eps', format='eps', dpi=500,bbox_inches = "tight",pad_inches=0.2)
