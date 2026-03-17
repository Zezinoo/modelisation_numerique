#!/usr/bin/python3


# *********************************************************************************************************
# nucpower.py
# *********************************************************************************************************
# This program calculates the evolution of a nuclear power plant power as a function of time
# when its reactivity is instantaneously raised from 0 (constant power) to a positive constant value at t=0
# Results of fixed-step Euler and RK4 methods as well as of an adaptative RK4 method can be compared
# to exact solutions given by Laplace transforms

from math import *
from numpy import *
import matplotlib.pyplot as plt


# *********************************************************************************************************
# Loading kinetic parameters
# *********************************************************************************************************

def read_dnparameters(filename,verbose=0):

    """
    Read Delayed Neutrons parameters from a file.
    Return arrays containing DN fractions and decay constants (/s).
    """
    
    betaList=[]
    halfList=[]
    betafile=open(filename,'r')
    for line in betafile:
        betaList.append(eval(line.split()[0]))
        halfList.append(eval(line.split()[1]))
    betafile.close()
    betaArray=array(betaList)*1.E-5
    halfArray=array(halfList)
    lamArray=log(2)/halfArray
    if verbose:
        print('beta tot = ',betaArray.sum())
        print('beta values ',betaArray,'\nlambda values ',lamArray)
	
    return betaArray,lamArray



# ***********************************************************************************
# Calculation of exact solution of the neutron population evolution with 6 groups
# Used to compare with Euler and RK4 results
# ***********************************************************************************

def numerator(p,bi,li,genTime):

    """
    Calculation of P(p).
    """
    
    ngroups=bi.size
    prod=1.
    asum=0.
    for i in range(ngroups):
        prod=prod*(p+li[i])
        prod2=1.
        for j in range(ngroups):
            if j!=i :
                prod2=prod2*(p+li[j])    
        asum=asum+bi[i]*prod2/genTime
    
    return prod+asum


def denominator(p,omegai):

    """
    Calculation of Q'(p).
    """
    
    nroots=omegai.size

    asum=0.
    for i in range(nroots):
        prod2=1.
        for j in range(nroots):
            if j!=i :
                prod2=prod2*(p-omegai[j])    
        asum=asum+prod2
    
    return asum


def nordheim(p,rho,bi,li,genTime):



def find_a_root(pmn,pmx,rho,bi,li,genTime,verbose=0):

    

def exact_pop(n0,rho,bi,li,genTime,verbose=0):



   
# *********************************************************************************************************************************
#                                             MAIN
# *********************************************************************************************************************************


n0=1.            # initial reactor power (a.u.)
rho=100.E-5      # reactivity step occuring at t=0
genTime=2.5E-5   # neutron mean generation time in second

# reading delayed neutron parameters from file
bi,li=read_dnparameters('dnu5_params.txt')
ngroups=bi.size


