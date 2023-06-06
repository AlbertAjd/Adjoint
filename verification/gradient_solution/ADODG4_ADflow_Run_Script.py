'''
This ADflow script runs the flow and adjoint for the ADODG4 case, the Common Research Model (CRM) wing

Objective function: CD
Design variables: Twists at eight spanwise locations and angle of attack
Mach number: 0.85
Reynolds number: 5.0e6
Lift coefficient: 0.5
Reference chord length: 1.0 m
Reference area: 3.407014 m^2
Angle of attack: 2.1837 degree
Mesh cells: 3,604,480
Reference:
Gaetan Kenway, Charles Mader, Ping He, and Joaquim Martins. 
Effective adjoint approaches in Computational Fluid Dynamics,
Progress in Aerospace Sciences, 2019

Before running this script, make sure you have installed the required codes listed in Code_Version.txt.
Download the mesh ADODG4_Multiblock_Structured_Mesh.cgns and the FFD file ADODG4_FFD.xyz along with this script

Run this command for the flow and adjoint: "mpirun -np 48 python ADODG4_Run_Script.py adjoint".
This will compute CD and its derivatives wrt to the eight twist variables and angle of attack.

Run this command to verify the adjoint derivatives: "mpirun -np 48 python ADODG4_Run_Script.py complex twist 0".
This will compute the derivatives of dCD/dEta0 using the complex step method.
Here Eta0 is the twist variable at the first spanwise location (wing root).
To compute derivatives for all the twist locations, you need to call the above command eight times,
varying the last parameter from 0 to 7.
To verify the derivatives wrt the angle of attack, run this command: "mpirun -np 48 python ADODG4_Run_Script.py complex alpha".
NOTE: To verify derivatives, make sure your flow and adjoint converge tightly, set L2TolFlow and L2TolAdj to 1e-14.
'''

import sys
import numpy
from mpi4py import MPI
from baseclasses import *
from pygeo import *
from pyspline import *

# determine if it is a complex step run
try:
    task=sys.argv[1]
except:
    print("Specify a task to run. Options are: adjoint or complex")
    exit()

if task=='complex':
    from adflow import ADFLOW_C
    from idwarp import USMesh_C
    ADFLOW = ADFLOW_C
    USMesh = USMesh_C
    CS=True
    try:
        perturbVar = sys.argv[2]
    except:
        print("The second argument not found! Options are: twist or alpha.")
        exit()
    if not perturbVar in ['twist','alpha']:
        print("The second argument not valid! Options are: twist or alpha.")
        exit()
elif task=='adjoint':
    from adflow import ADFLOW
    from idwarp import USMesh
    CS=False
else:
    print("The first argument not valid! The options are: adjoint or complex")

if CS==True:
    useNK=False
    useANK=False
    L2TolFlow=1e-14
    L2TolAdj=1e-14
    timeScheme='dadi'
    CFL=1.0
    MGStart=1
else:
    useNK=True
    useANK=True
    L2TolFlow=1e-14
    L2TolAdj=1e-14
    timeScheme='runge kutta'
    CFL=2.0
    MGStart=4

# ======================================================================
#         Input Information
# ======================================================================

if task=='complex' and perturbVar=='twist':
    try:
        perturbIdx = sys.argv[3]
    except:
        print("The third argument not found! Options are integers fromm 0 to 8.")
        exit()
    if not int(perturbIdx) in range(9):
        print("The third argument not valid! Options are integers fromm 0 to 8")
        exit()

# we have nine twists, initialize them
nTwist = 8
twistList = [0,0,0,0,0,0,0,0]

# if complex step and we want to perturb alpha, add 1e-40 to the img part
if CS and perturbVar=='alpha':
    AoA0=2.1837+1e-40j
else:
    AoA0=2.1837

# initialize the flow problem
ap = AeroProblem(name='ADODG4', mach=0.85, reynolds=5e6, reynoldsLength=1.0, T=326.45,
                 alpha=AoA0,areaRef=3.407014, chordRef=1.00,
                 xRef=1.20777, yRef=0, zRef=.007669, evalFuncs=['cd'])

# add alpha as the design variables
ap.addDV('alpha')

# ADflow options    
aeroOptions = {
        # Common Parameters
        'gridFile':'./ADODG4_Multiblock_Structured_Mesh.cgns',
        'outputDirectory':'./',
        'loadbalanceiter':50,
        
        # Physics Parameters
        'equationType':'rans',

        # Common Parameters
        'CFL':CFL,
        'CFLCoarse':0.5,
        'MGCycle':'3w',
        'MGStartLevel':MGStart,
        'nCyclesCoarse':500,
        'nCycles':100000,
        'nsubiterturb':4,
        'useblockettes':False,
        'usenksolver':useNK,
        'useanksolver':useANK,
        'usematrixfreedrdw':True,
        'smoother':timeScheme,
        # nk
        'nkadpc':True,
        'nkswitchtol':1.0e-4,
        'liftIndex': 3,

        # Convergence Parameters
        'L2Convergence':L2TolFlow,
        'L2ConvergenceCoarse':1e-4,

        # Adjoint Parameters
        'adjointL2Convergence':L2TolAdj,
        'ADPC':True,
        'adjointMaxIter': 1500,
        'adjointSubspaceSize':150,
        'ILUFill':2,
        'ASMOverlap':2,
        'outerPreconIts':3,
        }

# IDWarp options
meshOptions = {'gridFile':'./ADODG4_Multiblock_Structured_Mesh.cgns'}

# Create twist design variables
DVGeo = DVGeometry('ADODG4_FFD.xyz',complex=CS)
DVGeo.addRefAxis('wing', xFraction=.25, alignIndex='j', rotType=5)

def twist(val, geo):
    # Set all the twist values
    for i in xrange(nTwist):
        geo.rot_y['wing'].coef[i] = val[i]

# add twist design variables
DVGeo.addGeoDVGlobal('twist',twistList,twist,lower=-10.0, upper=10.0, scale=1.0)

# if we run complex step for twist DVs, add 1e-40 perturbation here
if task=='complex' and perturbVar=='twist':
    twistList[int(perturbIdx)] = 0+1e-40j
    DVGeo.setDesignVars({'twist':twistList})

# Create solver
comm = MPI.COMM_WORLD
CFDSolver = ADFLOW(options=aeroOptions, comm=comm)
CFDSolver.setDVGeo(DVGeo)
mesh = USMesh(options=meshOptions, comm=comm)
CFDSolver.setMesh(mesh)

# output slices
span = 3.758150834
pos = numpy.array([0.0235, 0.267, 0.557, 0.695, 0.828, 0.944])*span
CFDSolver.addSlices('y', pos, sliceType='absolute')

# call the flow
funcs={}
CFDSolver(ap)
CFDSolver.evalFunctions(ap,funcs)
if comm.rank==0:
    print ("\nComputed objective function value:")
    print (funcs)

# call the adjoint    
if not CS:
    funcsSens = {}
    CFDSolver.evalFunctionsSens(ap, funcsSens, ['cd'])

    if comm.rank==0:
        print ("\nComputed objective function derivatives:")
        numpy.set_printoptions(precision=16)
        print (funcsSens)

    CFDSolver.writeSurfaceSensitivity('sensitivity.dat', 'cd')


