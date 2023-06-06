# rst Imports
import numpy as np
import argparse #用于命令行操作
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

#命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="M6_analysis") #在当前文件夹下创建“output”文件夹
parser.add_argument("--gridFile", type=str, default="M6.cgns")    #网格文件，测试的时候用粗网格，是否需要网格无关性验证?
parser.add_argument("--task", choices=["analysis", "polar"], default="analysis")    #任务为流场求解，不求极曲线
args = parser.parse_args()

#创建output目录
comm = MPI.COMM_WORLD
if not os.path.exists(args.output):
    if comm.rank == 0:
        os.mkdir(args.output)

# rst ADflow options
aeroOptions = {
    # I/O Parameters
    "gridFile": args.gridFile,
    "outputDirectory": args.output,
    "monitorvariables": ["resrho", "resturb","cl", "cd"],   #监测 ρ和湍流模型的残差值以及cl、cd
    "writeTecplotSurfaceSolution": True,
   
   # Physics Parameters
    "equationType": "RANS",            #默认的湍流模型为SA模型
    "liftIndex": 2,                    #升力所在方向，1-x、2-y、3-z。ADflow默认的来流方向是怎样的？
    
    # Multigrid Solver Parameters：choose a basic solver (DADI or Runge Kutta) and set the CFL and multigrid parameters.
    #"MGCycle": "3w",                   #medium和fine有从好到差L0、L1、L2、L3四个级别，coarse只有L0、L1、L2三个级别。网格节点需要满足2^n+1。adflow从L1开始算
    #"smoother": "DADI",                #Multigrid技术默认使用D3ADI作为smoother，比Runge-Kutta更快，但可能鲁棒性更低。
    #"MGStartLevel":1,                #默认值，多重网格技术从最差的网格开始，但是对于RANS，通常不能从最差的网格开始，尤其是最差的网格数量很少的时候。
   
   # ANK Solver Parameters
    "useANKSolver": True,               #流场中存在流动分离，故使用ANK
    "ANKSwitchTol":1.0,                 #若使用ANK，则默认在相对收敛值为1，即最开始就使用ANK求解。但如果最粗网格可用且开启了多重网格技术，则还是以多重网格启动
    #"ANKCFL0":5.0,                     #默认ANK的CFL=5.0，可在收敛比较稳定时自适应提高CFL。
    #"ANKCFLLimit":1e5,                 #ANKCFL的默认上限值，更高的CFL非线性收敛更快，但是线性求解成本增大、鲁棒性降低
    #"ANKLinearSolveTol":0.05,          #默认为0.05，采用成本更低的非线性迭代来避免精确求解线性系统，需将线性残值降至相对收敛值为0.05，过大会造成不收敛。
    #"ANKPhysicalLSTol":0.2,            #默认为0.2，保证每个单元的密度和能量不超过原来的20%
    #"ANKPhysicalLSTolTurb":0.99,       #默认为0.99，湍流模型变量在每次迭代时只能减少99%，保证始终为正值。
    #"ANKSecondOrdSwitchTol": 1.5e-3,    #默认为1e-16，表示禁用，即默认近似求解Jacobian。在x阶收敛前近似求解Jacobian，后续准确求解。虽然近似雅克比矩阵精度低、非线性收敛慢，但更易于数值求解，故最好使用近似方法。近似在前3-4阶收敛更快，最佳转换点取决于试，精确求解可能会增加线性残差导致失败。
    "nSubiterTurb":3,                  #解耦模式（分开求解流场变量和湍流模型变量）下ANK使用turbDADI，默认值为3，通常为3~7，但是对于复杂情况可能为10。
    #"ANKCoupleSwitchTol":1e-16,        #耦合模式默认关闭，因为可能为造成收敛失败
    #"turbResScale":1e4,                #SA模型默认为1e4，将湍流模型残差放大到与流场变量的残差保持同一数量级，但不建议修改。
    
    # NK Solver Parameters
    "useNKSolver": True,                #最后收敛阶段使用NK，条件不错下，一次非线性收敛可达到2-3阶收敛。
    "NKSwitchTol": 1e-4,                #对于RANS，默认在4阶收敛时开启NK
    "NKLS": "cubic",                    #使用三次线搜索保证总残差减小，建议使用默认值cubic
    #"NKLinearSolveTol": 0.3,           #使用牛顿法近似求解线性系统的初始tolerance，默认为0.3，最高为0.8。若Lin Res长期为0.8，需要减小“NKSwitchTol”重试。
    #"NKUseEW":True,                    #默认使用EW算法，第一次以'NKLinearSolveTol'为tolerance，接下来据情况选择最佳线性tolerance。
    #"NKSubspaceSize": 60,              #NK求解器的GMRES子空间大小。Iter Tot的变化量不能大于该值。对于困难的问题，可以通过增加该值来提高收敛性，但要牺牲更多的内存。
    #"NKADPC": False,                   #默认使用有限差分计算Jacobian，可能因为有限差分的不准确性导致即使preconditioner很强也无法提高线性求解的性能。可将"NKADPC"打开，使用自动微分法计算解析偏导数，但成本显著增加。
    
    # Termination Criteria  
    "L2Convergence": 1e-9,              #降到6-12阶相对收敛即可，默认值1e-8
    #"L2ConvergenceCoarse": 1e-2,        #默认值1e-2，多重网格技术启动时粗网格上的收敛因子      
    "nCycles": 10000,                   #最大迭代步数
}
# rst Start ADflow
# Create solver
CFDSolver = ADFLOW(options=aeroOptions) #将刚设置的aeroOptions传进来

# Add features
CFDSolver.addLiftDistribution(150, "z") #用沿展长方向的150~250个点拟合升力、阻力等参数的分布曲线，所得.dat文件中共有26个变量，每个变量150个点
CFDSolver.addSlices("z", [0.23926,0.526372,0.777595,0.95704,1.07667,1.136485])#沿展长（y）方向创建10个等距截面，将翼型坐标和 cp 分布写入文本文件

# rst Create AeroProblem
ap = AeroProblem('m6_tunnel', mach=0.8395, reynolds=11.72e6, reynoldsLength=0.64607, areaRef=0.772893541, chordRef=0.64607, alpha=3.06, T=255.56) #采用半模计算，故参考面积0.1453/2 m²
# rst Run ADflow
if args.task == "analysis":
    # Solve
    CFDSolver(ap)
    # rst Evaluate and print
    funcs = {}
    CFDSolver.evalFunctions(ap, funcs)  #ADflow按规定填充funcs字典
    # Print the evaluated functions
    if comm.rank == 0:
        print(funcs)
# rst Create polar arrays
elif args.task == "polar":
    # Create an array of alpha values.
    # In this case we create 6 evenly spaced values from 0 - 5.
    alphaList = np.linspace(0, 5, 6)    #通过不同迎角得到极曲线

    # Create storage for the evaluated lift and drag coefficients
    CLList = []
    CDList = []
    # rst Start loop
    # Loop over the alpha values and evaluate the polar
    for alpha in alphaList:
        # rst update AP
        # Update the name in the AeroProblem. This allows us to modify the
        # output file names with the current alpha.
        ap.name = f"wing_body_{alpha:4.2f}"  #保留两位小数点，每个输出的数占4个位；如wing_0.80

        # Update the alpha in aero problem and print it to the screen.
        ap.alpha = alpha
        if comm.rank == 0:
            print(f"current alpha: {ap.alpha}")

        # rst Run ADflow polar
        # Solve the flow
        CFDSolver(ap)

        # Evaluate functions
        funcs = {}
        CFDSolver.evalFunctions(ap, funcs)

        # Store the function values in the output list
        CLList.append(funcs[f"{ap.name}_cl"])   #固定格式？怎么来的？
        CDList.append(funcs[f"{ap.name}_cd"])

    # rst Print polar
    # Print the evaluated functions in a table
    if comm.rank == 0:
        print("{:>6} {:>8} {:>8}".format("Alpha", "CL", "CD"))  #左对齐占8位
        print("=" * 24)
        for alpha, cl, cd in zip(alphaList, CLList, CDList):    #相对应位置的元素打包成元组
            print(f"{alpha:6.1f} {cl:8.4f} {cd:8.4f}")
''' 
默认为analysis，命令为：$ mpirun -np 4 python aero_run.py
极曲线（polar），命令为：$ mpirun -np 4 python aero_run.py --task polar --output polar
adflow默认网格单位为m
'''
