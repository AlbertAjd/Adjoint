# ======================================================================
#         导入模块Python库
# ======================================================================
import os
import numpy as np
import argparse
import ast
from mpi4py import MPI
from baseclasses import AeroProblem
from adflow import ADFLOW
from pygeo import DVGeometry, DVConstraints, geo_utils
from pyoptsparse import Optimization, OPT
from idwarp import USMesh
from multipoint import multiPointSparse

# 使用Python参数解析器设置命令行操作
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="wb_opt_solutions")
parser.add_argument("--opt", type=str, default="SLSQP", choices=["SLSQP", "SNOPT"])#可在多个优化器中选择
parser.add_argument("--gridFile", type=str, default="wb_vol_3.37m.cgns")
parser.add_argument("--optOptions", type=ast.literal_eval, default={}, help="additional optimizer options to be added")
args = parser.parse_args()

# ======================================================================
#         定义用于优化的参数
# ======================================================================
# cL 约束
mycl = 0.51

# ======================================================================
#         创建多点通信对象
# ======================================================================
MP = multiPointSparse(MPI.COMM_WORLD)
MP.addProcessorSet("wb_cruise", nMembers=1, memberSizes=MPI.COMM_WORLD.size)#由于是单点优化，所以处理器集数为1
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
if not os.path.exists(args.output):
    if comm.rank == 0:
        os.mkdir(args.output)

# ======================================================================
#         ADflow 参数设置
# ======================================================================
aeroOptions = {
    # I/O 参数
    "gridFile": args.gridFile,
    #"restartFile":args.gridFile,#热启动
    "outputDirectory": args.output,
    "monitorvariables": ["CPU", "resrho", "resturb","cl", "cd"],   #监测 ρ和湍流模型的残差值以及cl、cd
    "writeTecplotSurfaceSolution": True,
    "writeVolumeSolution": True,       #写出流场求解体网格求解文件，包含迭代信息
    
   # 物理参数
    "equationType": "RANS",            #默认的湍流模型为SA模型
    "liftIndex": 3,                    #升力所在方向，1-x、2-y、3-z。ADflow默认的来流方向为x正向
    "meshSurfaceFamily":'wing',        #'wing'由addFamilyGroup()设置，用于后续网格变形和曲面输出相关操作
    "designSurfaceFamily":'wing',      #定义设计表面，参与几何变形
    
    # 多重网格技术：选择DADI或者Ruge Kutta进行残差光顺
    #"MGCycle": "3w",                   #medium和fine有从好到差L0、L1、L2、L3四个级别，coarse只有L0、L1、L2三个级别。网格节点需要满足2^n+1。adflow从L1开始算
    #"smoother": "DADI",                #Multigrid技术默认使用D3ADI作为smoother，比Runge-Kutta更快，但可能鲁棒性更低。
    #"MGStartLevel":1,                  #默认值，多重网格技术从最差的网格开始，但是对于RANS，通常不能从最差的网格开始，尤其是最差的网格数量很少的时候。
   
   # ANK 求解器参数
    "useANKSolver": True,               #流场中存在流动分离，故使用ANK
    "ANKSwitchTol":1.0,                 #若使用ANK，则默认在相对收敛值为1，即最开始就使用ANK求解。但如果最粗网格可用且开启了多重网格技术，则还是以多重网格启动
    #"ANKCFL0":5.0,                     #默认ANK的CFL=5.0，可在收敛比较稳定时自适应提高CFL。
    #"ANKCFLLimit":1e5,                 #ANKCFL的默认上限值，更高的CFL非线性收敛更快，但是线性求解成本增大、鲁棒性降低
    #"ANKLinearSolveTol":0.05,          #默认为0.05，采用成本更低的非线性迭代来避免精确求解线性系统，需将线性残值降至相对收敛值为0.05，过大会造成不收敛。
    #"ANKPhysicalLSTol":0.2,            #默认为0.2，线搜索中保证每个单元的密度和能量不超过原来的20%
    #"ANKPhysicalLSTolTurb":0.99,       #默认为0.99，湍流模型变量在每次迭代时只能减少99%，保证始终为正值。
    "ANKSecondOrdSwitchTol": 1.5e-3,    #默认为1e-16，表示禁用，即默认近似求解雅可比矩阵。在x阶收敛前近似求解雅可比，后续准确求解。虽然近似雅克比矩阵精度低、非线性收敛慢，但更易于数值求解，故最好使用近似方法。近似在前3-4阶收敛更快，最佳转换点取决于试，精确求解可能会增加线性残差导致失败。
    "nSubiterTurb":10,                  #解耦模式（分开求解流场变量和湍流模型变量）下ANK使用turbDADI，默认值为3，通常为3~7，但是对于复杂情况可能为10。
    #"ANKCoupleSwitchTol":1e-16,        #耦合模式默认关闭，因为可能为造成收敛失败
    #"turbResScale":1e4,                #SA模型默认为1e4，将湍流模型残差放大到与流场变量的残差保持同一数量级，但不建议修改。
    
    # NK 求解器参数
    "useNKSolver": True,                #最后收敛阶段使用NK，条件不错下，一次非线性收敛可达到2-3阶收敛。
    "NKSwitchTol": 2e-5,                #对于RANS，默认在4阶收敛时开启NK，复杂案例建议在较小值启动
    "NKLS": "cubic",                    #使用三次线搜索保证总残差减小，建议使用默认值cubic
    #"NKLinearSolveTol": 0.3,           #使用牛顿法近似求解线性系统的初始容差，默认为0.3，最高为0.8。若Lin Res长期为0.8，需要减小“NKSwitchTol”重试。
    #"NKUseEW":True,                    #默认使用EW算法，第一次以'NKLinearSolveTol'为tolerance，接下来据情况选择最佳线性容差。
    "NKSubspaceSize": 400,              #NK求解器的GMRES子空间大小。“Iter Tot”的变化量不能大于该值。对于困难的问题，可以通过增加该值来提高收敛性，但要牺牲更多的内存。
    "NKADPC": True,                     #默认使用有限差分法计算雅可比矩阵，可能因为有限差分法的不准确性导致即使预处理子很强也无法提高线性求解的性能。可将"NKADPC"打开，使用自动微分法计算解析偏导数，但成本显著增加。
    "NKASMOverlap": 4,                  #NK求解器ASM重叠级别为4，增强预处理子性能
    "NKPCILUFill": 4,                   #NK求解器ILU填充级别为4，所需内存显著增大
    "NKJacobianLag": 5,                 #预处理子滞后于非线性迭代的次数，默认为20
    "NKInnerPreConIts": 3,              #内部预处理迭代次数，无需增大内存成本，每次迭代需要更多的计算。慎重修改，避免变化过大！
       
    # 伴随求解器参数 
    "adjointSolver": "GMRES",           #线性求解器中的伴随法类型：GMRES性能最好
    "adjointL2Convergence": 1e-6,       #伴随求解相对于初始残差值的相对收敛残值
    "ADPC": True,                       #对预处理子使用自动微分获得解析偏导数
    "adjointMaxIter": 1000,             #伴随求解的最大迭代次数，默认为500
    "adjointSubspaceSize": 400,         #Krylov子空间大小，默认为100
    "ILUFill": 3,                       #最大值为3，更大意味着更强的 预处理子，会有更少的（线性）迭代次数，但单个迭代将更昂贵，并消耗更多的内存,
    "ASMOverlap": 3,                    #伴随求解的加性Schwarz算法中重叠级别的数目。最大值为3，更大意味着更强的预处理子，代价是更昂贵的迭代和更多的内存。
    "outerPreconIts": 3,                #伴随求解的全局预处理迭代次数。更大可能有助于解决复杂问题。然而，每次迭代都需要更多的时间。默认值为3应该足以解决大多数问题。
    "NKOuterPreconits": 3,              #NK求解器的伴随求解的全局预处理迭代次数。更多的迭代可以帮助更快地收敛线性系统，默认值为1，最大值为3。
    "frozenTurbulence": False,          #是否在伴随中使用取消湍流假设。冻结湍流忽略了湍流模型的线性化。目前，只有 Spalart-Allmaras 模型是 ADed。使用 False 可能有助于高跨声速流动的收敛。然而，由此产生的灵敏度不太准确
    "restartADjoint": True,             #是否从以前的求解中重新启动伴随
    
    # Termination Criteria  
    "L2Convergence": 1e-9,              #降到6-12阶相对收敛即可，默认值1e-8      
    "nCycles": 10000,                   #SLSQP热启动时的最大迭代步数
    #"nCycles": 25000,                  #SNOPT热启动时的最大迭代步数,流场充分收敛防止优化失败
    
}

# 创建求解器
CFDSolver = ADFLOW(options=aeroOptions, comm=comm)
CFDSolver.addFamilyGroup(groupName='wing', families=['wingdn','wingle','wingtip','wingup'])#将网格中机翼的families添加为一组，只对机翼操作，需要在aeroOption中设置，不然参数化失败
CFDSolver.addLiftDistribution(200, "y", "wing") #用沿展长方向的150~250个点拟合升力、阻力等参数的分布曲线，所得.dat文件中共有26个变量，每个变量150个点；默认为所有壁面
CFDSolver.addSlices("y", [-0.0878475,-0.13997035,-0.19385015,-0.22079005,-0.24070215,-0.3010241,-0.3736447,-0.49604555])#将lift distribution写入dat文件

# ======================================================================
#         使用AeroProblem类快速设置流场条件 
# ======================================================================
ap = AeroProblem(name="wing_body", mach=0.75, reynolds=3.0e6, reynoldsLength=0.1412, areaRef=0.07265, chordRef=0.1412, alpha=0.49, T=300.0, evalFuncs=["cl", "cd"]) #采用半模计算，故参考面积0.1453/2 m²

# ======================================================================
#        几何设计变量设置
# ======================================================================
# 创建DVGeometry对象
FFDFile = "fitted_ffd_all.xyz"

DVGeo = DVGeometry(FFDFile,kmax=3)#B样条基函数最大次数为3，值越小影响范围越小
coords = CFDSolver.getSurfaceCoordinates('wing')#必须在ADFLOW的options里设置"meshSurfaceFamily":'wing'，不然点集没有全在FFD控制体内，会导致参数化失败，出现NaN。
DVGeo.addPointSet(coords, "wb_wing_coords")#DVGeo对机翼点集操作

#分段定义设计变量，防止网格变形失败
PS_root = geo_utils.PointSelect(psType = 'z', pt1=[0.03,-0.085,0], pt2=[0.3, -0.26, 0])#取j=2~4的FFD控制点为外形变量
DVGeo.addLocalDV("shape_root", lower=-0.001, upper=0.001, axis="z", scale=500, pointSelect=PS_root)
PS_middle = geo_utils.PointSelect(psType = 'z', pt1=[0.10,-0.27,0], pt2=[0.4, -0.43, 0])#取j=5~7的FFD控制点为外形变量
DVGeo.addLocalDV("shape_middle", lower=-0.00125, upper=0.00125, axis="z", scale=400, pointSelect=PS_middle)
PS_tip = geo_utils.PointSelect(psType = 'z', pt1=[0.2,-0.45,0], pt2=[0.4, -0.6, 0])#取j=8~10的FFD控制点为外形变量
DVGeo.addLocalDV("shape_tip", lower=-0.0002, upper=0.0002, axis="z", scale=2500, pointSelect=PS_tip)

# 将参数化的变量传给求解器
CFDSolver.setDVGeo(DVGeo)

# ======================================================================
#         容积约束
# ======================================================================
DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)#以DVGeo为基础创建几何约束

# 只有ADflow具有getTriangulatedSurface，获得三角化翼面：点-向量格式
DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface(groupName='wing'))#只在机翼内创建约束

# 容积约束
# 可通过多个角点定义，向机翼表面投影近似得到，点尽量位于一个平面且必须位于机翼内
leList = [[0.03991742, -0.07757637, 0.00786262], [0.1204299, -0.2342529, 0.02010563], [0.300878, -0.585926, 0.05232256]]#前缘三个角点的坐标
teList = [[0.2330884,-0.08151179,-0.007594167], [0.2347068, -0.2344674, 0.01767332], [0.3574723, -0.5858820, 0.05315291]]#后缘三个角点的坐标
DVCon.addVolumeConstraint(leList, teList, nSpan=50, nChord=15, lower=1.0, upper=1.15, scaled=True)

# 输出约束可视化文件
if comm.rank == 0:
    fileName = os.path.join(args.output, "vol_constraints.dat")
    DVCon.writeTecplot(fileName)
    DVCon.writeSurfaceTecplot(os.path.join(args.output,"TriangulatedSurface.dat"))

# ======================================================================
#        网格变形设置
# ======================================================================
meshOptions = {"gridFile": args.gridFile, "specifiedSurfaces":[b'wingdn',b'wingle',b'wingtip',b'wingup'],"errTol":0.0000001,}#指定机翼作为网格变形的对象，"errTol"越小，鲁棒性越高
mesh = USMesh(options=meshOptions, comm=comm, )#网格变形
CFDSolver.setMesh(mesh)#将网格对象传入CFD求解器

# ======================================================================
#         回调函数
# ======================================================================
def cruiseFuncs(x):#x为设计变量的字典
    if MPI.COMM_WORLD.rank == 0:
        print(x)
    # 设置设计变量
    DVGeo.setDesignVars(x)
    ap.setDesignVars(x)
    # 运行流场求解
    CFDSolver(ap)
    # 计算目标函数
    funcs = {}
    DVCon.evalFunctions(funcs)
    CFDSolver.evalFunctions(ap, funcs)
    CFDSolver.checkSolutionFailure(ap, funcs)#检查求解是否出错
    if MPI.COMM_WORLD.rank == 0:
        print(funcs)
    return funcs


def cruiseFuncsSens(x, funcs):
    funcsSens = {}
    DVCon.evalFunctionsSens(funcsSens)#约束函数导数
    CFDSolver.evalFunctionsSens(ap, funcsSens)#目标函数导数
    CFDSolver.checkAdjointFailure(ap, funcsSens)#伴随法求解是否出错
    if MPI.COMM_WORLD.rank == 0:
        print(funcsSens)
    return funcsSens


def objCon(funcs, printOK):#当用复数步长法时，printOK==False
    # 将目标函数和任何附加约束组合起来:
    funcs["obj"] = funcs[ap["cd"]]
    funcs["cl_con_" + ap.name] = funcs[ap["cl"]] - mycl#限制cl在给定的cl上
    if printOK:
        print("funcs in obj:", funcs)
    return funcs

# ======================================================================
#         优化问题设置
# ======================================================================
# 创建优化问题
optProb = Optimization("opt", MP.obj, comm=MPI.COMM_WORLD)

# 添加目标函数
optProb.addObj("obj", scale=1e2)

# 添加气动变量
ap.addVariablesPyOpt(optProb)

# 添加几何设计变量
DVGeo.addVariablesPyOpt(optProb)

# 添加几何约束
DVCon.addConstraintsPyOpt(optProb)
optProb.addCon("cl_con_" + ap.name, lower=0.0, scale=10.0)

# 并行求解
MP.setProcSetObjFunc("wb_cruise", cruiseFuncs)
MP.setProcSetSensFunc("wb_cruise", cruiseFuncsSens)
MP.setObjCon(objCon)
MP.setOptProb(optProb)
optProb.printSparsity()#在优化的最开始打印出约束的稀疏雅可比矩阵

# 设置优化器
if args.opt == "SLSQP":
    optOptions = {"MAXIT": 60, "ACC": 1e-4, "IFILE": os.path.join(args.output, "SLSQP.out")}
elif args.opt == "SNOPT":
    optOptions = {
        "Major feasibility tolerance": 1e-4,
        "Major optimality tolerance": 1e-4,
        "Hessian full memory": None,
        "Function precision": 1e-4,
        "Print file": os.path.join(args.output, "SNOPT_print.out"),
        "Summary file": os.path.join(args.output, "SNOPT_summary.out"),
    }
optOptions.update(args.optOptions)
opt = OPT(args.opt, options=optOptions)

# 启动优化
sol = opt(optProb, MP.sens, storeHistory=os.path.join(args.output, "opt.hst"))#迭代历史
if MPI.COMM_WORLD.rank == 0:
    print(sol)
