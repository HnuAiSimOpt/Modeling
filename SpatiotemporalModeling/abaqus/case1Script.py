from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
import numpy as np
import time
##### Cowper-Symonds Model 
plastic_strain_stress = (
                        (131.82, 0),
                        (139.96, 0.0026),
                        (149.36, 0.0067),
                        (162.33, 0.01534),
                        (171.18, 0.02319),
                        (178.25, 0.03144),
                        (183.72, 0.04064),
                        (187.39, 0.05029),
                        (189.83, 0.06013),
                        (190.62, 0.0648),
                        (191,    0.0658),
                        )
##### hyperparameters #####
n_samples = 3000
m = 2
mesh_size2 = 20
tp = 0.016
n_fo_inter = 10
l = 400.0
density = 2.7e-9
##### training set ranges
# r1_range = [50.0, 55.0]
# r2_range = [30.0, 35.0]
# th_range = [2.0, 3.0]
# elastic_modulus_range = [71000, 81000]
# Poissons_ratio_range = [0.304, 0.322]
# velo_range = [18800, 21200]
##### testing2 set ranges
# r1_range1 = [48.0, 50.0]
# r1_range2 = [55.0, 57.0]
# r2_range1 = [28.0, 30.0]
# r2_range2 = [35.0, 37.0]
# th_range1 = [1.8, 2.0]
# th_range2 = [3.0, 3.2]
# elastic_modulus_range1 = [69000, 71000]
# elastic_modulus_range2 = [81000, 83000]
# Poissons_ratio_range1 = [0.300, 0.304]
# Poissons_ratio_range2 = [0.322, 0.326]
# velo_range1 = [18600, 18800]
# velo_range2 = [21200, 21400]
##### testing set3 ranges
r1_range1 = [46.0, 48.0]
r1_range2 = [57.0, 59.0]
r2_range1 = [26.0, 28.0]
r2_range2 = [37.0, 39.0]
th_range1 = [1.6, 1.8]
th_range2 = [3.2, 3.4]
elastic_modulus_range1 = [67000, 69000]
elastic_modulus_range2 = [83000, 85000]
Poissons_ratio_range1 = [0.296, 0.300]
Poissons_ratio_range2 = [0.326, 0.330]
velo_range1 = [18400, 18600]
velo_range2 = [21400, 21600]

root_path = 'E:/SpatiotemporalModelingCMAMERevision/case1/testing3'
abaqus_working_path = 'F:/temp/Job-1.odb'
###########################
def name(i, last=''):
    if i <= 9:
        name = '000' + str(i) + last
    elif i <= 99:
        name = '00' + str(i) + last
    elif i <= 999:
        name = '0' + str(i) + last
    else:
        name = str(i) + last
    return name

def LHSample( D,bounds,N):
    '''
    :param D: dimensionality of sample space
    :param bounds: bounds for sample set
    :param N: number of samples
    :return: sample set
    '''
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # expanding
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]
    if np.any(lower_bounds > upper_bounds):
        print('error for boundary')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result,
                       (upper_bounds - lower_bounds),
                       out=result),
           lower_bounds,
           out=result)
    return result

variables1 = LHSample(
    D=12, 
    bounds=[r1_range1, 
            r2_range1, 
            th_range1, 
            th_range1, 
            th_range1, 
            elastic_modulus_range1, 
            Poissons_ratio_range1,
            elastic_modulus_range1, 
            Poissons_ratio_range1, 
            elastic_modulus_range1, 
            Poissons_ratio_range1,  
            velo_range1], 
    N=int(n_samples/2))
variables2 = LHSample(
    D=12, 
    bounds=[r1_range2, 
            r2_range2, 
            th_range2, 
            th_range2, 
            th_range2, 
            elastic_modulus_range2, 
            Poissons_ratio_range2,
            elastic_modulus_range2, 
            Poissons_ratio_range2, 
            elastic_modulus_range2, 
            Poissons_ratio_range2,  
            velo_range2], 
    N=int(n_samples/2))
variables = np.concatenate([variables1,variables2], axis=0)
np.random.shuffle(variables)
# variables = LHSample(
#     D=12, 
#     bounds=[r1_range, 
#             r2_range, 
#             th_range, 
#             th_range, 
#             th_range, 
#             elastic_modulus_range, 
#             Poissons_ratio_range,
#             elastic_modulus_range, 
#             Poissons_ratio_range, 
#             elastic_modulus_range, 
#             Poissons_ratio_range,  
#             velo_range], 
#     N=int(n_samples))
np.save(root_path+'/design_variables.npy', variables)
# variables = np.load(r"E:\SpatiotemporalModelingCMAMERevision\case1\training\design_variables.npy")
for n_s in range(n_samples):
    r1 = variables[n_s, 0]
    r2 = variables[n_s, 1]
    th1 = variables[n_s, 2]
    th2 = variables[n_s, 3]
    th3 = variables[n_s, 4]
    elastic_modulus_r1 = variables[n_s, 5]
    Poissons_ratio_r1 = variables[n_s, 6]
    elastic_modulus_r2 = variables[n_s, 7]
    Poissons_ratio_r2 = variables[n_s, 8]
    elastic_modulus_mid = variables[n_s, 9]
    Poissons_ratio_mid = variables[n_s, 10]
    velo = variables[n_s, 11]
    s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=200.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r1, 0.0))
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r2, 0.0))
    s.Line(point1=(-r2, 0.0), point2=(-r1, 0.0))
    s.HorizontalConstraint(entity=g[4], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[4], addUndoState=False)
    s.CoincidentConstraint(entity1=v[3], entity2=g[3], addUndoState=False)
    s.CoincidentConstraint(entity1=v[4], entity2=g[2], addUndoState=False)
    s.Line(point1=(0.0, r2), point2=(0.0, r1))
    s.VerticalConstraint(entity=g[5], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[5], addUndoState=False)
    s.CoincidentConstraint(entity1=v[5], entity2=g[3], addUndoState=False)
    s.CoincidentConstraint(entity1=v[6], entity2=g[2], addUndoState=False)
    s.Line(point1=(r2, 0.0), point2=(r1, 0.0))
    s.HorizontalConstraint(entity=g[6], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[6], addUndoState=False)
    s.Line(point1=(0.0, -r2), point2=(0.0, -r1))
    s.VerticalConstraint(entity=g[7], addUndoState=False)
    s.PerpendicularConstraint(entity1=g[3], entity2=g[7], addUndoState=False)
    s.CoincidentConstraint(entity1=v[7], entity2=g[3], addUndoState=False)
    s.CoincidentConstraint(entity1=v[8], entity2=g[2], addUndoState=False)
    p = mdb.models['Model-1'].Part(name='beam', dimensionality=THREE_D, 
        type=DEFORMABLE_BODY)
    p = mdb.models['Model-1'].parts['beam']
    p.BaseShellExtrude(sketch=s, depth=l)
    s.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['beam']
    del mdb.models['Model-1'].sketches['__profile__']
    p = mdb.models['Model-1'].parts['beam']
    v1, e, d1, n = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e[3], rule=CENTER))
    s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
        sheetSize=200.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.rectangle(point1=(-100.0, 100.0), point2=(100.0, -100.0))
    p = mdb.models['Model-1'].Part(name='wall', dimensionality=THREE_D, 
        type=DISCRETE_RIGID_SURFACE)
    p = mdb.models['Model-1'].parts['wall']
    p.BaseShell(sketch=s1)
    s1.unsetPrimaryObject()
    p = mdb.models['Model-1'].parts['wall']
    del mdb.models['Model-1'].sketches['__profile__']
    p = mdb.models['Model-1'].parts['wall']
    v2, e1, d2, n1 = p.vertices, p.edges, p.datums, p.nodes
    p.ReferencePoint(point=p.InterestingPoint(edge=e1[1], rule=MIDDLE))
    mdb.models['Model-1'].Material(name='al_r1')
    mdb.models['Model-1'].materials['al_r1'].Density(table=((2.7e-09, ), ))
    mdb.models['Model-1'].materials['al_r1'].Elastic(table=((elastic_modulus_r1, Poissons_ratio_r1), ))
    mdb.models['Model-1'].materials['al_r1'].Plastic(table=plastic_strain_stress)
    mdb.models['Model-1'].Material(name='al_r2')
    mdb.models['Model-1'].materials['al_r2'].Density(table=((2.7e-09, ), ))
    mdb.models['Model-1'].materials['al_r2'].Elastic(table=((elastic_modulus_r2, Poissons_ratio_r2), ))
    mdb.models['Model-1'].materials['al_r2'].Plastic(table=plastic_strain_stress)
    mdb.models['Model-1'].Material(name='al_mid')
    mdb.models['Model-1'].materials['al_mid'].Density(table=((2.7e-09, ), ))
    mdb.models['Model-1'].materials['al_mid'].Elastic(table=((elastic_modulus_mid, Poissons_ratio_mid), ))
    mdb.models['Model-1'].materials['al_mid'].Plastic(table=plastic_strain_stress)
    p = mdb.models['Model-1'].parts['beam']
    mdb.models['Model-1'].HomogeneousShellSection(name='r1', preIntegrate=OFF, 
        material='al_r1', thicknessType=UNIFORM, thickness=th1, thicknessField='', 
        nodalThicknessField='', idealization=NO_IDEALIZATION, 
        poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT, 
        useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)
    mdb.models['Model-1'].HomogeneousShellSection(name='r2', preIntegrate=OFF, 
        material='al_r2', thicknessType=UNIFORM, thickness=th2, thicknessField='', 
        nodalThicknessField='', idealization=NO_IDEALIZATION, 
        poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT, 
        useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)
    mdb.models['Model-1'].HomogeneousShellSection(name='mid', preIntegrate=OFF, 
        material='al_mid', thicknessType=UNIFORM, thickness=th3, thicknessField='', 
        nodalThicknessField='', idealization=NO_IDEALIZATION, 
        poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT, 
        useDensity=OFF, integrationRule=SIMPSON, numIntPts=5)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f0 ]', ), )
    region = p.Set(faces=faces, name='Set-1')
    p = mdb.models['Model-1'].parts['beam']
    p.SectionAssignment(region=region, sectionName='r1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f ]', ), )
    region = p.Set(faces=faces, name='Set-2')
    p = mdb.models['Model-1'].parts['beam']
    p.SectionAssignment(region=region, sectionName='r2', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#f00 ]', ), )
    region = p.Set(faces=faces, name='Set-3')
    p = mdb.models['Model-1'].parts['beam']
    p.SectionAssignment(region=region, sectionName='mid', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    p = mdb.models['Model-1'].parts['beam']
    r = p.referencePoints
    refPoints=(r[2], )
    region=p.Set(referencePoints=refPoints, name='Set-4')
    mdb.models['Model-1'].parts['beam'].engineeringFeatures.PointMassInertia(
        name='masspt', region=region, mass=m, alpha=0.0, composite=0.0)
    a = mdb.models['Model-1'].rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    p = mdb.models['Model-1'].parts['beam']
    a.Instance(name='beam-1', part=p, dependent=ON)
    p = mdb.models['Model-1'].parts['wall']
    a.Instance(name='wall-1', part=p, dependent=ON)
    a = mdb.models['Model-1'].rootAssembly
    a.translate(instanceList=('beam-1', ), vector=(0.0, 0.0, 1.0))
    #: The instance beam-1 was translated by 0., 0., 1. with respect to the assembly coordinate system
    mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
        timePeriod=tp, improvedDtMethod=ON)
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
        'S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U', 'V', 'A', 
        'RF', 'RT', 'RM', 'CF', 'SF', 'NFORC', 'NFORCSO', 'RBFOR', 'BF', 'GRAV', 
        'P', 'HP', 'IWCONWEP', 'TRSHR', 'TRNOR', 'VP', 'STAGP', 'SBF', 'CSTRESS', 
        'ENER', 'ELEN', 'ELEDEN', 'EDCDEN', 'EDT', 'EVF', 'MVF'))
    mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(
    numIntervals=170)
    mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(
        numIntervals=n_fo_inter)
    mdb.models['Model-1'].ContactProperty('IntProp-1')
    mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
        0.2, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
        fraction=0.005, elasticSlipStiffness=None)
    #: The interaction property "IntProp-1" has been created.
    mdb.models['Model-1'].ContactExp(name='Int-1', createStepName='Step-1')
    mdb.models['Model-1'].interactions['Int-1'].includedPairs.setValuesInStep(
        stepName='Step-1', useAllstar=ON)
    mdb.models['Model-1'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
    #: The interaction "Int-1" has been created.
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.instances['beam-1'].referencePoints
    refPoints1=(r1[2], )
    region1=a.Set(referencePoints=refPoints1, name='m_Set-1')
    a = mdb.models['Model-1'].rootAssembly
    s1 = a.instances['beam-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#55942918 ]', ), )
    region2=a.Surface(side1Edges=side1Edges1, name='s_Surf-1')
    mdb.models['Model-1'].Coupling(name='Constraint-1', controlPoint=region1, 
        surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
        localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
    a = mdb.models['Model-1'].rootAssembly
    r1 = a.instances['wall-1'].referencePoints
    refPoints1=(r1[2], )
    region = a.Set(referencePoints=refPoints1, name='Set-2')
    mdb.models['Model-1'].EncastreBC(name='BC-1', createStepName='Step-1', 
        region=region, localCsys=None)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['beam-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#fff ]', ), )
    e1 = a.instances['beam-1'].edges
    edges1 = e1.getSequenceFromMask(mask=('[#559d79bd ]', ), )
    v1 = a.instances['beam-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#9699 ]', ), )
    r1 = a.instances['beam-1'].referencePoints
    refPoints1=(r1[2], )
    region = a.Set(vertices=verts1, edges=edges1, faces=faces1, 
        referencePoints=refPoints1, name='Set-3')
    mdb.models['Model-1'].Velocity(name='Predefined Field-1', region=region, 
        field='', distributionType=MAGNITUDE, velocity1=0.0, velocity2=0.0, 
        velocity3=-velo, omega=0.0)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['beam-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#600 ]', ), )
    region = a.Set(faces=faces1, name='Set-4')
    mdb.models['Model-1'].XsymmBC(name='BC-2', createStepName='Initial', 
        region=region, localCsys=None)
    a = mdb.models['Model-1'].rootAssembly
    f1 = a.instances['beam-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#900 ]', ), )
    region = a.Set(faces=faces1, name='Set-5')
    mdb.models['Model-1'].YsymmBC(name='BC-3', createStepName='Initial', 
        region=region, localCsys=None)
    p = mdb.models['Model-1'].parts['beam']
    for i in range(79):
        p = mdb.models['Model-1'].parts['beam']
        p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=i*5+5)

    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#fff ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[85], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#fa7901 ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[84], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#a2101ab3 ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[83], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#1010d8d #821 ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[82], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#80cb4b #1010 ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[81], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#c80ab4b ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[80], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#a809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[79], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[78], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[77], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[76], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[75], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[74], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[73], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[72], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[71], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[70], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[69], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[68], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[67], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[66], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[65], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[64], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[63], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[62], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[61], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[60], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[59], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[58], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[57], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[56], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[55], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[54], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[53], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[52], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[51], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[50], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[49], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[48], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[47], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[46], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[45], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[44], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[43], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[42], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[41], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[40], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[39], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[38], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[37], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[36], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[35], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[34], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[33], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[32], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[31], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[30], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[29], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[28], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[27], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[26], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[25], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[24], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[23], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[22], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[21], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[20], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[19], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[18], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[17], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[16], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[15], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[14], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[13], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[12], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[11], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[10], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[9], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d1 = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d1[8], faces=pickedFaces)
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    pickedFaces = f.getSequenceFromMask(mask=('[#9809acb ]', ), )
    d = p.datums
    p.PartitionFaceByDatumPlane(datumPlane=d[7], faces=pickedFaces)

    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #1000000 ]', ), )
    p.Set(faces=faces, name='seg1-1')
    #: The set 'seg1-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #80000 ]', ), )
    p.Set(faces=faces, name='seg2-1')
    #: The set 'seg2-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #10000000 ]', ), )
    p.Set(faces=faces, name='seg3-1')
    #: The set 'seg3-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #40 ]', ), )
    p.Set(faces=faces, name='seg4-1')
    #: The set 'seg4-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #80000000 ]', ), )
    p.Set(faces=faces, name='seg5-1')
    #: The set 'seg5-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #40000 ]', ), )
    p.Set(faces=faces, name='seg6-1')
    #: The set 'seg6-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #200000 ]', ), )
    p.Set(faces=faces, name='seg7-1')
    #: The set 'seg7-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #400000 ]', ), )
    p.Set(faces=faces, name='seg8-1')
    #: The set 'seg8-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #10000 ]', ), )
    p.Set(faces=faces, name='seg9-1')
    #: The set 'seg9-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #1 ]', ), )
    p.Set(faces=faces, name='seg10-1')
    #: The set 'seg10-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #20 ]', ), )
    p.Set(faces=faces, name='seg11-1')
    #: The set 'seg11-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #40 ]', ), )
    p.Set(faces=faces, name='seg12-1')
    #: The set 'seg12-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #1 ]', ), )
    p.Set(faces=faces, name='seg13-1')
    #: The set 'seg13-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #10000 ]', ), )
    p.Set(faces=faces, name='seg14-1')
    #: The set 'seg14-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:24 #200000 ]', ), )
    p.Set(faces=faces, name='seg15-1')
    #: The set 'seg15-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #400000 ]', ), )
    p.Set(faces=faces, name='seg16-1')
    #: The set 'seg16-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #10000 ]', ), )
    p.Set(faces=faces, name='seg17-1')
    #: The set 'seg17-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:24 #1 ]', ), )
    p.Set(faces=faces, name='seg18-1')
    #: The set 'seg18-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #20 ]', ), )
    p.Set(faces=faces, name='seg19-1')
    #: The set 'seg19-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #40 ]', ), )
    p.Set(faces=faces, name='seg20-1')
    #: The set 'seg20-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #1 ]', ), )
    p.Set(faces=faces, name='seg21-1')
    #: The set 'seg21-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #10000 ]', ), )
    p.Set(faces=faces, name='seg22-1')
    #: The set 'seg22-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:21 #200000 ]', ), )
    p.Set(faces=faces, name='seg23-1')
    #: The set 'seg23-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #400000 ]', ), )
    p.Set(faces=faces, name='seg24-1')
    #: The set 'seg24-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #10000 ]', ), )
    p.Set(faces=faces, name='seg25-1')
    #: The set 'seg25-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:21 #1 ]', ), )
    p.Set(faces=faces, name='seg26-1')
    #: The set 'seg26-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #20 ]', ), )
    p.Set(faces=faces, name='seg27-1')
    #: The set 'seg27-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #40 ]', ), )
    p.Set(faces=faces, name='seg28-1')
    #: The set 'seg28-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #1 ]', ), )
    p.Set(faces=faces, name='seg29-1')
    #: The set 'seg29-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #10000 ]', ), )
    p.Set(faces=faces, name='seg30-1')
    #: The set 'seg30-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:18 #200000 ]', ), )
    p.Set(faces=faces, name='seg31-1')
    #: The set 'seg31-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #400000 ]', ), )
    p.Set(faces=faces, name='seg32-1')
    #: The set 'seg32-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #10000 ]', ), )
    p.Set(faces=faces, name='seg33-1')
    #: The set 'seg33-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:18 #1 ]', ), )
    p.Set(faces=faces, name='seg34-1')
    #: The set 'seg34-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #20 ]', ), )
    p.Set(faces=faces, name='seg35-1')
    #: The set 'seg35-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #40 ]', ), )
    p.Set(faces=faces, name='seg36-1')
    #: The set 'seg36-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #1 ]', ), )
    p.Set(faces=faces, name='seg37-1')
    #: The set 'seg37-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #10000 ]', ), )
    p.Set(faces=faces, name='seg38-1')
    #: The set 'seg38-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:15 #200000 ]', ), )
    p.Set(faces=faces, name='seg39-1')
    #: The set 'seg39-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #400000 ]', ), )
    p.Set(faces=faces, name='seg40-1')
    #: The set 'seg40-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #10000 ]', ), )
    p.Set(faces=faces, name='seg41-1')
    #: The set 'seg41-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:15 #1 ]', ), )
    p.Set(faces=faces, name='seg42-1')
    #: The set 'seg42-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #20 ]', ), )
    p.Set(faces=faces, name='seg43-1')
    #: The set 'seg43-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #40 ]', ), )
    p.Set(faces=faces, name='seg44-1')
    #: The set 'seg44-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #1 ]', ), )
    p.Set(faces=faces, name='seg45-1')
    #: The set 'seg45-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #10000 ]', ), )
    p.Set(faces=faces, name='seg46-1')
    #: The set 'seg46-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:12 #200000 ]', ), )
    p.Set(faces=faces, name='seg47-1')
    #: The set 'seg47-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #400000 ]', ), )
    p.Set(faces=faces, name='seg48-1')
    #: The set 'seg48-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #400000 ]', ), )
    p.Set(faces=faces, name='seg49-1')
    #: The set 'seg49-1' has been created (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #10000 ]', ), )
    p.Set(faces=faces, name='seg49-1')
    #: The set 'seg49-1' has been edited (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #10000 ]', ), )
    p.Set(faces=faces, name='seg50-1')
    #: The set 'seg50-1' has been created (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:12 #1 ]', ), )
    p.Set(faces=faces, name='seg50-1')
    #: The set 'seg50-1' has been edited (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #20 ]', ), )
    p.Set(faces=faces, name='seg51-1')
    #: The set 'seg51-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #40 ]', ), )
    p.Set(faces=faces, name='seg52-1')
    #: The set 'seg52-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #1 ]', ), )
    p.Set(faces=faces, name='seg53-1')
    #: The set 'seg53-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #10000 ]', ), )
    p.Set(faces=faces, name='seg54-1')
    #: The set 'seg54-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:9 #200000 ]', ), )
    p.Set(faces=faces, name='seg55-1')
    #: The set 'seg55-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #400000 ]', ), )
    p.Set(faces=faces, name='seg56-1')
    #: The set 'seg56-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #10000 ]', ), )
    p.Set(faces=faces, name='seg57-1')
    #: The set 'seg57-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:9 #1 ]', ), )
    p.Set(faces=faces, name='seg58-1')
    #: The set 'seg58-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #20 ]', ), )
    p.Set(faces=faces, name='seg59-1')
    #: The set 'seg59-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #40 ]', ), )
    p.Set(faces=faces, name='seg60-1')
    #: The set 'seg60-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #1 ]', ), )
    p.Set(faces=faces, name='seg61-1')
    #: The set 'seg61-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #10000 ]', ), )
    p.Set(faces=faces, name='seg62-1')
    #: The set 'seg62-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:6 #200000 ]', ), )
    p.Set(faces=faces, name='seg63-1')
    #: The set 'seg63-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #400000 ]', ), )
    p.Set(faces=faces, name='seg64-1')
    #: The set 'seg64-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #10000 ]', ), )
    p.Set(faces=faces, name='seg65-1')
    #: The set 'seg65-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:6 #1 ]', ), )
    p.Set(faces=faces, name='seg66-1')
    #: The set 'seg66-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #20 ]', ), )
    p.Set(faces=faces, name='seg67-1')
    #: The set 'seg67-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#0:7 #800000 ]', ), )
    p.Set(edges=edges, name='seg68-1')
    #: The set 'seg68-1' has been created (1 edge).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #40 ]', ), )
    p.Set(faces=faces, name='seg68-1')
    #: The set 'seg68-1' has been edited (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #1 ]', ), )
    p.Set(faces=faces, name='seg69-1')
    #: The set 'seg69-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #10000 ]', ), )
    p.Set(faces=faces, name='seg70-1')
    #: The set 'seg70-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:3 #200000 ]', ), )
    p.Set(faces=faces, name='seg71-1')
    #: The set 'seg71-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #400000 ]', ), )
    p.Set(faces=faces, name='seg72-1')
    #: The set 'seg72-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #10000 ]', ), )
    p.Set(faces=faces, name='seg73-1')
    #: The set 'seg73-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:3 #1 ]', ), )
    p.Set(faces=faces, name='seg74-1')
    #: The set 'seg74-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #20 ]', ), )
    p.Set(faces=faces, name='seg75-1')
    #: The set 'seg75-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #40 ]', ), )
    p.Set(faces=faces, name='seg76-1')
    #: The set 'seg76-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #1 ]', ), )
    p.Set(faces=faces, name='seg77-1')
    #: The set 'seg77-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #10000 ]', ), )
    p.Set(faces=faces, name='seg78-1')
    #: The set 'seg78-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#200000 ]', ), )
    p.Set(faces=faces, name='seg79-1')
    #: The set 'seg79-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    p.Set(faces=faces, name='seg80-1')
    #: The set 'seg80-1' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #2000 ]', ), )
    p.Set(faces=faces, name='seg1-3')
    #: The set 'seg1-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #400000 ]', ), )
    p.Set(faces=faces, name='seg2-3')
    #: The set 'seg2-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #8 ]', ), )
    p.Set(faces=faces, name='seg3-3')
    #: The set 'seg3-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #100 ]', ), )
    p.Set(faces=faces, name='seg4-3')
    #: The set 'seg4-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #4 ]', ), )
    p.Set(faces=faces, name='seg5-3')
    #: The set 'seg5-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #80000 ]', ), )
    p.Set(faces=faces, name='seg6-3')
    #: The set 'seg6-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #40000 ]', ), )
    p.Set(faces=faces, name='seg7-3')
    #: The set 'seg7-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #1000000 ]', ), )
    p.Set(faces=faces, name='seg8-3')
    #: The set 'seg8-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #40000 ]', ), )
    p.Set(faces=faces, name='seg9-3')
    #: The set 'seg9-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #8 ]', ), )
    p.Set(faces=faces, name='seg10-3')
    #: The set 'seg10-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #4 ]', ), )
    p.Set(faces=faces, name='seg11-3')
    #: The set 'seg11-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #100 ]', ), )
    p.Set(faces=faces, name='seg12-3')
    #: The set 'seg12-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #4 ]', ), )
    p.Set(faces=faces, name='seg13-3')
    #: The set 'seg13-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #80000 ]', ), )
    p.Set(faces=faces, name='seg14-3')
    #: The set 'seg14-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:24 #40000 ]', ), )
    p.Set(faces=faces, name='seg15-3')
    #: The set 'seg15-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #1000000 ]', ), )
    p.Set(faces=faces, name='seg16-3')
    #: The set 'seg16-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #40000 ]', ), )
    p.Set(faces=faces, name='seg17-3')
    #: The set 'seg17-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:24 #8 ]', ), )
    p.Set(faces=faces, name='seg18-3')
    #: The set 'seg18-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #4 ]', ), )
    p.Set(faces=faces, name='seg19-3')
    #: The set 'seg19-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #100 ]', ), )
    p.Set(faces=faces, name='seg20-3')
    #: The set 'seg20-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #4 ]', ), )
    p.Set(faces=faces, name='seg21-3')
    #: The set 'seg21-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #80000 ]', ), )
    p.Set(faces=faces, name='seg22-3')
    #: The set 'seg22-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:21 #40000 ]', ), )
    p.Set(faces=faces, name='seg23-3')
    #: The set 'seg23-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #1000000 ]', ), )
    p.Set(faces=faces, name='seg24-3')
    #: The set 'seg24-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #40000 ]', ), )
    p.Set(faces=faces, name='seg25-3')
    #: The set 'seg25-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:21 #8 ]', ), )
    p.Set(faces=faces, name='seg26-3')
    #: The set 'seg26-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #4 ]', ), )
    p.Set(faces=faces, name='seg27-3')
    #: The set 'seg27-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #100 ]', ), )
    p.Set(faces=faces, name='seg28-3')
    #: The set 'seg28-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #4 ]', ), )
    p.Set(faces=faces, name='seg29-3')
    #: The set 'seg29-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #80000 ]', ), )
    p.Set(faces=faces, name='seg30-3')
    #: The set 'seg30-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:18 #40000 ]', ), )
    p.Set(faces=faces, name='seg31-3')
    #: The set 'seg31-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #1000000 ]', ), )
    p.Set(faces=faces, name='seg32-3')
    #: The set 'seg32-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #40000 ]', ), )
    p.Set(faces=faces, name='seg33-3')
    #: The set 'seg33-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:18 #8 ]', ), )
    p.Set(faces=faces, name='seg34-3')
    #: The set 'seg34-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #4 ]', ), )
    p.Set(faces=faces, name='seg35-3')
    #: The set 'seg35-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #100 ]', ), )
    p.Set(faces=faces, name='seg36-3')
    #: The set 'seg36-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #4 ]', ), )
    p.Set(faces=faces, name='seg37-3')
    #: The set 'seg37-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #80000 ]', ), )
    p.Set(faces=faces, name='seg38-3')
    #: The set 'seg38-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:15 #40000 ]', ), )
    p.Set(faces=faces, name='seg39-3')
    #: The set 'seg39-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #1000000 ]', ), )
    p.Set(faces=faces, name='seg40-3')
    #: The set 'seg40-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #40000 ]', ), )
    p.Set(faces=faces, name='seg41-3')
    #: The set 'seg41-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:15 #8 ]', ), )
    p.Set(faces=faces, name='seg42-3')
    #: The set 'seg42-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #4 ]', ), )
    p.Set(faces=faces, name='seg43-3')
    #: The set 'seg43-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #100 ]', ), )
    p.Set(faces=faces, name='seg44-3')
    #: The set 'seg44-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #4 ]', ), )
    p.Set(faces=faces, name='seg45-3')
    #: The set 'seg45-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #80000 ]', ), )
    p.Set(faces=faces, name='seg46-3')
    #: The set 'seg46-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:12 #40000 ]', ), )
    p.Set(faces=faces, name='seg47-3')
    #: The set 'seg47-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #1000000 ]', ), )
    p.Set(faces=faces, name='seg48-3')
    #: The set 'seg48-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #40000 ]', ), )
    p.Set(faces=faces, name='seg49-3')
    #: The set 'seg49-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:12 #8 ]', ), )
    p.Set(faces=faces, name='seg50-3')
    #: The set 'seg50-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #4 ]', ), )
    p.Set(faces=faces, name='seg51-3')
    #: The set 'seg51-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #100 ]', ), )
    p.Set(faces=faces, name='seg52-3')
    #: The set 'seg52-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #4 ]', ), )
    p.Set(faces=faces, name='seg53-3')
    #: The set 'seg53-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #80000 ]', ), )
    p.Set(faces=faces, name='seg54-3')
    #: The set 'seg54-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:9 #40000 ]', ), )
    p.Set(faces=faces, name='seg55-3')
    #: The set 'seg55-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #40000 ]', ), )
    p.Set(faces=faces, name='seg57-3')
    #: The set 'seg57-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #1000000 ]', ), )
    p.Set(faces=faces, name='seg56-3')
    #: The set 'seg56-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:9 #8 ]', ), )
    p.Set(faces=faces, name='seg58-3')
    #: The set 'seg58-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #4 ]', ), )
    p.Set(faces=faces, name='seg59-3')
    #: The set 'seg59-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #100 ]', ), )
    p.Set(faces=faces, name='seg60-3')
    #: The set 'seg60-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #4 ]', ), )
    p.Set(faces=faces, name='seg61-3')
    #: The set 'seg61-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #80000 ]', ), )
    p.Set(faces=faces, name='seg62-3')
    #: The set 'seg62-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:6 #40000 ]', ), )
    p.Set(faces=faces, name='seg63-3')
    #: The set 'seg63-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #1000000 ]', ), )
    p.Set(faces=faces, name='seg64-3')
    #: The set 'seg64-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #40000 ]', ), )
    p.Set(faces=faces, name='seg65-3')
    #: The set 'seg65-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:6 #8 ]', ), )
    p.Set(faces=faces, name='seg66-3')
    #: The set 'seg66-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #4 ]', ), )
    p.Set(faces=faces, name='seg67-3')
    #: The set 'seg67-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #100 ]', ), )
    p.Set(faces=faces, name='seg68-3')
    #: The set 'seg68-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #4 ]', ), )
    p.Set(faces=faces, name='seg69-3')
    #: The set 'seg69-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #80000 ]', ), )
    p.Set(faces=faces, name='seg70-3')
    #: The set 'seg70-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:3 #40000 ]', ), )
    p.Set(faces=faces, name='seg71-3')
    #: The set 'seg71-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #1000000 ]', ), )
    p.Set(faces=faces, name='seg72-3')
    #: The set 'seg72-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #4 ]', ), )
    p.Set(faces=faces, name='seg73-3')
    #: The set 'seg73-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #100 ]', ), )
    p.Set(faces=faces, name='seg74-3')
    #: The set 'seg74-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #100 ]', ), )
    p.Set(faces=faces, name='seg75-3')
    #: The set 'seg75-3' has been created (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #40000 ]', ), )
    p.Set(faces=faces, name='seg73-3')
    #: The set 'seg73-3' has been edited (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:3 #8 ]', ), )
    p.Set(faces=faces, name='seg74-3')
    #: The set 'seg74-3' has been edited (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #4 ]', ), )
    p.Set(faces=faces, name='seg75-3')
    #: The set 'seg75-3' has been edited (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #100 ]', ), )
    p.Set(faces=faces, name='seg76-3')
    #: The set 'seg76-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #4 ]', ), )
    p.Set(faces=faces, name='seg77-3')
    #: The set 'seg77-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #80000 ]', ), )
    p.Set(faces=faces, name='seg78-3')
    #: The set 'seg78-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#40000 ]', ), )
    p.Set(faces=faces, name='seg79-3')
    #: The set 'seg79-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#8 ]', ), )
    p.Set(faces=faces, name='seg80-3')
    #: The set 'seg80-3' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #20000 ]', ), )
    p.Set(faces=faces, name='seg1-2')
    #: The set 'seg1-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:29 #20000000 ]', ), )
    p.Set(faces=faces, name='seg2-2')
    #: The set 'seg2-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #20000000 ]', ), )
    p.Set(faces=faces, name='seg3-2')
    #: The set 'seg3-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #10000 ]', ), )
    p.Set(faces=faces, name='seg4-2')
    #: The set 'seg4-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #1000 ]', ), )
    p.Set(faces=faces, name='seg5-2')
    #: The set 'seg5-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #20000 ]', ), )
    p.Set(faces=faces, name='seg6-2')
    #: The set 'seg6-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:28 #8 ]', ), )
    p.Set(faces=faces, name='seg7-2')
    #: The set 'seg7-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:27 #2 ]', ), )
    p.Set(faces=faces, name='seg8-2')
    #: The set 'seg8-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #8000000 ]', ), )
    p.Set(faces=faces, name='seg9-2')
    #: The set 'seg9-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #2 ]', ), )
    p.Set(faces=faces, name='seg10-2')
    #: The set 'seg10-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #20000 ]', ), )
    p.Set(faces=faces, name='seg11-2')
    #: The set 'seg11-2' has been created (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:26 #80000 ]', ), )
    p.Set(faces=faces, name='seg11-2')
    #: The set 'seg11-2' has been edited (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #20000 ]', ), )
    p.Set(faces=faces, name='seg12-2')
    #: The set 'seg12-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #800 ]', ), )
    p.Set(faces=faces, name='seg13-2')
    #: The set 'seg13-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:24 #20000 ]', ), )
    p.Set(faces=faces, name='seg14-2')
    #: The set 'seg14-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:25 #8 ]', ), )
    p.Set(faces=faces, name='seg15-2')
    #: The set 'seg15-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:24 #2 ]', ), )
    p.Set(faces=faces, name='seg16-2')
    #: The set 'seg16-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #8000000 ]', ), )
    p.Set(faces=faces, name='seg17-2')
    #: The set 'seg17-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #2 ]', ), )
    p.Set(faces=faces, name='seg18-2')
    #: The set 'seg18-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:23 #80000 ]', ), )
    p.Set(faces=faces, name='seg19-2')
    #: The set 'seg19-2' has been created (1 face).
    p1 = mdb.models['Model-1'].parts['beam']
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #20000 ]', ), )
    p.Set(faces=faces, name='seg20-2')
    #: The set 'seg20-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #800 ]', ), )
    p.Set(faces=faces, name='seg21-2')
    #: The set 'seg21-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:21 #20000 ]', ), )
    p.Set(faces=faces, name='seg22-2')
    #: The set 'seg22-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:22 #8 ]', ), )
    p.Set(faces=faces, name='seg23-2')
    #: The set 'seg23-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:21 #2 ]', ), )
    p.Set(faces=faces, name='seg24-2')
    #: The set 'seg24-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #8000000 ]', ), )
    p.Set(faces=faces, name='seg25-2')
    #: The set 'seg25-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #2 ]', ), )
    p.Set(faces=faces, name='seg26-2')
    #: The set 'seg26-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:20 #80000 ]', ), )
    p.Set(faces=faces, name='seg27-2')
    #: The set 'seg27-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #20000 ]', ), )
    p.Set(faces=faces, name='seg28-2')
    #: The set 'seg28-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #800 ]', ), )
    p.Set(faces=faces, name='seg29-2')
    #: The set 'seg29-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:18 #20000 ]', ), )
    p.Set(faces=faces, name='seg30-2')
    #: The set 'seg30-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:19 #8 ]', ), )
    p.Set(faces=faces, name='seg31-2')
    #: The set 'seg31-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:18 #2 ]', ), )
    p.Set(faces=faces, name='seg32-2')
    #: The set 'seg32-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #8000000 ]', ), )
    p.Set(faces=faces, name='seg33-2')
    #: The set 'seg33-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #2 ]', ), )
    p.Set(faces=faces, name='seg34-2')
    #: The set 'seg34-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:17 #80000 ]', ), )
    p.Set(faces=faces, name='seg35-2')
    #: The set 'seg35-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #20000 ]', ), )
    p.Set(faces=faces, name='seg36-2')
    #: The set 'seg36-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #800 ]', ), )
    p.Set(faces=faces, name='seg37-2')
    #: The set 'seg37-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:15 #20000 ]', ), )
    p.Set(faces=faces, name='seg38-2')
    #: The set 'seg38-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:16 #8 ]', ), )
    p.Set(faces=faces, name='seg39-2')
    #: The set 'seg39-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:15 #2 ]', ), )
    p.Set(faces=faces, name='seg40-2')
    #: The set 'seg40-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #8000000 ]', ), )
    p.Set(faces=faces, name='seg41-2')
    #: The set 'seg41-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #2 ]', ), )
    p.Set(faces=faces, name='seg42-2')
    #: The set 'seg42-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:14 #80000 ]', ), )
    p.Set(faces=faces, name='seg43-2')
    #: The set 'seg43-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #20000 ]', ), )
    p.Set(faces=faces, name='seg44-2')
    #: The set 'seg44-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #800 ]', ), )
    p.Set(faces=faces, name='seg45-2')
    #: The set 'seg45-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:12 #20000 ]', ), )
    p.Set(faces=faces, name='seg46-2')
    #: The set 'seg46-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:13 #8 ]', ), )
    p.Set(faces=faces, name='seg47-2')
    #: The set 'seg47-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:12 #2 ]', ), )
    p.Set(faces=faces, name='seg48-2')
    #: The set 'seg48-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #8000000 ]', ), )
    p.Set(faces=faces, name='seg49-2')
    #: The set 'seg49-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #2 ]', ), )
    p.Set(faces=faces, name='seg50-2')
    #: The set 'seg50-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:11 #80000 ]', ), )
    p.Set(faces=faces, name='seg51-2')
    #: The set 'seg51-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #20000 ]', ), )
    p.Set(faces=faces, name='seg52-2')
    #: The set 'seg52-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #800 ]', ), )
    p.Set(faces=faces, name='seg53-2')
    #: The set 'seg53-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:9 #20000 ]', ), )
    p.Set(faces=faces, name='seg54-2')
    #: The set 'seg54-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:10 #8 ]', ), )
    p.Set(faces=faces, name='seg55-2')
    #: The set 'seg55-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:9 #2 ]', ), )
    p.Set(faces=faces, name='seg56-2')
    #: The set 'seg56-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #8000000 ]', ), )
    p.Set(faces=faces, name='seg57-2')
    #: The set 'seg57-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #2 ]', ), )
    p.Set(faces=faces, name='seg58-2')
    #: The set 'seg58-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:8 #80000 ]', ), )
    p.Set(faces=faces, name='seg59-2')
    #: The set 'seg59-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #20000 ]', ), )
    p.Set(faces=faces, name='seg60-2')
    #: The set 'seg60-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #800 ]', ), )
    p.Set(faces=faces, name='seg61-2')
    #: The set 'seg61-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:6 #20000 ]', ), )
    p.Set(faces=faces, name='seg62-2')
    #: The set 'seg62-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:7 #8 ]', ), )
    p.Set(faces=faces, name='seg63-2')
    #: The set 'seg63-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:6 #2 ]', ), )
    p.Set(faces=faces, name='seg64-2')
    #: The set 'seg64-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #8000000 ]', ), )
    p.Set(faces=faces, name='seg65-2')
    #: The set 'seg65-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #2 ]', ), )
    p.Set(faces=faces, name='seg66-2')
    #: The set 'seg66-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:5 #80000 ]', ), )
    p.Set(faces=faces, name='seg67-2')
    #: The set 'seg67-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #20000 ]', ), )
    p.Set(faces=faces, name='seg68-2')
    #: The set 'seg68-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #800 ]', ), )
    p.Set(faces=faces, name='seg69-2')
    #: The set 'seg69-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:3 #20000 ]', ), )
    p.Set(faces=faces, name='seg70-2')
    #: The set 'seg70-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:4 #8 ]', ), )
    p.Set(faces=faces, name='seg71-2')
    #: The set 'seg71-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:3 #2 ]', ), )
    p.Set(faces=faces, name='seg72-2')
    #: The set 'seg72-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #8000000 ]', ), )
    p.Set(faces=faces, name='seg73-2')
    #: The set 'seg73-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #2 ]', ), )
    p.Set(faces=faces, name='seg74-2')
    #: The set 'seg74-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0:2 #80000 ]', ), )
    p.Set(faces=faces, name='seg75-2')
    #: The set 'seg75-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #20000 ]', ), )
    p.Set(faces=faces, name='seg76-2')
    #: The set 'seg76-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #800 ]', ), )
    p.Set(faces=faces, name='seg77-2')
    #: The set 'seg77-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#20000 ]', ), )
    p.Set(faces=faces, name='seg78-2')
    #: The set 'seg78-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#0 #8 ]', ), )
    p.Set(faces=faces, name='seg79-2')
    #: The set 'seg79-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#2 ]', ), )
    p.Set(faces=faces, name='seg80-2')
    #: The set 'seg80-2' has been created (1 face).
    p = mdb.models['Model-1'].parts['beam']
    p.seedPart(size=10.0, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-1'].parts['beam']
    p.generateMesh()
    a = mdb.models['Model-1'].rootAssembly
    a.regenerate()
    p = mdb.models['Model-1'].parts['wall']
    p = mdb.models['Model-1'].parts['wall']
    p.seedPart(size=20.0, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Model-1'].parts['wall']
    p.generateMesh()
    myjob = mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS,
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90,
        memoryUnits=PERCENTAGE, explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF,
        contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='',
        resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=6,
        activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=6)
    res = mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
    myjob.waitForCompletion()
    # shutil.copyfile('F:/temp/Job-1.odb', root_path+'/odbs/'+name(n_s, '.odb'))
    print('start to extract dataset from odb file')
    try:
        current_odb = session.openOdb(name=abaqus_working_path)
        part = current_odb.rootAssembly.instances['BEAM-1']
        segs = part.elementSets
        segs_keys = segs.keys()[:-4]
        # np.save(path + 'seg_keys.npy', segs_keys)
        ###### calculating the total mass of each segmentaion ###### 
        frames = current_odb.steps['Step-1'].frames
        frame100 = frames[100]
        ELSE = frame100.fieldOutputs['ELSE']
        ESEDEN = frame100.fieldOutputs['ESEDEN']
        mass_list = []
        for seg_key in segs_keys:
            seg = segs[seg_key]
            seg_ELSE = ELSE.getSubset(region=seg)
            seg_ESEDEN = ESEDEN.getSubset(region=seg)
            n_elements = len(seg_ELSE.values)
            seg_volume = 0
            for i in range(n_elements):
                seg_volume += seg_ELSE.values[i].data/seg_ESEDEN.values[i].data
            mass_list.append(seg_volume * density)
        ###### extracting the SEA ######
        st_inten = []
        for frame in frames:
            ELASE = frame.fieldOutputs['ELASE']
            ELSE = frame.fieldOutputs['ELSE']
            ELPD = frame.fieldOutputs['ELPD']
            ELCD = frame.fieldOutputs['ELCD']
            easeden = frame.fieldOutputs['EASEDEN']
            ecdden = frame.fieldOutputs['ECDDEN']
            epdden = frame.fieldOutputs['EPDDEN']
            eseden = frame.fieldOutputs['ESEDEN']
            SEG_INTEN = []
            for j, seg_key in enumerate(segs_keys):
                seg = segs[seg_key]
                seg_easeden = easeden.getSubset(region=seg)
                seg_ecdden = ecdden.getSubset(region=seg)
                seg_epdden = epdden.getSubset(region=seg)
                seg_eseden = eseden.getSubset(region=seg)
                seg_ELASE = ELASE.getSubset(region=seg)
                seg_ELSE = ELSE.getSubset(region=seg)
                seg_ELPD = ELPD.getSubset(region=seg)
                seg_ELCD = ELCD.getSubset(region=seg)
                n_elements = len(seg_ELSE.values)
                seg_int_e = 0
                for i in range(n_elements):
                    seg_int_e += seg_ELASE.values[i].data
                    seg_int_e += seg_ELSE.values[i].data
                    seg_int_e += seg_ELCD.values[i].data
                    seg_int_e += seg_ELPD.values[i].data
                seg_int_e = seg_int_e / mass_list[j]
                SEG_INTEN.append(seg_int_e)
            SEG_INTEN = np.stack(SEG_INTEN)
            st_inten.append(SEG_INTEN)
        np.save(root_path + '/energy/sea_npy/' + name(n_s, '.npy'), np.stack(st_inten, axis=0))
        # np.save('H:/abaqus_test/' + name(n, '.npy'), np.stack(st_inten, axis=0))
        ##### extracting the CF ######
        session.viewports['Viewport: 1'].setValues(displayedObject=current_odb)
        for seg_key in segs_keys:
            eLeaf = dgo.LeafFromElementSets(elementSets=("BEAM-1." + seg_key, ))
            nLeaf = dgo.LeafFromNodeSets(nodeSets=("BEAM-1." + seg_key, ))
            session.FreeBodyFromNodesElements(name=seg_key, elements=eLeaf, 
                nodes=nLeaf, summationLoc=CENTROID, componentResolution=NORMAL_TANGENTIAL)
        session.freeBodyReportOptions.setValues(reportFormat=COMMA_SEPARATED_VALUES)
        session.writeFreeBodyReport(fileName=root_path+'/force/force_txt/'+name(n_s, '.txt'), append=ON, step=0, 
            frame=200, stepFrame=ALL, odb=current_odb)
        with open(root_path+'/force/force_txt/'+name(n_s, '.txt')) as f:
            lines = f.readlines()
        force = []
        sec_force = []
        for j, line in enumerate(lines[1:]):
            seg_lable1 = line.split()[0]
            if j != 0:
                if seg_lable1 != seg_lable0:
                    change = True
                else:
                    change = False
                if change:
                    force.append(np.array(sec_force, dtype=np.float32))
                    sec_force = []
            line_splited = line.split()[-6:-3]
            vec = []
            for ele in line_splited:
                vec.append(float(ele[:-1]))
            # magnitude = (vec[0]**2+vec[1]**2+vec[2]**2)**(0.5)
            # sec_force.append(magnitude)
            sec_force.append(vec)
            seg_lable0 = line.split()[0]
        force.append(np.array(sec_force, dtype=np.float32))
        force = np.stack(force, axis=1)[1:,:]
        np.save(root_path+'/force/cf_npy/'+name(n_s, '.npy'), force)
        print('force shape',np.shape(force))
    except:
        np.save(root_path+'/force/cf_npy/'+name(n_s, '.npy'), np.zeros(shape=[3,3]))
        np.save(root_path + '/energy/sea_npy/' + name(n_s, '.npy'), np.zeros(shape=[3,3]))
        time.sleep(5)
    print('========{}th odb file has been extracted========'.format(n_s))
    current_odb.close()
    



