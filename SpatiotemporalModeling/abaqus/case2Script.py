from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
from odbAccess import *
import random
import os
import time
import numpy as np
import shutil
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
F_outputs_steps = 180
H_outputs_steps = 10
impact_time = 0.01
n_samples = 500
mesh_size = 4
THICK = 2.0
Length = 205.0
m = 2
density = 2.7e-9
root_path = 'E:/SpatiotemporalModelingCMAMERevision/data/case2/testing3'
abaqus_working_path = 'F:/temp/Job-1.odb'
##### training set ranges
r1_range = [5, 8]
r2_range = [5, 8]
r3_range = [5, 8]
d1_range = [20, 40]
d2_range = [70, 90]
d3_range = [130, 150]	
elastic_modulus_range = [71000, 81000]
Poissons_ratio_range = [0.304, 0.322]
velo_range = [14400, 15600]
##### testing2 set ranges
# r1_range1 = [4.8, 5.0]
# r1_range2 = [8.0, 8.2]
# r2_range1 = [4.8, 5.0]
# r2_range2 = [8.0, 8.2]
# r3_range1 = [4.6, 5.0]
# r3_range2 = [8.0, 8.4]
# d1_range1 = [18, 20]
# d1_range2 = [40, 42]
# d2_range1 = [68, 70]
# d2_range2 = [90, 92]
# d3_range1 = [126, 130]	
# d3_range2 = [150, 154]	
# elastic_modulus_range1 = [69000, 71000]
# elastic_modulus_range2 = [81000, 83000]
# Poissons_ratio_range1 = [0.300, 0.304]
# Poissons_ratio_range2 = [0.322, 0.326]
# velo_range1 = [14200, 14400]
# velo_range2 = [15600, 15800]
##### testing set3 ranges
# r1_range1 = [4.6, 4.8]
# r1_range2 = [8.2, 8.4]
# r2_range1 = [4.6, 4.8]
# r2_range2 = [8.2, 8.4]
# r3_range1 = [4.2, 4.6]
# r3_range2 = [8.4, 8.8]
# d1_range1 = [16, 18]
# d1_range2 = [42, 44]
# d2_range1 = [66, 68]
# d2_range2 = [92, 94]
# d3_range1 = [122, 126]	
# d3_range2 = [154, 158]	
# elastic_modulus_range1 = [67000, 69000]
# elastic_modulus_range2 = [83000, 85000]
# Poissons_ratio_range1 = [0.296, 0.300]
# Poissons_ratio_range2 = [0.326, 0.330]
# velo_range1 = [14000, 14200]
# velo_range2 = [15800, 16000]
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

def order(z):
    orderargs = np.argsort(z,axis=1)
    ordered = []
    for i in range(201):
        ordered.append(z[i, orderargs[i,:,1],0])
    ordered = np.array(ordered, dtype=np.float32)
    return ordered

# DVs = LHSample(D=11, bounds=[r1_range,	#r1
# 							 r2_range,	#dv_r2
# 							 r3_range,	#dv_r3
# 							 d1_range,  #d1
# 							 d2_range,  #dv_d2
# 							 d3_range,  #dv_d3
# 							 elastic_modulus_range,
# 							 Poissons_ratio_range,
# 							 elastic_modulus_range,
# 							 Poissons_ratio_range,
# 							 velo_range
#        						],  
# 							N=n_samples)
# DVs1 = LHSample(D=11, bounds=[r1_range1,	#r1
# 							 r2_range1,	#dv_r2
# 							 r3_range1,	#dv_r3
# 							 d1_range1,  #d1
# 							 d2_range1,  #dv_d2
# 							 d3_range1,  #dv_d3
# 							 elastic_modulus_range1,
# 							 Poissons_ratio_range1,
# 							 elastic_modulus_range1,
# 							 Poissons_ratio_range1,
# 							 velo_range1
#        						],  
# 							N=int(n_samples/2))
# DVs2 = LHSample(D=11, bounds=[r1_range2,	#r1
# 							 r2_range2,	#dv_r2
# 							 r3_range2,	#dv_r3
# 							 d1_range2,  #d1
# 							 d2_range2,  #dv_d2
# 							 d3_range2,  #dv_d3
# 							 elastic_modulus_range2,
# 							 Poissons_ratio_range2,
# 							 elastic_modulus_range2,
# 							 Poissons_ratio_range2,
# 							 velo_range2
#        						],  
# 							N=n_samples)
# DVs = np.concatenate([DVs1,DVs2], axis=0)
# np.random.shuffle(DVs)
# np.save(root_path+'/design_variables.npy', DVs)
DVs = np.load(root_path+'/design_variables.npy')
for n_s in range(39, n_samples):
	dv_r1 = DVs[n_s,0]
	dv_r2 = DVs[n_s,1]
	dv_r3 = DVs[n_s,2]
	dv_d1 = DVs[n_s,3]
	dv_d2 = DVs[n_s,4]
	dv_d3 = DVs[n_s,5]
	elastic_modulus_outer = DVs[n_s,6]
	poissions_ratio_outer = DVs[n_s,7]
	elastic_modulus_rib = DVs[n_s,8]
	poissions_ratio_rib = DVs[n_s,9]
	velo = DVs[n_s,10]
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=200.0)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=STANDALONE)
	s.Line(point1=(-56.25, 56.25), point2=(-32.5, 56.25))
	s.HorizontalConstraint(entity=g[2], addUndoState=False)
	s.Line(point1=(-32.5, 56.25), point2=(-18.75, 35.0))
	s.Line(point1=(-18.75, 35.0), point2=(-31.25, 15.0))
	s.Line(point1=(-31.25, 15.0), point2=(-55.0, 15.0))
	s.HorizontalConstraint(entity=g[5], addUndoState=False)
	s.Line(point1=(-55.0, 15.0), point2=(-70.0, 35.0))
	s.Line(point1=(-70.0, 35.0), point2=(-56.25, 56.25))
	s.ObliqueDimension(vertex1=v[0], vertex2=v[1], textPoint=(-51.0792274475098, 
		60.7162246704102), value=28.0)
	s.AngularDimension(line1=g[2], line2=g[3], textPoint=(-33.1843223571777, 
		49.3688735961914), value=120.0)
	s.ObliqueDimension(vertex1=v[1], vertex2=v[2], textPoint=(-16.0731353759766, 
		49.2384414672852), value=28.0)
	s.ObliqueDimension(vertex1=v[2], vertex2=v[3], textPoint=(-14.5057001113892, 
		22.1091499328613), value=28.0)
	s.AngularDimension(line1=g[3], line2=g[4], textPoint=(-21.9510192871094, 
		31.8913497924805), value=120.0)
	s.ObliqueDimension(vertex1=v[3], vertex2=v[4], textPoint=(-41.4133644104004, 
		2.54475402832031), value=28.0)
	s.ObliqueDimension(vertex1=v[4], vertex2=v[5], textPoint=(-70.5415725708008, 
		17.2832641601563), value=28.0)
	s.ObliqueDimension(vertex1=v[5], vertex2=v[0], textPoint=(-74.7214050292969, 
		45.9777069091797), value=28.0)
	s.copyMove(vector=(77.75, 1.74871130596426), objectList=(g[2], g[3], g[4], 
		g[5], g[6], g[7]))
	s.copyMove(vector=(-2.5, -62.5), objectList=(g[2], g[3], g[4], g[5], g[6], 
		g[7], g[8], g[9], g[10], g[11], g[12], g[13]))
	s.Line(point1=(-14.25, 32.0012886940357), point2=(7.5, 33.75))
	s.Line(point1=(-28.25, 7.75257738807145), point2=(-30.75, -6.25))
	s.Line(point1=(-16.75, -30.4987113059643), point2=(5.0, -28.75))
	s.Line(point1=(21.5, 9.50128869403571), point2=(19.0, -4.50128869403574))
	s.AngularDimension(line1=g[3], line2=g[26], textPoint=(-10.1952524185181, 
		36.3259429931641), value=120.0)
	s.AngularDimension(line1=g[5], line2=g[27], textPoint=(-32.5312271118164, 
		3.84904861450195), value=90.0)
	s.AngularDimension(line1=g[15], line2=g[28], textPoint=(-8.23595523834229, 
		-25.4975433349609), value=120.0)
	s.AngularDimension(line1=g[11], line2=g[29], textPoint=(24.9414749145508, 
		3.97947692871094), value=90.0)
	s.ObliqueDimension(vertex1=v[2], vertex2=v[11], textPoint=(-5.75417423248291, 
		42.5865478515625), value=28.0)
	s.ObliqueDimension(vertex1=v[3], vertex2=v[13], textPoint=(-26.9145736694336, 
		1.50131988525391), value=25.5)
	p = mdb.models['Model-1'].Part(name='Part-1', dimensionality=THREE_D, 
		type=DEFORMABLE_BODY)
	p = mdb.models['Model-1'].parts['Part-1']
	p.BaseShellExtrude(sketch=s, depth=Length)
	s.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts['Part-1']
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[26], sketchUpEdge=e[14], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-4.242052, 
		-35.845588, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=419.97, gridSpacing=10.49, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.Spot(point=(0.0, 70.8075))
	s1.Spot(point=(0.0, 18.3575))
	s1.Spot(point=(0.0, -39.3375))
	s1.CircleByCenterPerimeter(center=(0.0, 70.8075), point1=(2.6225, 60.3175))
	s1.CircleByCenterPerimeter(center=(0.0, 18.3575), point1=(2.6225, 10.49))
	s1.CircleByCenterPerimeter(center=(0.0, -39.3375), point1=(5.245, -49.8275))
	s1.DistanceDimension(entity1=v[8], entity2=g[10], textPoint=(92.9961413554688, 
		93.1195831298828), value=dv_d1)
	s1.DistanceDimension(entity1=v[9], entity2=g[10], textPoint=(109.338579099609, 
		96.7459259033203), value=dv_d2)
	s1.DistanceDimension(entity1=v[10], entity2=g[10], textPoint=(138.651140134766, 
		78.3551483154297), value=dv_d3)
	s1.Line(point1=(-14.0000001526196, -102.5), point2=(-14.0000001526196, 
		-84.0536880493164))
	s1.VerticalConstraint(entity=g[17], addUndoState=False)
	s1.ParallelConstraint(entity1=g[8], entity2=g[17], addUndoState=False)
	s1.CoincidentConstraint(entity1=v[17], entity2=g[8], addUndoState=False)
	s1.Line(point1=(-14.0000001526196, -84.0536880493164), point2=(
		13.9999998474382, -84.0536880493164))
	s1.HorizontalConstraint(entity=g[18], addUndoState=False)
	s1.PerpendicularConstraint(entity1=g[17], entity2=g[18], addUndoState=False)
	s1.CoincidentConstraint(entity1=v[18], entity2=g[4], addUndoState=False)
	s1.Line(point1=(13.9999998474382, -84.0536880493164), point2=(13.9999998473804, 
		-102.5))
	s1.VerticalConstraint(entity=g[19], addUndoState=False)
	s1.PerpendicularConstraint(entity1=g[18], entity2=g[19], addUndoState=False)
	s1.Line(point1=(13.9999998473804, -102.5), point2=(-14.0000001526196, -102.5))
	s1.HorizontalConstraint(entity=g[20], addUndoState=False)
	s1.PerpendicularConstraint(entity1=g[19], entity2=g[20], addUndoState=False)
	s1.DistanceDimension(entity1=g[12], entity2=g[18], textPoint=(98.7030200175781, 
		-86.6439361572266), value=20.0)
	s1.RadialDimension(curve=g[16], textPoint=(22.4384001903076, 
		-29.9174118041992), radius=dv_r1)
	s1.RadialDimension(curve=g[15], textPoint=(21.4007872752686, 36.9111022949219), 
		radius=dv_r2)
	s1.RadialDimension(curve=g[14], textPoint=(21.9196013621826, 83.5356597900391), 
		radius=dv_r3)
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[26], sketchUpEdge=e1[14], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s1, flipExtrudeDirection=OFF)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[1], sketchUpEdge=e[11], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-4.242052, 
		-35.845588, 112.024946))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=413.8, gridSpacing=10.34, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.rectangle(point1=(-1.52619605842119e-07, 92.975054), point2=(85.305, 
		-116.325))
	s.CoincidentConstraint(entity1=v[16], entity2=g[7], addUndoState=False)
	s.EqualDistanceConstraint(entity1=v[8], entity2=v[9], midpoint=v[16], 
		addUndoState=False)
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[1], sketchUpEdge=e1[11], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[16], sketchUpEdge=e[65], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(37.757948, 
		-60.0943, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=435.94, gridSpacing=10.89, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.rectangle(point1=(-35.3925, 103.455), point2=(49.005, -106.1775))
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[16], sketchUpEdge=e1[65], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s1, flipExtrudeDirection=OFF)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[14], sketchUpEdge=e[41], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-32.242052, 
		1.153123, 102.5))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=467.14, gridSpacing=11.67, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.rectangle(point1=(1.09619961108365e-07, 102.5), point2=(134.205, -105.03))
	s.CoincidentConstraint(entity1=v[8], entity2=g[10], addUndoState=False)
	s.EqualDistanceConstraint(entity1=v[5], entity2=v[2], midpoint=v[8], 
		addUndoState=False)
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	p.CutExtrude(sketchPlane=f1[14], sketchUpEdge=e1[41], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[9], sketchUpEdge=e[34], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-47.969944, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=443.91, gridSpacing=11.09, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f1[10], sketchUpEdge=e1[37], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-23.721233, 102.5))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=430.18, gridSpacing=10.75, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.rectangle(point1=(-29.5625, 104.8125), point2=(56.4375, -104.8125))
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	p.CutExtrude(sketchPlane=f[10], sketchUpEdge=e[37], sketchPlaneSide=SIDE1, 
		sketchOrientation=RIGHT, sketch=s, flipExtrudeDirection=OFF)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f1, e1 = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f1[9], sketchUpEdge=e1[34], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-47.969944, 102.5))
	s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=422.24, gridSpacing=10.55, transform=t)
	g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
	s1.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
	s1.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f, e = p.faces, p.edges
	t = p.MakeSketchTransform(sketchPlane=f[9], sketchUpEdge=e[34], 
		sketchPlaneSide=SIDE1, sketchOrientation=RIGHT, origin=(-67.242052, 
		-47.969944, 102.5))
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=422.24, gridSpacing=10.55, transform=t)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=SUPERIMPOSE)
	p = mdb.models['Model-1'].parts['Part-1']
	p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
	s.unsetPrimaryObject()
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-1']
	f1 = p.faces
	p.RemoveFaces(faceList = f1[8:11], deleteCells=False)
	mdb.models['Model-1'].Material(name='Material-1')
	mdb.models['Model-1'].materials['Material-1'].Density(table=((2.7e-09, ), ))
	mdb.models['Model-1'].materials['Material-1'].Elastic(table=((72000.0, 0.33), 
		))
	mdb.models['Model-1'].materials['Material-1'].Plastic(table=((131.8, 0.0), (
		140.0, 0.0023), (149.4, 0.0067), (162.3, 0.0153), (171.2, 0.0232), (178.3, 
		0.0314), (183.7, 0.0406), (187.4, 0.0503), (189.8, 0.0601), (190.6, 
		0.0648), (191.0, 0.0658)))
	mdb.models['Model-1'].HomogeneousShellSection(name='Section-1', 
		preIntegrate=OFF, material='Material-1', thicknessType=UNIFORM, 
		thickness=THICK, thicknessField='', nodalThicknessField='', 
		idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
		thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
		integrationRule=SIMPSON, numIntPts=5)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
	region = p.Set(faces=faces, name='Set-1')
	p = mdb.models['Model-1'].parts['Part-1']
	p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', 
		thicknessAssignment=FROM_SECTION)
	s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', 
		sheetSize=200.0)
	g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
	s.setPrimaryObject(option=STANDALONE)
	s.rectangle(point1=(-68.75, 61.25), point2=(73.75, -60.0))
	p = mdb.models['Model-1'].Part(name='Part-2', dimensionality=THREE_D, 
		type=DISCRETE_RIGID_SURFACE)
	p = mdb.models['Model-1'].parts['Part-2']
	p.BaseShell(sketch=s)
	s.unsetPrimaryObject()
	p = mdb.models['Model-1'].parts['Part-2']
	del mdb.models['Model-1'].sketches['__profile__']
	p = mdb.models['Model-1'].parts['Part-2']
	e = p.edges
	p.DatumPointByMidPoint(point1=p.InterestingPoint(edge=e[1], rule=MIDDLE), 
		point2=p.InterestingPoint(edge=e[3], rule=MIDDLE))
	p = mdb.models['Model-1'].parts['Part-2']
	v1, e1, d1, n = p.vertices, p.edges, p.datums, p.nodes
	p.ReferencePoint(point=d1[2])
	a = mdb.models['Model-1'].rootAssembly
	a.DatumCsysByDefault(CARTESIAN)
	p = mdb.models['Model-1'].parts['Part-1']
	a.Instance(name='Part-1-1', part=p, dependent=ON)
	p = mdb.models['Model-1'].parts['Part-2']
	a.Instance(name='Part-2-1', part=p, dependent=ON)
	p = mdb.models['Model-1'].parts['Part-1']
	e1 = p.edges
	p.DatumPointByMidPoint(point1=p.InterestingPoint(edge=e1[18], rule=MIDDLE), 
		point2=p.InterestingPoint(edge=e1[26], rule=MIDDLE))
	p = mdb.models['Model-1'].parts['Part-1']
	v1, e, d1, n1 = p.vertices, p.edges, p.datums, p.nodes
	p.ReferencePoint(point=d1[9])
	p = mdb.models['Model-1'].parts['Part-1']
	r = p.referencePoints
	refPoints=(r[10], )
	region=p.Set(referencePoints=refPoints, name='Set-2')
	mdb.models['Model-1'].parts['Part-1'].engineeringFeatures.PointMassInertia(
		name='Inertia-1', region=region, mass=m, alpha=0.0, composite=0.0)
	a = mdb.models['Model-1'].rootAssembly
	a.regenerate()
	a = mdb.models['Model-1'].rootAssembly
	a = mdb.models['Model-1'].rootAssembly
	r1 = a.instances['Part-2-1'].referencePoints
	a.DatumPointByOffset(point=r1[3], vector=(0.0, 0.0, 210.0))
	a1 = mdb.models['Model-1'].rootAssembly
	a1.translate(instanceList=('Part-1-1', ), vector=(48.742052, -37.526834, 5.0))
	mdb.models['Model-1'].ExplicitDynamicsStep(name='Step-1', previous='Initial', 
		timePeriod=impact_time, improvedDtMethod=ON)
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(
		numIntervals=F_outputs_steps)
	mdb.models['Model-1'].ContactProperty('IntProp-1')
	mdb.models['Model-1'].interactionProperties['IntProp-1'].TangentialBehavior(
		formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
		pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
		0.17, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
		fraction=0.005, elasticSlipStiffness=None)
	#: The interaction property "IntProp-1" has been created.
	mdb.models['Model-1'].ContactExp(name='Int-1', createStepName='Step-1')
	mdb.models['Model-1'].interactions['Int-1'].includedPairs.setValuesInStep(
		stepName='Step-1', useAllstar=ON)
	mdb.models['Model-1'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
		stepName='Step-1', assignments=((GLOBAL, SELF, 'IntProp-1'), ))
	a = mdb.models['Model-1'].rootAssembly
	r1 = a.instances['Part-1-1'].referencePoints
	refPoints1=(r1[10], )
	region1=a.Set(referencePoints=refPoints1, name='m_Set-1')
	a = mdb.models['Model-1'].rootAssembly
	s1 = a.instances['Part-1-1'].edges
	side1Edges1 = s1.getSequenceFromMask(mask=('[#24a44802 ]', ), )
	region2=a.Surface(side1Edges=side1Edges1, name='s_Surf-1')
	mdb.models['Model-1'].Coupling(name='Constraint-1', controlPoint=region1, 
		surface=region2, influenceRadius=WHOLE_SURFACE, couplingType=KINEMATIC, 
		localCsys=None, u1=ON, u2=ON, u3=ON, ur1=ON, ur2=ON, ur3=ON)
	a = mdb.models['Model-1'].rootAssembly
	f1 = a.instances['Part-1-1'].faces
	faces1 = f1.getSequenceFromMask(mask=('[#ff ]', ), )
	e1 = a.instances['Part-1-1'].edges
	edges1 = e1.getSequenceFromMask(mask=('[#2da6df87 ]', ), )
	v1 = a.instances['Part-1-1'].vertices
	verts1 = v1.getSequenceFromMask(mask=('[#28df06 ]', ), )
	r1 = a.instances['Part-1-1'].referencePoints
	refPoints1=(r1[10], )
	region = a.Set(vertices=verts1, edges=edges1, faces=faces1, 
		referencePoints=refPoints1, name='Set-2')
	mdb.models['Model-1'].Velocity(name='Predefined Field-1', region=region, 
		field='', distributionType=MAGNITUDE, velocity1=0.0, velocity2=0.0, 
		velocity3=-velo, omega=0.0)
	a = mdb.models['Model-1'].rootAssembly
	r1 = a.instances['Part-2-1'].referencePoints
	refPoints1=(r1[3], )
	region = a.Set(referencePoints=refPoints1, name='Set-3')
	mdb.models['Model-1'].DisplacementBC(name='BC-1', createStepName='Initial', 
		region=region, u1=SET, u2=SET, u3=SET, ur1=SET, ur2=SET, ur3=SET, 
		amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
	a = mdb.models['Model-1'].rootAssembly
	e1 = a.instances['Part-1-1'].edges
	edges1 = e1.getSequenceFromMask(mask=('[#1 ]', ), )
	region = a.Set(edges=edges1, name='Set-4')
	mdb.models['Model-1'].YsymmBC(name='BC-2', createStepName='Initial', 
		region=region, localCsys=None)
	a = mdb.models['Model-1'].rootAssembly
	e1 = a.instances['Part-1-1'].edges
	edges1 = e1.getSequenceFromMask(mask=('[#550 ]', ), )
	region = a.Set(edges=edges1, name='Set-5')
	mdb.models['Model-1'].XsymmBC(name='BC-3', createStepName='Initial', 
		region=region, localCsys=None)
	p = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	p.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)
	p = mdb.models['Model-1'].parts['Part-1']
	p.generateMesh()
	p = mdb.models['Model-1'].parts['Part-2']
	p = mdb.models['Model-1'].parts['Part-2']
	p.seedPart(size=14.0, deviationFactor=0.1, minSizeFactor=0.1)
	p = mdb.models['Model-1'].parts['Part-2']
	p.generateMesh()
	elemType1 = mesh.ElemType(elemCode=R3D4, elemLibrary=EXPLICIT)
	elemType2 = mesh.ElemType(elemCode=R3D3, elemLibrary=EXPLICIT)
	p = mdb.models['Model-1'].parts['Part-2']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
	pickedRegions =(faces, )
	p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
	p = mdb.models['Model-1'].parts['Part-1']
	elemType1 = mesh.ElemType(elemCode=S4R, elemLibrary=EXPLICIT, 
		secondOrderAccuracy=OFF, hourglassControl=DEFAULT)
	elemType2 = mesh.ElemType(elemCode=S3R, elemLibrary=EXPLICIT)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#ff ]', ), )
	pickedRegions =(faces, )
	p.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
	a = mdb.models['Model-1'].rootAssembly
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(
		variables=PRESELECT)
	mdb.models['Model-1'].fieldOutputRequests['F-Output-1'].setValues(variables=(
		'S', 'SVAVG', 'PE', 'PEVAVG', 'PEEQ', 'PEEQVAVG', 'LE', 'U', 'V', 'A', 
		'RF', 'RT', 'RM', 'CF', 'SF', 'NFORC', 'NFORCSO', 'RBFOR', 'BF', 'GRAV', 
		'P', 'HP', 'IWCONWEP', 'TRSHR', 'TRNOR', 'VP', 'STAGP', 'SBF', 'CSTRESS', 
		'ENER', 'ELEN', 'ELEDEN', 'EDCDEN', 'EDT', 'EVF'))
	mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(variables=(
		'ALLAE', 'ALLCD', 'ALLDC', 'ALLDMD', 'ALLFD', 'ALLIE', 'ALLKE', 'ALLPD', 
		'ALLSE', 'ALLVD', 'ALLWK', 'ALLCW', 'ALLMW', 'ALLPW', 'ETOTAL'))
	mdb.models['Model-1'].historyOutputRequests['H-Output-1'].setValues(
			numIntervals=H_outputs_steps)

	for i in range(5):
		p = mdb.models['Model-1'].parts['Part-1']
		p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=195.0+2*i)

	for i in range(6):
		p = mdb.models['Model-1'].parts['Part-1']
		p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=145.0+2*i)

	for i in range(6):
		p = mdb.models['Model-1'].parts['Part-1']
		p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=85.0+2*i)

	for i in range(23):
		p = mdb.models['Model-1'].parts['Part-1']
		p.DatumPlaneByPrincipalPlane(principalPlane=XYPLANE, offset=1 + 2*i)

	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedRegions = f.getSequenceFromMask(mask=('[#ff ]', ), )
	p.deleteMesh(regions=pickedRegions)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#ff ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[18], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#38ab ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[17], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#3002d5 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[16], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#200201d9 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[15], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[14], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[24], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[23], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #20000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[22], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[21], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[20], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[19], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0 #20000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[30], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:2 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[29], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0:2 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[28], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0:2 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[27], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:2 #20000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[26], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0:3 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[25], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0:3 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[53], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:3 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[52], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0:3 #20000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[51], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0:4 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[50], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:4 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[49], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0:4 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[48], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0:4 #20000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[47], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:5 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[46], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0:5 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[45], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0:5 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[44], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:5 #20000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[43], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#48b5 #0:6 #20 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[42], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#4004d5 #0:6 #2000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[41], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#401d9 #0:6 #200000 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[40], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#2455 #0:6 #10000000 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[39], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#22853 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[38], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#42869 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[37], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#41453 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[36], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#22853 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[35], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#42869 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[34], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#41453 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[33], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#22853 ]', ), )
	d = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d[32], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	pickedFaces = f.getSequenceFromMask(mask=('[#42869 ]', ), )
	d1 = p.datums
	p.PartitionFaceByDatumPlane(datumPlane=d1[31], faces=pickedFaces)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #20000000 ]', ), )
	p.Set(faces=faces, name='seg1-1')
	#: The set 'seg1-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #10000000 ]', ), )
	p.Set(faces=faces, name='seg1-2')
	#: The set 'seg1-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #10000 ]', ), )
	p.Set(faces=faces, name='seg1-3')
	#: The set 'seg1-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #400000 ]', ), )
	p.Set(faces=faces, name='seg1-4')
	#: The set 'seg1-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #40000 ]', ), )
	p.Set(faces=faces, name='seg1-5')
	#: The set 'seg1-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #800000 ]', ), )
	p.Set(faces=faces, name='seg1-6')
	#: The set 'seg1-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #100000 ]', ), )
	p.Set(faces=faces, name='seg1-7')
	#: The set 'seg1-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #1000000 ]', ), )
	p.Set(faces=faces, name='seg1-8')
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #80000000 ]', ), )
	p.Set(faces=faces, name='seg3-1')
	#: The set 'seg3-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #1 ]', ), )
	p.Set(faces=faces, name='seg3-2')
	#: The set 'seg3-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #8 ]', ), )
	p.Set(faces=faces, name='seg3-3')
	#: The set 'seg3-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #100 ]', ), )
	p.Set(faces=faces, name='seg3-4')
	#: The set 'seg3-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #4000000 ]', ), )
	p.Set(faces=faces, name='seg3-5')
	#: The set 'seg3-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #2000 ]', ), )
	p.Set(faces=faces, name='seg3-6')
	#: The set 'seg3-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #1000 ]', ), )
	p.Set(faces=faces, name='seg3-7')
	#: The set 'seg3-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #400 ]', ), )
	p.Set(faces=faces, name='seg3-8')
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #40 ]', ), )
	p.Set(faces=faces, name='seg4-1')
	#: The set 'seg4-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #8000 ]', ), )
	p.Set(faces=faces, name='seg4-2')
	#: The set 'seg4-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #800000 ]', ), )
	p.Set(faces=faces, name='seg4-3')
	#: The set 'seg4-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #20 ]', ), )
	p.Set(faces=faces, name='seg4-4')
	#: The set 'seg4-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #2000000 ]', ), )
	p.Set(faces=faces, name='seg4-5')
	#: The set 'seg4-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #4 ]', ), )
	p.Set(faces=faces, name='seg4-6')
	#: The set 'seg4-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #10000000 ]', ), )
	p.Set(faces=faces, name='seg4-7')
	#: The set 'seg4-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #40000000 ]', ), )
	p.Set(faces=faces, name='seg4-8')
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #8000000 ]', ), )
	p.Set(faces=faces, name='seg5-1')
	#: The set 'seg5-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #8000 ]', ), )
	p.Set(faces=faces, name='seg5-2')
	#: The set 'seg5-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #20000 ]', ), )
	p.Set(faces=faces, name='seg5-3')
	#: The set 'seg5-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #400000 ]', ), )
	p.Set(faces=faces, name='seg5-4')
	#: The set 'seg5-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #80000 ]', ), )
	p.Set(faces=faces, name='seg5-5')
	#: The set 'seg5-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #4000000 ]', ), )
	p.Set(faces=faces, name='seg5-6')
	#: The set 'seg5-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #2 ]', ), )
	p.Set(faces=faces, name='seg5-7')
	#: The set 'seg5-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #20000000 ]', ), )
	p.Set(faces=faces, name='seg5-8')
	#: The set 'seg5-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #80 ]', ), )
	p.Set(faces=faces, name='seg6-1')
	#: The set 'seg6-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #100 ]', ), )
	p.Set(faces=faces, name='seg6-2')
	#: The set 'seg6-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #800 ]', ), )
	p.Set(faces=faces, name='seg6-3')
	#: The set 'seg6-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #10000 ]', ), )
	p.Set(faces=faces, name='seg6-4')
	#: The set 'seg6-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #10 ]', ), )
	p.Set(faces=faces, name='seg6-5')
	#: The set 'seg6-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #200000 ]', ), )
	p.Set(faces=faces, name='seg6-6')
	#: The set 'seg6-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #100000 ]', ), )
	p.Set(faces=faces, name='seg6-7')
	#: The set 'seg6-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #40000 ]', ), )
	p.Set(faces=faces, name='seg6-8')
	#: The set 'seg6-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #4000 ]', ), )
	p.Set(faces=faces, name='seg7-1')
	#: The set 'seg7-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #1000000 ]', ), )
	p.Set(faces=faces, name='seg7-2')
	#: The set 'seg7-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #80000000 ]', ), )
	p.Set(faces=faces, name='seg7-3')
	#: The set 'seg7-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #2000 ]', ), )
	p.Set(faces=faces, name='seg7-4')
	#: The set 'seg7-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #2 ]', ), )
	p.Set(faces=faces, name='seg7-5')
	#: The set 'seg7-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #400 ]', ), )
	p.Set(faces=faces, name='seg7-6')
	#: The set 'seg7-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #10 ]', ), )
	p.Set(faces=faces, name='seg7-7')
	#: The set 'seg7-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #40 ]', ), )
	p.Set(faces=faces, name='seg7-8')
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #8 ]', ), )
	p.Set(faces=faces, name='seg8-1')
	#: The set 'seg8-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #8000 ]', ), )
	p.Set(faces=faces, name='seg9-1')
	#: The set 'seg9-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #400000 ]', ), )
	p.Set(faces=faces, name='seg10-1')
	#: The set 'seg10-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #800 ]', ), )
	p.Set(faces=faces, name='seg11-1')
	#: The set 'seg11-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #800000 ]', ), )
	p.Set(faces=faces, name='seg12-1')
	#: The set 'seg12-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #40000000 ]', ), )
	p.Set(faces=faces, name='seg13-1')
	#: The set 'seg13-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #80000 ]', ), )
	p.Set(faces=faces, name='seg14-1')
	#: The set 'seg14-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #80000000 ]', ), )
	p.Set(faces=faces, name='seg15-1')
	#: The set 'seg15-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #40 ]', ), )
	p.Set(faces=faces, name='seg16-1')
	#: The set 'seg16-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #8000000 ]', ), )
	p.Set(faces=faces, name='seg17-1')
	#: The set 'seg17-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #80 ]', ), )
	p.Set(faces=faces, name='seg18-1')
	#: The set 'seg18-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #4000 ]', ), )
	p.Set(faces=faces, name='seg19-1')
	#: The set 'seg19-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #8 ]', ), )
	p.Set(faces=faces, name='seg20-1')
	#: The set 'seg20-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #8000 ]', ), )
	p.Set(faces=faces, name='seg21-1')
	#: The set 'seg21-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #400000 ]', ), )
	p.Set(faces=faces, name='seg22-1')
	#: The set 'seg22-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #800 ]', ), )
	p.Set(faces=faces, name='seg23-1')
	#: The set 'seg23-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #800000 ]', ), )
	p.Set(faces=faces, name='seg24-1')
	#: The set 'seg24-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #40000000 ]', ), )
	p.Set(faces=faces, name='seg25-1')
	#: The set 'seg25-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #80000 ]', ), )
	p.Set(faces=faces, name='seg26-1')
	#: The set 'seg26-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #80000000 ]', ), )
	p.Set(faces=faces, name='seg27-1')
	#: The set 'seg27-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #80000000 ]', ), )
	p.Set(faces=faces, name='seg28-1')
	#: The set 'seg28-1' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #40 ]', ), )
	p.Set(faces=faces, name='seg28-1')
	#: The set 'seg28-1' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #80 ]', ), )
	p.Set(faces=faces, name='seg29-1')
	#: The set 'seg29-1' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #8000000 ]', ), )
	p.Set(faces=faces, name='seg29-1')
	#: The set 'seg29-1' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #80 ]', ), )
	p.Set(faces=faces, name='seg30-1')
	#: The set 'seg30-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #20000000 ]', ), )
	p.Set(faces=faces, name='seg31-1')
	#: The set 'seg31-1' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #4000 ]', ), )
	p.Set(faces=faces, name='seg31-1')
	#: The set 'seg31-1' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #20000000 ]', ), )
	p.Set(faces=faces, name='seg32-1')
	#: The set 'seg32-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #40000 ]', ), )
	p.Set(faces=faces, name='seg33-1')
	#: The set 'seg33-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #8 ]', ), )
	p.Set(faces=faces, name='seg34-1')
	#: The set 'seg34-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #100 ]', ), )
	p.Set(faces=faces, name='seg35-1')
	#: The set 'seg35-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#20000000 ]', ), )
	p.Set(faces=faces, name='seg36-1')
	#: The set 'seg36-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #4000 ]', ), )
	p.Set(faces=faces, name='seg37-1')
	#: The set 'seg37-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#80000 ]', ), )
	p.Set(faces=faces, name='seg38-1')
	#: The set 'seg38-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#100 ]', ), )
	p.Set(faces=faces, name='seg39-1')
	#: The set 'seg39-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#2000000 ]', ), )
	p.Set(faces=faces, name='seg40-1')
	#: The set 'seg40-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#10 ]', ), )
	p.Set(faces=faces, name='seg41-1')
	#: The set 'seg41-1' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #800000 ]', ), )
	p.Set(faces=faces, name='seg8-2')
	#: The set 'seg8-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #10000 ]', ), )
	p.Set(faces=faces, name='seg9-2')
	#: The set 'seg9-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #1 ]', ), )
	p.Set(faces=faces, name='seg10-2')
	#: The set 'seg10-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #80000000 ]', ), )
	p.Set(faces=faces, name='seg11-2')
	#: The set 'seg11-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #1000000 ]', ), )
	p.Set(faces=faces, name='seg12-2')
	#: The set 'seg12-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #100 ]', ), )
	p.Set(faces=faces, name='seg13-2')
	#: The set 'seg13-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #80 ]', ), )
	p.Set(faces=faces, name='seg14-2')
	#: The set 'seg14-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #1 ]', ), )
	p.Set(faces=faces, name='seg15-2')
	#: The set 'seg15-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #10000 ]', ), )
	p.Set(faces=faces, name='seg16-2')
	#: The set 'seg16-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #8000 ]', ), )
	p.Set(faces=faces, name='seg17-2')
	#: The set 'seg17-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #100 ]', ), )
	p.Set(faces=faces, name='seg18-2')
	#: The set 'seg18-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #1000000 ]', ), )
	p.Set(faces=faces, name='seg19-2')
	#: The set 'seg19-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #800000 ]', ), )
	p.Set(faces=faces, name='seg20-2')
	#: The set 'seg20-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #10000 ]', ), )
	p.Set(faces=faces, name='seg21-2')
	#: The set 'seg21-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #1 ]', ), )
	p.Set(faces=faces, name='seg22-2')
	#: The set 'seg22-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #80000000 ]', ), )
	p.Set(faces=faces, name='seg23-2')
	#: The set 'seg23-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #1000000 ]', ), )
	p.Set(faces=faces, name='seg24-2')
	#: The set 'seg24-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #100 ]', ), )
	p.Set(faces=faces, name='seg25-2')
	#: The set 'seg25-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #80 ]', ), )
	p.Set(faces=faces, name='seg26-2')
	#: The set 'seg26-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #1 ]', ), )
	p.Set(faces=faces, name='seg27-2')
	#: The set 'seg27-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #10000 ]', ), )
	p.Set(faces=faces, name='seg28-2')
	#: The set 'seg28-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #8000 ]', ), )
	p.Set(faces=faces, name='seg29-2')
	#: The set 'seg29-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #100 ]', ), )
	p.Set(faces=faces, name='seg30-2')
	#: The set 'seg30-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #1000000 ]', ), )
	p.Set(faces=faces, name='seg31-2')
	#: The set 'seg31-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #2 ]', ), )
	p.Set(faces=faces, name='seg32-2')
	#: The set 'seg32-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #80000 ]', ), )
	p.Set(faces=faces, name='seg33-2')
	#: The set 'seg33-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #40000000 ]', ), )
	p.Set(faces=faces, name='seg34-2')
	#: The set 'seg34-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #800 ]', ), )
	p.Set(faces=faces, name='seg35-2')
	#: The set 'seg35-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#40000000 ]', ), )
	p.Set(faces=faces, name='seg36-2')
	#: The set 'seg36-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #200 ]', ), )
	p.Set(faces=faces, name='seg37-2')
	#: The set 'seg37-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#400000 ]', ), )
	p.Set(faces=faces, name='seg38-2')
	#: The set 'seg38-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#200 ]', ), )
	p.Set(faces=faces, name='seg39-2')
	#: The set 'seg39-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#100000 ]', ), )
	p.Set(faces=faces, name='seg40-2')
	#: The set 'seg40-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#2 ]', ), )
	p.Set(faces=faces, name='seg41-2')
	#: The set 'seg41-2' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #2000000 ]', ), )
	p.Set(faces=faces, name='seg8-3')
	#: The set 'seg8-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #80000 ]', ), )
	p.Set(faces=faces, name='seg9-3')
	#: The set 'seg9-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #80 ]', ), )
	p.Set(faces=faces, name='seg10-3')
	#: The set 'seg10-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #2 ]', ), )
	p.Set(faces=faces, name='seg11-3')
	#: The set 'seg11-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #8000000 ]', ), )
	p.Set(faces=faces, name='seg12-3')
	#: The set 'seg12-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #8000 ]', ), )
	p.Set(faces=faces, name='seg13-3')
	#: The set 'seg13-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #200 ]', ), )
	p.Set(faces=faces, name='seg14-3')
	#: The set 'seg14-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #8 ]', ), )
	p.Set(faces=faces, name='seg15-3')
	#: The set 'seg15-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #800000 ]', ), )
	p.Set(faces=faces, name='seg16-3')
	#: The set 'seg16-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #20000 ]', ), )
	p.Set(faces=faces, name='seg17-3')
	#: The set 'seg17-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #800 ]', ), )
	p.Set(faces=faces, name='seg18-3')
	#: The set 'seg18-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #80000000 ]', ), )
	p.Set(faces=faces, name='seg19-3')
	#: The set 'seg19-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #2000000 ]', ), )
	p.Set(faces=faces, name='seg20-3')
	#: The set 'seg20-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #80000 ]', ), )
	p.Set(faces=faces, name='seg21-3')
	#: The set 'seg21-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #80 ]', ), )
	p.Set(faces=faces, name='seg22-3')
	#: The set 'seg22-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #2 ]', ), )
	p.Set(faces=faces, name='seg23-3')
	#: The set 'seg23-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #8000000 ]', ), )
	p.Set(faces=faces, name='seg24-3')
	#: The set 'seg24-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #8000 ]', ), )
	p.Set(faces=faces, name='seg25-3')
	#: The set 'seg25-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #200 ]', ), )
	p.Set(faces=faces, name='seg26-3')
	#: The set 'seg26-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #8 ]', ), )
	p.Set(faces=faces, name='seg27-3')
	#: The set 'seg27-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #8 ]', ), )
	p.Set(faces=faces, name='seg28-3')
	#: The set 'seg28-3' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #800000 ]', ), )
	p.Set(faces=faces, name='seg28-3')
	#: The set 'seg28-3' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #20000 ]', ), )
	p.Set(faces=faces, name='seg29-3')
	#: The set 'seg29-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #800 ]', ), )
	p.Set(faces=faces, name='seg30-3')
	#: The set 'seg30-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #1 ]', ), )
	p.Set(faces=faces, name='seg31-3')
	#: The set 'seg31-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:9 #8000000 ]', ), )
	p.Set(faces=faces, name='seg32-3')
	#: The set 'seg32-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #200000 ]', ), )
	p.Set(faces=faces, name='seg33-3')
	#: The set 'seg33-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #1000 ]', ), )
	p.Set(faces=faces, name='seg34-3')
	#: The set 'seg34-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #10000000 ]', ), )
	p.Set(faces=faces, name='seg35-3')
	#: The set 'seg35-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #1 ]', ), )
	p.Set(faces=faces, name='seg36-3')
	#: The set 'seg36-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#800000 ]', ), )
	p.Set(faces=faces, name='seg37-3')
	#: The set 'seg37-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #80 ]', ), )
	p.Set(faces=faces, name='seg38-3')
	#: The set 'seg38-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#800 ]', ), )
	p.Set(faces=faces, name='seg39-3')
	#: The set 'seg39-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#4 ]', ), )
	p.Set(faces=faces, name='seg40-3')
	#: The set 'seg40-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#40000 ]', ), )
	p.Set(faces=faces, name='seg41-3')
	#: The set 'seg41-3' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #40000000 ]', ), )
	p.Set(faces=faces, name='seg8-4')
	#: The set 'seg8-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #1000000 ]', ), )
	p.Set(faces=faces, name='seg9-4')
	#: The set 'seg9-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #200000 ]', ), )
	p.Set(faces=faces, name='seg10-4')
	#: The set 'seg10-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #40 ]', ), )
	p.Set(faces=faces, name='seg11-4')
	#: The set 'seg11-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #1 ]', ), )
	p.Set(faces=faces, name='seg12-4')
	#: The set 'seg12-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #20000000 ]', ), )
	p.Set(faces=faces, name='seg13-4')
	#: The set 'seg13-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #4000 ]', ), )
	p.Set(faces=faces, name='seg14-4')
	#: The set 'seg14-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #100 ]', ), )
	p.Set(faces=faces, name='seg15-4')
	#: The set 'seg15-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #20 ]', ), )
	p.Set(faces=faces, name='seg16-4')
	#: The set 'seg16-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #400000 ]', ), )
	p.Set(faces=faces, name='seg17-4')
	#: The set 'seg17-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #10000 ]', ), )
	p.Set(faces=faces, name='seg18-4')
	#: The set 'seg18-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #1000000 ]', ), )
	p.Set(faces=faces, name='seg19-4')
	#: The set 'seg19-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #200000 ]', ), )
	p.Set(faces=faces, name='seg20-4')
	#: The set 'seg20-4' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #2000 ]', ), )
	p.Set(faces=faces, name='seg19-4')
	#: The set 'seg19-4' has been edited (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #40000000 ]', ), )
	p.Set(faces=faces, name='seg20-4')
	#: The set 'seg20-4' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #1000000 ]', ), )
	p.Set(faces=faces, name='seg21-4')
	#: The set 'seg21-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #200000 ]', ), )
	p.Set(faces=faces, name='seg22-4')
	#: The set 'seg22-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #20000000 ]', ), )
	p.Set(faces=faces, name='seg25-4')
	#: The set 'seg25-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #40 ]', ), )
	p.Set(faces=faces, name='seg23-4')
	#: The set 'seg23-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #1 ]', ), )
	p.Set(faces=faces, name='seg24-4')
	#: The set 'seg24-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #4000 ]', ), )
	p.Set(faces=faces, name='seg26-4')
	#: The set 'seg26-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #100 ]', ), )
	p.Set(faces=faces, name='seg27-4')
	#: The set 'seg27-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #20 ]', ), )
	p.Set(faces=faces, name='seg28-4')
	#: The set 'seg28-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #400000 ]', ), )
	p.Set(faces=faces, name='seg29-4')
	#: The set 'seg29-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #10000 ]', ), )
	p.Set(faces=faces, name='seg30-4')
	#: The set 'seg30-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #2000 ]', ), )
	p.Set(faces=faces, name='seg31-4')
	#: The set 'seg31-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #80000000 ]', ), )
	p.Set(faces=faces, name='seg32-4')
	#: The set 'seg32-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #1000000 ]', ), )
	p.Set(faces=faces, name='seg33-4')
	#: The set 'seg33-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #800000 ]', ), )
	p.Set(faces=faces, name='seg34-4')
	#: The set 'seg34-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #400 ]', ), )
	p.Set(faces=faces, name='seg35-4')
	#: The set 'seg35-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #8 ]', ), )
	p.Set(faces=faces, name='seg36-4')
	#: The set 'seg36-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #4 ]', ), )
	p.Set(faces=faces, name='seg37-4')
	#: The set 'seg37-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#200000 ]', ), )
	p.Set(faces=faces, name='seg38-4')
	#: The set 'seg38-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#4000 ]', ), )
	p.Set(faces=faces, name='seg39-4')
	#: The set 'seg39-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#2000 ]', ), )
	p.Set(faces=faces, name='seg40-4')
	#: The set 'seg40-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
	p.Set(faces=faces, name='seg41-4')
	#: The set 'seg41-4' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #8000000 ]', ), )
	p.Set(faces=faces, name='seg8-5')
	#: The set 'seg8-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #1000 ]', ), )
	p.Set(faces=faces, name='seg9-5')
	#: The set 'seg9-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #200 ]', ), )
	p.Set(faces=faces, name='seg10-5')
	#: The set 'seg10-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #8 ]', ), )
	p.Set(faces=faces, name='seg11-5')
	#: The set 'seg11-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #100000 ]', ), )
	p.Set(faces=faces, name='seg12-5')
	#: The set 'seg12-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #2 ]', ), )
	p.Set(faces=faces, name='seg13-5')
	#: The set 'seg13-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #20000 ]', ), )
	p.Set(faces=faces, name='seg13-5')
	#: The set 'seg13-5' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #800 ]', ), )
	p.Set(faces=faces, name='seg14-5')
	#: The set 'seg14-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #10000000 ]', ), )
	p.Set(faces=faces, name='seg15-5')
	#: The set 'seg15-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #2000000 ]', ), )
	p.Set(faces=faces, name='seg16-5')
	#: The set 'seg16-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #80000 ]', ), )
	p.Set(faces=faces, name='seg17-5')
	#: The set 'seg17-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #10 ]', ), )
	p.Set(faces=faces, name='seg18-5')
	#: The set 'seg18-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #2 ]', ), )
	p.Set(faces=faces, name='seg19-5')
	#: The set 'seg19-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #8000000 ]', ), )
	p.Set(faces=faces, name='seg20-5')
	#: The set 'seg20-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #200 ]', ), )
	p.Set(faces=faces, name='seg21-5')
	#: The set 'seg21-5' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #1000 ]', ), )
	p.Set(faces=faces, name='seg21-5')
	#: The set 'seg21-5' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #200 ]', ), )
	p.Set(faces=faces, name='seg22-5')
	#: The set 'seg22-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #8 ]', ), )
	p.Set(faces=faces, name='seg23-5')
	#: The set 'seg23-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #100000 ]', ), )
	p.Set(faces=faces, name='seg24-5')
	#: The set 'seg24-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #20000 ]', ), )
	p.Set(faces=faces, name='seg25-5')
	#: The set 'seg25-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #800 ]', ), )
	p.Set(faces=faces, name='seg26-5')
	#: The set 'seg26-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #10000000 ]', ), )
	p.Set(faces=faces, name='seg27-5')
	#: The set 'seg27-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #2000000 ]', ), )
	p.Set(faces=faces, name='seg28-5')
	#: The set 'seg28-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #80000 ]', ), )
	p.Set(faces=faces, name='seg29-5')
	#: The set 'seg29-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #10 ]', ), )
	p.Set(faces=faces, name='seg30-5')
	#: The set 'seg30-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #4 ]', ), )
	p.Set(faces=faces, name='seg31-5')
	#: The set 'seg31-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #1000 ]', ), )
	p.Set(faces=faces, name='seg32-5')
	#: The set 'seg32-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #2000000 ]', ), )
	p.Set(faces=faces, name='seg33-5')
	#: The set 'seg33-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #2000 ]', ), )
	p.Set(faces=faces, name='seg34-5')
	#: The set 'seg34-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #400000 ]', ), )
	p.Set(faces=faces, name='seg35-5')
	#: The set 'seg35-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #10 ]', ), )
	p.Set(faces=faces, name='seg36-5')
	#: The set 'seg36-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#1000000 ]', ), )
	p.Set(faces=faces, name='seg37-5')
	#: The set 'seg37-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #2 ]', ), )
	p.Set(faces=faces, name='seg38-5')
	#: The set 'seg38-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#8000 ]', ), )
	p.Set(faces=faces, name='seg39-5')
	#: The set 'seg39-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#8 ]', ), )
	p.Set(faces=faces, name='seg40-5')
	#: The set 'seg40-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#1000 ]', ), )
	p.Set(faces=faces, name='seg41-5')
	#: The set 'seg41-5' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #4 ]', ), )
	p.Set(faces=faces, name='seg8-6')
	#: The set 'seg8-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #20000000 ]', ), )
	p.Set(faces=faces, name='seg9-6')
	#: The set 'seg9-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #40000 ]', ), )
	p.Set(faces=faces, name='seg10-6')
	#: The set 'seg10-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #400 ]', ), )
	p.Set(faces=faces, name='seg11-6')
	#: The set 'seg11-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #20 ]', ), )
	p.Set(faces=faces, name='seg12-6')
	#: The set 'seg12-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #40000 ]', ), )
	p.Set(faces=faces, name='seg13-6')
	#: The set 'seg13-6' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #4000000 ]', ), )
	p.Set(faces=faces, name='seg13-6')
	#: The set 'seg13-6' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #40000 ]', ), )
	p.Set(faces=faces, name='seg14-6')
	#: The set 'seg14-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #2000 ]', ), )
	p.Set(faces=faces, name='seg15-6')
	#: The set 'seg15-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #4 ]', ), )
	p.Set(faces=faces, name='seg16-6')
	#: The set 'seg16-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #4000000 ]', ), )
	p.Set(faces=faces, name='seg17-6')
	#: The set 'seg17-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #200000 ]', ), )
	p.Set(faces=faces, name='seg18-6')
	#: The set 'seg18-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #400 ]', ), )
	p.Set(faces=faces, name='seg19-6')
	#: The set 'seg19-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #4 ]', ), )
	p.Set(faces=faces, name='seg20-6')
	#: The set 'seg20-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #4 ]', ), )
	p.Set(faces=faces, name='seg21-6')
	#: The set 'seg21-6' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #20000000 ]', ), )
	p.Set(faces=faces, name='seg21-6')
	#: The set 'seg21-6' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #40000 ]', ), )
	p.Set(faces=faces, name='seg22-6')
	#: The set 'seg22-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #400 ]', ), )
	p.Set(faces=faces, name='seg23-6')
	#: The set 'seg23-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #20 ]', ), )
	p.Set(faces=faces, name='seg24-6')
	#: The set 'seg24-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #4000000 ]', ), )
	p.Set(faces=faces, name='seg25-6')
	#: The set 'seg25-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #40000 ]', ), )
	p.Set(faces=faces, name='seg26-6')
	#: The set 'seg26-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #2000 ]', ), )
	p.Set(faces=faces, name='seg27-6')
	#: The set 'seg27-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #4 ]', ), )
	p.Set(faces=faces, name='seg28-6')
	#: The set 'seg28-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #4000000 ]', ), )
	p.Set(faces=faces, name='seg29-6')
	#: The set 'seg29-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #200000 ]', ), )
	p.Set(faces=faces, name='seg30-6')
	#: The set 'seg30-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #400 ]', ), )
	p.Set(faces=faces, name='seg31-6')
	#: The set 'seg31-6' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #200 ]', ), )
	p.Set(faces=faces, name='seg8-7')
	#: The set 'seg8-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #10000000 ]', ), )
	p.Set(faces=faces, name='seg9-7')
	#: The set 'seg9-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #1000 ]', ), )
	p.Set(faces=faces, name='seg10-7')
	#: The set 'seg10-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #20000 ]', ), )
	p.Set(faces=faces, name='seg11-7')
	#: The set 'seg11-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #10 ]', ), )
	p.Set(faces=faces, name='seg12-7')
	#: The set 'seg12-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #100000 ]', ), )
	p.Set(faces=faces, name='seg13-7')
	#: The set 'seg13-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #2000000 ]', ), )
	p.Set(faces=faces, name='seg14-7')
	#: The set 'seg14-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #1000 ]', ), )
	p.Set(faces=faces, name='seg15-7')
	#: The set 'seg15-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #10000000 ]', ), )
	p.Set(faces=faces, name='seg16-7')
	#: The set 'seg16-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #2 ]', ), )
	p.Set(faces=faces, name='seg17-7')
	#: The set 'seg17-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #100000 ]', ), )
	p.Set(faces=faces, name='seg18-7')
	#: The set 'seg18-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #200 ]', ), )
	p.Set(faces=faces, name='seg20-7')
	#: The set 'seg20-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #10 ]', ), )
	p.Set(faces=faces, name='seg19-7')
	#: The set 'seg19-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #10000000 ]', ), )
	p.Set(faces=faces, name='seg21-7')
	#: The set 'seg21-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #1000 ]', ), )
	p.Set(faces=faces, name='seg22-7')
	#: The set 'seg22-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #20000 ]', ), )
	p.Set(faces=faces, name='seg23-7')
	#: The set 'seg23-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #10 ]', ), )
	p.Set(faces=faces, name='seg24-7')
	#: The set 'seg24-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #100000 ]', ), )
	p.Set(faces=faces, name='seg25-7')
	#: The set 'seg25-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #2000000 ]', ), )
	p.Set(faces=faces, name='seg26-7')
	#: The set 'seg26-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #1000 ]', ), )
	p.Set(faces=faces, name='seg27-7')
	#: The set 'seg27-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #10000000 ]', ), )
	p.Set(faces=faces, name='seg28-7')
	#: The set 'seg28-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #2 ]', ), )
	p.Set(faces=faces, name='seg29-7')
	#: The set 'seg29-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #100000 ]', ), )
	p.Set(faces=faces, name='seg30-7')
	#: The set 'seg30-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #10 ]', ), )
	p.Set(faces=faces, name='seg31-7')
	#: The set 'seg31-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #4000000 ]', ), )
	p.Set(faces=faces, name='seg32-7')
	#: The set 'seg32-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #200 ]', ), )
	p.Set(faces=faces, name='seg33-7')
	#: The set 'seg33-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #8000 ]', ), )
	p.Set(faces=faces, name='seg34-7')
	#: The set 'seg34-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #20 ]', ), )
	p.Set(faces=faces, name='seg35-7')
	#: The set 'seg35-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #100000 ]', ), )
	p.Set(faces=faces, name='seg36-7')
	#: The set 'seg36-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#4000000 ]', ), )
	p.Set(faces=faces, name='seg37-7')
	#: The set 'seg37-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#10000 ]', ), )
	p.Set(faces=faces, name='seg38-7')
	#: The set 'seg38-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#80000000 ]', ), )
	p.Set(faces=faces, name='seg39-7')
	#: The set 'seg39-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#20 ]', ), )
	p.Set(faces=faces, name='seg40-7')
	#: The set 'seg40-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#400 ]', ), )
	p.Set(faces=faces, name='seg41-7')
	#: The set 'seg41-7' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:8 #20 ]', ), )
	p.Set(faces=faces, name='seg8-8')
	#: The set 'seg8-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #4000000 ]', ), )
	p.Set(faces=faces, name='seg9-8')
	#: The set 'seg9-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #4000 ]', ), )
	p.Set(faces=faces, name='seg10-8')
	#: The set 'seg10-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #2000 ]', ), )
	p.Set(faces=faces, name='seg11-8')
	#: The set 'seg11-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:7 #4 ]', ), )
	p.Set(faces=faces, name='seg12-8')
	#: The set 'seg12-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #400000 ]', ), )
	p.Set(faces=faces, name='seg13-8')
	#: The set 'seg13-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #200000 ]', ), )
	p.Set(faces=faces, name='seg14-8')
	#: The set 'seg14-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:6 #400 ]', ), )
	p.Set(faces=faces, name='seg15-8')
	#: The set 'seg15-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #40000000 ]', ), )
	p.Set(faces=faces, name='seg16-8')
	#: The set 'seg16-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #20000000 ]', ), )
	p.Set(faces=faces, name='seg19-8')
	#: The set 'seg19-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #20000000 ]', ), )
	p.Set(faces=faces, name='seg17-8')
	#: The set 'seg17-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #40000 ]', ), )
	p.Set(faces=faces, name='seg18-8')
	#: The set 'seg18-8' has been created (1 face).
	p1 = mdb.models['Model-1'].parts['Part-1']
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #40 ]', ), )
	p.Set(faces=faces, name='seg19-8')
	#: The set 'seg19-8' has been edited (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:5 #20 ]', ), )
	p.Set(faces=faces, name='seg20-8')
	#: The set 'seg20-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #4000000 ]', ), )
	p.Set(faces=faces, name='seg21-8')
	#: The set 'seg21-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #4000 ]', ), )
	p.Set(faces=faces, name='seg22-8')
	#: The set 'seg22-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #2000 ]', ), )
	p.Set(faces=faces, name='seg23-8')
	#: The set 'seg23-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:4 #4 ]', ), )
	p.Set(faces=faces, name='seg24-8')
	#: The set 'seg24-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #400000 ]', ), )
	p.Set(faces=faces, name='seg25-8')
	#: The set 'seg25-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #200000 ]', ), )
	p.Set(faces=faces, name='seg26-8')
	#: The set 'seg26-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:3 #400 ]', ), )
	p.Set(faces=faces, name='seg27-8')
	#: The set 'seg27-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #40000000 ]', ), )
	p.Set(faces=faces, name='seg28-8')
	#: The set 'seg28-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #20000000 ]', ), )
	p.Set(faces=faces, name='seg29-8')
	#: The set 'seg29-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #40000 ]', ), )
	p.Set(faces=faces, name='seg30-8')
	#: The set 'seg30-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #40 ]', ), )
	p.Set(faces=faces, name='seg31-8')
	#: The set 'seg31-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #8000000 ]', ), )
	p.Set(faces=faces, name='seg32-8')
	#: The set 'seg32-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0:2 #20 ]', ), )
	p.Set(faces=faces, name='seg33-8')
	#: The set 'seg33-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #20000 ]', ), )
	p.Set(faces=faces, name='seg34-8')
	#: The set 'seg34-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #40 ]', ), )
	p.Set(faces=faces, name='seg35-8')
	#: The set 'seg35-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#0 #10000 ]', ), )
	p.Set(faces=faces, name='seg36-8')
	#: The set 'seg36-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#10000000 ]', ), )
	p.Set(faces=faces, name='seg37-8')
	#: The set 'seg37-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#20000 ]', ), )
	p.Set(faces=faces, name='seg38-8')
	#: The set 'seg38-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#8000000 ]', ), )
	p.Set(faces=faces, name='seg39-8')
	#: The set 'seg39-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#80 ]', ), )
	p.Set(faces=faces, name='seg40-8')
	#: The set 'seg40-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=('[#40 ]', ), )
	p.Set(faces=faces, name='seg41-8')
	#: The set 'seg41-8' has been created (1 face).
	p = mdb.models['Model-1'].parts['Part-1']
	p.generateMesh()
	del mdb.models['Model-1'].materials['Material-1']
	del mdb.models['Model-1'].sections['Section-1']
	p = mdb.models['Model-1'].parts['Part-1']
	del mdb.models['Model-1'].parts['Part-1'].sectionAssignments[0]
	mdb.models['Model-1'].Material(name='AL-OUTER')
	mdb.models['Model-1'].materials['AL-OUTER'].Density(table=((2.7e-09, ), ))
	mdb.models['Model-1'].materials['AL-OUTER'].Elastic(table=((elastic_modulus_outer, poissions_ratio_outer), ))
	mdb.models['Model-1'].materials['AL-OUTER'].Plastic(table=plastic_strain_stress)
	mdb.models['Model-1'].Material(name='AL-RIB')
	mdb.models['Model-1'].materials['AL-RIB'].Elastic(table=((elastic_modulus_rib, poissions_ratio_rib), ))
	mdb.models['Model-1'].materials['AL-RIB'].Plastic(table=plastic_strain_stress)
	mdb.models['Model-1'].materials['AL-RIB'].Density(table=((2.7e-09, ), ))
	mdb.models['Model-1'].HomogeneousSolidSection(name='Section-OUTER', 
		material='AL-OUTER', thickness=None)
	del mdb.models['Model-1'].sections['Section-OUTER']
	mdb.models['Model-1'].HomogeneousShellSection(name='Section-OUTER', 
		preIntegrate=OFF, material='AL-OUTER', thicknessType=UNIFORM, 
		thickness=2.0, thicknessField='', nodalThicknessField='', 
		idealization=NO_IDEALIZATION, poissonDefinition=DEFAULT, 
		thicknessModulus=None, temperature=GRADIENT, useDensity=OFF, 
		integrationRule=SIMPSON, numIntPts=3)
	mdb.models['Model-1'].HomogeneousShellSection(name='Section-RIB', 
		preIntegrate=OFF, material='AL-RIB', thicknessType=UNIFORM, thickness=2.0, 
		thicknessField='', nodalThicknessField='', idealization=NO_IDEALIZATION, 
		poissonDefinition=DEFAULT, thicknessModulus=None, temperature=GRADIENT, 
		useDensity=OFF, integrationRule=SIMPSON, numIntPts=3)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=(
		'[#ffdf9ffe #7e7ffbf3 #fb9edbff #dbfb9edb #9edbfb9e #fb9edbfb #dbfb9edb', 
		' #9edbfb9e #fb9edbfb #3f3d9edb ]', ), )
	region = p.Set(faces=faces, name='Set-314')
	p = mdb.models['Model-1'].parts['Part-1']
	p.SectionAssignment(region=region, sectionName='Section-OUTER', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', 
		thicknessAssignment=FROM_SECTION)
	p = mdb.models['Model-1'].parts['Part-1']
	f = p.faces
	faces = f.getSequenceFromMask(mask=(
		'[#206001 #8180040c #4612400 #24046124 #61240461 #4612404 #24046124', 
		' #61240461 #4612404 #c26124 ]', ), )
	region = p.Set(faces=faces, name='Set-315')
	p = mdb.models['Model-1'].parts['Part-1']
	p.SectionAssignment(region=region, sectionName='Section-RIB', offset=0.0, 
		offsetType=MIDDLE_SURFACE, offsetField='', 
		thicknessAssignment=FROM_SECTION)
	myjob = mdb.Job(name='Job-1', model='Model-1', description='', type=ANALYSIS, 
		atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
		memoryUnits=PERCENTAGE, explicitPrecision=SINGLE, 
		nodalOutputPrecision=SINGLE, echoPrint=OFF, modelPrint=OFF, 
		contactPrint=OFF, historyPrint=OFF, userSubroutine='', scratch='', 
		resultsFormat=ODB, parallelizationMethodExplicit=DOMAIN, numDomains=6, 
		activateLoadBalancing=False, multiprocessingMode=DEFAULT, numCpus=6)
	mdb.jobs['Job-1'].submit(consistencyChecking=OFF)
	myjob.waitForCompletion()
	####
	print('start to extract dataset from odb file')
	try:
		current_odb = session.openOdb(name=abaqus_working_path)
		part = current_odb.rootAssembly.instances['PART-1-1']
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
			eLeaf = dgo.LeafFromElementSets(elementSets=("PART-1-1." + seg_key, ))
			nLeaf = dgo.LeafFromNodeSets(nodeSets=("PART-1-1." + seg_key, ))
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
	except Exception as e:
		print(e)
		np.save(root_path+'/force/cf_npy/'+name(n_s, '.npy'), np.zeros(shape=[3,3]))
		np.save(root_path + '/energy/sea_npy/' + name(n_s, '.npy'), np.zeros(shape=[3,3]))
		time.sleep(5)
	print('========{}th odb file has been extracted========'.format(n_s))
	try:
		current_odb.close()
	except: pass

