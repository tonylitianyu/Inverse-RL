import pybullet as p
import time
import math
from datetime import datetime
import numpy as np
import random

clid = p.connect(p.SHARED_MEMORY)
if (clid<0):
	p.connect(p.GUI)
p.loadURDF("plane.urdf",[0,0,-.98])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
sawyerId = p.loadURDF("sawyer_robot/sawyer_description/urdf/sawyer.urdf",[0,0,0])
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)
p.resetBasePositionAndOrientation(sawyerId,[0,0,0],[0,0,0,1])

#bad, get it from name! sawyerEndEffectorIndex = 18
sawyerEndEffectorIndex = 16
numJoints = p.getNumJoints(sawyerId)
#joint damping coefficents
jd=[0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]

p.setGravity(0,0,0)
t=0.
prevPose=[0,0,0]
prevPose1=[0,0,0]
hasPrevPose = 0

useRealTimeSimulation = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 15



optimal_action = np.load('trained_model/example1.npy')
print(optimal_action.shape)

side_length = 0.5
unit_distance = side_length/float(optimal_action.shape[1]-1)


left_top_x = 0.9
left_top_y = 0.25
height_limit = side_length
corner = [[left_top_x,left_top_y,0.0],[left_top_x,left_top_y-side_length,0.0],[left_top_x-side_length,left_top_y-side_length,0.0],[left_top_x-side_length,left_top_y,0.0]]
curr_pos = [0.4,-0.25,0.5] #xyz
print(corner)
def moveForward():
    if curr_pos[0] < left_top_x:
        curr_pos[0] += unit_distance

def moveBackward():
    if curr_pos[0] > (left_top_x - side_length):
        curr_pos[0] -= unit_distance

def moveLeft():
    if curr_pos[1] < left_top_y:
        curr_pos[1] += unit_distance

def moveRight():
    if curr_pos[1] > (left_top_y - side_length):
        curr_pos[1] -= unit_distance

def moveUp():
    if curr_pos[2] < height_limit:
        curr_pos[2] += unit_distance

def moveDown():
    if curr_pos[2] > 0.0:
        curr_pos[2] -= unit_distance

def moveAction(str):
    if str == 'FORWARD':
        moveForward()
    elif str == 'BACKWARD':
        moveBackward()
    elif str == 'LEFT':
        moveLeft()
    elif str == 'RIGHT':
        moveRight()
    elif str == 'UP':
        moveUp()
    elif str == 'DOWN':
        moveDown()


def from_coor_to_index(pos):
    yi = int((left_top_x - pos[0])/unit_distance)
    xi = int((left_top_y - pos[1])/unit_distance)
    zi = int(pos[2]/unit_distance)

    return xi,yi,zi

def from_index_to_coor(xi,yi,zi):
    pos = [0.0, 0.0, 0.0]
    pos[0] = left_top_x - yi * unit_distance
    pos[1] = left_top_y - xi * unit_distance
    pos[2] = zi*unit_distance

    return pos


print(from_coor_to_index(curr_pos))
print(optimal_action[0])
print(optimal_action[1])
# print(optimal_action[0][2][2][1])  #gzyx
# print(from_index_to_coor(2,2,2))


#curr_pos = from_index_to_coor(2,2,2)
curr_pos = from_index_to_coor(random.randint(0,2),random.randint(0,2),random.randint(0,2))
gripper = 0
	
while 1:
    if (useRealTimeSimulation):
        dt = datetime.now()
        t = (dt.second/60.)*2.*math.pi
        print (t)
    else:
        t=t+0.01
        time.sleep(1.0)

    for i in range (1):
        #pos = [1.0,0.2*math.cos(t),0.+0.2*math.sin(t)]
        pos = curr_pos
        xi,yi,zi = from_coor_to_index(pos)
        print(xi,yi,zi)
        action_str = optimal_action[gripper][zi][yi][xi]
        print(action_str)
        moveAction(action_str)
        if action_str == 'OPEN' or action_str == 'CLOSE':
            gripper = int(not bool(gripper))

        #print(pos)
        #moveLeft()
        jointPoses = p.calculateInverseKinematics(sawyerId,sawyerEndEffectorIndex,pos,jointDamping=jd)

        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range (numJoints):
            jointInfo = p.getJointInfo(sawyerId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(sawyerId,i,jointPoses[qIndex-7])

    ls = p.getLinkState(sawyerId,sawyerEndEffectorIndex)
    if (hasPrevPose):
        p.addUserDebugLine(prevPose,pos,[0,0,0.3],1,trailDuration)
        p.addUserDebugLine(prevPose1,ls[4],[1,0,0],1,trailDuration)
    prevPose=pos
    prevPose1=ls[4]
    hasPrevPose = 1