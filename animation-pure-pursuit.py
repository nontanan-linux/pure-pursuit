########################################################################################################
########################################################################################################
####								Preject : Pure pursuit simulation								####
####								Developer : Nontanan Sommat										####
########################################################################################################
########################################################################################################

########################################################################################################
####									Impoer Library												####
########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import WaypointGenerator as wg
import time
# from IPython import display
########################################################################################################

########################################################################################################
####									Helper function												####
########################################################################################################
# Vehicle parameters (m)
LENGTH = 3.0 #vehicle lenght
WIDTH = 2.0 #vehicle width
BACKTOWHEEL = 0.5 #center back vehicle to back of car
WHEEL_LEN = 0.3
WHEEL_WIDTH = 0.2
TREAD = 0.7
WB = 2.0

def plotVehicle(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):
	"""
	The function is to plot the vehicle
	it is copied from https://github.com/AtsushiSakai/PythonRobotics/blob/187b6aa35f3cbdeca587c0abdb177adddefc5c2a/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py#L109
	"""
	outline = np.array(
		[
			[
				-BACKTOWHEEL,
				(LENGTH - BACKTOWHEEL),
				(LENGTH - BACKTOWHEEL),
				-BACKTOWHEEL,
				-BACKTOWHEEL,
			],
			[WIDTH / 2, WIDTH / 2, -WIDTH / 2, -WIDTH / 2, WIDTH / 2],
		]
	)

	fr_wheel = np.array(
		[
			[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
			[
				-WHEEL_WIDTH - TREAD,
				-WHEEL_WIDTH - TREAD,
				WHEEL_WIDTH - TREAD,
				WHEEL_WIDTH - TREAD,
				-WHEEL_WIDTH - TREAD,
			],
		]
	)
	# print(outline)
	# print(fr_wheel)

	rr_wheel = np.copy(fr_wheel)
	fl_wheel = np.copy(fr_wheel)
	fl_wheel[1, :] *= -1
	rl_wheel = np.copy(rr_wheel)
	rl_wheel[1, :] *= -1

	Rot1 = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
	Rot2 = np.array([[math.cos(steer), math.sin(steer)], [-math.sin(steer), math.cos(steer)]])

	fr_wheel = (fr_wheel.T.dot(Rot2)).T
	fl_wheel = (fl_wheel.T.dot(Rot2)).T
	fr_wheel[0, :] += WB
	fl_wheel[0, :] += WB

	fr_wheel = (fr_wheel.T.dot(Rot1)).T
	fl_wheel = (fl_wheel.T.dot(Rot1)).T

	outline = (outline.T.dot(Rot1)).T
	rr_wheel = (rr_wheel.T.dot(Rot1)).T
	rl_wheel = (rl_wheel.T.dot(Rot1)).T

	outline[0, :] += x
	outline[1, :] += y
	fr_wheel[0, :] += x
	fr_wheel[1, :] += y
	rr_wheel[0, :] += x
	rr_wheel[1, :] += y
	fl_wheel[0, :] += x
	fl_wheel[1, :] += y
	rl_wheel[0, :] += x
	rl_wheel[1, :] += y

	plt.plot(
		np.array(outline[0, :]).flatten(),
		np.array(outline[1, :]).flatten(), 
		truckcolor
	)
	plt.plot(
		np.array(fr_wheel[0, :]).flatten(),
		np.array(fr_wheel[1, :]).flatten(),
		truckcolor,
	)
	plt.plot(
		np.array(rr_wheel[0, :]).flatten(),
		np.array(rr_wheel[1, :]).flatten(),
		truckcolor,
	)
	plt.plot(
		np.array(fl_wheel[0, :]).flatten(),
		np.array(fl_wheel[1, :]).flatten(),
		truckcolor,
	)
	plt.plot(
		np.array(rl_wheel[0, :]).flatten(),
		np.array(rl_wheel[1, :]).flatten(),
		truckcolor,
	)
	plt.plot(x, y, "*")

class Vehicle:
	def __init__(self, x, y, yaw, vel=0):
		"""
		Define a vehicle class
		:param x: float, x position
		:param y: float, y position
		:param yaw: float, vehicle heading
		:param vel: float, velocity
		"""
		self.x = x
		self.y = y
		self.yaw = yaw
		self.vel = vel

	def update(self, acc, delta):
		"""
		Vehicle motion model, here we are using simple bycicle model
		:param acc: float, acceleration
		:param delta: float, heading control
		"""
		self.x += self.vel * math.cos(self.yaw) * dt
		self.y += self.vel * math.sin(self.yaw) * dt
		self.yaw += self.vel * math.tan(delta) / WB * dt
		self.vel += acc * dt

class PID:
	def __init__(self, kp=0.8, ki=0.1, kd=0.001):
		"""
		Define a PID controller class
		:param kp: float, kp coeff
		:param ki: float, ki coeff
		:param kd: float, kd coeff
		:param ki: float, ki coeff
		"""
		self.kp = kp
		self.ki = ki
		self.kd = kd
		self.Pterm = 0.0
		self.Iterm = 0.0
		self.Dterm = 0.0
		self.last_error = 0.0
	def control(self, error):
		"""
		PID main function, given an input, this function will output a control unit
		:param error: float, error term
		:return: float, output control
		"""
		self.Pterm = self.kp * error
		self.Iterm += self.ki*error * dt
		self.Dterm += self.kd*error/dt
		self.last_error = error
		output = self.Pterm + self.ki * self.Iterm
		return output

def pt_to_pt_distance (pt1,pt2):
	distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
	return distance

# returns -1 if num is negative, 1 otherwise
def sgn (num):
	if num >= 0:
		return 1
	else:
		return -1
########################################################################################################

########################################################################################################
####											Path												####
########################################################################################################
path = [[0.0, 0.0], [0.571194595265405, -0.4277145118491421], [1.1417537280142898, -0.8531042347260006], [1.7098876452457967, -1.2696346390611464], [2.2705328851607995, -1.6588899151216996], [2.8121159420106827, -1.9791445882187304], [3.314589274316711, -2.159795566252656], [3.7538316863009027, -2.1224619985315876], [4.112485112342358, -1.8323249172947023], [4.383456805594431, -1.3292669972090994], [4.557386228943757, -0.6928302521681386], [4.617455513800438, 0.00274597627737883], [4.55408382321606, 0.6984486966257434], [4.376054025556597, 1.3330664239172116], [4.096280073621794, 1.827159263675668], [3.719737492364894, 2.097949296701878], [3.25277928312066, 2.108933125822431], [2.7154386886417314, 1.9004760368018616], [2.1347012144725985, 1.552342808106984], [1.5324590525923942, 1.134035376721349], [0.9214084611203568, 0.6867933269918683], [0.30732366808208345, 0.22955002391894264], [-0.3075127599907512, -0.2301742560363831], [-0.9218413719658775, -0.6882173194028102], [-1.5334674079795052, -1.1373288016589413], [-2.1365993767877467, -1.5584414896876835], [-2.7180981380280307, -1.9086314914221845], [-3.2552809639439704, -2.1153141204181285], [-3.721102967810494, -2.0979137913841046], [-4.096907306768644, -1.8206318841755131], [-4.377088212533404, -1.324440752295139], [-4.555249804461285, -0.6910016662308593], [-4.617336323713965, 0.003734984720118972], [-4.555948690867849, 0.7001491248072772], [-4.382109193278264, 1.3376838311365633], [-4.111620918085742, 1.8386823176628544], [-3.7524648889185794, 2.1224985058331005], [-3.3123191098095615, 2.153588702898333], [-2.80975246649598, 1.9712114570096653], [-2.268856462266256, 1.652958931009528], [-1.709001159778989, 1.2664395490411673], [-1.1413833971013372, 0.8517589252820573], [-0.5710732645795573, 0.4272721367616211], [0, 0], [0.571194595265405, -0.4277145118491421]]
# path1 = [[-4,-2],[-4,2],[4,2],[4,-2],[-4,-2]]
traj_x = np.arange(0, 100, 5)
traj_y = [math.sin(x / 10.0) * x / 2.0 for x in traj_x]
upper_y = [math.sin(x / 10.0) *2* x / 2.0 for x in traj_x]
lower_y = [math.sin(x / 10.0) * x / 2.0 /2 for x in traj_x]
path2 = []
for i in range(0, len(traj_y)):
	path2.append([traj_x[i],traj_y[i]])

########################################################################################################
####								Pure pursuit function											####
########################################################################################################
# this function needs to return 3 things IN ORDER: goalPt, lastFoundIndex, turnVel
# think about this function as a snapshot in a while loop
# given all information about the robot's current state, what should be the goalPt, lastFoundIndex, and turnVel?
# the LFindex takes in the value of lastFoundIndex as input. Looking at it now I can't remember why I have it.
# it is this way because I don't want the global lastFoundIndex to get modified in this function, instead, this function returns the updated lastFoundIndex value 
# this function will be feed into another function for creating animation
def pure_pursuit_step (path, currentPos, currentHeading, lookAheadDis, LFindex) :
	# extract currentX and currentY
	currentX = currentPos[0]
	currentY = currentPos[1]

	# use for loop to search intersections
	lastFoundIndex = LFindex
	intersectFound = False
	startingIndex = lastFoundIndex

	for i in range (startingIndex, len(path)-1):
		# beginning of line-circle intersection code
		x1 = path[i][0] - currentX
		y1 = path[i][1] - currentY
		x2 = path[i+1][0] - currentX
		y2 = path[i+1][1] - currentY
		dx = x2 - x1
		dy = y2 - y1
		dr = math.sqrt (dx**2 + dy**2)
		D = x1*y2 - x2*y1
		discriminant = (lookAheadDis**2) * (dr**2) - D**2

		if discriminant >= 0:
			sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
			sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
			sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
			sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2

			sol_pt1 = [sol_x1 + currentX, sol_y1 + currentY]
			sol_pt2 = [sol_x2 + currentX, sol_y2 + currentY]
			# end of line-circle intersection code

			minX = min(path[i][0], path[i+1][0])
			minY = min(path[i][1], path[i+1][1])
			maxX = max(path[i][0], path[i+1][0])
			maxY = max(path[i][1], path[i+1][1])

			# if one or both of the solutions are in range
			if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
				foundIntersection = True
				# if both solutions are in range, check which one is better
				if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
					# make the decision by compare the distance between the intersections and the next point in path
					if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
						goalPt = sol_pt1
					else:
						goalPt = sol_pt2

				# if not both solutions are in range, take the one that's in range
				else:
					# if solution pt1 is in range, set that as goal point
					if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
						goalPt = sol_pt1
					else:
						goalPt = sol_pt2
		  
				# only exit loop if the solution pt found is closer to the next pt in path than the current pos
				if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
					# update lastFoundIndex and exit
					lastFoundIndex = i
					break
				else:
		  			# in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
		  			lastFoundIndex = i+1
			# if no solutions are in range
			else:
				foundIntersection = False
				# no new intersection found, potentially deviated from the path
				# follow path[lastFoundIndex]
				goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

	# obtained goal point, now compute turn vel
	# initialize proportional controller constant
	Kp = 1

	# calculate absTargetAngle with the atan2 function
	absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/pi
	if absTargetAngle < 0: absTargetAngle += 360

	# compute turn error by finding the minimum angle
	turnError = absTargetAngle - currentHeading
	if turnError > 180 or turnError < -180 :
		turnError = -1 * sgn(turnError) * (360 - abs(turnError))
  
	# apply proportional controller
	turnVel = Kp*turnError
  
	return goalPt, lastFoundIndex, turnVel

def pure_pursuit_step2 (path, currentPos, currentHeading, lookAheadDis, LFindex) :
	# extract currentX and currentY
	currentX = currentPos[0]
	currentY = currentPos[1]

	# use for loop to search intersections
	lastFoundIndex = LFindex
	intersectFound = False
	startingIndex = lastFoundIndex

	for i in range (startingIndex, len(path)-1):
		# beginning of line-circle intersection code
		j = 1
		x1 = path[i][0] - currentX
		y1 = path[i][1] - currentY
		x2 = path[i+j][0] - currentX
		y2 = path[i+j][1] - currentY
		line_slope = (y2-y1)/(x2-x1)
		line_intercept = y1 - line_slope*x1
		A = 1 + line_slope**2
		B = 2*line_slope*line_intercept - 2*line_slope*currentY -2*currentX
		C = currentX**2 + currentY**2 + line_intercept**2 -2*line_intercept*currentY - lookAheadDis**2
		discriminant = B**2 - 4 * A * C

		if discriminant >= 0:
			sol_x1 = (-B + math.sqrt(discriminant)) / (2 * A)
			sol_x2 = (-B - math.sqrt(discriminant)) / (2 * A)
			sol_y1 = line_slope * x1 + line_intercept
			sol_y2 = line_slope * x2 + line_intercept

			sol_pt1 = [sol_x1, sol_y1 + currentY]
			sol_pt2 = [sol_x2, sol_y2 + currentY]
	  		# end of line-circle intersection code

			# minX = min(x1,x2)
			# minY = min(y1,y2)
			# maxX = max(x1,x2)
			# maxY = max(y1,y2)
			minX = min(path[i][0], path[i+1][0])
			minY = min(path[i][1], path[i+1][1])
			maxX = max(path[i][0], path[i+1][0])
			maxY = max(path[i][1], path[i+1][1])

			# if one or both of the solutions are in range
			if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) or ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
				foundIntersection = True
			
				# if both solutions are in range, check which one is better
				if ((minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY)) and ((minX <= sol_pt2[0] <= maxX) and (minY <= sol_pt2[1] <= maxY)):
					# make the decision by compare the distance between the intersections and the next point in path
					if pt_to_pt_distance(sol_pt1, path[i+1]) < pt_to_pt_distance(sol_pt2, path[i+1]):
						goalPt = sol_pt1
					else:
						goalPt = sol_pt2
				# if not both solutions are in range, take the one that's in range
				else:
					# if solution pt1 is in range, set that as goal point
					if (minX <= sol_pt1[0] <= maxX) and (minY <= sol_pt1[1] <= maxY):
						goalPt = sol_pt1
					else:
						goalPt = sol_pt2
  
				# only exit loop if the solution pt found is closer to the next pt in path than the current pos
				if pt_to_pt_distance (goalPt, path[i+1]) < pt_to_pt_distance ([currentX, currentY], path[i+1]):
		  			# update lastFoundIndex and exit
					lastFoundIndex = i
					break
				else:
					# in case for some reason the robot cannot find intersection in the next path segment, but we also don't want it to go backward
					lastFoundIndex = i+1
		# if no solutions are in range
		else:
			foundIntersection = False
			# no new intersection found, potentially deviated from the path
			# follow path[lastFoundIndex]
			goalPt = [path[lastFoundIndex][0], path[lastFoundIndex][1]]

	# obtained goal point, now compute turn vel
	# initialize proportional controller constant
	Kp = 3

	# calculate absTargetAngle with the atan2 function
	absTargetAngle = math.atan2 (goalPt[1]-currentPos[1], goalPt[0]-currentPos[0]) *180/pi
	if absTargetAngle < 0: absTargetAngle += 360

	# compute turn error by finding the minimum angle
	turnError = absTargetAngle - currentHeading
	if turnError > 180 or turnError < -180 :
		turnError = -1 * sgn(turnError) * (360 - abs(turnError))

	# apply proportional controller
	turnVel = Kp*turnError
	return goalPt, lastFoundIndex, turnVel
########################################################################################################

########################################################################################################
####									Setup for simulation										####
########################################################################################################
# the code below is for animation
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------
# THIS IS DIFFERENT THAN BEFORE! initialize variables here
# you can also change the Kp constant which is located at line 113
# for the sake of my sanity
pi = np.pi
currentPos = [0, 0]
currentHeading = 330
yaw = currentHeading*2*pi/360
lastFoundIndex = 0
lookAheadDis = 4
linearVel = 2
dt = 0.5
prev_time = 0


# set this to true if you use rotations
using_rotation = False

# this determines how long (how many frames) the animation will run. 400 frames takes around 30 seconds.
numOfFrames = 400

########################################################################################################

########################################################################################################
####									Animation function											####
########################################################################################################
def main():
	# define globals
	global currentPos
	global currentHeading
	global lastFoundIndex
	global linearVel
	global yaw
	global prev_time

	path_gen = wg.add_more_points2(path2,0.2)
	path_gen = wg.autoSmooth(path_gen,70)
	gen_x = []
	gen_y = []

	for i in range(0,len(path_gen)):
		gen_x.append(path_gen[i][0])
		gen_y.append(path_gen[i][1])

	# path_gen = path2

	reach_point = path_gen[len(path_gen)-1]

	# real trajectory
	traj_ego_x = []
	traj_ego_y = []

	# model: 200rpm drive with 18" width
	#               rpm   /s  circ   feet
	maxLinVelfeet = 200 / 60 * pi*4 / 12
	#               rpm   /s  center angle   deg
	maxTurnVelDeg = 200 / 60 * pi*4 / 9 *180/pi
	# for the animation to loop
	if lastFoundIndex <= len(path_gen)-2 :
		plt.figure(figsize=(12, 8))
		while(pt_to_pt_distance(currentPos, reach_point) >= lookAheadDis):
	 		# call pure_pursuit_step to get info
			goalPt, lastFoundIndex, turnVel = pure_pursuit_step (path_gen, currentPos, currentHeading, lookAheadDis, lastFoundIndex)
			print("index : {}	|	current pose : {}	|	heading : {}".format(lastFoundIndex,currentPos,currentHeading))
			print("goal point : {}".format(goalPt))
			# print("Reach : {}".format(pt_to_pt_distance(currentPos, reach_point) >= lookAheadDis))
			# model: 200rpm drive with 18" width
			#               rpm   /s  circ   feet
			# update x and y, but x and y stays constant here
			yaw_err = math.atan2(goalPt[1] - currentPos[0], goalPt[0] - currentPos[0])
			# stepDis = linearVel * maxLinVelfeet * dt
			stepDis = linearVel*dt
			currentPos[0] += stepDis * np.cos(currentHeading*pi/180)
			currentPos[1] += stepDis * np.sin(currentHeading*pi/180)
			# yaw_err = math.atan2(goalPt[1] - currentPos[0], goalPt[0] - currentPos[0])
			# currentHeading += turnVel/10 * maxTurnVelDeg * dt
			# 360 	=	2*pi
			# x		=	(x*2*pi)/360
			currentHeading += turnVel*dt
			if using_rotation == False :
				currentHeading = currentHeading%360
				if currentHeading < 0: currentHeading += 360
			yaw_err = yaw_err*2*pi/360
			yaw = currentHeading*2*pi/360
			# store the trajectory
			traj_ego_x.append(currentPos[0])
			traj_ego_y.append(currentPos[1])

			# plots
			plt.cla()
			plt.plot(traj_x, traj_y,"-*",color= "black", linewidth=1, label="original course")
			plt.plot(gen_x, gen_y,"--",color= "grey", linewidth=1, label="generate course")
			plt.plot(traj_ego_x, traj_ego_y,"-",color= "blue", linewidth=1, label="trajectory")
			plt.plot(currentPos[0], currentPos[1],"o" ,color = "red",label="currentPos")
			plt.plot(goalPt[0], goalPt[1], "og", ms=5, label="target point")
			plotVehicle(x=currentPos[0], y=currentPos[1], yaw=yaw, steer=yaw_err)
			plt.xlabel("x[m]")
			plt.ylabel("y[m]")
			plt.axis("equal")
			plt.legend()
			plt.grid(True)
			plt.pause(0.1)
		# plt.show()

if __name__ == "__main__":
	# anim = animation.FuncAnimation (fig, pure_pursuit_animation, frames = numOfFrames, interval = 50)
	main()
	plt.show()