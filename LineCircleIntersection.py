################################################################
####			Import Necessary Libraries					####
################################################################
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from IPython import display
################################################################

################################################################
####			Helper Functions for Graphing				####
################################################################
# Below are some helper functions to help visualize the output of pure pursuit and path generator
# modify these functions to graph lines in the style you like
def add_line (path) : 
	for i in range (0,len(path)):
		plt.plot(path[i][0],path[i][1],'.',color='red',markersize=10)
	for i in range(0,len(path)-1):
		plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],color='b')
		plt.axis('scaled')
		# plt.show()

def add_complicated_line (path,lineStyle,lineColor,lineLabel) :
	for i in range (0,len(path)):
		plt.plot(path[i][0],path[i][1],'.',color='red',markersize=10)
	for i in range(0,len(path)-1):
		if(i == 0):
			# plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],color='b')
			plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],lineStyle,color=lineColor,label=lineLabel)    
		else:
			plt.plot([path[i][0],path[i+1][0]],[path[i][1],path[i+1][1]],lineStyle,color=lineColor)        
	plt.axis('scaled')
			
def highlight_points (points, pointColor):
	for point in points :
		plt.plot(point[0], point[1], '.', color = pointColor, markersize = 10)
		
def draw_circle (x, y, r, circleColor):
	xs = []
	ys = []
	angles = np.arange(0, 2.2*np.pi, 0.2)        
	for angle in angles :
		xs.append(r*np.cos(angle) + x)
		ys.append(r*np.sin(angle) + y)
	plt.plot(xs, ys, '-', color = circleColor)
################################################################

################################################################
####				Line-Circle Intersection				####
################################################################
# helper function: sgn(num)
# returns -1 if num is negative, 1 otherwise
def sgn (num):  
	if num >= 0:
		return 1
	else:
		return -1
	
# currentPos: [currentX, currentY]
# pt1: [x1, y1]
# pt2: [x2, y2]
def line_circle_intersection (currentPos, pt1, pt2, lookAheadDis):
	# extract currentX, currentY, x1, x2, y1, and y2 from input arrays
	currentX = currentPos[0]
	currentY = currentPos[1]
	x1 = pt1[0]  
	y1 = pt1[1]
	x2 = pt2[0]  
	y2 = pt2[1]
	
	# boolean variable to keep track of if intersections are found
	intersectFound = False
	
	# output (intersections found) should be stored in arrays sol1 and sol2
	# if two solutions are the same, store the same values in both sol1 and sol2
	
	# subtract currentX and currentY from [x1, y1] and [x2, y2] to offset the system to origin
	x1_offset = x1 - currentX
	y1_offset = y1 - currentY
	x2_offset = x2 - currentX
	y2_offset = y2 - currentY  
	
	# calculate the discriminant using equations from the wolframalpha article
	dx = x2_offset - x1_offset
	dy = y2_offset - y1_offset
	dr = math.sqrt (dx**2 + dy**2)
	D = x1_offset*y2_offset - x2_offset*y1_offset
	discriminant = (lookAheadDis**2) * (dr**2) - D**2  
	
	# if discriminant is >= 0, there exist solutions
	if discriminant >= 0:
		# calculate the solutions
		sol_x1 = (D * dy + sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
		sol_x2 = (D * dy - sgn(dy) * dx * np.sqrt(discriminant)) / dr**2
		sol_y1 = (- D * dx + abs(dy) * np.sqrt(discriminant)) / dr**2
		sol_y2 = (- D * dx - abs(dy) * np.sqrt(discriminant)) / dr**2
		
		# add currentX and currentY back to the solutions, offset the system back to its original position
		sol1 = [sol_x1 + currentX, sol_y1 + currentY]
		sol2 = [sol_x2 + currentX, sol_y2 + currentY]
		
		# find min and max x y values
		minX = min(x1, x2)
		maxX = max(x1, x2)
		minY = min(y1, y2)
		maxY = max(y1, y2)
		
		# check to see if any of the two solution points are within the correct range
		# for a solution point to be considered valid, its x value needs to be within minX and maxX AND its y value needs to be between minY and maxY
		# if sol1 OR sol2 are within the range, intersection is found
		if (minX <= sol1[0] <= maxX and minY <= sol1[1] <= maxY) or (minX <= sol2[0] <= maxX and minY <= sol2[1] <= maxY) :
			intersectFound = True 
			# now do a more detailed check to determine which point is valid, which is not
			if (minX <= sol1[0] <= maxX and minY <= sol1[1] <= maxY) :
				print ('solution 1 is valid!') 
			if (minX <= sol2[0] <= maxX and minY <= sol2[1] <= maxY) :
				print ('solution 1 is valid!')
		return sol1,sol2
	return 0,0
	# graphing functions to visualize the outcome
	# ----------------------------------------------------------------------------------------------------------------------------------------
	# plt.plot ([x1, x2], [y1, y2], '--', color='grey')
	# draw_circle (currentX, currentY, lookAheadDis, 'orange')
	# if intersectFound == False :
	# 	print ('No intersection Found!')
	# else:
	# 	print ('Solution 1 found at [{}, {}]'.format(sol1[0], sol1[1]))
	# 	print ('Solution 2 found at [{}, {}]'.format(sol2[0], sol2[1]))
	# 	plt.plot (sol1[0], sol1[1], '.', markersize=10, color='red', label='sol1')
	# 	plt.plot (sol2[0], sol2[1], '.', markersize=10, color='blue', label='sol2')
	# 	plt.legend()
	
	# plt.axis('scaled')
	# plt.show()

# import math

def line_circle_intersection2(circle_center, circle_radius, line_slope, line_intercept):
	# Extract circle center coordinates (a, b) from the circle_center tuple
	a, b = circle_center

	# Calculate the coefficients of the quadratic equation for intersection points
	A = 1 + line_slope**2
	B = -2 * (a + line_slope * (line_intercept - b))
	C = a**2 + (line_intercept - b)**2 - circle_radius**2

	# Calculate the discriminant
	discriminant = B**2 - 4 * A * C

	# Initialize lists to store the intersection points
	intersection_points = []

	# Check if the line and the circle intersect
	if discriminant >= 0:
		# Calculate the x-coordinates of the intersection points
		x1 = (-B + math.sqrt(discriminant)) / (2 * A)
		x2 = (-B - math.sqrt(discriminant)) / (2 * A)

		# Calculate the y-coordinates of the intersection points
		y1 = line_slope * x1 + line_intercept
		y2 = line_slope * x2 + line_intercept

		# Add the intersection points to the list
		intersection_points.append((x1, y1))
		intersection_points.append((x2, y2))

	return intersection_points

def generate_linrear_equation(ptr0,ptr1,eqfrom = ""):
	# extract x1,y1 and x2,y2
	x1,y1 = ptr0[0],ptr0[1]
	x2,y2 = ptr1[0],ptr1[1]
	if eqfrom == "slope-intersept":
		# generate slope
		m = (y2 - y1)/(x2 - x1)
		# generate y-interseption
		b = y1 - m*x1
		# y = mx + b
		return m,b
	if eqfrom == "general-form":
		DX = y2-y1
		DY = x2-x1
		CY = y1*(x2-x1) - y1*(y2-y1)
		# DX*x + DY*y + CY = 0
		return DX,DY,CY

def find_interseption(currentPose,ptr0,ptr1,lookAheadDis):
	a = currentPose[0]
	b = currentPose[1]
	line_slope = (ptr1[1] - ptr0[1])/(ptr1[0] - ptr0[0])
	line_intercept = ptr0[1]-line_slope*ptr0[0]
	circle_radius = lookAheadDis
	A = 1 + line_slope**2
	B = -2 * (a + line_slope * (line_intercept - b))
	C = a**2 + (line_intercept - b)**2 - circle_radius**2
	discriminant = B**2 - 4 * A * C
	if discriminant >= 0:
		x1 = (-B + math.sqrt(discriminant)) / (2 * A)
		x2 = (-B - math.sqrt(discriminant)) / (2 * A)
		y1 = line_slope * x1 + line_intercept
		y2 = line_slope * x2 + line_intercept
		sol1 = {"x":x1, "y":y1}
		sol2 = {"x":x2, "y":y2}
		return sol1,sol2

def find_interseption2(currentPose,ptr0,ptr1,lookAheadDis):
	a = currentPose[0]
	b = currentPose[1]
	line_slope = (ptr1[1] - ptr0[1])/(ptr1[0] - ptr0[0])
	line_intercept = ptr0[1]-line_slope*ptr0[0]
	circle_radius = lookAheadDis
	A = 1 + line_slope**2
	B = 2*line_slope*line_intercept - 2*line_slope*b -2*a
	C = a**2 + b**2 + line_intercept**2 -2*line_intercept*b - circle_radius**2
	discriminant = B**2 - 4 * A * C
	if discriminant >= 0:
		x1 = (-B + math.sqrt(discriminant)) / (2 * A)
		x2 = (-B - math.sqrt(discriminant)) / (2 * A)
		y1 = line_slope * x1 + line_intercept
		y2 = line_slope * x2 + line_intercept
		sol1 = {"x":x1, "y":y1}
		sol2 = {"x":x2, "y":y2}
		return sol1,sol2

# Example usage:
circle_center = (0, 0)
circle_radius = 2
line_slope = 1
line_intercept = 1

# intersections = line_circle_intersection2(circle_center, circle_radius, line_slope, line_intercept)

# now call this function and see the results!
intersections = line_circle_intersection ([0, 1], [2, 3], [0, 1], 1)
print("Intersection points:", intersections)
c = find_interseption(currentPose=[0, 1], ptr0=[2, 3], ptr1=[0, 1], lookAheadDis=1)
print(c)
d = find_interseption2(currentPose=[0, 1], ptr0=[2, 3], ptr1=[0, 1], lookAheadDis=1)
print(d)