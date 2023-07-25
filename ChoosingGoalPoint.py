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
####					Choosing Goal Point					####
################################################################
path1 = [[0.0, 0.0], [0.011580143395790051, 0.6570165243709267], [0.07307496243411533, 1.2724369146199181], [0.3136756819515748, 1.7385910188236868], [0.8813313906933087, 1.9320292911046681], [1.6153051608455251, 1.9849785681091774], [2.391094224224885, 1.9878393390954208], [3.12721333474683, 1.938831731115573], [3.685011039017028, 1.7396821576569221], [3.9068092597113266, 1.275245079016133], [3.9102406525571713, 0.7136897450501469], [3.68346383786099, 0.2590283720040381], [3.1181273273535957, 0.06751996250999465], [2.3832776875784316, 0.013841087641154892], [1.5971423891000605, 0.0023698980178599423], [0.7995795475309813, 0.0003490964043320208], [0, 0]]

# helper functions
def pt_to_pt_distance (pt1,pt2) :
    distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    return distance

# returns -1 if num is negative, 1 otherwise
def sgn (num):
    if num >= 0:
        return 1
    else:
        return -1
        
def goal_pt_search (path, currentPos, lookAheadDis, lastFoundIndex) :

    # extract currentX and currentY
    currentX = currentPos[0]
    currentY = currentPos[1]
    
    # initialize goalPt in case no intersection is found
    goalPt = [None, None]
    
    # use for loop to search intersections
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
                    # make the decision by comparing the distance between the intersections and the next point in path
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
                    
    # visualize outcome
    # ---------------------------------------------------------------------------------------------------------------------------------------
    # traveled path (path omitted for searching) will be marked in brown
    field = plt.figure()
    xscale,yscale = (1.5, 1.5) # <- modify these values to scale the plot
    path_ax = field.add_axes([0,0,xscale,yscale])
    add_complicated_line(path[0:lastFoundIndex+1],'--','brown','traveled path')
    add_complicated_line(path[lastFoundIndex:len(path)],'--','grey','remaining path')
    highlight_points(path[0:lastFoundIndex], 'brown')
    highlight_points(path[lastFoundIndex:len(path)], 'grey')
    
    xMin, yMin, xMax, yMax = (-1, -1, 5, 3)  # <- modify these values to set plot boundaries
                                             # (minX, minY, maxX, maxY)

    # plot field
    path_ax.plot([xMin,xMax],[yMin,yMin],color='black')
    path_ax.plot([xMin,xMin],[yMin,yMax],color='black')
    path_ax.plot([xMax,xMax],[yMin,yMax],color='black')
    path_ax.plot([xMax,xMin],[yMax,yMax],color='black')
    
    # set grid
    xTicks = np.arange(xMin, xMax+1, 2)
    yTicks = np.arange(yMin, yMax+1, 2)
    
    path_ax.set_xticks(xTicks)
    path_ax.set_yticks(yTicks)
    path_ax.grid(True)
    
    path_ax.set_xlim(xMin-0.25,xMax+0.25)
    path_ax.set_ylim(yMin-0.25,yMax+0.25)
    
    # plot start and end
    path_ax.plot(path[0][0],path[0][1],'.',color='blue',markersize=15,label='start')
    path_ax.plot(path[-1][0],path[-1][1],'.',color='green',markersize=15,label='end')
    
    # plot current position and goal point
    draw_circle (currentX, currentY, lookAheadDis, 'orange')
    plt.plot (currentX,currentY, '.', markersize=15, color='orange', label='current position')  
    if goalPt != [None, None] :
        plt.plot (goalPt[0], goalPt[1], '.', markersize=15, color='red', label='goal point')
        add_complicated_line([currentPos, goalPt], '-', 'black', 'look ahead distance')
        print('Goal point found at {}'.format(goalPt))
    else:
        print('No intersection found!')
        
    path_ax.legend()

# call the function to see the results
goal_pt_search (path1, [1, 2.2], 0.6, 3)
