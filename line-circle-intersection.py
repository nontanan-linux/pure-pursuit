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
####			Helper Functions for Graphing				####
################################################################
