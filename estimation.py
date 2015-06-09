import cv2
import numpy as np
import sys
import getopt
import os
import math

#
# Read in an image file, errors out if we can't find the file
#
def readImage(filename, greyScale):
    img = cv2.imread(filename, greyScale)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...' + filename)
        return img
#
# Read and return raw depth data
#
def readRawDepthInfo(filename):
    f = None
    x = None
    try:
        f = open(filename, 'r')
    except:
        print("Could not retrieve raw depth file: " + filename)
    if f is not None:
        x = np.zeros((480, 640))
        count = 0
        lines = f.readlines()
        for line in lines:
            elements = [int(i) for i in line.split('\t') if i != '\n']
            #print len(elements)
            #print type(elements[0])
            x[count,:] = elements
            count+=1
    return x

#
# Finds the leftmost and rightmost column of the human
#
def findLeftandRight(depthImg):
    left = None;
    right = None;
    for i in range(depthImg.shape[1]):
        for j in range(depthImg.shape[0]):
            if left is None and depthImg.item(j, i) != 255:
                left = i
            if left is not None and depthImg.item(j, i) != 255:
                right = i
    return left, right

#
# Finds the top and bottom row of the human
#
def findTopandBottom(depthImg):
    top = None;
    btm = None;
    for i in range(depthImg.shape[0]):
        for j in range(depthImg.shape[1]):
            if top is None and depthImg.item(i, j) != 255:
                top = i
            if top is not None and depthImg.item(i, j) != 255:
                btm = i
    return top, btm
    
    
#
# Finds the pixel with the minimum depth given a row or column
#
def findMinDepth(vector):
    min = 999999
    for n in vector:
        if n < min and n != 0:
            min = n
            
    return min
#
# Finds the correct depth points to use in estimating a person's height and weight
#
def findBorderPoints(depthImg):
    top, btm = findTopandBottom(depthImg)
    left, right = findLeftandRight(depthImg)
    offset = 3
    
    top += offset
    btm -= offset
    left += offset
    right -= offset
    
    for i in range(left, right):
        if(depthImg.item(top,i) == findMinDepth(depthImg[top][left:right])):#np.amin(depthImg[top][np.nonzero(depthImg[top][left:right])])):
            t = (top, i)
        if(depthImg.item(btm,i) == findMinDepth(depthImg[btm][left:right])):
            b = (btm, i)
            
    for i in range(top, btm):
        if(depthImg.item(i,left) == findMinDepth(depthImg[:,left][top:btm])):
            l = (i, left)
        if(depthImg.item(i,right) == findMinDepth(depthImg[:,right][top:btm])):
            r = (i, right)
    
    return t, b, l, r
    
#
# Distance formula for two 3D points
#
def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
    
#
# Converts a point on the depth image to a 3D point
#
def depthTo3D(rawDepth, x_d, y_d):

    fx_d = 5.9421434211923247e+02
    fy_d = 5.9104053696870778e+02
    cx_d = 3.3930780975300314e+02
    cy_d = 2.4273913761751615e+02
    
    x = (x_d - cx_d) * rawDepth.item(y_d,x_d) / fx_d
    y = (y_d - cy_d) * rawDepth.item(y_d,x_d) / fy_d
    z = rawDepth.item(y_d,x_d) 
        
    return x, y, z
    
#
# Esimate a person's height from their image
#
def estimateHeight(depthImg, rawDepth):
    t, b, l, r = findBorderPoints(depthImg)
    top = depthTo3D(rawDepth, t[1], t[0])
    btm = depthTo3D(rawDepth, b[1], b[0])
    left = depthTo3D(rawDepth, l[1], l[0])    
    right = depthTo3D(rawDepth, r[1], r[0])
    
    height = distance(top, btm)
    width = distance(left, right)
    
    return height, width
    
def getRawDepth(rawDepth, point):
    return rawDepth.item(point)
# 
# Find and return 200 dimension sideview feature
#
def getSideviewShape(depthImg, top, btm):
    sideview = []
    for i in range(top, btm+1):
        result = []
        for j in range(depthImg.shape[1]):
            if(depthImg.item(i, j) < 255):
                result.append(depthImg.item(i, j))
        sideview.append(np.mean(result))

    #calculate 200 sideview features
    sideview[:] = [255 - point for point in sideview]
    sideview[:] = [point/min(sideview) for point in sideview]
    sideview = sideview[:len(sideview)/2]
    num_int_points = 201 - len(sideview)
    int_distance = float(len(sideview)) / float(num_int_points)
    x_points = range(1, len(sideview)+1)
    int_points = [int_distance * i for i in range(1, num_int_points)]
    int_values = np.interp(int_points, x_points, sideview)
    final_sideview = [(value, sideview[value-1]) for value in x_points]
    for i in range(len(int_points)):
        final_sideview.append((int_points[i], int_values[i]))
    final_sideview = sorted(final_sideview, key=lambda tup: tup[0])
    final_sideview = [x[1] for x in final_sideview]

    return final_sideview
    
def getColorImgName(img_name):
    color_list = os.listdir('./color')
    color_img_name = None
    for name in color_list:
            if img_name in name:
                color_img_name = name
    return color_img_name
#
# Main parses argument list and runs the functions
#
def main():
    args, folder_name = getopt.getopt(sys.argv[1:],'', [''])
    args = dict(args)
    folder_name = ['depth', 'color', 'raw_depth']
    
    images = {}
    #depth_list = os.listdir(folder_name[1])
    #color_list = os.listdir(folder_name[0])
    #raw_depth_list = os.listdir(folder_name[2])    
    depth_list = os.listdir('./depth')
    color_list = os.listdir('./color')
    raw_depth_list = os.listdir('./raw_depth')
    #read our images and keep them in a tuple
    for img_name in depth_list:
        #depth_img = readImage(folder_name[1] + '/' + img_name, 0)
        depth_img = readImage('depth/' + img_name, 0)
        color_img_name = getColorImgName(img_name)

        if color_img_name is not None:
            #color_img = readImage(folder_name[0] + '/' + color_img_name, 1)
            color_img = readImage('color/' + color_img_name, 1)

        fname, ext = os.path.splitext(img_name)
        #raw_depth_name = folder_name[2] + '/' + fname + ".txt"
        raw_depth_name = 'raw_depth/' + fname + ".txt"
        raw_depth = readRawDepthInfo(raw_depth_name)
        if raw_depth is not None:
            images[img_name] = ([color_img, depth_img, raw_depth])

    print len(images)
    
    f = open('features.txt', 'w')
    f2 = open('heightwidth.txt', 'w')
    for img_name in depth_list:
        if img_name in images.keys():
            t, b = findTopandBottom(images[img_name][1])
            l, r = findLeftandRight(images[img_name][1])
            top, btm, left, right = findBorderPoints(images[img_name][1])
            
            height, width = estimateHeight(images[img_name][1], images[img_name][2])
            f2.write(img_name + ': \n')
            f2.write('top, btm, left, right = (' + str(top) + ', ' + str(btm) + ', ' + str(left) + ', ' + str(right) + ')' + '\n')
            f2.write('top depth: ' + str(getRawDepth(images[img_name][2], top)) + '\n')
            f2.write('btm depth: ' + str(getRawDepth(images[img_name][2], btm)) + '\n')
            f2.write('left depth: ' + str(getRawDepth(images[img_name][2], left)) + '\n')
            f2.write('right depth: ' + str(getRawDepth(images[img_name][2], right)) + '\n')
            f2.write('height, width = ' + str((height, width)) + '\n' + '\n')
            sv = getSideviewShape(images[img_name][1], t, b)
            color_img_name = getColorImgName(img_name)
            if color_img_name is not None and img_name in images.keys():                
                f.write(color_img_name[1:4])
                for i in range(1, len(sv)):
                    f.write(' ' + str(i) + ':' + str(sv[i]))
                f.write(' 201:' + str(height * width) + '\n')
            print sv
        
    f.close()
    f2.close()

    l, r = findLeftandRight(images['dornoosh7.bmp'][1])
    leftCol = images['dornoosh7.bmp'][2][:,[l]]
    rightCol = images['dornoosh7.bmp'][2][:,[r]]
    topRow = images['dornoosh7.bmp'][2][t,:]
    bottomRow = images['dornoosh7.bmp'][2][b,:]
#    print np.amin(leftCol[np.nonzero(leftCol)])
#    print np.amin(rightCol[np.nonzero(rightCol)])
#    print np.amin(topRow[np.nonzero(topRow)])
#    print np.amin(bottomRow[np.nonzero(bottomRow)])
    
#    print('height, width = ' + str(estimateHeight(images['Shahzor0.bmp'][1], images['Shahzor0.bmp'][2])))

if __name__ == "__main__":
    main()