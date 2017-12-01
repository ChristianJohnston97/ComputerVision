#####################################################################

# RANSAC Computer Vision Assignment

# Author : Christian Johnston, christian.johnston@durham.ac.uk

#####################################################################

import cv2
import os
import numpy as np
import random
import csv
import math

#####################################################################

# Path to dataset, currently set to temp folder
# Please edit as required

master_path_to_dataset = "C:/temp/TTBB-durham-02-10-17-sub10";

#####################################################################


# At the top of my program I have included all the functions I used
# during my implementation, irrespective of whether I have included
# them in my final solution as this shows how my approach to the task
# developed througout

# I have analysed these approaches in my report.

#####################################################################

# Function which finds canny edges of an image
# Upper and lower thresholds based on mean value of equalised histogram of image

def getEdges(img):
    blur = cv2.blur(img,(5,5))
    imgEq = cv2.equalizeHist(blur)
    average = cv2.mean(imgEq)[0]
    topQ = 1.33*average
    lowQ = 0.66*average
    canny = cv2.Canny(imgEq,lowQ , topQ)
    return canny

#####################################################################


# Function to remove the green colours from the images
# This is an example of region of interest extraction methodology
# And a heuristic to speed up processing times


def filterGreenColour(img):
    # Blur the image 
    blur = cv2.GaussianBlur(img, (15,15), 2)
    # Convert to HSV colour space, can isolate Hue channel
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # open cv hue is 0-180, sat and value 0-255
    lower_green = np.array([30,70,0])
    upper_green = np.array([80,255,255])
    # Keeps the pixels in the given range
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Morphological opening (erosion follwoed by dilation)
    kernel = np.ones((5,5), np.uint8)
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    res = cv2.bitwise_and(img,img, mask = mask);
    img = img - res
    return img


#####################################################################

# Function to plot the planar normal direction vector in the middle of the image

def plotNormal(img, coefficients):
    height, width = img.shape[:2]
    x = int(width/2)
    y = int(height/2)
   #  Scaled accordingly 
    oldX = int(coefficients[0][0]*700)
    oldY = int(coefficients[1][0]*700)
    

    secondPointX = x + oldX
    secondPointY = y + oldY
   
    cv2.line(img,(x,y),(secondPointX,secondPointY), (255,0,0),3)    
    
    return img


####################################################################

# Function to 'crop' image
# Create a 'box' of image, setting pixels outside this box to 0
# This is an example of region of interest extraction methodology
# And a heuristic to speed up processing times


def cropImage(img):
    width, height = img.shape[:2]
    for i in range(0, width):
        for j in range(0, height):
            if i > width*0.8 or i < width*0.2 or j > height*0.8 or j < height*0.2:
                img[i][j] = [255,255,255]
    return img

#####################################################################

# Function implementing Harris Corner Detection

def harris(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01*dst.max()] = [0,0,255]
    return img

#####################################################################

# Function which implements openCV's Contrast Limited Adaptive Histogram Equalisation
# Includes convertion from BGR colour space to LAB (luminance channel) colour space
# Stack Overflow help, credit to @ Jeru Luke.


def convertIllumination(img):
    
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
 
    #-----Converting image from LAB Color model to RGB model--------------------
    illume = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return illume

#####################################################################


# Differnce of Gaussian

def diffOfGaussian(img):
    sigmaU = 2;
    sigmaL = 1;
    
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    smoothedU = cv2.GaussianBlur(gray_frame,(0,0),sigmaU);
    smoothedL = cv2.GaussianBlur(gray_frame,(0,0),sigmaL);
    DoG = cv2.absdiff(smoothedU, smoothedL);
    DoG = DoG * (np.max(DoG) / 255);
    return DoG

#####################################################################


# Function for automatic detecting and highlighting obstacles that rise above the road surface plane
# HG pedestrian detection
# Credit to lecture course examples.

def hog(img):
    hog = cv2.HOGDescriptor();
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() );
     # perform HOG based pedestrain detection

    found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
    found_filtered = []

    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and inside(r, q):
                break
            else:
                found_filtered.append(r)

    draw_detections(img, found)
    draw_detections(img, found_filtered, 3)
    
    return img

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)


#####################################################################
        
# Function which gets gets the plane given 3 points

def getPlane(points):

    # selecting 3 non-colinear points
    crossProduct = np.array([0,0,0]);
    while crossProduct[0] == 0 and crossProduct[1] ==  0 and crossProduct[2] == 0:
        # Takes a random sample of 3 points from the given input of points
        [P1,P2,P3] = random.sample(points,3);
        P1 = np.array(P1)[:3]
        P2 = np.array(P2)[:3]
        P3 = np.array(P3)[:3]

        # check colinearity
        crossProduct = np.cross(P1-P2, P2-P3);

    #calculating the plane from these points

    # Coefficients_abc gives the surface normal coefficients
    
    coefficients_abc = np.dot(np.linalg.inv(np.array([P1,P2,P3])), np.ones([3,1]))

    coefficient_d = math.sqrt(coefficients_abc[0]*coefficients_abc[0]+coefficients_abc[1]*coefficients_abc[1]+coefficients_abc[2]*coefficients_abc[2])

    # measures distance of all points from plane given the plane coefficients 

    dist = abs((np.dot(points, coefficients_abc) - 1)/coefficient_d)

    return P1, P2, P3, dist, coefficients_abc

#####################################################################

# Main ransac function
# Data consists of inliers which fit a 'model'
# Estimates parameters of a model by random sampling of observed data

# This function calls the previously defined function getPlane
# ransac function checks which elements of the points are consistent
# with the plane from the previous function


def ransac(points):

    #threshold value to determine when a point fits the model
    # Can set this variable to whaatever to determine accruacy of model
    threshold = 0.01;

    # set of best fit points
    bestSetOfInliers = [];
    bestPlane = None;
    for i in range(40):
    # can alter this value for speed of processing, is number of iterations

        P1, P2, P3, dist, coefficients_abc  = getPlane(points);
        # This is a set of potential bets inliers
        inliers = [];
        for x in range(len(dist)):
            # Computes the distance between the points from the fitting plane
            d = dist[x];
            # If this distance is less than our prescribed threshold, add
            # to the set of inliers (not best inliers)
            if d <= threshold:
                inliers.append(points[x]);
        if len(inliers) > len(bestSetOfInliers):
            # Update best set of inliers if better model is found
            bestSetOfInliers = inliers
            # return the best plane
            bestPlane = [P1, P2, P3]

    return bestPlane, np.array(bestSetOfInliers),coefficients_abc  ;

#####################################################################

# Directory to image set data-path

directory_to_cycle_left = "left-images";     
directory_to_cycle_right = "right-images";   

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

crop_disparity = False; # display full or cropped disparity image
pause_playback = False; # pause until key press after each image


#####################################################################

# fixed camera parameters for this stereo setup (from calibration)

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

image_centre_h = 262.0;
image_centre_w = 474.5;

#####################################################################

## project_disparity_to_3d : project a given disparity image
## (uncropped, unscaled) to a set of 3D points with optional colour

def project_disparity_to_3d(disparity, max_disparity, rgb=[]):

    points = [];

    f = camera_focal_length_px;
    B = stereo_camera_baseline_m;

    height, width = disparity.shape[:2];

    # assume a minimal disparity of 2 pixels is possible to get Zmax
    # and then get reasonable scaling in X and Y output

    Zmax = ((f * B) / 2);

    for y in range(height): # 0 - height is the y axis index
        for x in range(width): # 0 - width is the x axis index

            # if we have a valid non-zero disparity

            if (disparity[y,x] > 0):

                # calculate corresponding 3D point [X, Y, Z]

                # stereo lecture - slide 22 + 25

                Z = (f * B) / disparity[y,x];

                X = ((x - image_centre_w) * Zmax) / f;
                Y = ((y - image_centre_h) * Zmax) / f;

                # add to points

                if(len(rgb) > 0):
                    points.append([X,Y,Z,rgb[y,x,2], rgb[y,x,1],rgb[y,x,0]]);
                else:
                    points.append([X,Y,Z]);

    return points;

#####################################################################

# project a set of 3D points back the 2D image domain

def project_3D_points_to_2D_image_points(points):

    points2 = [];

    Zmax = (camera_focal_length_px * stereo_camera_baseline_m) / 2;

    for i1 in range(len(points)):
        
        x = ((points[i1][0] * camera_focal_length_px) / Zmax) + image_centre_w;
        y = ((points[i1][1] * camera_focal_length_px) / Zmax) + image_centre_h;
        points2.append([x,y]);

    return points2;

#####################################################################


# Function to calculate the disparity between the left and right images


def disparity(illumeL, illumeR):
    # convert to grayscale (as the disparity matching works on grayscale)
    grayL = cv2.cvtColor(illumeL, cv2.COLOR_BGR2GRAY);
    grayR = cv2.cvtColor(illumeR, cv2.COLOR_BGR2GRAY)

    # compute disparity image from undistorted and rectified stereo images that we have loaded, scale by 16

    disparity = stereoProcessor.compute(grayL,grayR);

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5; # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter);

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity,0, max_disparity * 16, cv2.THRESH_TOZERO);
    disparity_scaled = (disparity / 16.).astype(np.uint8);


            
    # crop disparity to chop out left part where there are with no disparity
    # as this area is not seen by both cameras and also
    # chop out the bottom area (where we see the front of car bonnet)

    if (crop_disparity):
        width = np.size(disparity_scaled, 1);
        disparity_scaled = disparity_scaled[0:390,135:width];

    # display image (scaling it to the full 0->255 range based on the number
    # of disparities in use for the stereo part)

    disparity_scaled = (disparity_scaled * (256. / max_disparity)).astype(np.uint8);
        
    return disparity_scaled

############################################################################


# resolve full directory location of data set for left / right images

full_path_directory_left =  os.path.join(master_path_to_dataset, directory_to_cycle_left);
full_path_directory_right =  os.path.join(master_path_to_dataset, directory_to_cycle_right);

# get a list of the left image files and sort them (by timestamp in filename)

left_file_list = sorted(os.listdir(full_path_directory_left));

# setup the disparity stereo processor to find a maximum of 128 disparity values
# (adjust parameters if needed - this will effect speed to processing)

max_disparity = 128;
stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21);

for filename_left in left_file_list:

    # skip forward to start a file we specify by timestamp (if this is set)

    if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename_left)):
        continue;
    elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename_left)):
        skip_forward_file_pattern = "";

    # from the left image filename get the correspondoning right image

    filename_right = filename_left.replace("_L", "_R");
    full_path_filename_left = os.path.join(full_path_directory_left, filename_left);
    full_path_filename_right = os.path.join(full_path_directory_right, filename_right);


    # check the file is a PNG file (left) and check a correspondoning right image actually exists

    if ('.png' in filename_left) and (os.path.isfile(full_path_filename_right)) :

        # read left and right images and display in windows
        # N.B. despite one being grayscale both are in fact stored as 3-channel
        # RGB images so load both as such

        imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
        imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

        print("-- files loaded successfully");
        print();


###################################################################################################

        # *** MAIN Part of pre-processing and implementation *** #

###################################################################################################

        # Contrast Limited Adaptive Histogram Equalisation
        illumeL = convertIllumination(imgL);
        illumeR = convertIllumination(imgR);

        # Crop image
        cropL = cropImage(illumeL)
        cropR = cropImage(illumeR)

        # Remove Green Pixels from selection

        colourCropL = filterGreenColour(cropL)
        colourCropR = filterGreenColour(cropR)

        # Gaussian Blur 
        illumeL = cv2.GaussianBlur(colourCropL, (5, 5),0)
        illumeR = cv2.GaussianBlur(colourCropR, (5, 5),0)
 
                
        # Calculate Disparity from left and right image channels
        disparity_scaled = disparity(illumeL, illumeR)
        cv2.imshow("dispsarity", disparity_scaled)



        # project to a 3D colour point cloud 
        points = project_disparity_to_3d(disparity_scaled, max_disparity);


        plane, inliers, coefficients = ransac(points)

        points2D = project_3D_points_to_2D_image_points(inliers)


        # Create a new image of the same size called blank
        
        height, width = imgL.shape[:2]
        # Create a new blank image
        blank = np.zeros((height,width,3), np.uint8)



        # Here is a method to crop the image
        # Loops through all the points and if the points are in a certain range/area then
        # draw a black circle around these points
    
        
        for point in points2D:
            x = int(point[0])
            y = int(point[1])
            if x < width*0.75 and x > width*0.25 and y < height*0.75 and y > 250:
                cv2.circle(blank, (x,y), 2, (255,255,255), -1)

        # Convert this new image with circles on to gray
        
        gray = cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY)

        # Gaussian blur

        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Threshold the image
        #Kept this very low
        
        ret, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        # Dilation

        kernel = np.ones((5,5), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations = 1)



        # Contoured the image
         
        newImg, contours, hierachy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        

        # Sort the contours by area in decreasing order
        contours = sorted(contours, key=cv2.contourArea, reverse = True)

        

        biggestContours = contours[:20]


        # biggestContours = np.array(biggestContours)
        # Draw a convex hull around these contours
        twoDConvex = np.vstack(biggestContours).squeeze()
        hull = cv2.convexHull(twoDConvex)
        cv2.polylines(imgL, [hull], True, (0,0,255), 3)

        
        
    
###############################################################################################


        # Prints the normal coefficients of the detected plane
        
        newCos = "(" + str(coefficients[0][0]) + ", " + str(coefficients[1][0]) + ", " + str(coefficients[2][0]) + ")"
        
        print(full_path_filename_left);
        print(full_path_filename_right + " : " + "road surface normal " + newCos);
        print();

        imgL = plotNormal(imgL, coefficients) 
        imgL = hog(imgL)    
        cv2.imshow('Final Image', imgL);

###############################################################################################

        # keyboard input for exit (as standard), save disparity and cropping
        # exit - x
        # save - s
        # crop - c
        # pause - space

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            break; # exit
        elif (key == ord('s')):     # save
            cv2.imwrite("sgbm-disparty.png", disparity_scaled);
            cv2.imwrite("left.png", imgL);
            cv2.imwrite("right.png", imgR);
        elif (key == ord('c')):     # crop
            crop_disparity = not(crop_disparity);
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback);
    else:
            print("-- files skipped (perhaps one is missing or not PNG)");
            print();

# close all windows

cv2.destroyAllWindows()

#####################################################################
