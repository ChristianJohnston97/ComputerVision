# ComputerVision
Road Surface Region Detection in 3D Stereo using RANSAC algorithm

Task was to develop a road surface detection system that correctly detects the 3D planar orientation and bounds (i.e. edges) of any free (unoccupied) road surface region immediately infront of the vehicle in which the autonomous vehicle needs to operate (e.g. for staying in the correct lane / staying on the road itself / avoiding obstacles / automatic braking).


I was given a set of still image pairs (left and right) extracted from on-board forward facing stereo video footage under varying illumination conditions and road surface environments. 

System was based on shape detection using the Random Sample and Consensus (RANSAC) algorithms. 

This is a real-world task, comprising a real-world image set. (not provided as size of file was 2GB)

Included automatically detecting and highlighting obstacles that rise above the road surface plane
(vehicles, pedestrians, bollards etc.) as they appear directly in front of the vehicle.
