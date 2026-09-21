# V1
--------------------------------------------------------------------------------------

This python code performs simulation of interneurons in 2D.
Interneurons are modelled as chains, eventually branched, of nodes connected by springs. So far level 2 and level 3 branches are implemented.
Leading and tailing processes are allowed to perform random jumps. Jump acceptance probability obeys a Boltzmann-like distribution based on the local value of pre-defined 2D affinity fields and a 'temperature' value.
Jumps introduce kinetic energy in the system, which is equilibrated thanks to the integration of Newton's equation of moment together with a friction coefficient.
The geometry of the confining universe is defined by MASKS.
With this project, 3 different masking geometries are given: BRAIN, SLICE and CORTEX.
The parameters are provided for 3 different sets of parameters : HUMAN (H42), MOUSE (M43) and Mouse+Gain of function (GOF4).
Parameters are provided as is. It is not advised to modify the proposed default values.
The code uses numba parallelism by default and is limited to 400 interneuron surrogates.
Runing the codes requires the following libraries to be available in the python3 environment :
numba, numpy, opencv, imageio.v2, pathlib, time, tkinter, matplotlib, threading

---------------------------------------------------------------------------------------

Running the codes requires the presence of the following files in the runtime directory :
A) for the BRAIN mask:
Brain_only_BW.png
Full_black.png
Lines_negative_brain_V11.png
Lines_positive_brain_V11.png
Striatum_small.png
multiline_affinity_139pts_full_v12-800x1200.txt

B) for the CORTEX mask:
Lines_bkg_cortex_V11.png
Full_black.png
Lines_negative_cortex_V11.png
Lines_positive_cortex_V11.png
multiline_affinity_60pts_cortex_v12-528x463.txt

C) for the SLICE mask
Slice_affinity_limits_V11.png
Slice_affinity_negative_V11.png
Slice_affinity_positive_V11.png

To run the code, simply enter
python ./Interneurons_v1.0.py

or
python3 ./Interneurons_v1.0.py

depending on your environment.

At startup, the code reads the 'default_data.txt' file, hopefully producing an error message in case of an incorrect syntax. Lines contaning the # character are omitted
Frequent error occurs when a blank line is found (do not leave blank lines ...)

Two graphic windows open. One with the simulation 2D box, and one with some options menu. 
A)'START' does as it says
B)'STOPS' interrupts the calculation and finalizes the output of the video 'simulation_xxx.mp4' file. This file is not readable is the calculations stops abruptly or is cancelled by some other action.
  after a 'STOP' instruction: 'STARTS can be used to restart the calculation (with apparition of IN surrogates as defined in the 'batch' keyword, but with theiur lat position as starting point)
                              the trajectory file is updated
                              a new 'simulation_xxx.mp4' file is created


If several 'simulation, simulation1 simulation2 ' etc... files have been produced during a single simulation, these can be collated in a single long mp4 file using the concatenate.py script. This requires installation of the ffmpeg plugin.
C) 'Field to plot' allows to toggle between the various fileds of the simulation. Post relevant are 'potential' and 'affinity' fields.

----------------------------------------------------------------------

several post-processing are given.

---------------------------------------------------------------------

                              
