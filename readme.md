1.Install Azure Kinect SDK (K4A), you can find it here: https://docs.microsoft.com/en-us/azure/kinect-dk/sensor-sdk-download

2.Install open3d 0.15.1  instruction: http://www.open3d.org/docs/latest/introduction.html

3.Install argparse:  pip install argparse

4.Run azure_kinect_recorder.py, use --help for argparse insturctions
manual mode: press [space] to record 1 frames of data
auto mode: record at a certain fps

set --save_ply True to save ply frame by frame when recording(not recommend to use under auto mode)

P.S.
The intrinsic.json is the intrinsic for 
"color_resolution":"K4A_COLOR_RESOLUTION_1536P"

"depth_mode":"K4A_DEPTH_MODE_WFOV_UNBINNED"

--align_depth_to_color Ture.

For different camera setting, get the instrintc by following http://www.open3d.org/docs/latest/tutorial/sensor/azure_kinect.html#open3d-azure-kinect-recorder

Camera settings can be adjusted in default_config.json according to https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/include/k4a/k4atypes.h