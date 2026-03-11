# Real-Time-Video-Defogging-System-Based-on-Deep-Learning-Algorithm
##1. axi_dma folder contains the whole Vivado project previously used to build system on ZYNQ. You can refer to the block design in it to learn about how to build up a ZYNQ system for our project. You can open it by Vivado2017.4.
##2. executable_files_zynq folder contains the source code and executable file that can be directly run on zynq. It handles video receiving and data transmit between PS and PL, and video sending.
##3. gstreamer_video_process folder contains the video sender, video receiver and displaying part, along with script to run these files. They could be used directly after proper environment setting.
