# License-Plate-Detection with Deep Learning

This project purpose is to detect license plate of vehicle from mutiple video sources by using deep learning models. Since YOLOv3 has a better performance on object detection, this model I built is based on this network structure. The whole code contains three parts:

1. Obtain an orginial video and break it to frames with specific fps

2. Feed frames to model and got the output

3. Generate the output video by using the output frames


### Prerequisites

The requirements.txt is all prerequistes for this project. Use the following code for installation:

```
pip3 install -U -r requirements.txt
```

### How to use the code

1. Put the video into the main folder

2. Test video with the the following code:

```
python3 detect.py --cfg cfg/yolov3.cfg --weights weight.pt --images ./test2ot/
```

3. Link of weight file: 

 https://drive.google.com/file/d/1euMylyzsEKeKvL4n8CGR33D840rZcqwG/view?usp=sharing



