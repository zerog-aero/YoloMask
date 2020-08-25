# zeroFaceMask
Detect Faces with or without mask using yolov5.  [Original Code](https://github.com/ultralytics/yolov5)


# Description: 


In times of global pandemics like Corona, automatic systems to detect people wearing masks are becoming more and more important.
Be it for governments who might want to know how many people are actually wearing masks in crowded places like public trains; or businesses who are required  by law to enforce the usage of masks within their facilities. 

This projects aims to provide an easy framework to set up such a mask detection system with minimal effort.
We provide a pre-trained model trained for people relatively close to the camera which you can use as a quick start option.

But even if your use case is not covered by the pre-trained model, training your own is also quite easy (also a reasonable recent GPU is highly recommended) and a you should be able to do this by following the short guide provided in this README. 


# Installation:

0. The pre-trained model can be downloaded from https://drive.google.com/file/d/1--KabFrjWq42uktpWtV6w-IegydsncNS/view?usp=sharing

1. All the needed python packages can be found in the requirements.txt file. The commands needed to install them are also provided there.  
   **IMPORTANT**: The required numpy version is 1.17. [https://github.com/cocodataset/cocoapi/issues/356](https://github.com/cocodataset/cocoapi/issues/356)

   **pip**: If you use pip, just type in "pip install -U -r requirements.txt" and all the required libraries should be installed.  
   
   **anaconda**: For anaconda "conda install --file requirements.txt" should work. If it does not, you have to install them step by step:  
	* conda update base -c defaults conda
	* conda install anaconda opencv matplotlib tqdm pillow ipython
	* conda install numpy=1.17
	* conda install conda-forge scikit-image pycocotools tensorboard
	* conda install pytorch torchvision cudatoolkit=10.2 -c pytorch (see [here](https://pytorch.org/get-started/locally/) for differenct CUDA Version)

2. Docker 
    
   A Docker image is provided that can run inference on single images and videos. Data is pushed into the docker via a RestAPI
   which runs on port 80 in the docker.

   1. Building the docker container:
      docker build . -t "image_name":latest    

   2. Now we have to start the docker:
      docker run  --name "container_name" -p 80:80 -d "image_name"
      with working cuda: Add the options: --gpus all  
   
   3. Once started, you can send images/video to the docker by using curl in your favourite terminal:
      * curl -X POST http://127.0.0.1:80/annotate_image  --data-binary @"path to the image file" --output "name of the output file"  
      The @ charackter ist important and must be included. This is the command that I run on my local machine:  
      curl -X POST http://127.0.0.1:80/annotate  --data-binary @"C:\Users\U734813\Documents\GitLab\zero_mask\inference\images\with_
      
      The following endpoints are avaliable:
      * annotate_image: Annotates and draws bounding boxes (shows mask/no mask)      
       Usage: curl -X POST http://127.0.0.1:80/annotate_image  --data-binary @"path to the image file" --output "name of the output file"
      
      * annotate_image_demo: This is demo shows a possible use case. See section "DEMO" for details.  
       Usage: curl -X POST http://127.0.0.1:80/annotate_image_demo  --data-binary @"path to the image file" --output "name of the output file"  
       The Position of the image and the information panel(STOP, HAVE A NICE DAY, COME CLOSER)  can be switched by supplying the paramter "info_screen_small":  
       Usage: curl -X POST http://127.0.0.1:80/annotate_image_demo?info_screen_small=False  --data-binary @"path to the image file" --output "name of the output file"
      
      * annotate_image_json: Returns a json string containing all recognized elements and their bounding boxes. 
       Usage: curl -X POST http://127.0.0.1:80/annotate_image_json  --data-binary @"path to the image file"
        
       
      * annotate_video: Annotates and draws bounding boxes (shows mask/no mask) for videos   
       Usage: curl -X POST http://127.0.0.1:80/annotate_video  --data-binary @"path to the video file" --output "name of the output file"
      
      * annotate_video_demo: This is demo shows a possible use case. See section "DEMO" for details
       Usage: curl -X POST http://127.0.0.1:80/annotate_video_demo  --data-binary @"path to the video file" --output "name of the output file"
       The Position of the image and the information panel (STOP, HAVE A NICE DAY, COME CLOSER)  can be switched by supplying the paramter "info_screen_small":  
       Usage: curl -X POST http://127.0.0.1:80/annotate_video_demo?info_screen_small=False  --data-binary @"path to the image file" --output "name of the output file"
       
       * annotate_video_json: Returns a json string containing all recognized elements and their bounding boxes for each frame
       Usage: curl -X POST http://127.0.0.1:80/annotate_video_json  --data-binary @"path to the image file"

   4. If somethings goes wrong it might be helpful to check the docker log using the following command:
      docker logs -f "container_name" 

   5. Finally to stop the running container and (optionally delete it): docker stop  zero_mask_container && docker rm zero_mask_container                                                           


# Usage of the library

## Using the pre-trained model

The `detect.py` script can be used to run inference on images, video files or video streams. 
Lets start with a simple example. Multiple pictures are provided in "inference/images" which we will use for a first test run. 

`detect.py` can take multiple arguments, but for now we use the default parameter and just run:

```shell
python detect.py
```

This will use the following default values (check out the `detect.py` script to see how it works):

-  `--weights weights/yolov5l_fm_opt.pt`: The pre-trained model provided with this repository
-  `--source inference/images`: Path to a folder or filename that you want to run inference on. 
-  `--output inference/output`: Output folder where the inferred files are stored
-  `--img-size' 480`: The model is trained on an image-size of 480. This is good for close range detection.  If you want to detect people standing further away, then changing this image size to a higher value might work

Since "inference/images" is the default input folder, we only have to run `python detect.py` to run inference on all the images in this folder. 
The results can then be found in `inference/output`
An example can be found here: 
![alt text](https://github.com/zeroG-AI-in-Aviation/zero_mask/blob/master/inference/output/with_mask_short_range.jpg)
It's easy to change the input folder/file and output folder by just running the script with different arguments:

```shell
python detect.py --source myexamplefolder/images/my_image.jpg --output somefolderpath
```

which runs the inference script for the file named `my_image.jpg` found in `myexamplefolder/images/` and stores the result in `somefolderpath`.

Videos can also be used as input. Just give the source to the video using:
```shell
python detect.py  --source myexamplefolder/images/my_video.mp4
```

### Limitations of the Model

The model was trained on images from a combination of datasets
-  [Real-World Masked Face Dataset](https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset)
-  https://github.com/prajnasb/observations 
-  [Real and Fake Face Detection](https://www.kaggle.com/ciplab/real-and-fake-face-detection)

The images mainly contain one person at a time at a close distance.
Due to this the model does not perform that well on a big crowds at longer distances.
But even then the model performs reasonable well.

![alt text](inference/output/with_mask_group_of_people.jpg?raw=true)

## Training your own model
After you tried the provided model, you might find that it does not fit your use case.
In this case it might be helpful to train your own model. In this section we give a short overview on how to do this.

### Acquiring and labeling images
To train your own models you need enough labeled images featuring people with and without masks. For reference about 1000 images were used to train the model provided here.

#### Image sources and labels
Our model is mainly based on images from 3 datasets (see [Limitations of the Model](#limitations-of-the-model)).

If you are lucky your dataset is already labeled. In this case you just have to make sure that the label format matches the Darknet format:

```
0	0.2232 0.4654 0.1213 0.2054
```

So what does this actually mean? 

-  `0`: This is the object-id defined in "TODO". In our case this would be for example 0 for "person_without_mask" and 1 for "person_with_mask"
-  `0.2232 0.4654`: This is the center point of the bounding box relative to height and width of the image.
                   For example: The point (0.4 0.6) is located at 40% of the image width and 60% of the image height.
-  `0.1213 0.2054`: Size if the bounding box, 0.1213 is the width and 0.2054 is the height. Both again are relative to the real image size.
                   So (0.2 0.3) would mean that the bounding size  is 20% of the image width  and 30% of the image height.

If you do not have labeled images, you have to label them first. There are a lot of different tools out there to achieve this and most of them
will do the job just fine. We used [labelme](https://github.com/wkentaro/labelme) which does NOT  output its labels in the Darknet format so you have to convert it first.


### Creating Training and Validation sets
The labeled images have to be split in a training and a validation set. Splitting it 80-20 should be reasonable.

The folder structure is as follows:

```
data  
├── images  
│   ├── train  
│   └── val  
├── labels  
    ├── train  
    └── val  
```

You can find another detailed guide to create your own dataset [here](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data).

#### Training

To start the training simply run the `train.py` script.  
Again this script can take a number of arguments, but for a first run you can just start it with the default parameters.
The following options are needed and have default values:

-  `--epochs default=300`: Number of times that the whole dataset is iterated through. 30 was enough for our pre-trained model.
-  `--batch-size default=16`: the number of training examples in one forward/backward pass. The higher the batch size, the more GPU memory is needed.
-  `--cfg default='models/yolov5s.yaml`: general config file used for training
-  `--data' default='data/coco128.yaml`: yaml file containing information about training data
-  `--img-size default=[640, 640]`: image size for training. Images will be resized automatically to this resolution. Higher resolutions might lead to significantly longer training times.


# License: 
Check the LICENSE file
