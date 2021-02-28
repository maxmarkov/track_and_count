# The electronic observer project (track & count)

## Project objective and description

<p align="justify">
<b>Voting</b> is a key procedure enabling society to select their representatives and to hold them accountable for their performance in office. It plays a vital role in the communication of the society's needs and demands straight up to the political institutes. Both society and politicians are interested in a transparent and trustable procedure guaranteeing the legitimacy of the choice made. One of the mechanisms to assure the fairness of the procedure is observation. Usually, observers are people representing different political parties and public organizations whose primary task is to monitor the fairness of the election procedure as well as the correct counting of votes. Good observation prevents or at least limits the fraud and increases the legitimacy of the result. Here we propose a computer vision algorithm that aims to count the number of unique people voting during the election day. Shortly speaking, this is <b>an electronic observer</b>. At the end of the day, the counted number of votes can be compared with the official turnout at the polling station. A large discrepancy between the two is a signature of the fraud and signalize that the video should be more carefully examined by independent observers to look for any stuffing evidence. 
</p>

<p align="justify">
Traditional methods used to determine electoral fraud are based on a statistical analysis of irregularities in Vote-Turnout distributions. Among the most observed anomalies are <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2934485">coarse vote percentages</a>, zero-dispersion distribution of votes within one electorate, and a peak in the distribution of votes at high turnout rates for one candidate. Such electoral statistical methods are very developed in Russia where a large array of data is collected and analyzed by Dr. Sergey Shpilkin (rus. <a href="https://www.facebook.com/sergey.shpilkin">Сергей Шпилькин</a>). However, the statistical analysis is relatively difficult to explain to a general audience with a highly varying level of mathematical education. Our algorithm, in turn, provides visual and simple interpretable results: the demonstration of ballot staffing on a video is a clear argument that is difficult to reject. Importantly, our algorithm does not gather any personal information since it does not use face recognition technology. We test our algorithm on short video samples available publicly on YouTube. These samples were recorded by video cameras installed in polling stations <a href="https://aceproject.org/electoral-advice/archive/questions/replies/291099047">in Russia</a> where they had been installed in 2012.  
</p>

![Gif example](https://github.com/maxmarkov/track_and_count/blob/master/example/example_count.gif)

## Important notes on implementation

Table of contents
=================
- [Custom object detection](#custom-detection)
- [Tracking](#tracking)
- [Count](#count)
- [Reidentification](#reid)
- [How to run the trackers](#run-tracker)
- [How to detect urns](#detect-urn)
- [Literature](#lit)
- [Codes](#codes)

<a name="custom-detection"></a>
## Custom object detection 

The implementation of custom object detection could be found in a folder *urn_detection_yolov5*. 
First, the dataset of urn pictures was collected (see *urn_detection_yolov5/collecting_urn_dataset.doc*
for details). Note that the dataset has already been augmented with different brightness levels to simulate the 
effect of illumination in a room and/or bad camera settings. The dataset can be downloaded with curl.
Then, the **YOLOv5 detector** is applied with 2 classes of objects specified: an urn (a custom object) 
and a person (a coco object). The neural network is then fine-tuned to learn about the custom 
object class. Finally, the inference is done on a subset of data and the result is visualized. 

**Example of urn detection with YOLOv5**

<img src="example/urn_detection_inference.jpeg" width="400" class="centerImage">

*NB*: Since an urn is a stationary object (i.e. its position is not supposed to change in time),
the detection can be performed on a single (initial) video frame. Then, the urn's coordinates could
be easily passed further to other frames without performing the detection task repeatedly. 


<a name="tracking"></a>
## Tracking

In the second part of the project, we track people in a room using the tracking-by-detection paradigm.
As it has been done earlier in the custom object detection section, **YOLOv5** performs a person
detection on each single video frame. Then, the detections on different frames must be associated 
with each other to re-identify the same person. **The SORT tracker** combines the linear Kalman filter
to predict the state of the object (*the motion model*) and the Hungarian algorithm to associate objects 
from the previous frames with objects in the current frame. The tracker does not consider any details
of the object's appearance. My implementation of the SORT tracker inside the YOLOv5 inference script could be found in 
*track_yolov5_sort.py*. The Jupyter notebook *colabs/run_sort_tracker_on_colab.ipynb* shows how to run the
tracker on **Google Colab**.

**Example of tracking in a room using SORT and YOLOv5**

![Gif example](https://github.com/maxmarkov/track_and_count/blob/master/example/tracker_example.gif)

A nice alternative to the SORT tracker is a [Deep SORT](https://arxiv.org/pdf/1703.07402.pdf).
**The Deep SORT** extends the SORT tracker adding a deep association metric to build an appearance
model in addition to the motion model. According to the authors, this extension enables to track objects
through longer periods of occlusions, effectively reducing the number of identity switches. My implementation
of the tracker inside the YOLOv5 inference script could be found in *track_yolov5_deepsort.py*. The Jupyter
notebook *colabs/run_deepsort_tracker_on_colab.ipynb* shows how to run the tracker on **Google Colab**.

<a name="count"></a>
## Count

<p align="justify">
Since our primary task is to count the number of unique voters but not the total number of people in a room (like kids who just accompany the adults),
it is important to define the voting act in a more precise way. Both an urn and voters are
identified using the YOLOv5 detector which puts a bounding box around each of them. To vote, a person must come close to an urn and
spend a certain amount of time around (i.e. the distance between the object centroids must be within a certain critical radius). This
"certain amount of time" is necessary to distinguish the people who pass by and the ones who vote. This approach requires two
predefined <b>*parameters</b>:
</p>

- Critical radius
- Minimum interaction time

<p align="justify">
The person whose motion satisfies the conditions defined above can be then tracked until he/she disappears from the camera view. The
tracking is necessary in case the person stays in a room hanging around for a while. To further ensure that we count the unique people only,
one can save an image of each tracked person inside the bound box building a database of voters in a video. When the dataset of images with
voters is built, one can run a neural network to find the unique voters based on the similarity of their appearance.
</p>

<a name="reid"></a>
## Reidentification

<p align="justify">
Both trackers listed above possess only short-term memory. The object's track is erased from memory after max_age number of frames
without associated detections. Typically, max_age is around 10-100 frames. If a person leaves a room and comes back in a while, the
tracker will not re-identify the person assigning a new ID instead. To solve this issue, one needs long-term memory. Here we implement
long-term memory by means of appearance features from the Deep Sort algorithm. An appearance feature vector is a 1D array with 512 components. 
For each track ID we create a separate folder into which we write feature vectors. Feature vectors files are labeled in their names with 
frame number index where the object has been detected. When a new track is identified, one can compute the cosine distance between this 
track and all saved tracks in appearance space. If the distance is smaller than some threshold value, an old ID could be reassigned to 
a new object. Long-term memory enables us to exclude the security guards or the election board members who approach an urn frequently.
</p>

Feature extractor script is *deepsort_features.py*. Besides the standard output video file, it also writes features and corresponding cropped
images of tracked objects being saved into inference/features and inference/image_crops folders, respectively. The log file with the dictionary 
storing the history of object detections is in inference/features/log_detection.txt. The keys of this dictionary are track IDs
and values are the lists with frame numbers where the corresponding track has been registered. Moreover, we save frames per second rate which enables
us to restore the time (instead of frame number) when the track is detected.

**Content:**

- track_yolov5_sort.py implements the SORT tracker in YOLOv5
- track_yolov5_deepsort.py implements the Deep SORT tracker in YOLOv5
- colabs/run_sort_tracker_on_colab.ipynb and colabs/run_deepsort_tracker_on_colab.ipynb shows how to run the trackers on google colab. 
- track_yolov5_counter.py runs a counter
- deepsort_features.py implements the feature extractor
- folder 'theory' contains the slides with summary of theoretical approaches  

<a name="detect-urn"></a>
## How to detect urns.

1. Extract some snapshot frames into snapshot_frames folder

     python3 utils/extract_frames.py --source video_examples/election_2018_sample_1.mp4 --destination snapshot_frames --start 1 --end 10000 --step 1000

2. Run the detector which saves the coordinates into .txt file in urn_coordinates folder

     python3 yolov5/detect.py --weights urn_detection_yolov5/weights_best_urn.pt --img 416 --conf 0.2 --source snapshot_frames --output urn_coordinates --save-txt

<a name="run-tracker"></a>
## How to run the trackers

1. Follow the installation steps described in INSTALL.md

2. Run tracker: YOLOv5 + (SORT or Deep SORT)

     python3 track_yolov5_sort.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt --conf 0.4 --max_age 50 --min_hits 10 --iou_threshold 0.3

     python3 track_yolov5_deepsort.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt

3. Run tracker with pose estimator

     python3 track_yolov5_pose.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt

4. Run the counter

     python3 track_yolov5_counter.py --source video_examples/election_2018_sample_1.mp4 --weights yolov5/weights/yolov5s.pt

5. Run the feature extractor

     python3 deepsort_features.py --source example/running.mp4 --weights yolov5/weights/yolov5s.pt


<a name="lit"></a>
## Literature

- [Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763)
- [Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT)](https://arxiv.org/pdf/1703.07402.pdf)
- [Real-Time Multiple Object Tracking: A Study on the Importance of Speed by S.Murray](https://arxiv.org/pdf/1709.03572.pdf)
- [Real time multiple camera person detection and tracking by D.Baikova](https://repositorio.iscte-iul.pt/handle/10071/17743)
- [Detection-based Multi-Object Trackingin Presence of Unreliable Appearance Features by A.Kumar(UCL)](https://sites.uclouvain.be/ispgroup/uploads//Main/PHDAKC_thesis.pdf)
- [Slides on "Re-identification for multi-person tracking" by V. Sommers (UCL)](https://sites.uclouvain.be/ispgroup/uploads//ISPS/ABS220720_slides.pdf)
- [Kalman and Bayesian Filters in Python (pdf)](https://elec3004.uqcloud.net/2015/tutes/Kalman_and_Bayesian_Filters_in_Python.pdf)
- [Kalman and Bayesian Filters in Python (codes)](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python)
- [Deep Cosine Metric Learning for Person Re-Identification](https://elib.dlr.de/116408/1/WACV2018.pdf)

<a name="codes"></a>
## Codes

- [SORT](https://github.com/abewley/sort)
- [Deep SORT (TF)](https://github.com/nwojke/deep_sort), [Deep SORT (PyTorch)](https://github.com/ZQPei/deep_sort_pytorch)
- [YOLOv5+DeepSORT](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)
- [Deep person reid (UCL)](https://github.com/VlSomers/deep-person-reid)
- [YOLOv4](https://github.com/AlexeyAB/darknet), [YOLOv4 PyTorch](https://github.com/Tianxiaomo/pytorch-YOLOv4)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [FilterPy library (the Kalman filter)](https://filterpy.readthedocs.io/en/latest/)

## Habr

- [Как работает Object Tracking на YOLO и DeepSort](https://habr.com/en/post/514450/)
- [Самая сложная задача в Computer Vision](https://habr.com/en/company/recognitor/blog/505694/) 
