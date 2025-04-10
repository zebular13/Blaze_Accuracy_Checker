# Blaze_Accuracy_Checker
Accuracy checker for models in [blaze_app_python](https://github.com/AlbertaBeef/blaze_app_python).
**Note:** Currently this script only supports accuracy checking for the pose application.


## Instructions
### Set up
Clone this repo: 
```
git clone git@github.com:zebular13/Blaze_Accuracy_Checker.git
```
Clone [blaze_app_python](https://github.com/AlbertaBeef/blaze_app_python) into the same parent folder as Blaze_Accuracy_Checker:
```
git clone git@github.com:AlbertaBeef/blaze_app_python.git
```
Clone your reference dataset, in my case, [Yoga_Poses_Dataset](https://github.com/Manoj-2702/Yoga_Poses-Dataset), into the same parent folder as Blaze_Accuracy_Checker:
```
git clone git@github.com:Manoj-2702/Yoga_Poses-Dataset.git
```

All three repos should be in the same parent directory. 


Copy ```dataset_comparison.py``` and ```meanerror.py``` from this repository into ```blaze_app_python/blaze_tflite```
```
cd Blaze_Accuracy_Checker
cp dataset_comparison.py ../blaze_app_python/blaze_tflite/dataset_comparison.py
cp meanerror.py ../blaze_app_python/blaze_tflite/meanerror.py
```

### Run

Change directory to ```blaze_app_python/blaze_tflite``` and run ```dataset_comparison.py```
```
cd ../blaze_app_python/blaze_tflite
python3 dataset_comparison.py
```

### Launch Arguments

| -Argument | --Argument    | Description                               | 
| :-------: | :-----------: | :---------------------------------------- | 
|  -d       | --debug       | Enable Debug Mode.  Default is off        |
|  -a       | --an       | Export Annotated Images. Default is off |
|  -v       | --view      | View Annotated Images. Default is off  |
|  -r       | --rd       | Export Representative Dataset. Default is off  |
|  -t       | --thresh      | Error threshold at which to save images to representative dataset. Default is 0.10 (90% accuracy)"   |

### Results

After running, you should see the average mean error for your model for all images, e.g.:
```
Final Results for detection model models/pose_detection.tflite and landmark model models/pose_landmark_full.tflite
Total Mean Error:  0.3362397581013624
Total per row mean error:  0.34446449079582736
Total per row mean error minus vis for detection model:  0.15524349497385875
Total processed images: 482
Original total images:  484
```


Your representative dataset and Annotated Images will be in the results folder, E.G. for Yoga Poses Dataset:
```
.
├── Landmark_Images
│   ├── ArdhaChandrasana
│   ├── BaddhaKonasana
│   ├── Downward_Dog
│   ├── Natarajasana
│   ├── Triangle
│   ├── UtkataKonasana
│   ├── Veerabhadrasana
│   └── Vrukshasana
└── Representative_Dataset
    ├── ArdhaChandrasana
    ├── BaddhaKonasana
    ├── Downward_Dog
    ├── Natarajasana
    ├── Triangle
    ├── UtkataKonasana
    ├── Veerabhadrasana
    └── Vrukshasana
```