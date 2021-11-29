<img src="/logo.png" width="100%"/>

[![Downloads](https://static.pepy.tech/personalized-badge/auto-tensorflow?period=total&units=none&left_color=grey&right_color=brightgreen&left_text=Downloads)](https://pepy.tech/project/auto-tensorflow)
[![Generic badge](https://img.shields.io/pypi/v/auto-tensorflow.svg?logo=pypi&logoColor=white&color=orange)](https://pypi.org/project/auto-tensorflow/)
![Generic badge](https://img.shields.io/badge/python-v3.6%20%7C%203.7%20%7C%203.8-blue)
![example workflow](https://github.com/rafiqhasan/auto-tensorflow/actions/workflows/codeql-analysis.yml/badge.svg)
![Open issues](https://img.shields.io/github/issues-raw/rafiqhasan/auto-tensorflow?color=red)

### **Auto Tensorflow - Mission:**
**Build Low Code Automated Tensorflow, What-IF explainable models in just 3 lines of code.**

To make Deep Learning on Tensorflow absolutely easy for the masses with its low code framework and also increase trust on ML models through What-IF model explainability.

### **Under the hood:**
Built on top of the powerful **Tensorflow** ecosystem tools like **TFX** , **TF APIs** and **What-IF Tool** , the library automatically does all the heavy lifting internally like EDA, schema discovery, feature engineering, HPT, model search etc. This empowers developers to focus only on building end user applications quickly without any knowledge of Tensorflow, ML or debugging. Built for handling large volume of data / BigData - using only TF scalable components. Moreover the models trained with auto-tensorflow can directly be deployed on any cloud like GCP / AWS / Azure.

<img src="/header.png" width="100%"/>

### **Official Launch**: https://youtu.be/sil-RbuckG0

### **Features:**
1. Build Classification / Regression models on CSV data
2. Automated Schema Inference
3. Automated Feature Engineering 
    - Discretization
    - Scaling
    - Normalization
    - Text Embedding
    - Category encoding
5. Automated Model build for mixed data types( Continuous, Categorical and Free Text )
6. Automated Hyper-parameter tuning
7. Automated GPU Distributed training
8. Automated UI based What-IF analysis( Fairness, Feature Partial dependencies, What-IF )
9. Control over complexity of model
10. No dependency over Pandas / SKLearn
11. Can handle dataset of any size - including multiple CSV files

### **Tutorials**:
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafiqhasan/auto-tensorflow/blob/main/tutorials/TFAuto_%7C_Classification.ipynb) - Auto Classification on CSV data
2. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafiqhasan/auto-tensorflow/blob/main/tutorials/TFAuto_%7C_Regression.ipynb) - Auto Regression on CSV data

### **Setup:**
1. Install library
    - PIP(Recommended): ```pip install auto-tensorflow```
    - Nightly: ```pip install git+https://github.com/rafiqhasan/auto-tensorflow.git```
2. Works best on UNIX/Linux/Debian/Google Colab/MacOS

### **Usage:**
1. Initialize TFAuto Engine
```
from auto_tensorflow.tfa import TFAuto
tfa = TFAuto(train_data_path='/content/train_data/', test_data_path='/content/test_data/', path_root='/content/tfauto')
```

2. Step 1 - Automated EDA and Schema discovery
```
tfa.step_data_explore(viz=True) ##Viz=False for no visualization
```

3. Step 2 - Automated ML model build and train
```
tfa.step_model_build(label_column = 'price', model_type='REGRESSION', model_complexity=1)
```

4. Step 3 - Automated What-IF Tool launch
```
tfa.step_model_whatif()
```

### **API Arguments:**
- Method **TFAuto**
  - ```train_data_path```: Path where training data is stored
  - ```test_data_path```: Path where Test / Eval data is stored
  - ```path_root```: Directory for running TFAuto( Directory should NOT exist )

- Method **step_data_explore**
  - ```viz```: Is data visualization required ? - True or False( Default )

- Method **step_model_build**
  - `label_column`: The feature to be used as Label
  - `model_type`: Either of 'REGRESSION'( Default ), 'CLASSIFICATION'
  - `model_complexity`:
    - `0` : Model with default hyper-parameters
    - `1` (Default): Model with automated hyper-parameter tuning
    - `2` : Complexity 1 + Advanced fine-tuning of Text layers

### **Current limitations:**
There are a few limitations in the initial release but we are working day and night to resolve these and **add them as future features**.
1. Doesn't support Image / Audio data

### **Future roadmap:**
1. Add support for Timeseries / Audio / Image data
2. Add feature to download full pipeline model Python code for advanced tweaking

### **Release History:**
**1.3.2** - 27/11/2021 - [Release Notes](https://github.com/rafiqhasan/auto-tensorflow/releases/tag/1.3.2)

**1.3.1** - 18/11/2021 - [Release Notes](https://github.com/rafiqhasan/auto-tensorflow/releases/tag/1.3.1)

**1.2.0** - 24/07/2021 - [Release Notes](https://github.com/rafiqhasan/auto-tensorflow/releases/tag/1.2.0)

**1.1.1** - 14/07/2021 - [Release Notes](https://github.com/rafiqhasan/auto-tensorflow/releases/tag/1.1.1)

**1.0.1** - 07/07/2021 - [Release Notes](https://github.com/rafiqhasan/auto-tensorflow/releases/tag/1.0.1)
