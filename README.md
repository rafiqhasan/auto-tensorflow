<img src="/logo.png" width="100%"/>

### **Mission:**
**Build Low Code Automated Tensorflow explainable models in just 3 lines of code.**

We aim to make Deep Learning on Tensorflow absolutely easy for the masses with our low code framework and also increase trust on ML models through What-IF tool simulation.

### **Under the hood:**
Built on top of the powerful **Tensorflow** ecosystem like **TFX** , **TF APIs** and **What-IF Tool** , the library automatically does all the heavy lifting internally like EDA, schema discovery, HPT, model search etc. This empowers developers to focus only on building end user applications quickly without any knowledge of Tensorflow, ML or debugging. There is no dependency on Pandas / SKLearn or other libraries which makes the whole pipeline highly scalable on any volume of data. Moreover the models trained with auto-tensorflow can directly be deployed on any cloud like GCP / AWS / Azure.

<img src="/header.png" width="100%"/>

### **Features:**
1. Build Classification / Regression models on CSV data
2. Automated Schema Inference
3. Automated EDA and visualization
4. Automated Model build for mixed data types( Continuous, Categorical and Free Text )
5. Automated Hyper-parameter tuning
6. Automated UI based What-IF analysis
7. Control over complexity of model
8. No dependency over Pandas / SKLearn
9. Can handle dataset of any size - including multiple CSV files

### **Setup:**
1. Install library using - ```!pip install auto-tensorflow```
2. Works best on UNIX/Linux/Debian/Google Colab

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

**Tutorials**: https://github.com/rafiqhasan/auto-tensorflow/tree/main/tutorials

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
  - `model_complexity`: 0 to 1 (0: Model without HPT, 1(Default): Model with HPT) -> More will be added in future

### **Current limitations:**
There are a few limitations in the initial release but we are working day and night to resolve these and **add them as future features**.
1. Doesn't support Image / Audio data
2. Doesn't support - quote delimited CSVs( TFX doesn't support qCSV yet )
3. Classification only supports integer labels from 0 to N

### **Future roadmap:**
1. Add support for Timeseries / Audio / Image data
2. Add support for quoted CSVs
3. Add feature to download full pipeline model Python code for advanced tweaking
