##Owner: Hasan Rafiq
##URL: https://www.linkedin.com/in/sam04/
import os
import tempfile
import re
import json
import urllib

import absl
import tensorflow as tf
import tensorflow.keras as keras
import tfx
import tensorflow_hub as hub
import kerastuner as kt
import tensorflow_model_analysis as tfma
import witwidget
import tensorflow_data_validation as tfdv
tf.get_logger().propagate = False

from tfx.components import CsvExampleGen
from typing import Dict, List, Text
from tfx.components import Evaluator, ExampleValidator, Pusher, ResolverNode, SchemaGen, Trainer, StatisticsGen, Transform
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input
from tfx.orchestration.local.local_dag_runner import LocalDagRunner
from ml_metadata.proto import metadata_store_pb2
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.types import standard_component_specs
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_data_validation.utils import io_util, display_util
from google.protobuf import text_format
from google.protobuf.json_format import MessageToDict
from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget

class TFAutoUtils():
  def __init__(self, data_path, path_root='/tfx'):
    ##Define all constants
    self._tfx_root = os.path.join(os.getcwd(), path_root)
    self._pipeline_root = os.path.join(self._tfx_root, 'pipelines');      # Join ~/tfx/pipelines/
    self._metadata_db_root = os.path.join(self._tfx_root, 'metadata.db');    # Join ~/tfx/metadata.db
    self._metadata = os.path.join(self._tfx_root, 'metadata');    # Join ~/tfx/metadata
    self._log_root = os.path.join(self._tfx_root, 'logs');
    self._model_root = os.path.join(self._tfx_root, 'model');
    self._data_path = data_path

  def check_directories(self):
    if os.path.exists(self._tfx_root):
      raise Exception("Root Directory: {} already exists. Please make sure directory doesn't exist!".format(self._tfx_root))

  def create_directories(self):
    self.check_directories()

    directories = [self._tfx_root, self._pipeline_root, self._metadata,
                   self._log_root, self._model_root]
    [ print("Creating {}".format(d)) for d in directories ]
    [ os.mkdir(d) for d in directories ]

  def load_anomalies_binary(self, input_path: Text) -> anomalies_pb2.Anomalies:
    """Loads the Anomalies proto stored in text format in the input path.
    Args:
      input_path: File path from which to load the Anomalies proto.
    Returns:
      An Anomalies protocol buffer.
    """
    anomalies_proto = anomalies_pb2.Anomalies()

    anomalies_proto.ParseFromString(io_util.read_file_to_string(
        input_path, binary_mode=True))

    return anomalies_proto

  def _get_latest_execution(self, metadata, pipeline_name, component_id):
    """Gets the execution objects for the latest run of the pipeline."""
    node_context = metadata.store.get_context_by_type_and_name(
        'node', f'{pipeline_name}.{component_id}')
    executions = metadata.store.get_executions_by_context(node_context.id)
    # Pick the latest one.
    return max(executions, key=lambda e: e.last_update_time_since_epoch)

  #Get all running artifact details from MLMD
  def _get_artifacts_for_component_id(self, metadata, execution):
    return execution_lib.get_artifacts_dict(metadata, execution.id,
                                          metadata_store_pb2.Event.OUTPUT)
  
  #Get all running artifact directories from MLMD for later uses
  def get_artifacts_directories(self, component_name='StatisticsGen'):
    metadata_connection_config = metadata.sqlite_metadata_connection_config(self._metadata_db_root)

    with metadata.Metadata(metadata_connection_config) as metadata_handler:
      execution = self._get_latest_execution(metadata_handler, 'data_pipeline', component_name)
      output_directory = self._get_artifacts_for_component_id(metadata_handler, execution)
    
    return output_directory

class TFAutoData():
  def __init__(self):
    ##Define all constants
    self.features_list = []     #Features used for training
    self._train_data_path = ''  #Training data path
    self.schema = ''            #Schema details of data
    self.stats_train = ''       #Statistics of Train data
    self.stats_eval = ''        #Statistics of Eval data
    self.anom_train = ''
    self.anom_eval = ''
    self.file_headers = []      #Headers of CSV train file
    self._len_train = 0         #Training data size
    self._run = False           #Run flag

  def collect_feature_details(self, schema):
    features_list = []
    features_dict = display_util.get_schema_dataframe(schema).to_dict('index')
    features_stats = MessageToDict(self.stats_train)
    self._len_train = features_stats['datasets'][0]['numExamples']

    for f_ in features_dict.keys():
      features_dict[f_]['feature'] = re.sub( r"\'", "", f_)
      #Feature has a domain( categorical feature )
      if features_dict[f_]['Domain'] != '-':
        features_dict[f_]['categorical_values'] = [ v for v in tfdv.get_domain(schema, features_dict[f_]['feature']).value ]
        features_dict[f_]['num_categorical_values'] = len(tfdv.get_domain(schema, features_dict[f_]['feature']).value)

        #Handle for free Text( If ratio of unique values with rows > 0.2 )
        if int(features_dict[f_]['num_categorical_values']) / int(self._len_train) > 0.2:
          features_dict[f_]['Type'] = 'STRING'
          features_dict[f_]['categorical_values'] = ""
          features_dict[f_]['num_categorical_values'] = 0
        else:
          features_dict[f_]['Type'] = 'CATEGORICAL'

      #Min/Max for numerical features
      for feat in features_stats['datasets'][0]['features']:
        curr_feat = feat['path']['step'][0]
        if curr_feat == features_dict[f_]['feature'] and features_dict[f_]['Type'] in ['INT','FLOAT']:
          features_dict[f_]['min'] = feat['numStats'].get('min', 0.0)
          features_dict[f_]['max'] = feat['numStats'].get('max', 0.0)
          features_dict[f_]['mean'] = feat['numStats'].get('mean', 0.0)
          features_dict[f_]['std_dev'] = feat['numStats'].get('stdDev', 1)
      
      features_list.append(features_dict[f_])
    self.features_list = features_list
    return self.features_list

  def get_columns_from_file_header(self, path, num_cols):
    record_defaults=[]
    #Create dataset input functions
    if os.path.isdir(path):
      path = path + "*"
    elif os.path.isfile(path):
      path = path

    for _ in range(num_cols):
      record_defaults.append('')

    # Create list of files that match pattern
    file_list = tf.io.gfile.glob(path)

    # Create dataset from file list
    dataset = tf.data.experimental.CsvDataset(file_list, header=False, record_defaults=record_defaults, use_quote_delim=False)

    for example in dataset.take(1):
      return ([e.numpy().decode('utf-8') for e in example])

  def run_initial(self, _data_path, _tfx_root, _metadata_db_root, tfautils, viz=False):
    """Run all data steps in pipeline and generate visuals"""
    self.example_gen = CsvExampleGen(input=external_input(_data_path))

    self.statistics_gen = StatisticsGen(examples=self.example_gen.outputs['examples'])

    self.infer_schema = SchemaGen(
        statistics=self.statistics_gen.outputs['statistics'], infer_feature_shape=False)

    self.validate_stats = ExampleValidator(
      statistics=self.statistics_gen.outputs['statistics'],
      schema=self.infer_schema.outputs['schema'])
    
    #Create pipeline
    self.pipeline = pipeline.Pipeline(
      pipeline_name=  'data_pipeline',
      pipeline_root=  _tfx_root,
      components=[
          self.example_gen, self.statistics_gen, self.infer_schema, self.validate_stats
      ],
      metadata_connection_config = metadata.sqlite_metadata_connection_config(_metadata_db_root),
      enable_cache=True,
      beam_pipeline_args=['--direct_num_workers=%d' % 0, '--direct_running_mode=multi_threading'],
    )

    #Run data pipeline
    print("Data: Pipeline execution started...")
    LocalDagRunner().run(self.pipeline)
    self._run = True

    #Get directories after run
    dir_stats = tfautils.get_artifacts_directories('StatisticsGen')
    dir_anom = tfautils.get_artifacts_directories('ExampleValidator')
    dir_schema = tfautils.get_artifacts_directories('SchemaGen')

    #Get statistics
    stats_url_train = str(dir_stats['statistics'][0].uri) + "/Split-train/FeatureStats.pb"
    self.stats_train = tfdv.load_stats_binary(stats_url_train)
    stats_url_eval = str(dir_stats['statistics'][0].uri) + "/Split-eval/FeatureStats.pb"
    self.stats_eval = tfdv.load_stats_binary(stats_url_eval)

    #Get data anomalies
    anom_url_train = str(dir_anom['anomalies'][0].uri) + "/Split-train/SchemaDiff.pb"
    self.anom_train = tfautils.load_anomalies_binary(anom_url_train)
    anom_url_eval = str(dir_anom['anomalies'][0].uri) + "/Split-eval/SchemaDiff.pb"
    self.anom_eval = tfautils.load_anomalies_binary(anom_url_eval)

    #Get Schema and Features details, generate config JSON
    schema_url = str(dir_schema['schema'][0].uri) + "/schema.pbtxt"
    self.schema = tfdv.load_schema_text(schema_url)
    self.features_list = self.collect_feature_details(self.schema)

    #Get columns from training file
    self.file_headers = self.get_columns_from_file_header(_data_path, len(self.features_list))

    # Visualize results using TFDV
    if viz==True:
      #Show Schema Gen
      print("\n### Generating schema visuals")
      tfdv.display_schema(self.schema)

      #Show Train Schema Stats
      print("\n### Generating Train Data Statistics Visuals...")
      tfdv.visualize_statistics(self.stats_train)

      #Show Eval Schema Stats
      print("\n### Generating Test Data Statistics Visuals...")
      tfdv.visualize_statistics(self.stats_eval)

      #Show Train Anomalies
      print("\n### Generating Train Data Anomaly Visuals...")
      tfdv.display_anomalies(self.anom_train)

      #Show Eval Anomalies
      print("\n### Generating Test Data Anomaly Visuals...")
      tfdv.display_anomalies(self.anom_eval)

    return self.pipeline

class TextEncoder(tf.keras.Model):
  def __init__(self):
    super(TextEncoder, self).__init__()
    self.encoder = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

  def __call__(self, inp):
    #Preprocess text
    # encoder_inputs = self.preprocessor(inp)

    #Encode text
    embedding = self.encoder(inp)

    return embedding

class TFAutoModel():
  def __init__(self, _tfx_root, train_data_path, test_data_path):
    ##Define all constants
    self._tfx_root = _tfx_root
    self._config_json = ''
    self._pipeline_root = os.path.join(self._tfx_root, 'pipelines');      # Join ~/tfx/pipelines/
    self._log_root = os.path.join(self._tfx_root, 'logs');
    self._model_root = os.path.join(self._tfx_root, 'model');
    self._label = ''                                                      #Label
    self._features = []                                                   #List of features to be used for modeling
    self._train_data_path = train_data_path                               #Training data
    self._test_data_path = test_data_path                                 #Test data
    self._model_type = ''
    self._model_complexity = 1
    self._defaults = []
    self._run = False           #Run flag
    self.hpt_config = {
        0:{
            'deep_neurons':{
                'min':64,
                'max':64
            },
           'wide_neurons':{
                'min':64,
                'max':64
           },
           'prefinal_dense': {
                'min':32,
                'max':32
           },
           'learning_rate':[0.01]
        },
        1:{
            'deep_neurons':{
                'min':32,
                'max':512
            },
           'wide_neurons':{
                'min':32,
                'max':512
           },
           'prefinal_dense': {
                'min':32,
                'max':128
           },
           'learning_rate':[0.01, 0.005, 0.002, 0.001, 0.0005]
        },
    }
    
    ##GPU Strategy
    self.strategy = tf.distribute.MirroredStrategy()

  def load_config_json(self):
    with open(os.path.join(self._tfx_root, 'config.json')) as f:
      self._config_json = json.load(f)

    #Create list of features
    for feats in self._config_json['data_schema']:
      #Don't include ignored features
      if feats['feature'] in self._config_json['ignore_features'] or feats['feature'] == self._label:
        continue
      else:
        self._features.append(feats['feature'])

  def make_input_fn(self, filename, mode, vnum_epochs = None, batch_size = 512):
    CSV_COLUMNS = [ feats['feature'] for feats in self._config_json['data_schema'] ]
    LABEL_COLUMN = self._label

    # Set default values for required only CSV columns + LABEL
    # This has to be in sequence of CSV columns 0 to N
    DEFAULTS = []
    for f_ in self._config_json['file_headers']:
      for feats in self._config_json['data_schema']:
        if feats['feature'] != f_:
          continue

        #Logic for default values
        if feats['Type'] in [ 'CATEGORICAL', 'STRING' ]:
          DEFAULTS.append([''])
        elif feats['Type'] == 'FLOAT':
          DEFAULTS.append([tf.cast(0, tf.float32)])
        elif feats['Type'] == 'INT':
          DEFAULTS.append([tf.cast(0, tf.int64)])

        # print("Default for {} is {}".format(f_, DEFAULTS[-1]))
        
        break

    self._defaults = DEFAULTS

    ###############################
    ##Feature engineering functions
    def feature_engg_features(features):
      #Apply data type conversions
      for feats in self._config_json['data_schema']:
        if feats['feature'] == self._label or not feats['feature'] in self._features:
          continue

        #Convert dtype of all tensors as per requested schema
        if feats['Type'] in [ 'CATEGORICAL', 'STRING' ] and features[feats['feature']].dtype != tf.string:
          features[feats['feature']] = tf.strings.as_string(features[feats['feature']])
        elif feats['Type'] == 'FLOAT' and features[feats['feature']].dtype != tf.float32:
          #Needs special handling for strings
          if features[feats['feature']].dtype != tf.string:
            features[feats['feature']] = tf.cast(features[feats['feature']], dtype=tf.float32)
        elif feats['Type'] == 'INT' and features[feats['feature']].dtype != tf.int64:
          #Needs special handling for strings
          if features[feats['feature']].dtype != tf.string:
            features[feats['feature']] = tf.cast(features[feats['feature']], dtype=tf.int64)

      return(features)

    #To be called from TF
    def feature_engg(features, label):
      #Add new features
      features = feature_engg_features(features)
      return(features, label)

    def _input_fn(v_test=False):     
        # Create list of files that match pattern
        file_list = tf.io.gfile.glob(filename)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = vnum_epochs # indefinitely
        else:
            num_epochs = 1 # end-of-input after this        
        
        # Create dataset from file list
        dataset = tf.data.experimental.make_csv_dataset(file_list,
                                                   batch_size=batch_size,
                                                   column_names=self._config_json['file_headers'],
                                                   column_defaults=DEFAULTS,
                                                   label_name=LABEL_COLUMN,
                                                   num_epochs = num_epochs,
                                                   num_parallel_reads=30)
        
        dataset = dataset.prefetch(buffer_size = batch_size)

        #Feature engineering
        dataset = dataset.map(feature_engg)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = vnum_epochs # indefinitely
            dataset = dataset.shuffle(buffer_size = batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs)       
        
        #Begins - Uncomment for testing only -----------------------------------------------------<
        if v_test == True:
          print(next(dataset.__iter__()))
          
        #End - Uncomment for testing only -----------------------------------------------------<
        return dataset
    return _input_fn

  # def make_input_fn_gz(self, dir_uri, mode, vnum_epochs = None, batch_size = 512):
  #   def decode_tfr(serialized_example):
  #     schema = {}
  #     features = {}
  #     for feats in self._config_json['data_schema']:
  #       #1. Create GZIP TFR parser schema
  #       # if feats['Type'] == 'CATEGORICAL':
  #       #   schema[feats['feature']] =  tf.io.FixedLenFeature([], tf.string, default_value="")
  #       # elif feats['Type'] == 'FLOAT':
  #       #   schema[feats['feature']] =  tf.io.FixedLenFeature([], tf.float32, default_value=0.0 )
  #       # elif feats['Type'] == 'INT':
  #       #   schema[feats['feature']] =  tf.io.FixedLenFeature([], tf.int64, default_value=0)

  #       if feats['Type'] == 'CATEGORICAL':
  #         schema[feats['feature']] =  tf.io.VarLenFeature(tf.string)
  #       elif feats['Type'] == 'FLOAT':
  #         schema[feats['feature']] =  tf.io.VarLenFeature(tf.float32)
  #       elif feats['Type'] == 'INT':
  #         schema[feats['feature']] =  tf.io.VarLenFeature(tf.int64)

  #     # 1. define a parser
  #     features = tf.io.parse_example(
  #       serialized_example,
  #       # Defaults are not specified since both keys are required.
  #       features=schema)

  #     return features, features[self._label]

  #   def _input_fn(v_test=False):
  #     # Get the list of files in this directory (all compressed TFRecord files)
  #     tfrecord_filenames = tf.io.gfile.glob(dir_uri)

  #     # Create a `TFRecordDataset` to read these files
  #     dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type="GZIP")

  #     if mode == tf.estimator.ModeKeys.TRAIN:
  #       num_epochs = vnum_epochs # indefinitely
  #     else:
  #       num_epochs = 1 # end-of-input after this

  #     dataset = dataset.batch(batch_size)
  #     dataset = dataset.prefetch(buffer_size = batch_size)

  #     #Convert TFRecord data to dict
  #     dataset = dataset.map(decode_tfr)

  #     #Feature engineering
  #     # dataset = dataset.map(feature_engg)

  #     if mode == tf.estimator.ModeKeys.TRAIN:
  #         num_epochs = vnum_epochs # indefinitely
  #         dataset = dataset.shuffle(buffer_size = batch_size)
  #     else:
  #         num_epochs = 1 # end-of-input after this

  #     dataset = dataset.repeat(num_epochs)       
      
  #     #Begins - Uncomment for testing only -----------------------------------------------------<
  #     if v_test == True:
  #       print(next(dataset.__iter__()))
        
  #     #End - Uncomment for testing only -----------------------------------------------------<
  #     return dataset
  #   return _input_fn

  def create_feature_cols(self):
    #Keras format features
    feats_dict = {}
    keras_dict_input = {}
    for feats in self._config_json['data_schema']:
      #Only include features
      if feats['feature'] not in self._features:
        continue

      #Create feature columns list
      if feats['Type'] in [ 'CATEGORICAL', 'STRING' ]:
        feats_dict[feats['feature']] = tf.keras.Input(name=feats['feature'], shape=(1,), dtype=tf.string)
      elif feats['Type'] == 'FLOAT':
        feats_dict[feats['feature']] = tf.keras.Input(name=feats['feature'], shape=(1,), dtype=tf.float32)
      elif feats['Type'] == 'INT':
        feats_dict[feats['feature']] = tf.keras.Input(name=feats['feature'], shape=(1,), dtype=tf.int32)

      for k_ in feats_dict.keys():
        keras_dict_input[k_] = feats_dict[k_]

    self._feature_cols = {'K' : keras_dict_input}
    return self._feature_cols

  def create_keras_model_classification(self, hp):
    # with self.strategy.scope():
      # params = self.params_default
      feature_cols = self._feature_cols

      #Number of classes
      for feats in self._config_json['data_schema']:
        #Only include features
        if feats['feature'] == self._label:
          num_classes = int(feats['max'] + 1)
          break

      METRICS = [
          # tf.keras.metrics.AUC(multi_label=True, num_labels=num_classes),
          'sparse_categorical_accuracy'
      ]

      #Input layers
      input_feats = []
      for inp in feature_cols['K'].keys():
        input_feats.append(feature_cols['K'][inp])

      ##Input processing
      ##https://keras.io/examples/structured_data/structured_data_classification_from_scratch/
      ##https://github.com/tensorflow/community/blob/master/rfcs/20191212-keras-categorical-inputs.md

      ##Automated feature handling
      #Handle categorical attributes( One-hot encoding )
      feat_cat = []
      for feats in self._config_json['data_schema']:
        if feats['feature'] in self._features and feats['Type'] == 'CATEGORICAL':
          feat_cat.append('')
          cat_len = feats['num_categorical_values']
          cat = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=feats['categorical_values'], mask_token=None, oov_token = '~UNK~')(feature_cols['K'][feats['feature']])
          feat_cat[-1] = tf.keras.layers.experimental.preprocessing.CategoryEncoding(num_tokens = cat_len + 1)(cat)

      #Handle numerical attributes
      feat_numeric = []
      for feats in self._config_json['data_schema']:
        if feats['feature'] in self._features and feats['Type'] not in [ 'CATEGORICAL', 'STRING' ]:
          feat_numeric.append('')
          
          #Apply normalization
          feat_numeric[-1] = ( tf.cast(feature_cols['K'][feats['feature']], tf.float32) - feats['mean'] ) / feats['std_dev']

      #Handle Text attributes
      feat_text = []
      text_emb = TextEncoder()
      for feats in self._config_json['data_schema']:
        if feats['Type'] == 'STRING':
          feat_text.append('')
          
          #Apply Text Encoding from TFHub
          feat_text[-1] = text_emb(tf.squeeze(tf.cast(feature_cols['K'][feats['feature']], tf.string)))

      ###Create MODEL
      ####Concatenate all features( Numerical input )
      numeric_features_count = 0
      if len(feat_numeric) > 0:
        numeric_features_count += 1
        x_input_numeric = tf.keras.layers.concatenate(feat_numeric)
        
        #DEEP - This Dense layer connects to input layer - Numeric Data
        deep_neurons = hp.Int('deep_neurons', min_value=self.hpt_config[self._model_complexity]['deep_neurons']['min'], 
                                              max_value=self.hpt_config[self._model_complexity]['deep_neurons']['max'],
                              step=32)
        x_numeric = tf.keras.layers.Dense(deep_neurons, activation='relu', kernel_initializer="he_uniform")(x_input_numeric)
        x_numeric = tf.keras.layers.BatchNormalization()(x_numeric)

      ####Concatenate all Categorical features( Categorical converted )
      text_features_count = 0
      if len(feat_text) > 0:
        text_features_count += 1
        x_input_text = tf.keras.layers.concatenate(feat_text)

      ####Concatenate all Categorical features( Categorical converted )
      categ_features_count = 0
      if len(feat_cat) > 0:
        categ_features_count += 1
        x_input_categ = tf.keras.layers.concatenate(feat_cat)    
        
        #WIDE - This Dense layer connects to input layer - Categorical Data
        wide_neurons = hp.Int('wide_neurons', min_value=self.hpt_config[self._model_complexity]['wide_neurons']['min'],
                                              max_value=self.hpt_config[self._model_complexity]['wide_neurons']['max'], 
                              step=32)
        x_categ = tf.keras.layers.Dense(wide_neurons, activation='relu', kernel_initializer="he_uniform")(x_input_categ)

      ####Concatenate both Wide and Deep layers
      if numeric_features_count > 0 and categ_features_count > 0 and text_features_count > 0:
        x = tf.keras.layers.concatenate([x_numeric, x_categ, x_input_text])
      elif numeric_features_count == 0 and categ_features_count > 0 and text_features_count > 0:
        x = tf.keras.layers.concatenate([x_categ, x_input_text])
      elif numeric_features_count > 0 and categ_features_count == 0 and text_features_count > 0:
        x = tf.keras.layers.concatenate([x_numeric, x_input_text])
      elif numeric_features_count > 0 and categ_features_count > 0 and text_features_count == 0:
        x = tf.keras.layers.concatenate([x_numeric, x_categ])
      elif numeric_features_count > 0 and categ_features_count == 0 and text_features_count == 0:
        x = x_numeric
      elif numeric_features_count == 0 and categ_features_count > 0 and text_features_count == 0:
        x = x_categ
      elif numeric_features_count == 0 and categ_features_count == 0 and text_features_count > 0:
        x = x_input_text 

      prefinal_dense = hp.Int('prefinal_dense', min_value=self.hpt_config[self._model_complexity]['prefinal_dense']['min'], 
                                                max_value=self.hpt_config[self._model_complexity]['prefinal_dense']['max'],
                              step=32)
      x = tf.keras.layers.Dense(prefinal_dense, activation='relu', kernel_initializer="he_uniform",
                                activity_regularizer=tf.keras.regularizers.l2(0.00001))(x)
      x = tf.keras.layers.BatchNormalization()(x)

      #Final Layer
      # out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(x)
      out = tf.keras.layers.Dense(num_classes, activation='softmax', name='out')(x)
      model = tf.keras.Model(input_feats, out)

      #Set optimizer
      hp_learning_rate = hp.Choice('learning_rate', values=self.hpt_config[self._model_complexity]['learning_rate'], ordered=True)
      opt = tf.keras.optimizers.Adam(lr = hp_learning_rate)

      #Compile model
      model.compile(loss='sparse_categorical_crossentropy',  optimizer=opt, metrics = METRICS)

      return model

  def create_keras_model_regression(self, hp):
    # with self.strategy.scope():
      METRICS = [
              keras.metrics.RootMeanSquaredError(name='rmse'),
              keras.metrics.MeanAbsolutePercentageError(name='mape')
      ]

      # params = self.params_default
      feature_cols = self._feature_cols

      #Input layers
      input_feats = []
      for inp in feature_cols['K'].keys():
        input_feats.append(feature_cols['K'][inp])

      ##Input processing
      ##https://keras.io/examples/structured_data/structured_data_classification_from_scratch/
      ##https://github.com/tensorflow/community/blob/master/rfcs/20191212-keras-categorical-inputs.md

      ##Automated feature handling
      #Handle categorical attributes( One-hot encoding )
      feat_cat = []
      for feats in self._config_json['data_schema']:
        if feats['feature'] in self._features and feats['Type'] == 'CATEGORICAL':
          feat_cat.append('')
          cat_len = feats['num_categorical_values']
          cat = tf.keras.layers.experimental.preprocessing.StringLookup(vocabulary=feats['categorical_values'], mask_token=None, oov_token = '~UNK~')(feature_cols['K'][feats['feature']])
          feat_cat[-1] = tf.keras.layers.experimental.preprocessing.CategoryEncoding(num_tokens = cat_len + 1)(cat)

      #Handle numerical attributes
      feat_numeric = []
      for feats in self._config_json['data_schema']:
        if feats['feature'] in self._features and feats['Type'] not in [ 'CATEGORICAL', 'STRING' ]:
          feat_numeric.append('')
          
          #apply normalization
          feat_numeric[-1] = ( tf.cast(feature_cols['K'][feats['feature']], tf.float32) - feats['mean'] ) / feats['std_dev']
      
      #Handle Text attributes
      feat_text = []
      text_emb = TextEncoder()
      for feats in self._config_json['data_schema']:
        if feats['Type'] == 'STRING':
          feat_text.append('')
          
          #Apply Text Encoding from TFHub
          feat_text[-1] = text_emb(tf.squeeze(tf.cast(feature_cols['K'][feats['feature']], tf.string)))

      ###Create MODEL
      ####Concatenate all features( Numerical input )
      numeric_features_count = 0
      if len(feat_numeric) > 0:
        numeric_features_count += 1
        x_input_numeric = tf.keras.layers.concatenate(feat_numeric)
        
        #DEEP - This Dense layer connects to input layer - Numeric Data
        deep_neurons = hp.Int('deep_neurons', min_value=self.hpt_config[self._model_complexity]['deep_neurons']['min'], 
                                              max_value=self.hpt_config[self._model_complexity]['deep_neurons']['max'],
                              step=32)
        x_numeric = tf.keras.layers.Dense(deep_neurons, activation='relu', kernel_initializer="he_uniform")(x_input_numeric)
        x_numeric = tf.keras.layers.BatchNormalization()(x_numeric)

      ####Concatenate all Categorical features( Categorical converted )
      text_features_count = 0
      if len(feat_text) > 0:
        text_features_count += 1
        x_input_text = tf.keras.layers.concatenate(feat_text)

      ####Concatenate all Categorical features( Categorical converted )
      categ_features_count = 0
      if len(feat_cat) > 0:
        categ_features_count += 1
        x_input_categ = tf.keras.layers.concatenate(feat_cat)        
        
        #WIDE - This Dense layer connects to input layer - Categorical Data
        wide_neurons = hp.Int('wide_neurons', min_value=self.hpt_config[self._model_complexity]['wide_neurons']['min'], 
                                              max_value=self.hpt_config[self._model_complexity]['wide_neurons']['max'], 
                              step=32)
        x_categ = tf.keras.layers.Dense(wide_neurons, activation='relu', kernel_initializer="he_uniform")(x_input_categ)

      ####Concatenate both Wide and Deep layers
      if numeric_features_count > 0 and categ_features_count > 0 and text_features_count > 0:
        x = tf.keras.layers.concatenate([x_numeric, x_categ, x_input_text])
      elif numeric_features_count == 0 and categ_features_count > 0 and text_features_count > 0:
        x = tf.keras.layers.concatenate([x_categ, x_input_text])
      elif numeric_features_count > 0 and categ_features_count == 0 and text_features_count > 0:
        x = tf.keras.layers.concatenate([x_numeric, x_input_text])
      elif numeric_features_count > 0 and categ_features_count > 0 and text_features_count == 0:
        x = tf.keras.layers.concatenate([x_numeric, x_categ])
      elif numeric_features_count > 0 and categ_features_count == 0 and text_features_count == 0:
        x = x_numeric
      elif numeric_features_count == 0 and categ_features_count > 0 and text_features_count == 0:
        x = x_categ
      elif numeric_features_count == 0 and categ_features_count == 0 and text_features_count > 0:
        x = x_input_text 

      prefinal_dense = hp.Int('prefinal_dense', min_value=self.hpt_config[self._model_complexity]['prefinal_dense']['min'], 
                                                max_value=self.hpt_config[self._model_complexity]['prefinal_dense']['max'],
                              step=32)
      x = tf.keras.layers.Dense(prefinal_dense, activation='relu', kernel_initializer="he_uniform",
                                activity_regularizer=tf.keras.regularizers.l2(0.00001))(x)
      x = tf.keras.layers.BatchNormalization()(x)

      #Final Layer
      out = tf.keras.layers.Dense(1, activation='relu', name='out')(x)
      model = tf.keras.Model(input_feats, out)

      #Set optimizer
      hp_learning_rate = hp.Choice('learning_rate', values=self.hpt_config[self._model_complexity]['learning_rate'], ordered=True)
      opt = tf.keras.optimizers.Adam(lr = hp_learning_rate)

      #Compile model
      model.compile(loss='mean_squared_error',  optimizer=opt, metrics = METRICS)

      return model

  def keras_train_and_evaluate(self, model, epochs=100, mode='Train'):
    #Add callbacks
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=5, min_lr=0.00001, verbose = 1)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

    #Create dataset input functions
    if os.path.isdir(self._train_data_path):
      train_file_path = self._train_data_path + "*"
    elif os.path.isfile(self._train_data_path):
      train_file_path = self._train_data_path

    if os.path.isdir(self._test_data_path):
      test_file_path = self._test_data_path + "*"
    elif os.path.isfile(self._test_data_path):
      test_file_path = self._test_data_path

    train_dataset = self.make_input_fn(filename = train_file_path,
                        mode = tf.estimator.ModeKeys.TRAIN,
                        batch_size = 128)()
    # eval_file = '/content/tfauto/CsvExampleGen/examples/1/Split-train/*'
    # train_dataset = self.make_input_fn_gz(dir_uri = eval_file,
    #                 mode = tf.estimator.ModeKeys.TRAIN,
    #                 batch_size = 10)()

    validation_dataset = self.make_input_fn(filename = test_file_path,
                        mode = tf.estimator.ModeKeys.EVAL,
                        batch_size = 512)()
    # validation_dataset = self.make_input_fn_gz(dir_uri = eval_file,
    #                 mode = tf.estimator.ModeKeys.TRAIN,
    #                 batch_size = 10)()
    
    #Train and Evaluate
    if mode == 'Train':
      if self._model_type == 'REGRESSION':
        print("Training a regression model...")
      elif self._model_type == 'CLASSIFICATION':
        print("Training a classification model...")

      #Start training loop
      self._model.fit(train_dataset, 
                      validation_data = validation_dataset,
                      epochs=epochs,
                      # validation_steps = 3,   ###Keep this none for running evaluation on full EVAL data every epoch
                      steps_per_epoch = 100,   ###Has to be passed - Cant help it :) [ Number of batches per epoch ]
                      callbacks=[reduce_lr, #modelsave_callback, #tensorboard_callback, 
                                keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=True)]
                      )
    else:
      if self._model_type == 'REGRESSION':
        print("Hyper-Tuning a regression model...")
        mod_func = self.create_keras_model_regression
      elif self._model_type == 'CLASSIFICATION':
        print("Hyper-Tuning a classification model...")
        mod_func = self.create_keras_model_classification

      ###Create Tuner
      ###########################################
      tuner = kt.Hyperband(
                      mod_func,
                      objective='val_loss',
                      max_epochs=10)
      stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

      tuner.search(train_dataset, validation_data=validation_dataset, epochs=50, steps_per_epoch = 100, callbacks=[stop_early])
      # print(f"""
      # The hyperparameter search is complete. The optimal number of units in the first densely-connected
      # layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
      # is {best_hps.get('learning_rate')}.
      # """)
      best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
      print("Best LR: ", best_hps.get('learning_rate'))
      self._model = tuner.hypermodel.build(best_hps)

    return self._model
    
  def save_model(self):
    version = "1"  #{'serving_default': call_output}
    tf.saved_model.save(
        self._model,                        #Model
        self._model_root + "/"  + version   #Location
    )
  
  def search_hpt(self):
    #Create feature columns dynamically and model too
    self._feature_cols = self.create_feature_cols()

    #Start HPT
    self._model = self.keras_train_and_evaluate(None, epochs=50, mode='Tune')

    #Print summary
    print(self._model.summary())

  def start_train(self):
    #Create feature columns dynamically and model too
    self._feature_cols = self.create_feature_cols()

    #Start training loop on HPT best found model
    try:
      tf.keras.utils.plot_model(self._model, show_shapes=True, rankdir="LR")
    except:
      1 - 1

    self._model = self.keras_train_and_evaluate(self._model, epochs=9999, mode='Train')

    #Save model
    self.save_model()

  def generate_examples_for_wit(self):
    max_samples = 100
    examples = []
    record_defaults=[]
    out = {}
    path = self._test_data_path
    #Create dataset input functions
    if os.path.isdir(path):
      path = path + "*"
    elif os.path.isfile(path):
      path = path

    # Create list of files that match pattern
    file_list = tf.io.gfile.glob(path)

    # Create dataset from file list
    dataset = tf.data.experimental.make_csv_dataset(file_list, header=True, batch_size=max_samples
                                                    ,num_epochs=1, column_defaults=self._defaults)
    
    #Get first batch
    for features in dataset.take(1):
      for i, (name, value) in enumerate(features.items()):
        out[name] = value.numpy()

    #Generate examples
    for row in range(max_samples):
      example = tf.train.Example()
      #For each column in file
      for f_ in self._config_json['file_headers']:
        for feats in self._config_json['data_schema']:
          if feats['feature'] != f_:
            continue

          #Prepare example data
          if feats['Type'] in [ 'CATEGORICAL', 'STRING' ]:
            example.features.feature[f_].bytes_list.value.append(out[f_][row])
          elif feats['Type'] == 'FLOAT':
            example.features.feature[f_].float_list.value.append(out[f_][row])
          elif feats['Type'] == 'INT':
            example.features.feature[f_].int64_list.value.append(int(out[f_][row]))
      examples.append(example)
    return examples

  #Create prediction function to link in WIT
  def wit_prediction_fn_dyn(self, examples):
    version = "1"
    out = []

    #LOCAL: Predict using Keras prediction function
    saved_mod = tf.saved_model.load(self._model_root + "/"  + version)   #Location)

    #Get prediction function from serving
    mod_fn = saved_mod.signatures['serving_default']

    for ex in examples:
      #Extract features from each example
      keyword_args = {}
      test_data = ex.features

      for f_ in self._config_json['file_headers']:
        if f_ == self._label or f_ in self._config_json['ignore_features'] :
          continue
        for feats in self._config_json['data_schema']:
          if feats['feature'] != f_:
            continue

          #Prepare example data
          if feats['Type'] in [ 'CATEGORICAL', 'STRING' ]:
            keyword_args[f_] = tf.convert_to_tensor([test_data.feature[f_].bytes_list.value])
          elif feats['Type'] == 'FLOAT':
            keyword_args[f_] = tf.convert_to_tensor([test_data.feature[f_].float_list.value])
          elif feats['Type'] == 'INT':
            keyword_args[f_] = tf.convert_to_tensor([test_data.feature[f_].int64_list.value])

      #Run prediction function on saved model
      # print(keyword_args)
      # break
      pred = mod_fn(**keyword_args)

      p_ = pred['out'].numpy()
      out.append(p_[0])
    
    return out

  def call_wit(self):
    #Generate examples for WIT
    examples_wit = self.generate_examples_for_wit()
    if self._model_type == 'REGRESSION':
      wit_type = 'regression'
    elif self._model_type == 'CLASSIFICATION':
      wit_type = 'classification'

    config_builder = (WitConfigBuilder(examples_wit, self._config_json['file_headers'])
                      .set_custom_predict_fn(self.wit_prediction_fn_dyn)
                      .set_model_type(wit_type))
    
    WitWidget(config_builder)
  
  def prechecks(self):
    '''Set of tests to run before training'''
    success_flag = True
    #Test 1 -> Label Data Type check
    for feats in self._config_json['data_schema']:
      if feats['feature'] != self._label:
        continue

      #Only allow numerical values
      if feats['Type'] in [ 'CATEGORICAL', 'STRING' ]:
        print("Error: Label should be numerical only")
        success_flag = False
        return success_flag
      
      if self._model_type == "CLASSIFICATION" and feats['Type'] != 'INT':
        print("Error: CLASSIFICATION - Label data type is not correct")
        success_flag = False
        return success_flag

    #Test 2 -> Label values check
    for feats in self._config_json['data_schema']:
      if feats['feature'] != self._label:
        continue

      #For classification, minimum value should be 0 and type should be INT
      if self._model_type == "CLASSIFICATION":
        if int(feats['min']) != 0:
          print("Error: CLASSIFICATION - Label values are not correct")
          success_flag = False
          return success_flag
    
    return success_flag

  def run_initial(self, label_column, model_type='REGRESSION', model_complexity=1):
    """Run all modeling steps in pipeline and generate results"""
    self._label = label_column
    self._model_type = model_type
    self._model_complexity = model_complexity
    self.load_config_json()
    self._run = True                                  #Run flag

    #Prechecks
    if self.prechecks() == False:
      raise Exception("Error: Precheck failed for Training start")

    #Run HPT
    self.search_hpt()

    #Run Trainining and Evaluation
    self.start_train()

class TFAuto():
  def __init__(self, train_data_path, test_data_path, path_root='/tfx'):
    '''
    Initialize TFAuto engine
    train_data_path: Path where Training data is stored
    test_data_path: Path where Test / Eval data is stored
    path_root: Directory for running TFAuto( Directory should NOT exist )
    '''
    ##Define all constants
    self._tfx_root = os.path.join(os.getcwd(), path_root)
    self._pipeline_root = os.path.join(self._tfx_root, 'pipelines');      # Join ~/tfx/pipelines/
    self._metadata_db_root = os.path.join(self._tfx_root, 'metadata.db');    # Join ~/tfx/metadata.db
    self._metadata = os.path.join(self._tfx_root, 'metadata');    # Join ~/tfx/metadata
    self._log_root = os.path.join(self._tfx_root, 'logs');
    self._model_root = os.path.join(self._tfx_root, 'model');
    self._data_path = train_data_path

    self._input_fn_module_file = 'inputfn_trainer.py'
    self._constants_module_file = 'constants_trainer.py'
    self._model_trainer_module_file = 'model_trainer.py'

    #Instantiate other services
    self.tfautils = TFAutoUtils(data_path=train_data_path, path_root=path_root)
    self.tfadata = TFAutoData()
    self.tfamodel = TFAutoModel(self._tfx_root, train_data_path, test_data_path)

    #Create all required directories
    self.tfautils.create_directories()

    #Set interactive context
    # self.context = InteractiveContext(pipeline_root=self._tfx_root)

    #Output
    print("TF initialized...")
    print("All paths setup at {}".format(self._tfx_root))

  def generate_config_json(self):
    #Generate JSON for data modeling etc
    config_dict = {}
    config_json = os.path.join(self._tfx_root, 'config.json')
    config_dict['root_path'] = self._tfx_root
    config_dict['data_schema'] = self.tfadata.features_list
    config_dict['ignore_features'] = ['ADD_FEATURES_TO_IGNORE_FROM_MODEL']
    config_dict['file_headers'] = list(self.tfadata.file_headers)

    #Write JSON file
    with open(config_json, 'w') as fp:
      json.dump(config_dict, fp, indent = 4)

  def step_data_explore(self, viz=False):
    '''
    Method to automatically estimate schema of Data
    Viz: (False) Is data visualization required ?
    '''
    self.pipeline = self.tfadata.run_initial(self._data_path, self._tfx_root, self._metadata_db_root, self.tfautils, viz)
    self.generate_config_json()

  def step_model_build(self, label_column, model_type='REGRESSION', model_complexity=1):
    '''
    Method to automatically create models from data
    Parameters
    label_column: The feature to be used as Label
    model_type: Either of 'REGRESSION', 'CLASSIFICATION'
    model_complexity: 0 to 1 (0: Model without HPT, 1: Model with HPT) -> More will be added in future
    '''
    # #Run Modeling steps
    if self.tfadata._run == True:
      print("Success: Started AutoML Training")
      self.tfamodel.run_initial(label_column, model_type, model_complexity)
    else:
      print("Error: Please run Step 1 - step_data_explore")

    print("Success: Model Training complete. Exported to {}")

  def step_model_whatif(self):
    '''
    Run What-IF tool for trained model
    '''
    # #Run Modeling steps
    if self.tfadata._run == True:
      self.tfamodel.call_wit()
    else:
      print("Error: Please run Step 2 - step_model_build")
