import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import *
from fastai.tabular import *
from fastai.tabular.all import *


def build_and_train_custom_learner(dataset_list,test_dataset_list,config,data_loader_function, image_generator_function,
n_epochs=50,generate_training_images=False,generate_test_images=False,rescale=False):
  """ Build and train a FastAI learner for battery SoC classification task using custom data_loader and image_generator functions"""
  learn= build_custom_learner(dataset_list,test_dataset_list,config,
  data_loader_function, image_generator_function,
  generate_training_images,generate_test_images)
  lr_obj = learn.lr_find()
  print(f"Valley: {lr_obj.valley:.2e}")
  learn.fine_tune(n_epochs,lr_obj.valley)
  return learn

def build_and_train_learner(config,n_epochs):
  """ Build and train a FastAI learner for battery SoC classification existing images"""
  learn= build_learner(config)
  lr_obj = learn.lr_find()
  print(f"Valley: {lr_obj.valley:.2e}")
  learn.fine_tune(n_epochs,lr_obj.valley)
  return learn

def build_custom_learner(measurement_list,test_measurement_list,config,
data_loader_function,generate_image_function,
generate_training_images=False,generate_test_images=False,rescale=True):
  """ Train a classifier model using a custom data loader and  image generator functions """
  #Train - Validation
  dataset,feature_col_names=data_loader_function(measurement_list,config["soc_list"],config['DATASETS_DIR'])

  #Test dataset
  test_dataset,feature_col_names=data_loader_function(test_measurement_list,config["soc_list"],config['DATASETS_DIR'])

  splitter = config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41) RandomSplitter(valid_pct=0.3, seed=41)

  #FastAI image pipeline
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]

  rePat=config['rePat'] 

  #Build FastAI DataBlock
  dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_items_func,
                   get_y=RegexLabeller(rePat),
                   splitter=splitter,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)

  dblock.summary(config["IMAGES_PATH"])
  dl= dblock.dataloaders(config["IMAGES_PATH"],bs=32)
  learn = cnn_learner(dl, resnet18, metrics=accuracy)
  return learn


def build_learner(config):
  """ Train a classifier model using a custom data loader and  image generator functions """

  splitter = config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41) RandomSplitter(valid_pct=0.3, seed=41)

  #FastAI image pipeline
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]

  rePat=config['rePat'] 

  #Build FastAI DataBlock
  dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_items_func,
                   get_y=RegexLabeller(rePat),
                   splitter=splitter,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)

  dblock.summary(config["IMAGES_PATH"])
  dl= dblock.dataloaders(config["IMAGES_PATH"],bs=32)
  learn = cnn_learner(dl, resnet18, metrics=accuracy)
  return learn

   
     
# first method that return second method
def get_items_func(imagePath,rePat=r'^.*_(\d+).png$'):
    '''get_items function for FastAI data block'''

    #IMAGES_PATH+"/"+ExperimentName+"/"+str(rowIndex)+"_"+str(socValue)+".png"
    #TODO: questo deve diventare un parametro configurabile. Ora lo legge dall'oggetto config globale
    #rePat=config['rePat'] #
    regExfilter = re.compile(rePat)
     
    return get_image_files_filtered(imagePath, regExfilter) 

#Get only image that match a specific regex filter in name
def get_image_files_filtered(imagePath, regExfilter):
  fnames = get_image_files(imagePath)
  return fnames.filter(lambda o:regExfilter.match(o.name))

def get_analog_value(prediction_output,learner):
  ind = np.argsort(prediction_output)
  value1 = prediction_output[ind[len(prediction_output)-1]]
  print(value1)
  value2 = prediction_output[ind[len(prediction_output)-2]]
  print(value2)

  classIndex1= ind[len(prediction_output)-1]
  print(classIndex1)

  classIndex2= ind[len(prediction_output)-2]
  print(classIndex2)

  avg_value= (int(learner.dls.vocab[classIndex2]) * value2) + (int(learner.dls.vocab[classIndex1]) * value1)
  return avg_value

def get_root_path():
  import os
  return os.path.dirname(os.path.abspath(__file__))

from fastai.vision.all import *
def score_model(saved_weights,data_loader):
  learn = cnn_learner(data_loader, resnet18, metrics=accuracy)
  learn = learn.load(saved_weights)
  interpretation = ClassificationInterpretation.from_learner(learn)
  interpretation.plot_confusion_matrix(normalize=False, figsize=(30, 20))
  cm=interpretation.confusion_matrix()
  print(cm)
  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  interpretation.plot_confusion_matrix(normalize=true, figsize=(30, 20))
  print(cm_norm)
  print("interpretation.most_confused()")
  interpretation.most_confused()
  print("interpretation.top_losses()")
  interpretation.plot_top_losses(10)
  print("interpretation.print_classification_report()")
  interpretation.print_classification_report()
  print("learn.show_results()")
  learn.show_results()
  print("learn.validate()")
  learn.validate()
  predictions,targets = learn.get_preds()
  model_accuracy= accuracy(predictions,targets)
  print(f"Model accuracy: {model_accuracy:.2f}")
  print("learn.get_preds()")
  learn.get_preds(with_decoded=True,with_loss=True)
  return model_accuracy

from datetime import datetime
def save_model_weights(learner,model_path,model_name,add_timestamp=True):
  dateTimeObj = datetime.now()
  dateTimeObj.timestamp()
  filename = model_name
  if(add_timestamp):
    filename = model_name+"_"+str(dateTimeObj.timestamp())+"_SAVED"
  else:
    filename = model_name+"_SAVED"
    
  learner.save(filename, with_opt=true)
  filename_pth= filename+".pth"
  print("saved filename: "+filename_pth)
  return filename
 
def export_model(learner,model_path):
  dateTimeObj = datetime.now()
  dateTimeObj.timestamp()
  filename = str(dateTimeObj.timestamp())+".pkl"
  print("model filename: " +filename)
  learner.export(fname=filename)
  path = Path()
  path.ls(file_exts='.pkl')

#Define Custom get_items function for FastAI Datablock
from fastai.vision.all import *

def build_data_loader(config):
  #Build FastAI DataBlock
  splitter=config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41)
  rePat=config['rePat'] 
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]
  dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                    get_items=get_items_func,
                    get_y=RegexLabeller(rePat),
                    splitter=splitter,
                    item_tfms=item_tfms,
                    batch_tfms=batch_tfms)

  dblock.summary(config['IMAGES_PATH'])
  #Build FastAI DataLoader
  dl= dblock.dataloaders(config['IMAGES_PATH'],bs=32)
  return dl

def inference_on_test_dataset(learner,config,reFilter):
  for testImgIdx in range(0,9,1): #TODO: calcolare range in base al numero di immagini di test
    print(testImgIdx)
    test_fnames = get_image_files_filtered(config['TEST_IMAGES_PATH'],reFilter)
    resultLabel, classIndex, values = learner.predict(test_fnames[testImgIdx])
    print("test case: "+str(test_fnames[testImgIdx]))
    print("classification predicte label:: "+resultLabel)
    print(values)

def rescale_dataset(df):
  ''' rescale dataset to 0-1 range and return new rescaled dataset and sklearn scaler object'''
  from sklearn import preprocessing
  scaler = preprocessing.MinMaxScaler()
  names = df.columns
  #print(names)
  scaled = scaler.fit_transform(df)
  scaled_df = pd.DataFrame(scaled, columns=names)
  return scaled_df, scaler

def plottingfunction_bar(dataX,dataY, ax=None, show=True):
    plt.ioff()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # do something with fig and ax here, e.g.
    line, = ax.plot(dataX,dataY,'o-',kind='bar')

    if show:
        plt.show()

    return fig, ax, line

def plottingfunction(dataX,dataY, ax=None, show=True):
    plt.ioff()
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    #plt.ioff()
    # do something with fig and ax here, e.g.
    line = ax.plot(dataX,dataY,'o-')

    if show:
        plt.show()

    return fig, ax, line

def show_pair_plot(data,var_columns,category_columns,title):
    import seaborn as sns;
    sns.pairplot(data,vars=var_columns,hue=category_columns,kind='hist')
    plt.show()

def show_coorelation_plot(data):
    #obtain the correlations of each features in dataset
    import seaborn as sns;
    import matplotlib.pyplot as plt
    corrmat = data.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
    plt.show()

def build_tabular_learner(dataset,splits,model_path,dep_var,cat_names,cont_names,bs=32):
    to = TabularPandas(dataset, procs=[Categorify, FillMissing,Normalize],cat_names = cat_names,cont_names = cont_names, y_names=dep_var,splits=splits)
    dls = to.dataloaders(bs)
    learn = tabular_learner(dls, layers=[300,200, 100, 50],metrics= rmse,path=model_path)
    return learn

