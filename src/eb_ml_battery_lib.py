from audioop import minmax
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.preprocessing import MinMaxScaler
from eb_ml_utils import get_items_func,rescale_dataset,plottingfunction

#DEFAULT OFFSET FOR DATA AUGMENTATION CSV FILE NAME GENERATION 
AUGMENTATION_OFFSET = 1000
DATA_AUGMENTATION_FACTOR_OFFSET = 100

def build_and_train_battery_learner_from_EIS(battery_list,test_battery_list,config,n_epochs=50,generate_training_images=False,generate_test_images=False,rescale=False):
  """ Build and train a FastAI learner for battery SoC classification task """
  learn= build_battery_soc_model_learner(battery_list,test_battery_list,config,generate_training_images,generate_test_images)
  lr_obj = learn.lr_find()
  print(f"Valley: {lr_obj.valley:.2e}")
  learn.fine_tune(n_epochs,lr_obj.valley)
  return learn

def build_and_train_battery_learner_from_EC(battery_list,test_battery_list,config,n_epochs=50,generate_training_images=False,generate_test_images=False,rescale=True):
  """ Build and train a FastAI learner for battery SoC classification task """
  learn= build_battery_soc_model_learner_ec(battery_list,test_battery_list,config,generate_training_images,generate_test_images,rescale)
  lr_obj = learn.lr_find()
  print(f"Valley: {lr_obj.valley:.2e}")
  learn.fine_tune(n_epochs,lr_obj.valley)
  return learn


def build_battery_soc_model_learner(battery_list,test_battery_list,config,
generate_training_images=False,generate_test_images=False):
  """ Train a battery SOC classifier model """
  #Train - Validation
  dataset,eis_col_names=load_soc_dataset(battery_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_training_images):
    generate_image_files_from_eis(dataset,eis_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

  #Test dataset
  test_dataset,eis_col_names=load_soc_dataset(test_battery_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_test_images):
    generate_image_files_from_eis(test_dataset,eis_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=1)
  
  splitter = config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41) RandomSplitter(valid_pct=0.3, seed=41)

  #FastAI image pipeline
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]
  rePat=config['rePat'] #r'^.*_(\d+).png$'

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

def build_battery_soc_model_learner_ec(battery_list,test_battery_list,config,
generate_training_images=False,generate_test_images=False,rescale=True):
  """ Train a battery SOC classifier model """
  rePat=config['rePat'] #r'^.*_(\d+).png$'

  #Train - Validation
  dataset,feature_col_names=load_soc_dataset_ec(battery_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_training_images):
    generate_image_from_ec(dataset,feature_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

  #Test dataset
  test_dataset,feature_col_names=load_soc_dataset_ec(test_battery_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_test_images):
    generate_image_from_ec(test_dataset,feature_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=1)
  
  splitter = config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41)

  #FastAI image pipeline
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]

  #Build FastAI DataBlock
  dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files_filtered,
                   get_y=RegexLabeller(rePat),
                   splitter=splitter,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)

  dblock.summary(config["IMAGES_PATH"])
  dl= dblock.dataloaders(config["IMAGES_PATH"],bs=32)
  learn = cnn_learner(dl, resnet18, metrics=accuracy)
  return learn    

def build_image_dataset_from_eis(battery_list,test_battery_list,config,generate_training_images=False,generate_test_images=False):
  #Train - Validation
  dataset,eis_col_names=load_soc_dataset(battery_list,config['soc_list'],config['DATASETS_DIR'])
  if(generate_training_images):
    generate_image_files_from_eis(dataset,eis_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

  #Test dataset
  test_dataset,eis_col_names=load_soc_dataset(test_battery_list,config['soc_list'],config['DATASETS_DIR'])
  if(generate_test_images):
    generate_image_files_from_eis(test_dataset,eis_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=1)

def load_soc_dataset(battery_list,soc_list, dataset_path,show_data=False):
  dataset = pd.DataFrame(columns=['SOC','BATTERY'])
  for battery_index, battery_value in enumerate(battery_list):
    if show_data:
      print("battery: "+str(battery_value))
    #Create a Pandas dataframe from CSV
    df_original= pd.read_csv(dataset_path+"/BATT_"+str(battery_value)+"_ALL_SOC.csv",names=soc_list, low_memory=False)
    #note: csv from matlab are in format 12-64i.
    #      'i" must be replaced with "j" into the CVS file
    df = df_original.apply(lambda col: col.apply(lambda val: val.replace('i','j')))
    #Parse complex number in format: 123-56j, 432+56j
    df = df.apply(lambda col: col.apply(lambda val: complex(val)))
    df_rows=df.transpose()

    eis_col_names= []
    for colIndex in range(0,df_rows.shape[1],1):
      eis_col_names.append("Z_f"+str(colIndex))
    
    if show_data:
      print(eis_col_names)
    
    df_rows.columns=eis_col_names

    #for rowIndex, row in enumerate(df_rows):
    df_rows['SOC']=soc_list
    df_rows['BATTERY']=battery_value
    dataset= dataset.append(df_rows)
    if show_data:
      print(df_rows)

  return dataset,eis_col_names

def load_soc_dataset_ec(battery_list,soc_list, dataset_path,show_data=False):
  dataset = pd.DataFrame(columns=['SOC','BATTERY'])
  for battery_index, battery_value in enumerate(battery_list):
    if show_data:
      print("battery: "+str(battery_value))
    #Create a Pandas dataframe from CSV
    df= pd.read_csv(dataset_path+"/BATT_"+str(battery_value)+"_EC.csv",names=soc_list, low_memory=False)
    df_rows=df.transpose()

    eis_col_names= []
    for colIndex in range(0,df_rows.shape[1],1):
      eis_col_names.append("ec_param"+str(colIndex))
    
    if show_data:
      print(eis_col_names)
    
    df_rows.columns=eis_col_names

    #for rowIndex, row in enumerate(df_rows):
    df_rows['SOC']=soc_list
    df_rows['BATTERY']=battery_value
    dataset= dataset.append(df_rows)
    if show_data:
      print(df_rows)

  return dataset,eis_col_names

def generate_image_files_from_eis(dataset,eis_col_names,IMAGES_PATH,experimentName,rescale=False,DATA_AUGMENTATION_FACTOR=1,NOISE_AMOUNT=1e-4):

  row_number=dataset.shape[0]
  print("dataset row number: "+str(row_number))
  print("start image file generation. IMAGE_PATH: "+IMAGES_PATH)

  df=dataset[eis_col_names]
  df_real= df.apply(lambda col: col.apply(lambda val: np.real(val)))
  df_img= df.apply(lambda col: col.apply(lambda val: np.imag(val)))
  #print(df_img)

  print("df_real shape: " + str(df_real.shape))
  print("df_img shape: " + str(df_img.shape))

  #Create a root folder for image dataset
  import os
  if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)
    
  if not os.path.exists(IMAGES_PATH+"/"+experimentName):
    os.mkdir(IMAGES_PATH+"/"+experimentName)

  for rowIndex in range(0,row_number,1):
      soc_label=dataset["SOC"].iloc[rowIndex]
      print("soc: "+str(soc_label))
      battery_value=dataset["BATTERY"].iloc[rowIndex]
      battery_name_str=str(battery_value)
      print("battery: "+battery_name_str)

      
      for augmentation_index in range(0,DATA_AUGMENTATION_FACTOR,1):
        print("augmentation_index: "+str(augmentation_index))
        df_real_copy = df_real.copy(deep=True)
        df_img_copy = df_img.copy(deep=True)
 
        if augmentation_index>0:
          # apply offset to image file name for file generated with data augmentation                     
          augmented_battery_value=AUGMENTATION_OFFSET+(DATA_AUGMENTATION_FACTOR_OFFSET*augmentation_index)+battery_value
          battery_name_str=str(augmented_battery_value)

          # AWG noise must be added before rescaling    
          df_real_copy= df_real_copy + np.random.normal(0, NOISE_AMOUNT, df_real.shape)
          df_img_copy = df_img_copy+ np.random.normal(0, NOISE_AMOUNT, df_img.shape)          

        # After adding the desidered amount of noise the values of EIS can be rescaled to 0-1 range      
        if rescale:
          df_real_copy,scaler= rescale_dataset(df_real_copy)
          df_img_copy, scaler= rescale_dataset(df_img_copy)

        #Get EIS data for a single SoC value from the dataset                    
        EIS_real=df_real_copy.iloc[rowIndex,:]
        EIS_img=df_img_copy.iloc[rowIndex,:] 

        img_file_name=IMAGES_PATH+"/"+experimentName+"/Batt_"+battery_name_str+"_"+str(soc_label)+".png"
        print(img_file_name)
        plotAndSave_complex(EIS_real,EIS_img,img_file_name)

def generate_image_from_ec(dataset,feature_col_names,IMAGES_PATH,experimentName,rescale=True,DATA_AUGMENTATION_FACTOR=1,NOISE_AMOUNT=1e-5,image_mode="plotAndSave"):

  row_number=dataset.shape[0]
  print("dataset row number: "+str(row_number))
  print("start image file generation. IMAGE_PATH: "+IMAGES_PATH)

  df=dataset[feature_col_names]

  #normalization
  if rescale:
    [df, scaler]= rescale_dataset(df)
  print("df shape: " + str(df.shape))

  #Create a root folder for image dataset
  import os
  if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)
    
  if not os.path.exists(IMAGES_PATH+"/"+experimentName):
    os.mkdir(IMAGES_PATH+"/"+experimentName)

  for rowIndex in range(0,row_number,1):
      soc_label=dataset["SOC"].iloc[rowIndex]
      print("soc: "+str(soc_label))
      battery_value=dataset["BATTERY"].iloc[rowIndex]
      print("battery: "+str(battery_value))
      img_file_name=IMAGES_PATH+"/"+experimentName+"/Batt_"+str(battery_value)+"_"+str(soc_label)+".png"
      print(img_file_name)
      if(image_mode=="plotAndSave"):
        plotAndSave(df.iloc[rowIndex,:],img_file_name)
      else:
        convertToImageAndSave(df.iloc[rowIndex,:],img_file_name)


      if DATA_AUGMENTATION_FACTOR>1:
        original=df.iloc[rowIndex,:]
        augmented_battery_value=1000+(battery_value*DATA_AUGMENTATION_FACTOR)
        for index in range(1,DATA_AUGMENTATION_FACTOR,1):
          battery_value=augmented_battery_value+index
          df_noise= np.random.rand(original.shape[0])*NOISE_AMOUNT
          df_with_noise= original + df_noise
          img_file_name=IMAGES_PATH+"/"+experimentName+"/Batt_"+str(battery_value)+"_"+str(soc_label)+".png"
          print(img_file_name)
          if(image_mode=="plotAndSave"):
            plotAndSave(df_with_noise,img_file_name)
          else:
            convertToImageAndSave(df_with_noise,img_file_name)

def plotAndSave(df,img_file_name):
    fig, ax, _ = plottingfunction(range(0,df.shape[0]),df,show=False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    fig.savefig(img_file_name)
    matplotlib.pyplot.close(fig)

def plotAndSave_complex(df_real,df_img,img_file_name):
  fig, ax, _ = plottingfunction(df_real,df_img,show=False)
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)
  fig.savefig(img_file_name)
  matplotlib.pyplot.close(fig)

def convertToImageAndSave(df,img_file_name):
  normalized = df/(df.max()/255.0)
  img_array = normalized.to_numpy().astype(np.uint8).T
  im = Image.fromarray(img_array)
  #plt.imshow(img_array, cmap='Greys')
  im.save(img_file_name)