import numpy as np
import pandas as pd
import os
from xml.etree import cElementTree
from dictify import dictify
import joblib
from ImageTransformer import ImageTransformer

# takes path to xml file, return a dict of relevant image metadata
def extract_pvoc_metadata(dir, filename):
    # concatenate our directory and filename
    xml_path = f"{dir}/{filename}"
    
    # read our xml in as a dictionary    
    print(xml_path)
    tree = cElementTree.parse(xml_path)
    root = tree.getroot()
    xml_dict = dictify(root)

    # extract info from our xml dict
    image = {}
    image['filename'] = xml_dict['annotation']['filename'][0]['_text']
    image['directory'] = dir
    try:
        image['label'] = xml_dict['annotation']['object'][0]['name'][0]['_text']
    except KeyError:
        image['label'] = "NoAnimal" 
    image['height'] = xml_dict['annotation']['size'][0]['height'][0]['_text']
    image['width'] = xml_dict['annotation']['size'][0]['width'][0]['_text']

    # bounding box values
    try:
        xmin = int(xml_dict['annotation']['object'][0]['bndbox'][0]['xmin'][0]['_text'])
        xmax = int(xml_dict['annotation']['object'][0]['bndbox'][0]['xmax'][0]['_text'])
        ymin = int(xml_dict['annotation']['object'][0]['bndbox'][0]['ymin'][0]['_text'])
        ymax = int(xml_dict['annotation']['object'][0]['bndbox'][0]['ymax'][0]['_text'])

        image['bndbox'] = (xmin, xmax, ymin, ymax)
    except KeyError:
        image['bndbox'] = None

    # return dictionary
    return image


# takes directory and dataframe, scans recursively and stores metadata for each dataset xml file
def collect_metadata_from_folder(dir, df, verbose=False):

    # get list of files in dir
    files = os.listdir(dir)

    # for each file inside dir
    for filename in files:
        # concatenate dir + filename
        filepath = f"{dir}/{filename}"
        if verbose: print(f"Current file: {filename}")
        
        # if file is folder
        if os.path.isdir(filepath):
            if verbose: print(f"is folder: {filename}")
            
            # df becomes result of recursive call
            df = collect_metadata_from_folder(filepath, df, verbose)

        #else if extension is '.xml'
        elif filename[-4:] == '.xml':
            if verbose: print(f"is xml {filename}")
            
            # extract metadata from xml file
            record_dict = extract_pvoc_metadata(dir, filename)
            if verbose: print(record_dict)

            # make dataframe from our metadata
            record_df = pd.DataFrame([record_dict]) # wrapped in list 

            # append new record to df
            df = pd.concat([df, record_df], axis=0, ignore_index=True)

    return df


## BEGIN MAIN PROGRAM

#
# list of directories to scan 
dir_list = [
    'C:/Users/user/Documents/Academic/Fall 2025/final project/datasets/trailcam-dataset',
    'C:/Users/user/Documents/Academic/Fall 2025/final project/datasets/Moose detection.v1i.voc',
    'C:/Users/user/Documents/Academic/Fall 2025/final project/datasets/Animal Detection.v31.voc',
    'C:/Users/user/Documents/Academic/Fall 2025/final project/datasets/deerDetector.v4i.voc',
    'C:/Users/user/Documents/Academic/Fall 2025/final project/datasets/Trail Camera Detection_v1.v5i.voc'
]

model_path = 'C:/Users/user/Documents/Academic/Fall 2025/ML I/final_project/daynight_classifier.pkl'

# create our dataset index (dataframe)
df = pd.DataFrame()

# run recursive metadata collection to populate our index df
for dir in dir_list:
    df = collect_metadata_from_folder(dir, df, verbose=True)


## CLEANING UP DATA
# make a mask where day = True for any moose records labeled as day images
mask_day = df['label'].str.contains(r"^day(.*)moose(.*)", regex=True)

# add the mask as a column to our dataframe
# NOTE that this will leave all non-moose records classified as night images by default
# we need to make sure that our classifier will correct this afterwards.
df['day'] = mask_day

# combine moose subclasses into label "Moose"
df['label'] = df['label'].str.replace(r"(.*)moose(.*)", "Moose", regex=True)

# combine deer subclasses into label "Deer"
df['label'] = df['label'].str.replace(r"(.*)deer(.*)", "Deer", regex=True, case=False)
df['label'] = df['label'].str.replace(r"(.*)doe(.*)", "Deer", regex=True)
df['label'] = df['label'].str.replace(r"(.*)buck(.*)", "Deer", regex=True)

# label person as "NoAnimal"
df['label'] = df['label'].str.replace("person", "NoAnimal")

# rename badger to Badger
df['label'] = df['label'].str.replace("badger", "Badger")

# rename fox to Fox
df['label'] = df['label'].str.replace("fox", "Fox")

# rename raccoon to Raccoon
df['label'] = df['label'].str.replace("raccoon", "Raccoon")

## DAY/NIGHT CLASSIFICATION
# make a copy of our dataframe without the moose class
df_daynight = df[df['label'] != 'Moose']

# import model
clf = joblib.load(model_path)

# run inference on df_daynight
print("\nRunning inference for day/night tagging...")
result = clf.predict(df_daynight)
print("Inference complete.")
df_daynight = df_daynight.drop(columns=['day'])

# add inference results to df_daynight
df_daynight['day'] = result

# recombine df_daynight with the moose portion
df = pd.concat([df[df['label'] == 'Moose'], df_daynight], axis=0, ignore_index=True)


##DATA SUMMARY
# print data summary
print("\n\n---Data Summary---")
print("\n" + str(df.head()))
print("\n" + str(df.info()))
print("\n" + str(df.describe()))

# print unique classes
print("\n\n---Unique Classes---:")
print("\n" + str(df['label'].value_counts()))

# export our index to a csv file
df.to_csv('dataset_index.csv', index=False)