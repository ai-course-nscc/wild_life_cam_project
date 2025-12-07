To generate the dataset index csv:
 
 1) open build_dataset_index.py in an editor, and update the 'dir_list' entries to match your local filesystem.
 2) underneath 'dir_list', you'll also want to update 'model_path'.
 2) run collect_from_subfolder_3.py to generate a dataframe index of dataset images with labels and locations. The script will automatically export the dataframe to a CSV called 'dataset_index.csv"

To load each image during training:

 1) Import 'dataset_index.csv' as a dataframe using pd.read_csv()
 2) To get the file path of a given record:
    path = str(df['directory'][i]) + "/" + str(df['filename'][i])
        # where i is the index number of our record.

NOTES ABOUT NEW FEATURES:
- the 'day' column is a boolean feature. it will reflect the results of inference from the classifier (for all classes except 'Moose'.)
- the 'bndbox' feature is a tuple of 4 int values representing (xmin, xmax, ymin, ymax) of the bounding box. Most images should have bounding boxes, but approx. 1000 did not include bounding box data.