from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import cv2

class ImageTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, size: int, interpolation: int = cv2.INTER_LINEAR, grayscale: bool = True):
        self.size = size
        self.interpolation = interpolation
        self.grayscale = grayscale
    def fit(self, X, y=None):
        return self # fit method does nothing in our case
    
    def transform(self, X: pd.DataFrame):
        
        def record_to_df(row):
            #print(row)
            
            # import image
            img = cv2.imread(row['directory'] + "/" + row['filename'])
            
            # if grayscale, convert
            if self.grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # scale our image
            img = cv2.resize(img, (self.size,self.size), self.interpolation)

            # flatten to 1d array
            img_flat = img.flatten()

            return pd.DataFrame(np.array([img_flat]))
        
        # build 'out' df from first record
        out = record_to_df(X.iloc[0])
        
        # for each record in X
        for i, row in X.iloc[1:].iterrows():


            record = record_to_df(row)

            # put array of image into column
            out = pd.concat([out, record], ignore_index=True)
            #print(f"iteration {i}: length {len(out)}")
            #print(out)
        return out