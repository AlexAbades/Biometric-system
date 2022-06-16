import numpy as np 
import pandas as pd 
import matplotlib as plt 
from arcface import ArcFace
import os
import skimage.io
import os 
import auxiliar as ax
import aux_dataframe as axdf



if __name__ == "__main__":


    # Load Data Frame
    df = pd.read_csv(r'C:\Users\G531\Documents\8 - Github\Biometric-system\data\demo_attributes_frgc.csv')
    # Erase the two las columns
    df = df[df.columns[:-2]]
    # Specify reference path 
    reference_path = r'data\reference/'
    # Specify probe path 
    probe_path = mypath = r'data\probe/'
    # Create a Dataframe to store Values 
    idx_df = ['Emb_dist', 'Morph', 'Name', 'Sex', 'Race', 'Skin Type', 'Age']
    D1 = pd.DataFrame(columns=idx_df)

    # Specify 1st conditions 
    sex = 'male'
    race = 'caucasian'
    skin = '2'
    age = 'young'
    # Select DataFrame based on the conditions based on condition 
    df_tmp = df[(df.Sex == 'male') & (df.Age == 'young') & (df.Race == 'caucasian') & (df['Skin Type'] == '2')]
    # Create a new data frame with the morphing options
    D1 = axdf.createDataFrame(df, df_tmp, D1, reference_path, probe_path)

    # Specify 2nd conditions 
    sex = 'female'
    age = 'middle'
    race = 'asian'
    skin = '4'
    # Select based on condition 
    df_tmp1 = df[(df.Sex == sex) & (df.Age == age) & (df.Race == race) & (df['Skin Type'] == skin)]
    # Append newest selections to the existing DataFrame
    D1 = axdf.createDataFrame(df, df_tmp1, D1, reference_path, probe_path)

    # Save DataFrame to a csv file 
    D1.to_csv('selected_photos.csv', index=False)

