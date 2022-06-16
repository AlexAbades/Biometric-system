import pandas as pd 
import numpy as np 
import auxiliar as ax 
from numpy.random import randint as rd 
from os import listdir
from os.path import isfile, join


def selec_pair_idx(dataframe:pd.DataFrame, n:int):
    """
    Given a dataframe and n number of pairs, select n radom pairs from the dataframe

    ATRIBUTES
    --------
    dataframe:
    Pandas dataFrame 
    n: number of pairs

    RETURNS
    -------
    idx: tuple of pairs of indexes
    """
    rows = np.random.choice(dataframe.index.values, n*2, replace=False)
    return rows.reshape((n,2))



def createDataFrame(df_o, df_c, D, reference_path, probes_path):
    """
    
    ATRIBUTES 
    ---------
    df: Original DataFrame.
    df_c: Data Frame with the conditions on the parameters.
    D: Data Frame where we are going to append the values. Has to be an empty DataFrame with the columns specified
    reference_path: Path to the reference folder to get the photos 
    probes_path: Path to the Probes folder, photos we wish to compare 
    
    """
    # Get all the files names from the probes directory 
    onlyfiles = [f for f in listdir(probes_path) if isfile(join(probes_path, f))]
    
    # Select a number of Random points with same conditions
    N = selec_pair_idx(df_c, 3)
    
    # Counter to get rows from two by two 
    c = len(D.index.values)
    # Loop over all the randomly selected indexes 
    for idx1, idx2 in N:
        # Append the selected dataframe rows 
        D = D.append([df_o.iloc[idx1]], ignore_index = True)
        D = D.append([df_o.iloc[idx2]], ignore_index = True)
        # Obtain the names of the photos
        photo1, photo2 = D.loc[c:c+1, 'Name'].unique()
        # Create the complete path 
        path1 = reference_path+photo1
        path2 = reference_path+photo2
        # Calculate the embedding distance 
        s = ax.comparefaces(path1, path2)
        # Add the scores to the dataframe
        D.loc[c:c+1, 'Morph'] = int(c/2)
        # Set dummy variable for the groupby
        D.loc[c:c+1, 'Emb_dist'] = s
        c += 2

    # Get the photo identifier to search for matches in the probes folder
    D[['photo_id', 'extra']] = D['Name'].str.split('d', expand=True)
    
    # Get the files names of the selected photos on our DataFrame 
    for id in D.photo_id:
        ph = [file for file in onlyfiles if id in file]
        ph = ','.join(ph)
        D.loc[D.photo_id == id, 'Probes'] = ph

    return D