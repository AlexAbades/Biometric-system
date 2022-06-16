
import numpy as np 
import pandas as pd 
import auxiliar as ax 
from os import listdir
from os.path import isfile, join
from tqdm import tqdm 






if __name__ == "__main__":

    # specify path
    path = r'data\reference/'

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    # Create a dataframe 
    idx = onlyfiles
    idx_2 = idx.copy()

    df_emb = pd.DataFrame(index=idx, columns=idx)
    df_cos = df_emb.copy()

    match_emb = []
    notmatch_emb = []

    match_cos = []
    notmatch_cos = []

    for file1 in tqdm(idx):

        for file2 in idx_2:
            # Get full paths of each photo
            path1 = path+file1
            path2 = path+file2
            
            # Calculate the embeding distance and cos similiratity for each pair of photos
            emb_dist, cos_sim  = ax.comparefaces(path1, path2)

            # Store the embeding distance in the dataframe
            df_emb[file1][file2] = emb_dist
            df_emb[file2][file1] = emb_dist

            # Store the Cos similarity in the dataframe
            df_cos[file1][file2] = cos_sim
            df_cos[file2][file1] = cos_sim

            # Check for matches
            if file1.split('d')[0] == file2.split('d')[0]:
                if emb_dist:
                    match_emb.append(emb_dist)
                    match_cos.append(cos_sim)
            # Check for not macthes
            else:
                # It's not a match
                notmatch_emb.append(emb_dist)
                notmatch_cos.append(cos_sim)


        # Drop first index
        idx_2.pop(0)

    # Save txt files 
    np.savetxt('match_emb.txt', match_emb, delimiter=',')
    np.savetxt('match_cos.txt', match_cos, delimiter=',')
    np.savetxt('notmatch_emb.txt', notmatch_emb, delimiter=',')
    np.savetxt('notmatch_cos.txt', notmatch_cos, delimiter=',')
    # Save dataframes
    df_emb.to_csv('matrix_distance.csv', index=False)
    df_cos.to_csv('matrix_cos_similiaity.csv', index=False)