import numpy as np 
import pandas as pd 
from tqdm import tqdm 
from arcface import ArcFace

if __name__ == "__main__":
    # specify path
    path = r'../biometric_system/data/reference/'

    # Create a dataframe 
    df_emb = pd.read_csv('./HPC_results/matrix_distance.csv', index_col=0)
    
    # Create indexes 
    idx = np.array(df_emb.columns[937:]) # Last one not included 
    # Erase problematic photo: '04876d62.png'
    erase = np.where(idx == '04876d62.png')
    idx = np.delete(idx, erase)
    # Create a copy and make it list for easer delation
    idx_2 = list(idx.copy())

    # create matches list 
    match_emb = []
    
    # Create Arcface object
    face_rec = ArcFace.ArcFace()

    # Specify how many photos we want to calculate 
    num_photos = 200

    for i in tqdm(range(num_photos)): # As it was idx[937:1137]
        
        file1 = idx[i]
        print(file1)
        for file2 in idx_2:

            # Get full paths of each photo
            path1 = path+file1
            path2 = path+file2
            
            # Calculate the embeding distance and cos similiratity for each pair of photos
            try:
                # create embeding objects 
                emb1 = face_rec.calc_emb(path1)
                emb2 = face_rec.calc_emb(path2)

                emb_dist = face_rec.get_distance_embeddings(emb1, emb2)
                
                # Store the embeding distance in the dataframe
                df_emb[file1][file2] = emb_dist
                df_emb[file2][file1] = emb_dist
            
            except FileNotFoundError: 
                pass

            # Check for matches
            if file1.split('d')[0] == file2.split('d')[0]:
                if emb_dist:
                    match_emb.append(emb_dist)


        # Drop first index
        idx_2.pop(0)

    # Save txt files 
    np.savetxt('./HPC_results/match_emb1.txt', match_emb, delimiter=',', fmt='%s')
    # Save dataframes
    df_emb.to_csv('./HPC_results/matrix_distance1.csv')
    print('Programe Finish, results Saved')
