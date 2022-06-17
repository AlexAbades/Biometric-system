from arcface import ArcFace
import numpy as np


def cosine_similarity(x, y):
    
    return np.dot(x, y) / (np.sqrt(np.dot(x, x)) * np.sqrt(np.dot(y, y)))


def comparefaces(path_im1:str, path_im2:str)->int:
    """
    Using arcface we compare two images and compute a score of similaruty. 
    We are using embedings, we are comparing its Euclidean Distace. If they are perfectly matched, the distance will be 0
    ATRIBUTES
    ---------
    im1: path to image 1 
    im2: path to image 2 

    RETURNS
    -------
    S: Score of similarity between 2 images
    cos_sim: Cosine similarity between 2 images 
    """
    
    face_rec = ArcFace.ArcFace()
    emb1 = face_rec.calc_emb(path_im1)
    emb2 = face_rec.calc_emb(path_im2)

    s = face_rec.get_distance_embeddings(emb1, emb2)
    cos_sim = cosine_similarity(emb1, emb2)
    
    return s, cos_sim


    