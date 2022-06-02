from arcface import ArcFace
import os
import skimage.io


face_rec = ArcFace.ArcFace()
emb1 = face_rec.calc_emb("data\est-image-gates.png")
emb2 = face_rec.calc_emb("data\est-image-gates2.png")

print(face_rec.get_distance_embeddings(emb1, emb2))