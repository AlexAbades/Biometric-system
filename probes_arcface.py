from arcface import ArcFace
import skimage.io
face_rec = ArcFace.ArcFace()


path = r'data\reference/'
photo1 = "02463d252.png"
photo2 = '04822d02.png'

photo3 = '04339d98.png'
photo4 = '04894d48.png'

emb1 = face_rec.calc_emb(path+photo1)


# 2 Objects 
face_rec = ArcFace.ArcFace()
emb1 = face_rec.calc_emb(path+photo1)
emb2 = face_rec.calc_emb(path+photo2)
d1 = face_rec.get_distance_embeddings(emb1, emb2)

face_rec = ArcFace.ArcFace()
emb3 = face_rec.calc_emb(path+photo3)
emb4 = face_rec.calc_emb(path+photo4)
d2 = face_rec.get_distance_embeddings(emb3, emb4)


# One object 
face_rec = ArcFace.ArcFace()
emb11 = face_rec.calc_emb(path+photo1)
emb22 = face_rec.calc_emb(path+photo2)
d11 = face_rec.get_distance_embeddings(emb11, emb22)

emb33 = face_rec.calc_emb(path+photo3)
emb44 = face_rec.calc_emb(path+photo4)
d22 = face_rec.get_distance_embeddings(emb33, emb44)

if d1 == d11 and d2 == d22:
    print('Same')