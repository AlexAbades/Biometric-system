from arcface import ArcFace
import skimage.io
face_rec = ArcFace.ArcFace()
path = r'data\reference/'
photo1 = "02463d252.png"
emb1 = face_rec.calc_emb(path+photo1)
# print(emb1)
emb2 = face_rec.calc_emb("./data/reference/04202d392.png")
face_rec.get_distance_embeddings(emb1, emb2)


# I = skimage.io.imread("./data/reference/02463d252.png")