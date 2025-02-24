import cv2
import numpy as np
import os
import glob
from math import atan2,pi
from scipy.spatial.transform import Rotation as Rot

#################################################################################################
#Partie 3
#################################################################################################

CHECKERBOARD = (7, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #c'est comme ça pas autrement
#vecteurs qui contiennent les tableaux de positions 3D et 2D images
objp = []
imgp = []

#Définition points 3D d'une image
pos = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
for i in range(49):
    pos[0][i][0] = 0.02+0.02 * (i % 7)
    pos[0][i][1] = 0.02+0.02 * (i // 7)
images = glob.glob("./images/*.jpg")
images.sort()
for fname in images:
	
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret: #Si on a bien les coordonnées 2D pour l'image
        objp.append(pos)
        corners_accurate = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        imgp.append(corners_accurate)
		# Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners_accurate, ret)
        
    cv2.imshow(f'img{fname}',img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

#merci cv2 qui fait tout
ret, camera_matrix, dist, rotation, translation = cv2.calibrateCamera(objp, imgp, gray.shape[::-1], None, None) 

if ret:
    print("Camera matrix : \n")
    print(camera_matrix)
    print("dist : \n")
    print(dist)
    R= [cv2.Rodrigues(r)[0] for r in rotation]
    for i in range(len(images)):

        print("R : \n")
        print(R[i])
        print("tvecs : \n")
        print(translation[i])
    ax= 2*atan2(320,camera_matrix[0][0])
    ay = 2*atan2(240,camera_matrix[1][1])
    print(ax*180/pi)
    print(ay*180/pi)

rvec =rotation[0]
tvec = translation[0]

# Récupérer tous les points 3D du premier damier détecté
points_3D = np.array(objp[0], dtype=np.float32)  # (1, 49, 3)

# Projeter les points 3D en 2D
points_2D, _ = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist)  # (49, 1, 2)

# Enlever les dimensions inutiles pour avoir (49, 2)
points_2D = points_2D.squeeze()

# Récupérer les points détectés correspondants (2D réels)
points_reels = np.array(imgp[0], dtype=np.float32).squeeze()  # (49, 2)

# Calculer l'erreur pour chaque point
erreurs = np.linalg.norm(points_2D - points_reels, axis=1)  # Erreur pour chaque point

# Calculer l'erreur moyenne
erreur_moyenne = np.mean(erreurs)

# Afficher les résultats
print(f"Erreur de reprojection moyenne : {erreur_moyenne:.2f} pixels")
print(f"Erreur maximale : {np.max(erreurs):.2f} pixels")
print(f"Erreur minimale : {np.min(erreurs):.2f} pixels")


#Créer T

def create_homogeneous_matrix(R, t):
    T = np.eye(4)  # Crée une matrice identité 4x4
    T[:3, :3] = R  # Insère la matrice de rotation
    T[:3, 3] = t.flatten()  # Insère le vecteur de translation
    return T

# Exemple avec plusieurs images
T_matrices = []  # Stocker toutes les matrices homogènes

for R1, t in zip(R, translation):  # R_list et t_list contiennent R et t pour chaque image
    T = create_homogeneous_matrix(R1, t)
    T_matrices.append(T)
    print(f"Matrice homogène T :\n{T}\n")

#################################################################################################
#Partie 4
#################################################################################################

#Définition des matrices nécessaires.

def construction_matrix():
    f = open("images/cart_poses.txt")
    ligne = f.readline()
    
    ligne_elts = ligne.split(',')
    bTm = np.zeros([4, 4])
    del ligne_elts[0]
    for i in range(len(ligne_elts)):
        ligne_elts[i] = ligne_elts[i].strip('(')
        ligne_elts[i] = ligne_elts[i].strip(')\n')
        bTm[i%4][i//4] = float(ligne_elts[i])

        
    ligne = f.readline()
    matrices_pos_robot = []
    while ligne:
        bTo = np.zeros([4, 4])
        oTb = np.zeros([4, 4])
        ligne_elts = ligne.split(',')
        del ligne_elts[0]
        for i in range(len(ligne_elts)):
            ligne_elts[i] = ligne_elts[i].strip('(')
            ligne_elts[i] = ligne_elts[i].strip(')\n')
            bTo[i%4][i//4] = float(ligne_elts[i])
        oTb = np.linalg.inv(bTo)
        matrices_pos_robot.append(oTb)
        ligne = f.readline()

    f.close()
    return bTm, matrices_pos_robot

bTm, matrices_pos_robot = construction_matrix()

matrices_glob = []
for i in range(len(matrices_pos_robot)):
    M = np.dot(np.dot(matrices_pos_robot[i], bTm), np.linalg.inv(T_matrices[i]))
    matrices_glob.append(M)

def sum_vectors(l1, l2):
    l = []
    for i in range(len(l1)):
        l.append(l1[i] + l2[i])
    return l

tx, ty, tz = 0, 0, 0
r = np.array([0, 0, 0])
for m in matrices_glob:
    tx += m[0][3]
    ty += m[1][3]
    tz += m[2][3]
    r = r + Rot.from_matrix(m[:3,:3]).as_rotvec()

n = len(matrices_glob)
tx /= n
ty /= n
tz /= n
r1 = r[0]/n
r2 = r[1]/n
r3 = r[2]/n

r = [r1, r2, r3]
R_mat = Rot.from_rotvec(r).as_matrix()
oTc = np.eye(4)
oTc[:3, :3] = R_mat
oTc[:3, 3] = [tx, ty, tz]
print("Matrice homogène de la caméra dans le repère de l'outil")
print(oTc) #:)

# vérification des résultats
matctm = np.linalg.inv(matrices_pos_robot[0]@bTm)@oTc
matmtc = np.linalg.inv(matctm)
tvec=np.array([matmtc[0][3],matmtc[1][3],matmtc[2][3]])

rvec, _ =cv2.Rodrigues(matmtc[:3,:3])


# Récupérer tous les points 3D du premier damier détecté
points_3D = np.array(objp[0], dtype=np.float32)  # (1, 49, 3)

# Projeter les points 3D en 2D
points_2D, _ = cv2.projectPoints(points_3D, rvec, tvec, camera_matrix, dist)  # (49, 1, 2)

# Enlever les dimensions inutiles pour avoir (49, 2)
points_2D = points_2D.squeeze()

# Récupérer les points détectés correspondants (2D réels)
points_reels = np.array(imgp[0], dtype=np.float32).squeeze()  # (49, 2)

# Calculer l'erreur pour chaque point
erreurs = np.linalg.norm(points_2D - points_reels, axis=1)  # Erreur pour chaque point

# Calculer l'erreur moyenne
erreur_moyenne = np.mean(erreurs)

# Afficher les résultats
print(f"Erreur de reprojection moyenne : {erreur_moyenne:.2f} pixels")
print(f"Erreur maximale : {np.max(erreurs):.2f} pixels")
print(f"Erreur minimale : {np.min(erreurs):.2f} pixels")

img=cv2.imread(images[0])
for points in points_2D:
    img = cv2.drawMarker(img, tuple(points.astype(int)), (0,0,255),markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
cv2.imshow('img',img)
cv2.waitKey()
#################################################################################################
#Partie 5
#################################################################################################

