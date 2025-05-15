import numpy as np
import sympy as sm
import scipy as sp
import math
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from scipy.optimize import root_scalar
import pandas as pd
import seaborn as sns
import time
import multiprocessing
import os
import warnings
warnings.filterwarnings('ignore')

# ======================================================================================================================
# ============================================== FONCTIONS_REQUISES ====================================================
# ======================================================================================================================

# Fonction pour définir une rotation selon l'axe x
def rot_x(theta):
    t = np.radians(theta)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)],[0, np.sin(t), np.cos(t)]])
        
# Fonction pour définir une rotation selon l'axe y
def rot_y(theta):
    t = np.radians(theta)
    return np.array([[np.cos(t), 0, np.sin(t)],[0, 1, 0],[-np.sin(t), 0, np.cos(t)]])
        
# Fonction pour définir une rotation selon l'axe z
def rot_z(theta):
    t = np.radians(theta)
    return np.array([[np.cos(t), -np.sin(t), 0],[np.sin(t), np.cos(t), 0],[0, 0, 1]])
    
# Fonction pour calculer les angles de rotation pour arrivar à un vecteur (0,0,1) à partir de n'importe quel vecteur
def Determiner_angles_rotation(vec_cible,vec_original = np.array([0, 0, 1])):
    vec_cible = vec_cible / np.linalg.norm(vec_cible)
    axe = np.cross(vec_original, vec_cible)
    angle = np.arccos(np.dot(vec_original, vec_cible))
    if np.allclose(axe, 0):
        if np.dot(vec_original, vec_cible) > 0:
            rot = R.identity()
        else:
            rot = R.from_rotvec(np.pi * np.array([1, 0, 0]))
    else:
        axe = axe / np.linalg.norm(axe)
        rot = R.from_rotvec(angle * axe)
    angles = rot.as_euler('zyx', degrees=True)
    return angles

# Fonction pour définir dans une liste les paramètres importants du plan en question
def Definition_plan(id_plan,Point_C,vec_L_xy,vec_angle_rotation):
    #l_xx = vec_L_xy[0] \ \ \ \ L_yy = vec_L_xy[1]
    #Point_C = np.array([x_c, y_c, z_c])
    #vec_angle_rotation = np.array([angle_rotation_x, angle_rotation_y, angle_rotation_z])
    u = np.array([1, 0, 0]) * vec_L_xy[0] / 2
    v = np.array([0, 1, 0]) * vec_L_xy[1] / 2
    n= np.array([0, 0, 1])
    R_total = rot_x(vec_angle_rotation[2]) @ rot_y(vec_angle_rotation[1]) @ rot_z(vec_angle_rotation[0])
    u_rot = R_total @ u
    v_rot = R_total @ v
    n_rot = R_total @ n
    #0 : id_plan / / 1 : Point_C / / 2 : vec_L_xy / / 3 : vec_angle_rotation / / 4 : R_total / / 5 : u_rot / / 6 : v_rot / / 7 : n_rot  
    return [id_plan,Point_C,vec_L_xy,vec_angle_rotation,R_total,u_rot,v_rot,n_rot]

# Fonction pour définir dans une liste les paramètres importants du plan de la parabole en question
def Definition_plan_parabole(id_plan,Point_C,vec_L_xy,vec_angle_rotation,vec_normal,points):
    #l_xx = vec_L_xy[0] \ \ \ \ L_yy = vec_L_xy[1]
    #Point_C = np.array([x_c, y_c, z_c])
    #vec_angle_rotation = np.array([angle_rotation_x, angle_rotation_y, angle_rotation_z])
    Am, Bm, Cm, Dm = points
    u = np.array([1, 0, 0]) * vec_L_xy[0] / 2
    v = np.array([0, 1, 0]) * vec_L_xy[1] / 2
    n= np.array([0, 0, 1])
    R_total = rot_x(vec_angle_rotation[2]) @ rot_y(vec_angle_rotation[1]) @ rot_z(vec_angle_rotation[0])
    n_rot = vec_normal
    u_rot = (Bm - Am)/2
    v_rot = (Cm - Bm)/2 
    #0 : id_plan / / 1 : Point_C / / 2 : vec_L_xy / / 3 : vec_angle_rotation / / 4 : R_total / / 5 : u_rot / / 6 : v_rot / / 7 : n_rot  
    return [id_plan,Point_C,vec_L_xy,vec_angle_rotation,R_total,u_rot,v_rot,n_rot]

# Fonction pour définir dans une liste les paramètres importants du plan avec un trou en question
def Definition_plan_special(id_plan,Point_C,vec_L_xy,vec_angle_rotation,region_creuse):
    #vec_L_xy = np.array([L_xx,L_yy])
    #Point_C = np.array([x_c, y_c, z_c])
    #vec_angle_rotation = np.array([angle_rotation_x, angle_rotation_y, angle_rotation_z])
    #region_creuse = np.array([x_h,y_h,L_xx,L_yy])
    u = np.array([1, 0, 0]) * vec_L_xy[0] / 2
    v = np.array([0, 1, 0]) * vec_L_xy[1] / 2
    n= np.array([0, 0, 1])
    R_total = rot_x(vec_angle_rotation[2]) @ rot_y(vec_angle_rotation[1]) @ rot_z(vec_angle_rotation[0])
    u_rot = R_total @ u
    v_rot = R_total @ v
    n_rot = R_total @ n
    #0 : id_plan / / 1 : Point_C / / 2 : vec_L_xy / / 3 : vec_angle_rotation / / 4 : R_total / / 5 : u_rot / / 6 : v_rot / / 7 : n_rot / / 8 : region_creuse  
    return [id_plan,Point_C,vec_L_xy,vec_angle_rotation,R_total,u_rot,v_rot,n_rot,region_creuse]

# Fonction pour définir dans une liste les paramètres importants du plan qui refracte en question
def Definition_plan_refraction(id_plan,Point_C,vec_L_xy,vec_angle_rotation,n_materiel):
    #l_xx = vec_L_xy[0] \ \ \ \ L_yy = vec_L_xy[1]
    #Point_C = np.array([x_c, y_c, z_c])
    #vec_angle_rotation = np.array([angle_rotation_x, angle_rotation_y, angle_rotation_z])
    u = np.array([1, 0, 0]) * vec_L_xy[0] / 2
    v = np.array([0, 1, 0]) * vec_L_xy[1] / 2
    n= np.array([0, 0, 1])
    R_total = rot_x(vec_angle_rotation[2]) @ rot_y(vec_angle_rotation[1]) @ rot_z(vec_angle_rotation[0])
    u_rot = R_total @ u
    v_rot = R_total @ v
    n_rot = R_total @ n
    #0 : id_plan / / 1 : Point_C / / 2 : vec_L_xy / / 3 : vec_angle_rotation / / 4 : R_total / / 5 : u_rot / / 6 : v_rot / / 7 : n_rot / / 8 : n_material    
    return [id_plan,Point_C,vec_L_xy,vec_angle_rotation,R_total,u_rot,v_rot,n_rot,n_materiel]

# Fonction permettant de définir les paramètres d'une ligne dans l'espace 3D
def Definition_ligne_droite(id_recta,vec_O,vec_d):
    #vec_O = np.array([x_c, y_c, z_c])
    #vec_d = np.array([d_x, d_y, d_z])
    #0 : id_recta / / 1 : vec_O / / 2 : vec_d
    return [id_recta,vec_O,vec_d]


# ======================================================================================================================
# ============================================== FONCTIONS_INTERSECTION ================================================
# ======================================================================================================================

# Fonction permettant de déterminer l'intersection et la réflexion d'un rayon avec un plan normal
def Determiner_intersection_plan(Parametros_linea,Parametros_plano):
    Num = np.dot((Parametros_plano[1]-Parametros_linea[1]),Parametros_plano[7]) 
    Den = np.dot(Parametros_linea[2],Parametros_plano[7])
    t = Num/Den # [(Point_C - Point_O) . vec_n] / [vec_d . vec_n]
    if (t<0.0):
        Intersection = False
        return Intersection
    Point_Int = Parametros_linea[2]*t + Parametros_linea[1] # R(t) = vec_d*t + vec_O
    # Validation si cette intersection se produit dans le plan
    dP = Point_Int - Parametros_plano[1]
    valeur_u = np.dot(dP, Parametros_plano[5]/ np.linalg.norm(Parametros_plano[5])) # vec_dP . u_rot
    valeur_v = np.dot(dP, Parametros_plano[6]/ np.linalg.norm(Parametros_plano[6])) # vec_dP . v_rot
    Dans_x = (np.abs(valeur_u)) <= Parametros_plano[2][0]/2
    Dans_y = (np.abs(valeur_v)) <= Parametros_plano[2][1]/2
    if Dans_x and Dans_y:
        Intersection = True
        nouveau_vec_d = Parametros_linea[2] - 2*np.dot(Parametros_linea[2],Parametros_plano[7])*Parametros_plano[7]
        return Intersection,t,Point_Int,nouveau_vec_d
    else:
        Intersection = False
        return Intersection
    f
# Fonction permettant de déterminer l'intersection et la réflexion d'un rayon avec un plan comportant un trou
def Determiner_intersection_plan_special(Parametros_linea,Parametros_plano):
    Num = np.dot((Parametros_plano[1]-Parametros_linea[1]),Parametros_plano[7]) 
    Den = np.dot(Parametros_linea[2],Parametros_plano[7])
    t = Num/Den # [(Point_C - Point_O) . vec_n] / [vec_d . vec_n]
    if (t<0.0):
        Intersection = False
        return Intersection
    Point_Int = Parametros_linea[2]*t + Parametros_linea[1] # R(t) = vec_d*t + vec_O
    # Validation si cette intersection se produit dans le plan
    dP = Point_Int - Parametros_plano[1]
    valeur_u = np.dot(dP, Parametros_plano[5]/ np.linalg.norm(Parametros_plano[5])) # vec_dP . u_rot
    valeur_v = np.dot(dP, Parametros_plano[6]/ np.linalg.norm(Parametros_plano[6])) # vec_dP . v_rot
    Dans_rectangle_x = (Parametros_plano[8][0] - Parametros_plano[8][2]/2) <= valeur_u <= (Parametros_plano[8][0] + Parametros_plano[8][2]/2)
    Dans_rectangle_y = (Parametros_plano[8][1] - Parametros_plano[8][3]/2) <= valeur_v <= (Parametros_plano[8][1] + Parametros_plano[8][3]/2)
    Rectangle_cond = Dans_rectangle_x and Dans_rectangle_y
    Dans_x = ((np.abs(valeur_u)) <= Parametros_plano[2][0]/2)
    Dans_y = ((np.abs(valeur_v)) <= Parametros_plano[2][1]/2)
    if Dans_x and Dans_y and not Rectangle_cond:
        Intersection = True
        nouveau_vec_d = Parametros_linea[2] - 2*np.dot(Parametros_linea[2],Parametros_plano[7])*Parametros_plano[7]
        return Intersection,t,Point_Int,nouveau_vec_d
    else:
        Intersection = False
        return Intersection
    
# Fonction permettant de déterminer la réfraction de la lumière le long d'un plan en utilisant la loi de Snell
def Determiner_refraction_plan(Parametros_linea,Parametros_plano):
    n_entrada= 1.0
    Num = np.dot((Parametros_plano[1]-Parametros_linea[1]),Parametros_plano[7]) 
    Den = np.dot(Parametros_linea[2],Parametros_plano[7])
    t = Num/Den # [(Point_C - Point_O) . vec_n] / [vec_d . vec_n]
    if (t<0.0):
        Intersection = False
        return Intersection
    Point_Int = Parametros_linea[2]*t + Parametros_linea[1] # R(t) = vec_d*t + vec_O
    # Validation si cette intersection se produit dans le plan
    dP = Point_Int - Parametros_plano[1]
    valeur_u = np.dot(dP, Parametros_plano[5]/ np.linalg.norm(Parametros_plano[5])) # vec_dP . u_rot
    valeur_v = np.dot(dP, Parametros_plano[6]/ np.linalg.norm(Parametros_plano[6])) # vec_dP . v_rot
    Dans_x = (np.abs(valeur_u)) <= Parametros_plano[2][0]/2
    Dans_y = (np.abs(valeur_v)) <= Parametros_plano[2][1]/2
    if Dans_x and Dans_y:
        Intersection = True
        n_rot = Parametros_plano[7]/np.linalg.norm(Parametros_plano[7])
        vec_d = Parametros_linea[2]/np.linalg.norm(Parametros_linea[2])
        n_eta = n_entrada/Parametros_plano[8]
        if np.dot(n_rot,vec_d)>0:
            n_rot = -1*n_rot
        cos_theta_i = 1*np.dot(n_rot,vec_d)
        Discriminant = 1-(n_eta**2)*(1-cos_theta_i**2)
        if Discriminant < 0:
            Intersection = False
            return Intersection
        nouveau_vec_d = n_eta*(vec_d - cos_theta_i*n_rot)-n_rot*np.sqrt(Discriminant)
        return Intersection,t,Point_Int,nouveau_vec_d
    else:
        Intersection = False
        return Intersection
        
# ======================================================================================================================
# =============================================== FONCTIONS_CREATION ===================================================
# ======================================================================================================================

# Fonction permettant de calculer la définition des rayons lumineux initiaux incidents sur le réflecteur de Scheffler
def Creation_rayon_initial(num_rayos,x_h,y_h,r,vec_d,z_0,deviation,angle_rad,foyer_y):
    rotation = R.from_rotvec(angle_rad * np.array([0, 1, 0]))
    rayons = r * np.sqrt(np.random.rand(num_rayos)) 
    angles = 2 * np.pi * np.random.rand(num_rayos)
    x = x_h + rayons * np.cos(angles)
    y = y_h + rayons * np.sin(angles)
    Matrice_rayons = np.zeros((num_rayos,3), dtype=object)
    for i in range(num_rayos):
        vec_O = np.array([x[i],y[i],z_0])
        dx = np.random.uniform(-deviation, deviation)
        dy = np.random.uniform(-deviation, deviation)
        vec_d_cor = np.array([dx, dy, 0]) +  vec_d
        vec_O_tr = vec_O - np.array([0, 0, foyer_y])
        vec_O_rot = rotation.apply(vec_O_tr) + np.array([0, 0, foyer_y])
        vec_d_rot = rotation.apply(vec_d_cor)
        Matrice_rayons[i, 0] = i+1
        Matrice_rayons[i, 1] = vec_O_rot
        Matrice_rayons[i, 2] = vec_d_rot
    return Matrice_rayons

# Fonction qui permet de créer tous les plans présents dans la simulation
def Creation_plans(n_normales,n_speciaux,n_refraction,Matrice_Point_C,Matrice_L_xy,Matrice_vec_normal,Matrice_Point_C_special,Matrice_L_xy_special,Matrice_vec_normal_special,Matrice_region_creuse_special,Matrice_Point_C_refraction,Matrice_L_xy_refraction,Matrice_vec_normal_refraction,Matrice_n_refraction):
    Matrice_plans_normaux = np.zeros((n_normales,8), dtype=object)
    Matrice_plans_speciaux = np.zeros((n_speciaux,9), dtype=object)
    Matrice_plans_refraction = np.zeros((n_refraction,9), dtype=object)
    for i in range(n_normales):
        Angles_zyx = Determiner_angles_rotation(Matrice_vec_normal[i,:])
        Parametres_plan_normal = Definition_plan(i+1,Matrice_Point_C[i,:],Matrice_L_xy[i,:],Angles_zyx)
        Matrice_plans_normaux[i,0] = Parametres_plan_normal[0]
        Matrice_plans_normaux[i,1] = Parametres_plan_normal[1]
        Matrice_plans_normaux[i,2] = Parametres_plan_normal[2]
        Matrice_plans_normaux[i,3] = Parametres_plan_normal[3]
        Matrice_plans_normaux[i,4] = Parametres_plan_normal[4]
        Matrice_plans_normaux[i,5] = Parametres_plan_normal[5]
        Matrice_plans_normaux[i,6] = Parametres_plan_normal[6]
        Matrice_plans_normaux[i,7] = Parametres_plan_normal[7]
    for i in range(n_speciaux):
        Angles_zyx = Determiner_angles_rotation(Matrice_vec_normal_special[i,:])
        Parametres_plan_special = Definition_plan_special(i+1,Matrice_Point_C_special[i,:],Matrice_L_xy_special[i,:],Angles_zyx,Matrice_region_creuse_special[i,:])
        Matrice_plans_speciaux[i,0] = Parametres_plan_special[0]
        Matrice_plans_speciaux[i,1] = Parametres_plan_special[1]
        Matrice_plans_speciaux[i,2] = Parametres_plan_special[2]
        Matrice_plans_speciaux[i,3] = Parametres_plan_special[3]
        Matrice_plans_speciaux[i,4] = Parametres_plan_special[4]
        Matrice_plans_speciaux[i,5] = Parametres_plan_special[5]
        Matrice_plans_speciaux[i,6] = Parametres_plan_special[6]
        Matrice_plans_speciaux[i,7] = Parametres_plan_special[7]
        Matrice_plans_speciaux[i,8] = Parametres_plan_special[8]
    for i in range(n_refraction):
        Angles_zyx = Determiner_angles_rotation(Matrice_vec_normal_refraction[i,:])
        Parametres_plan_refraction = Definition_plan_refraction(i+1,Matrice_Point_C_refraction[i,:],Matrice_L_xy_refraction[i,:],Angles_zyx, Matrice_n_refraction[i])
        Matrice_plans_refraction[i,0] = Parametres_plan_refraction[0]
        Matrice_plans_refraction[i,1] = Parametres_plan_refraction[1]
        Matrice_plans_refraction[i,2] = Parametres_plan_refraction[2]
        Matrice_plans_refraction[i,3] = Parametres_plan_refraction[3]
        Matrice_plans_refraction[i,4] = Parametres_plan_refraction[4]
        Matrice_plans_refraction[i,5] = Parametres_plan_refraction[5]
        Matrice_plans_refraction[i,6] = Parametres_plan_refraction[6]
        Matrice_plans_refraction[i,7] = Parametres_plan_refraction[7]
        Matrice_plans_refraction[i,8] = Parametres_plan_refraction[8]
    return Matrice_plans_normaux, Matrice_plans_speciaux, Matrice_plans_refraction

# Fonction permettant de créer les plans qui font partie du réflecteur de Scheffler
def Creation_plans_parabole(n_normales_reflecteur,Matrice_Point_C_parabole, Matrice_L_xy_parabole, Matrice_vec_normal_parabole,M_tile):
    Matrice_plans_normaux_parabole = np.zeros((n_normales_reflecteur,8), dtype=object)
    for i in range(n_normales_reflecteur):
        Angles_zyx = Determiner_angles_rotation(Matrice_vec_normal_parabole[i,:])
        Parametres_plan_normal_parabole = Definition_plan_parabole(i+1,Matrice_Point_C_parabole[i,:],Matrice_L_xy_parabole[i,:],Angles_zyx,Matrice_vec_normal_parabole[i,:], M_tile[i])
        Matrice_plans_normaux_parabole[i,0] = Parametres_plan_normal_parabole[0]
        Matrice_plans_normaux_parabole[i,1] = Parametres_plan_normal_parabole[1]
        Matrice_plans_normaux_parabole[i,2] = Parametres_plan_normal_parabole[2]
        Matrice_plans_normaux_parabole[i,3] = Parametres_plan_normal_parabole[3]
        Matrice_plans_normaux_parabole[i,4] = Parametres_plan_normal_parabole[4]
        Matrice_plans_normaux_parabole[i,5] = Parametres_plan_normal_parabole[5]
        Matrice_plans_normaux_parabole[i,6] = Parametres_plan_normal_parabole[6]
        Matrice_plans_normaux_parabole[i,7] = Parametres_plan_normal_parabole[7]
    return Matrice_plans_normaux_parabole

# Fonction qui permet de définir tous les paramètres nécessaires à la simulation RayTracing
def Creer_simulation(num_rayos,parametres_creation_simulation,matrices_creation_simulation):
    # ----- Définition des plans -----
    Centre_spatial,Longueur_four,Largeur_four,Hauteur_four,foyer_y,A_approx,n_crossbars_real,gap_crossbars,n_year,lon_miroir,z_0_rayons,Longueur_creuse,Largeur_creuse,n_refraction_materiel,deviation_rayons,alpha_year,foyer_y = parametres_creation_simulation
    Matrice_Point_C_ad,Matrice_L_xy_ad,Matrice_vec_normal_ad,Matrice_Point_C_special_ad,Matrice_L_xy_special_ad,Matrice_vec_normal_special_ad,Matrice_region_creuse_special_ad,Matrice_Point_C_refraction_ad,Matrice_L_xy_refraction_ad,Matrice_vec_normal_refraction_ad,Matrice_n_refraction_ad = matrices_creation_simulation
    
    # °1 : Côté droit / °2 : Côté gauche / °3 ​​: Côté arrière / °4 : Plafond / °5 : Sol
    Matrice_Point_C = np.array([[Centre_spatial[0],Centre_spatial[1]-Largeur_four/2,Centre_spatial[2]],
                            [Centre_spatial[0],Centre_spatial[1]+Largeur_four/2,Centre_spatial[2]],
                            [Centre_spatial[0]-Longueur_four/2,Centre_spatial[1],Centre_spatial[2]],
                            [Centre_spatial[0],Centre_spatial[1],Centre_spatial[2]+Hauteur_four/2],
                            [Centre_spatial[0],Centre_spatial[1],Centre_spatial[2]-Hauteur_four/2]])
    Matrice_L_xy = np.array([[Longueur_four,Hauteur_four],
                            [Longueur_four,Hauteur_four],
                            [Hauteur_four,Largeur_four],
                            [Longueur_four,Largeur_four],
                            [Longueur_four,Largeur_four]])
    Matrice_vec_normal = np.array([[0,1,0],
                                [0,-1,0],
                                [1,0,0],
                                [0,0,1],
                                [0,0,-1]])
    # °1 : Face avant (devant)
    Matrice_Point_C_special = np.array([[Centre_spatial[0]+Longueur_four/2,Centre_spatial[1],Centre_spatial[2]]])
    Matrice_L_xy_special = np.array([[Hauteur_four,Largeur_four]])
    Matrice_vec_normal_special = np.array([[-1,0,0]])
    Matrice_region_creuse_special = np.array([[0,0,Largeur_creuse,Longueur_creuse]])
    # °1 : Face avant (avant) [Diffuseur]
    Matrice_Point_C_refraction = np.array([[Centre_spatial[0]+Longueur_four/2,Centre_spatial[1],Centre_spatial[2]]])
    Matrice_L_xy_refraction = np.array([[Largeur_creuse,Longueur_creuse]])
    Matrice_vec_normal_refraction = np.array([[1,0,0]])
    Matrice_n_refraction = np.array([n_refraction_materiel])
    
    # --- Union des matrices ---
    if Matrice_Point_C_ad.size > 0:
        Matrice_Point_C = np.vstack((Matrice_Point_C, Matrice_Point_C_ad))
    if Matrice_L_xy_ad.size > 0:
        Matrice_L_xy = np.vstack((Matrice_L_xy, Matrice_L_xy_ad))
    if Matrice_vec_normal_ad.size > 0:
        Matrice_vec_normal = np.vstack((Matrice_vec_normal, Matrice_vec_normal_ad))
    if Matrice_Point_C_special_ad.size > 0:
        Matrice_Point_C_special = np.vstack((Matrice_Point_C_special, Matrice_Point_C_special_ad))
    if Matrice_L_xy_special_ad.size > 0:
        Matrice_L_xy_special = np.vstack((Matrice_L_xy_special, Matrice_L_xy_special_ad))
    if Matrice_vec_normal_special_ad.size > 0:
        Matrice_vec_normal_special = np.vstack((Matrice_vec_normal_special, Matrice_vec_normal_special_ad))
    if Matrice_region_creuse_special_ad.size > 0:
        Matrice_region_creuse_special = np.vstack((Matrice_region_creuse_special, Matrice_region_creuse_special_ad))
    if Matrice_Point_C_refraction_ad.size > 0:
        Matrice_Point_C_refraction = np.vstack((Matrice_Point_C_refraction, Matrice_Point_C_refraction_ad))
    if Matrice_L_xy_refraction_ad.size > 0:
        Matrice_L_xy_refraction = np.vstack((Matrice_L_xy_refraction, Matrice_L_xy_refraction_ad))
    if Matrice_vec_normal_refraction_ad.size > 0:
        Matrice_vec_normal_refraction = np.vstack((Matrice_vec_normal_refraction, Matrice_vec_normal_refraction_ad))
    if Matrice_n_refraction_ad.size > 0:
        Matrice_n_refraction = np.vstack((Matrice_n_refraction, Matrice_n_refraction_ad))

    n_normales = len(Matrice_vec_normal)
    n_speciaux = len(Matrice_vec_normal_special)
    n_refraction = len(Matrice_vec_normal_special)
    Matrice_plans_normaux, Matrice_plans_speciaux, Matrice_plans_refraction = Creation_plans(n_normales,n_speciaux,n_refraction,Matrice_Point_C,Matrice_L_xy,Matrice_vec_normal,Matrice_Point_C_special,Matrice_L_xy_special,Matrice_vec_normal_special,Matrice_region_creuse_special,Matrice_Point_C_refraction,Matrice_L_xy_refraction,Matrice_vec_normal_refraction,Matrice_n_refraction)

    # ----- Définition de la parabole -----
    Matrice_Point_C_parabole, Matrice_L_xy_parabole, Matrice_vec_normal_parabole, Points_graph_parabole, Parametres_ligne_saison = Calculer_parabole_réfléchissante_adéquate(foyer_y, A_approx, n_crossbars_real, gap_crossbars, n_year, lon_miroir)
    Matrice_Point_C_parabole = np.array(Matrice_Point_C_parabole)
    Matrice_L_xy_parabole = np.array(Matrice_L_xy_parabole)
    Matrice_vec_normal_parabole = np.array(Matrice_vec_normal_parabole)
    n_normales_reflecteur = len(Matrice_vec_normal_parabole)
    Matrice_plans_normaux_parabole = Creation_plans_parabole(n_normales_reflecteur,Matrice_Point_C_parabole, Matrice_L_xy_parabole, Matrice_vec_normal_parabole, Points_graph_parabole)
    
    # ----- Définition des rayons initiaux -----
    y_h_par = 0
    x_h_par = (Parametres_ligne_saison[4][0]+Parametres_ligne_saison[3][0])/2
    r_par = (Parametres_ligne_saison[4][0]-Parametres_ligne_saison[3][0])/2
    vec_d = np.array([0,0,-1])
    Matrice_rayons = Creation_rayon_initial(num_rayos,x_h_par,y_h_par,r_par,vec_d,z_0_rayons,deviation_rayons,alpha_year,foyer_y)
    
    return Matrice_plans_normaux, Matrice_plans_speciaux,Matrice_plans_refraction,Matrice_plans_normaux_parabole,Matrice_rayons,n_normales,n_speciaux,n_refraction,n_normales_reflecteur, Points_graph_parabole


# ======================================================================================================================
# ============================================== FONCTIONS_EVOLUTION ===================================================
# ======================================================================================================================

# Fonction permettant de calculer l'évolution de chaque collision pour chaque rayon analysé
def Evolution_rayons(Matrice_plans_normaux,Matrice_plans_speciaux,Matrice_plans_refraction,Matrice_rayons,Matrice_plans_normaux_parabole,n_normales,n_speciaux,n_refraction,n_normales_reflecteur,num_rayos,num_max_rebonds,t_tol=0.00000005,t_iter_ref=100000):
    # Initialisation
    trajectoires_rayons = np.zeros((num_rayos,num_max_rebonds), dtype=object)
    id_rayons_actifs = np.arange(num_rayos)
    nouveaux_actifs = []
    # Données initiales
    ii=0
    for i in range(num_rayos):
        trajectoires_rayons[i,ii] = Matrice_rayons[i,1]
        
    # Collision avec la parabole
    ii = 1
    for k in id_rayons_actifs:
        # Itérer sur les plans normaux
        id_plan = 0
        t_iter = t_iter_ref
        Existe_Intersection = False
        for j in range(n_normales_reflecteur):
            Resultat = Determiner_intersection_plan(Matrice_rayons[k,:], Matrice_plans_normaux_parabole[j,:])
            if Resultat != False:
                Intersection,t,Point_Int,nouveau_vec_d = Resultat
                if (t<t_iter) and (t>t_tol):
                    t_iter = t
                    id_plan = j
                    Existe_Intersection = True
        if Existe_Intersection:            
            Intersection,t,Point_Int,nouveau_vec_d = Determiner_intersection_plan(Matrice_rayons[k,:], Matrice_plans_normaux_parabole[id_plan,:])        
            Matrice_rayons[k,1] = Point_Int
            Matrice_rayons[k,2] = nouveau_vec_d
            trajectoires_rayons[k,ii] = Matrice_rayons[k,1]
            nouveaux_actifs.append(k)
    id_rayons_actifs = nouveaux_actifs    
    
    # Itérations pour calculer les collisions après le paraboloïde
    ii = 2
    while (ii < num_max_rebonds):
        nouveaux_actifs = []  
        # Itérer sur les plans normaux
        for k in id_rayons_actifs:
            id_plan = 0
            t_iter = t_iter_ref
            Existe_Intersection = False
            Special = False
            Refraction = False
            for j in range(n_normales):
                Resultat = Determiner_intersection_plan(Matrice_rayons[k,:], Matrice_plans_normaux[j,:])
                if Resultat != False:
                    Intersection,t,Point_Int,nouveau_vec_d = Resultat
                    if (t<t_iter) and (t>t_tol):
                        t_iter = t
                        id_plan = j
                        Special = False
                        Refraction = False
                        Existe_Intersection = True
            for j in range(n_speciaux):
                Resultat = Determiner_intersection_plan_special(Matrice_rayons[k,:], Matrice_plans_speciaux[j,:])
                if Resultat != False:
                    Intersection,t,Point_Int,nouveau_vec_d = Resultat
                    if (t<t_iter) and (t>t_tol):
                        t_iter = t
                        id_plan = j
                        Special = True
                        Refraction = False
                        Existe_Intersection = True
            for j in range(n_refraction):
                Resultat = Determiner_refraction_plan(Matrice_rayons[k,:], Matrice_plans_refraction[j,:])
                if Resultat != False:
                    Intersection,t,Point_Int,nouveau_vec_d = Resultat
                    if (t<t_iter) and (t>t_tol):
                        t_iter = t
                        id_plan = j
                        Special = False
                        Refraction = True
                        Existe_Intersection = True
            if Existe_Intersection:
                if Special == False:
                    if Refraction == True:
                        Intersection,t,Point_Int,nouveau_vec_d = Determiner_refraction_plan(Matrice_rayons[k,:], Matrice_plans_refraction[id_plan,:])
                    else:
                        Intersection,t,Point_Int,nouveau_vec_d = Determiner_intersection_plan(Matrice_rayons[k,:], Matrice_plans_normaux[id_plan,:])
                elif Special == True:
                    Intersection,t,Point_Int,nouveau_vec_d = Determiner_intersection_plan_special(Matrice_rayons[k,:], Matrice_plans_speciaux[id_plan,:])
                Matrice_rayons[k,1] = Point_Int
                Matrice_rayons[k,2] = nouveau_vec_d
                trajectoires_rayons[k,ii] = Matrice_rayons[k,1]
                if id_plan != 4: #(5-1)
                    nouveaux_actifs.append(k)
        id_rayons_actifs = nouveaux_actifs
        ii = ii+1
    return trajectoires_rayons, Matrice_rayons

# Fonction permettant de déterminer lequel de tous les rayons simulés a fini par entrer en collision avec la plaque du four
def Determiner_intersection_plaque(trajectoires_rayons,num_rayos,num_max_rebonds,z_reference,tol_z = 0.001):
    points_xy = []
    for i in range(num_rayos):
        k = num_max_rebonds-1
        Array_final = isinstance(trajectoires_rayons[i,k], np.ndarray)
        if not Array_final:
            k = 0
            while (k < (num_max_rebonds-1)) and isinstance(trajectoires_rayons[i,k+1], np.ndarray):
                k = k+1
        if (abs(trajectoires_rayons[i,k][2]-z_reference)<tol_z):
            points_xy.append(trajectoires_rayons[i,k][:2])
    return points_xy


# ======================================================================================================================
# ============================================= FONCTIONS_GRAPHIQUES ===================================================
# ======================================================================================================================

def Graphique_general(Matrice_plans_normaux,Matrice_plans_speciaux,Matrice_plans_normaux_parabole,n_normales,n_speciaux,n_normales_reflecteur,num_max_rebonds,num_rayons_graph,trajectoires_rayons,angulo_elev,angulo_azim,Matrice_rayons,Points_graph_parabole,x_lim,y_lim,z_lim):
    fig = plt.figure(figsize=(16,13))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Simulation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=angulo_elev, azim=angulo_azim)
    Tracer_paraboloide(ax,Matrice_plans_normaux_parabole,n_normales_reflecteur,Points_graph_parabole)
    Tracer_plans_normales(ax,Matrice_plans_normaux,n_normales)
    Tracer_plans_speciaux(ax,Matrice_plans_speciaux,n_speciaux)
    Tracer_trajectoires_rayons(ax,trajectoires_rayons,num_rayons_graph,num_max_rebonds)
    #for i in range(num_rayons_graph):
    #    tt = 10
    #    point_depart = Matrice_rayons[i,1]
    #    point_final = point_depart + tt*Matrice_rayons[i,2]
    #    Tracer_ligne_droite(ax, point_depart, point_final)
    # Foco de la parabola (Foyer)
    #ax.scatter(Matriz_parabola[1][3], Matriz_parabola[1][4], Matriz_parabola[1][2] + 1/(4*Matriz_parabola[1][0]), color='red')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    plt.tight_layout()
    plt.show()
    
def Graphique_points_collision(Intersections_plaque,num_max_rayons_KDE,Parametros_plano):
    points = np.array(Intersections_plaque)
    num_rayons_graph_col = len(Intersections_plaque)
    
    if num_rayons_graph_col > num_max_rayons_KDE:
        num_rayons_graph_col = num_max_rayons_KDE
    
    Point_C = Parametros_plano[1][:2]
    vec_u_ref = Parametros_plano[5]/ np.linalg.norm(Parametros_plano[5])
    vec_v_ref = Parametros_plano[6]/ np.linalg.norm(Parametros_plano[6])
    vec_u_ref = vec_u_ref[:2]
    vec_v_ref = vec_v_ref[:2]
    u_vals = np.zeros(num_rayons_graph_col)
    v_vals = np.zeros(num_rayons_graph_col)
    
    for i in range(num_rayons_graph_col):
        dP = points[i,:] - Point_C
        u_vals[i] = np.dot(dP, vec_u_ref)
        v_vals[i] = np.dot(dP, vec_v_ref)
        #plt.scatter(valeur_u, valeur_v, color='blue', s=30)

    plt.figure(figsize=(8, 6))
    sns.kdeplot(x=u_vals, y=v_vals, fill=True, cmap="viridis", bw_adjust=0.5, levels=100, thresh=0)
    #plt.scatter(u_vals, v_vals, s=5, color='white')
    plt.title("KDE de los puntos")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.xlim(-Parametros_plano[2][0]/2, Parametros_plano[2][0]/2)
    plt.ylim(-Parametros_plano[2][1]/2, Parametros_plano[2][1]/2)
    plt.show()

    
def Tracer_paraboloide(ax,Matrice_plans_normaux_parabole,n_normales_reflecteur,Points_graph_parabole):
    for i in range(n_normales_reflecteur):
        A, B, C, D = Points_graph_parabole[i] 
        X = np.array([[A[0], B[0]], [D[0], C[0]]])
        Y = np.array([[A[1], B[1]], [D[1], C[1]]])
        Z = np.array([[A[2], B[2]], [D[2], C[2]]])
        ax.plot_surface(X, Y, Z, alpha=0.5, color='cyan')
        #ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='k')
        #ax.plot([B[0], C[0]], [B[1], C[1]], [B[2], C[2]], color='k')
        #ax.plot([C[0], D[0]], [C[1], D[1]], [C[2], D[2]], color='k')
        #ax.plot([D[0], A[0]], [D[1], A[1]], [D[2], A[2]], color='k')
        #Tracer_ligne_droite(ax,Matrice_plans_normaux_parabole[i,1],Matrice_plans_normaux_parabole[i,1]+Matrice_plans_normaux_parabole[i,5])
        #Tracer_ligne_droite(ax,Matrice_plans_normaux_parabole[i,1],Matrice_plans_normaux_parabole[i,1]+Matrice_plans_normaux_parabole[i,6])
    
def Tracer_plans_normales(ax,Matrice_plans_normaux,n_normales):
    nx = 2
    ny = 2
    for i in range(n_normales):
        x = np.linspace(-Matrice_plans_normaux[i,2][0]/2, Matrice_plans_normaux[i,2][0]/2, nx)
        y = np.linspace(-Matrice_plans_normaux[i,2][1]/2, Matrice_plans_normaux[i,2][1]/2, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        points_tournes = Matrice_plans_normaux[i,4] @ points
        # Recolocar puntos en el centro (x_c, y_c, z_c)
        X_rot = points_tournes[0,:].reshape(X.shape) + Matrice_plans_normaux[i,1][0]
        Y_rot = points_tournes[1,:].reshape(Y.shape) + Matrice_plans_normaux[i,1][1]
        Z_rot = points_tournes[2,:].reshape(Z.shape) + Matrice_plans_normaux[i,1][2]
        ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, color='cyan')

def Tracer_plans_speciaux(ax,Matrice_plans_speciaux,n_speciaux):
    nx = 200
    ny = 200
    for i in range(n_speciaux):
        x = np.linspace(-Matrice_plans_speciaux[i,2][0]/2, Matrice_plans_speciaux[i,2][0]/2, nx)
        y = np.linspace(-Matrice_plans_speciaux[i,2][1]/2, Matrice_plans_speciaux[i,2][1]/2, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, np.nan)
        X_c, Y_c = Matrice_plans_speciaux[i,8][0], Matrice_plans_speciaux[i,8][1]
        largeur, hauteur = Matrice_plans_speciaux[i,8][2], Matrice_plans_speciaux[i,8][3]
        mask = ~(
            (X >= X_c - largeur/2) & (X <= X_c + largeur/2) &
            (Y >= Y_c - hauteur/2) & (Y <= Y_c + hauteur/2)
        )
        Z[mask] = 0
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
        points_tournes = Matrice_plans_speciaux[i, 4] @ points
        # Recolocar puntos en el centro (x_c, y_c, z_c)
        X_rot = points_tournes[0,:].reshape(X.shape) + Matrice_plans_speciaux[i,1][0]
        Y_rot = points_tournes[1,:].reshape(Y.shape) + Matrice_plans_speciaux[i,1][1]
        Z_rot = points_tournes[2,:].reshape(Z.shape) + Matrice_plans_speciaux[i,1][2]
        ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, color='cyan')
        #Tracer_ligne_droite(ax,Matrice_plans_speciaux[i,1],Matrice_plans_speciaux[i,1]+Matrice_plans_speciaux[i,5])
        #Tracer_ligne_droite(ax,Matrice_plans_speciaux[i,1],Matrice_plans_speciaux[i,1]+Matrice_plans_speciaux[i,6])

def Tracer_ligne_droite(ax, point_depart, point_final, color='k', epaisseur=1):
    x = [point_depart[0], point_final[0]]
    y = [point_depart[1], point_final[1]]
    z = [point_depart[2], point_final[2]]
    ax.plot(x, y, z, color=color, linewidth=epaisseur)
    
def Tracer_trajectoires_rayons(ax,trajectoires_rayons,num_rayons_graph,num_max_rebonds):
    for i in range(num_rayons_graph):
        k = 0
        while (k < (num_max_rebonds-1)) and isinstance(trajectoires_rayons[i,k+1], np.ndarray):
            Tracer_ligne_droite(ax, trajectoires_rayons[i,k], trajectoires_rayons[i,k+1])
            k = k+1
            

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ============================================== FONCTIONS_REFLECTEUR ==================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================

# Fonction pour intégrer la fonction sqrt(1 + (x/2f)^2)
def integrate_line(a, b, foyer_y, n_points):
    if a > b:
        a, b = b, a
    h = (b - a) / n_points
    total = 0
    for i in range(n_points):
        x = a + (i + 0.5) * h
        total += np.sqrt(1 + (x/(2*foyer_y))**2)
    return total * h

# Fonction pour calculer la valeur spatiale dans la parabole du réflecteur de Scheffler
def Parabole_y_n(x,m_p,C_p):
    P = m_p*x**2 + C_p #m
    return P

# Fonction permettant de calculer la valeur spatiale sur la ligne du plan de coupe du paraboloïde
def Ligne_droite(x,m_g,C_g):
    G = m_g*x + C_g #m
    return G

# Calcul de la valeur de l'angle alpha comme dans l'article de Reddy
def Calculer_pente_Reddy_1(foyer_y, R_a):
    alpha_Reddy_1 = 45 - 0.036*(R_a/foyer_y)-1.75*(R_a/foyer_y)**2
    return alpha_Reddy_1

# Calcul de la valeur de l'angle alpha comme dans l'article de Reddy
def Calculer_pente_Reddy_2(foyer_y, x_int):
    alpha_Reddy_2 = 40.1 + 4.055*(x_int/foyer_y)+0.8396*(x_int/foyer_y)**2
    return alpha_Reddy_2

# Fonction permettant de calculer la valeur de l'angle alpha pour une estimation initiale
def Calculer_pente_droite(foyer_y, R_a, Type_Evaluation):
    if Type_Evaluation == 0:
        # Doit itérer d'un angle de 42° à 44,9°
        alpha_l_min = 42 #deg
        alpha_l_max = 44.9 #deg
        scale_factor = np.pi/180 #deg -> rad
        # Informations pour le calcul itératif
        error_tol = 0.00001 #m
        alpha_l = alpha_l_min
        error = 1000
        n_points = 40
        step_min = 1e-9
        step_max = 0.05
        while (alpha_l<alpha_l_max) and (error>error_tol):
            x_E1 = 2*foyer_y*np.tan(alpha_l*scale_factor) - R_a
            x_E2 = 2*foyer_y*np.tan(alpha_l*scale_factor) + R_a
            error = abs(integrate_line(x_E1, 2*foyer_y, foyer_y, n_points) - integrate_line(2*foyer_y, x_E2, foyer_y, n_points))
            alpha_step = max(step_min, min(step_max, error * 0.1))
            alpha_l += alpha_step
        if (alpha_l>alpha_l_max):
            print("WARNING: Une valeur de pente adéquate n'a pas été obtenue")
    if Type_Evaluation == 1:
        alpha_l = Calculer_pente_Reddy_1(foyer_y, R_a)    
    return alpha_l

# Fonction permettant de calculer les paramètres de l'ellipse formée dans le plan de coupe
def Calcul_parametres_ellipse(Parametres_parabole, Parametres_ligne):
    a = np.sqrt((Parametres_ligne[0]/(2*Parametres_parabole[0]))**2 - (Parametres_parabole[1]-Parametres_ligne[1])/Parametres_parabole[0])
    ratio_angle = np.arctan(Parametres_ligne[1])
    ratio_ellipse = np.cos(ratio_angle)
    b = a/ratio_ellipse
    h = ((b-a)/(b+a))**2
    perimetre = np.pi*(a+b)*(1 + (3*h)/(10 + np.sqrt(4-3*h)))
    Parametres_ellipse = np.array([a, b, ratio_ellipse, perimetre])
    return Parametres_ellipse

# Fonction pour calculer les traverses que la structure aura autour du réflecteur
def Distribution_crossbars(Parametres_ellipse, n_crossbars_real, gap_crossbars, Parametres_ligne, Parametres_parabole, x_E1, x_E2):
    # Calcul de la distribution de la position spatiale des barres réflectrices
    x_lim = gap_crossbars*(n_crossbars_real-1)/2.
    if x_lim > Parametres_ellipse[1]:
        x_lim = Parametres_ellipse[1]
        print("WARNING: L'espacement entre les barres est incorrect")
        x_n = np.linspace(-x_lim, x_lim, n_crossbars_real+2) 
    else:
        x_n = np.linspace(-x_lim, x_lim, n_crossbars_real)    
        x_n = np.concatenate(([-Parametres_ellipse[1]], x_n, [Parametres_ellipse[1]]))
    y_n = Parametres_ellipse[2]*np.sqrt((Parametres_ellipse[1]**2 - x_n**2))
    # Calcul de la ligne imaginaire qui simule la barre transversale présente au milieu de l'ellipse
    perpendicular_angle = np.arctan(Parametres_ligne[0]) - (np.pi/2) #Pente de la ligne perpendiculaire au plan de coupe
    m_q_n = np.tan(perpendicular_angle)
    x_q_n_middle = (x_E2 + x_E1)/2
    y_q_n_middle = Ligne_droite(x_q_n_middle, Parametres_ligne[0], Parametres_ligne[1]) 
    C_q_n_middle = y_q_n_middle - x_q_n_middle*m_q_n
    ratio_ellipse_qn = np.cos(abs(perpendicular_angle))
    coefficients_quad = np.array([Parametres_parabole[0], (-m_q_n), (Parametres_parabole[1] - C_q_n_middle)]) # Intersection: y = m_p x^2 + C_p = m_g x + C_g 
    roots = np.roots(coefficients_quad)
    real_roots = roots[np.isreal(roots)].real
    x_intersect = real_roots[real_roots>=0][0]
    y_intersect = Parabole_y_n(x_intersect, Parametres_parabole[0], Parametres_parabole[1]) 
    # Calcul des autres crossbars
    C_q_vec = np.zeros(n_crossbars_real)
    a_qn_vec = np.zeros(n_crossbars_real)
    b_qn_vec = np.zeros(n_crossbars_real)
    x_n_i_middle = 0
    i_middle = int((n_crossbars_real+1)/2)
    for i in range(n_crossbars_real):
        C_q_vec[i] = (x_n[i+1] - x_n_i_middle)/np.cos(abs(perpendicular_angle)) + C_q_n_middle
        a_qn_vec[i] = np.sqrt( ((m_q_n)/(2*Parametres_parabole[0]))**2 - (Parametres_parabole[1] - C_q_vec[i])/Parametres_parabole[0] )
        b_qn_vec[i] = a_qn_vec[i]/ratio_ellipse_qn
    return x_n, y_n, m_q_n, C_q_vec, a_qn_vec, b_qn_vec, perpendicular_angle, x_intersect, y_intersect

# Processus de calcul des informations géométriques pour dimensionner les traverses du réflecteur
def Crossbars_information(n_crossbars_real, y_n, a_qn_vec, perpendicular_angle):
    delta_n = np.zeros(n_crossbars_real)
    R_n = np.zeros(n_crossbars_real)
    beta_n = np.zeros(n_crossbars_real)
    arc_n = np.zeros(n_crossbars_real)
    for i in range(n_crossbars_real):
        delta_n[i] = (a_qn_vec[i] - np.sqrt( a_qn_vec[i]**2 - y_n[i+1]**2 ) )/np.cos(perpendicular_angle)
        R_n[i] = (delta_n[i]**2 + y_n[i+1]**2)/(2*delta_n[i])
        beta_n[i] = np.arcsin(y_n[i+1]/R_n[i])
        arc_n[i] = 2*R_n[i]*(beta_n[i])    
    return delta_n, R_n, beta_n, arc_n

# Fonction permettant de déterminer l'angle d'incidence de la lumière solaire tout au long de l'année
def alpha_solar(n_year):
    principal_angle = (n_year-1)*(2*np.pi)/(365)
    alpha_rad = 0.006918 - 0.399912*np.cos(principal_angle) + 0.070257*np.sin(principal_angle) - 0.006758*np.cos(2*principal_angle) + 0.000907*np.sin(2*principal_angle) -0.002769*np.cos(3*principal_angle) + 0.00148*np.sin(3*principal_angle) 
    alpha_deg = (180/np.pi)*(alpha_rad)
    return alpha_rad

# Fonction pour calculer la valeur z spatiale du paraboloïde calculé    
def Paraboloide_rev(x,y,Parametres_parabole):
    f = Parametres_parabole[0]*(x**2+y**2) + Parametres_parabole[1]
    return f

# Fonction permettant de déterminer la dérivée partielle de la fonction paraboloïde par rapport à l'axe des x
def dev_Paraboloide_rev_x(x,y,Parametres_parabole):
    dz_dx = 2*Parametres_parabole[0]*x
    return dz_dx

# Fonction permettant de déterminer la dérivée partielle de la fonction paraboloïde par rapport à l'axe des y
def dev_Paraboloide_rev_y(x,y,Parametres_parabole):
    dz_dy = 2*Parametres_parabole[0]*y
    return dz_dy

# Fonction permettant de déterminer l'aire de la surface dans l'espace 3D
def valeur_surface(Parametres_parabole, a_cercle, r_cercle, n_x, n_y):
    delta_x = (2*r_cercle)/(n_x-1)
    delta_y = (2*r_cercle)/(n_y-1)
    dA = delta_x*delta_y
    sum_int = 0
    for j in range(n_x-1):
        for i in range(n_y-1):
            x_j = a_cercle - r_cercle + (j)*delta_x/2 + delta_x/2
            y_i = r_cercle - (i)*delta_y/2 - delta_y/2
            if (x_j - a_cercle)**2 + y_i**2 <= r_cercle**2:
                sum_int = sum_int + dA*np.sqrt(1 + dev_Paraboloide_rev_x(x_j,y_i,Parametres_parabole)**2 + dev_Paraboloide_rev_y(x_j,y_i,Parametres_parabole)**2)
    return sum_int

# Fonction permettant de déterminer une rotation d'un point bidimensionnel avec un angle alpha (règle de la main droite) et autour du point focal
def determine_rotation(x_parabola, y_parabola, alpha, f_point_x, f_point_y):
    x_parabola_tr = x_parabola - f_point_x
    y_parabola_tr = y_parabola - f_point_y
    Matrix = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
    coordinates = np.array([x_parabola_tr, y_parabola_tr])
    new_coordinates = np.dot(coordinates, Matrix)
    x_parabola_new = new_coordinates[0] + f_point_x
    y_parabola_new = new_coordinates[1] + f_point_y
    return x_parabola_new, y_parabola_new

# Determinar los parámetros de la parabola en el equinnocio
def Determiner_Parabole_equinnoxe(foyer_y, A_approx, n_crossbars_real, gap_crossbars):
    # Calculer les paramètres du réflecteur de Scheffleur
    y_p = foyer_y
    x_p = 2*foyer_y 
    dev_P = 1
    m_p = dev_P/(2*x_p)
    C_p = 0
        # Calculer le foyer de la parabole
    f_point_x_equinnoxe = 0.0 # x_h = 0
    f_point_y_equinnoxe = 1./(4*m_p)
    f_spatial_point_equinnoxe = np.array([f_point_x_equinnoxe, f_point_y_equinnoxe])
    Point_P_equinnoxe = np.array([x_p, y_p])
    Parametres_parabole_equinnoxe = np.array([m_p, C_p, f_spatial_point_equinnoxe, Point_P_equinnoxe], dtype=object)
    
    # Processus itératif pour calculer les valeurs des paramètres de la ligne du plan de coupe du paraboloïde
    R_a = np.sqrt(A_approx/np.pi)
    scale_factor = np.pi/180 #deg -> rad
    alpha_deg = Calculer_pente_droite(foyer_y, R_a, 0) #0 : Itératif, 1: Reddy's formula
    x_E1 = 2*foyer_y*np.tan(alpha_deg*scale_factor) - R_a
    x_E2 = 2*foyer_y*np.tan(alpha_deg*scale_factor) + R_a
    y_E1 = Parabole_y_n(x_E1, Parametres_parabole_equinnoxe[0], Parametres_parabole_equinnoxe[1]) 
    y_E2 = Parabole_y_n(x_E2, Parametres_parabole_equinnoxe[0], Parametres_parabole_equinnoxe[1])
    m_g = (y_E2 - y_E1)/(x_E2 - x_E1) #41° - 44.9°
    C_g = y_E1-m_g*x_E1
    Parametres_ligne_equinnoxe = np.array([m_g, C_g, alpha_deg, np.array([x_E1,y_E1]), np.array([x_E2,y_E2])], dtype=object)
    
    # Calculer les paramètres de l'ellipse formée dans le plan de coupe
    a = (x_E2-x_E1)/2 #np.sqrt((Parametres_ligne_equinnoxe[0]/(2*Parametres_parabole_equinnoxe[0]))**2 - (Parametres_parabole_equinnoxe[1]-Parametres_ligne_equinnoxe[1])/Parametres_parabole_equinnoxe[0])
    ratio_angle = np.arctan(m_g)
    ratio_ellipse = np.cos(ratio_angle) #r_e = a/b
    b = a/ratio_ellipse
    b_n = np.sqrt((x_E2-x_E1)**2 + (y_E2-y_E1)**2)/2
    Surface_equinnoxe_plan_coupe = np.pi*a*b
    h = ((b-a)/(b+a))**2
    perimetre = np.pi*(a+b)*(1 + (3*h)/(10 + np.sqrt(4-3*h)))
    Parametres_ellipse_equinnoxe = np.array([a, b, ratio_ellipse, perimetre, Surface_equinnoxe_plan_coupe], dtype=object)
    
    # Calcul des informations des crossbars
    x_n, y_n, m_q_n, C_q_vec, a_qn_vec, b_qn_vec, perpendicular_angle, x_intersect, y_intersect = Distribution_crossbars(Parametres_ellipse_equinnoxe, n_crossbars_real, gap_crossbars, Parametres_ligne_equinnoxe, Parametres_parabole_equinnoxe, x_E1, x_E2)
    delta_n, R_n, beta_n, arc_n = Crossbars_information(n_crossbars_real, y_n, a_qn_vec, perpendicular_angle)
    Parametres_crossbars_equinnoxe = np.array([x_n, y_n, m_q_n, C_q_vec, a_qn_vec, b_qn_vec, perpendicular_angle, delta_n, R_n, beta_n, arc_n], dtype=object)
    
    # Calcul de la surface du réflecteur
    r_cercle = Parametres_ellipse_equinnoxe[0]
    a_cercle = x_E1 + r_cercle
    surface = valeur_surface(Parametres_parabole_equinnoxe, a_cercle, r_cercle, 1000, 1000)
    return Parametres_parabole_equinnoxe, Parametres_ligne_equinnoxe, Parametres_ellipse_equinnoxe, Parametres_crossbars_equinnoxe

# Fonction permettant de calculer la forme de la parabole, du plan de coupe et de la forme de l'ellipse résultante en fonction du jour de l'année
def Determiner_Parabole_saison(n_year, Parametres_parabole_equinnoxe, Parametres_ligne_equinnoxe, Parametres_ellipse_equinnoxe):
    # Equinnnoxe information
    f_point_y_equinnoxe = Parametres_parabole_equinnoxe[2][1]
    x_p_equinnoxe = Parametres_parabole_equinnoxe[3][0]
    y_p_equinnoxe = Parametres_parabole_equinnoxe[3][1]
    alpha_year = alpha_solar(n_year)
    f_n = f_point_y_equinnoxe*(1-np.cos(np.pi/2 - alpha_year))
    m_saison = 1/(4*f_n)
    C_saison = f_point_y_equinnoxe-f_n
    f_spatial_point_saison = np.array([Parametres_parabole_equinnoxe[2][0], f_n])
    Point_P_saison = np.array([x_p_equinnoxe*np.cos(alpha_year),(x_p_equinnoxe*np.cos(alpha_year))**2/(4*f_n)+ (f_point_y_equinnoxe-f_n)])
    Parametres_parabole_saison = np.array([m_saison, C_saison, f_spatial_point_saison, Point_P_saison], dtype=object)
    
    # Calculer les nouveaux points d'intersection du plan
    n_points = 100
    I_E1_equinnoxe = integrate_line(Parametres_ligne_equinnoxe[3][0], x_p_equinnoxe, f_point_y_equinnoxe, n_points)
    I_E2_equinnoxe = integrate_line(x_p_equinnoxe, Parametres_ligne_equinnoxe[4][0], f_point_y_equinnoxe, n_points)
    def objective_E1(x_inic_new):
        result = integrate_line(x_inic_new, Point_P_saison[0], f_n, n_points)
        return result - I_E1_equinnoxe
    def objective_E2(x_inic_new):
        result = integrate_line(Point_P_saison[0], x_inic_new, f_n, n_points)
        return result - I_E2_equinnoxe
    tol_method = 1e-8
    x_E1_n = root_scalar(objective_E1, bracket=[0, Point_P_saison[0]], method='brentq', xtol=tol_method).root
    x_E2_n = root_scalar(objective_E2, bracket=[Point_P_saison[0], Parametres_ligne_equinnoxe[4][0]*4], method='brentq', xtol=tol_method).root
    y_E1_n = Parabole_y_n(x_E1_n, Parametres_parabole_saison[0], Parametres_parabole_saison[1]) 
    y_E2_n = Parabole_y_n(x_E2_n, Parametres_parabole_saison[0], Parametres_parabole_saison[1])
    b_n = np.sqrt((x_E2_n-x_E1_n)**2 + (y_E2_n-y_E1_n)**2)/2
    a_n = Parametres_ellipse_equinnoxe[4]/(np.pi*b_n)
    ratio_ellipse_n = a_n/b_n
    Surface_n_plan_coupe = np.pi*a_n*b_n
    h = ((b_n-a_n)/(b_n+a_n))**2
    perimetre_n = np.pi*(a_n+b_n)*(1 + (3*h)/(10 + np.sqrt(4-3*h)))
    m_g_n = (y_E2_n - y_E1_n)/(x_E2_n - x_E1_n) #41° - 44.9°
    C_g_n = y_E1_n-m_g_n*x_E1_n
    alpha_deg_n = np.arctan(m_g_n)*180/np.pi
    Parametres_ligne_saison = np.array([m_g_n, C_g_n, alpha_deg_n, np.array([x_E1_n,y_E1_n]), np.array([x_E2_n,y_E2_n])], dtype=object)
    
    # Calculer des vecteurs pour identifier le plan de coupe spatialement
    alpha_rad_n = alpha_deg_n*np.pi/180
    U_ellipse_n = np.array([np.cos(alpha_rad_n),np.sin(alpha_rad_n),0])
    V_ellipse_n = np.array([0,0,1])
    C_ellipse_n = np.array([(x_E2_n + x_E1_n)/2,(y_E2_n + y_E1_n)/2,0])
    Parametres_ellipse_saison = np.array([a_n, b_n, ratio_ellipse_n, perimetre_n, Surface_n_plan_coupe, U_ellipse_n, V_ellipse_n, C_ellipse_n], dtype=object)
    return Parametres_parabole_saison, Parametres_ligne_saison, Parametres_ellipse_saison

# Fonction permettant de déterminer le nombre de miroirs à placer dans chaque section de miroir
def Determiner_nombre_miroirs(Parametres_parabole_saison, Parametres_ligne_saison, Parametres_ellipse_saison, Parametres_parabole_equinnoxe, lon_miroir, step_y=0.0001):
    alpha_rad_n = Parametres_ligne_saison[2]*np.pi/180
    a_n = Parametres_ellipse_saison[0] #(b_n para nosotros)
    b_n = Parametres_ellipse_saison[1] #(a_n para nosotros)
    f_n = Parametres_parabole_saison[2][1]
    f_equinnoxe = Parametres_parabole_equinnoxe[2][1]
    U_ellipse_n = Parametres_ellipse_saison[5]
    V_ellipse_n = Parametres_ellipse_saison[6]
    C_ellipse_n = Parametres_ellipse_saison[7]
    
    # Fonction de calcul en x renvoyant une valeur de la parabole y (demi-axe positif)
    def y_to_x(y):
        return np.sqrt(4*f_n*(y-(f_equinnoxe-f_n)))
    
    # Calculer les coordonnées y [ymin,ymax] du profil parabolique où la largeur de la crossbar est égale à lon_miroir
    x_val = b_n*np.sqrt(1-(lon_miroir/(2*a_n))**2)
    x = np.array([x_val,-x_val])
    yl = x*np.sin(alpha_rad_n)+C_ellipse_n[1]
    xl = y_to_x(yl)
    
    # Processus pour diviser le profil parabolique entre ymin, ymax en segments de taille lon_miroir
    y_vals = [yl[0]]
    x_vals = [xl[0]]
    i = 0
    while y_vals[i] >= yl[1]:
        x_i = y_to_x(y_vals[i])
        y2 = y_vals[i]
        x2 = x_i
        d = 0
        # Chaque crossbar est supposée reposer sur le coin supérieur de chaque segment
        while d <= lon_miroir:
            d = np.sqrt((x_vals[i]-x2)**2+(y_vals[i]-y2)**2)
            y2 -= step_y
            x2 = y_to_x(y2)
        if y2 >= yl[1]:
            y_vals.append(y2)
            x_vals.append(x2)
            i += 1
        else:
            break
    # Calculer les informations géométriques des crossbars dans l'espace
    Rayon_crossbar = np.array(x_vals)
    x_proj = (np.array(y_vals)-C_ellipse_n[1])/np.sin(alpha_rad_n) # Distance de C_ellipse_n le long de l'axe majeur
    #t = np.arccos(x_proj/a_n)
    coords_spatial_x = np.abs(np.array(y_vals)/np.tan(alpha_rad_n))
    coords_spatial_y = np.array(y_vals)
    coords_spatial_z = a_n*np.sqrt(1-(x_proj/b_n)**2)
    positions_spaciales_miroirs = np.vstack((coords_spatial_x,coords_spatial_y,coords_spatial_z)).T
    demiangle_soustendu_crossbars = np.degrees(np.arcsin(positions_spaciales_miroirs[:,2]/Rayon_crossbar))
    angle_soustendu_miroirs = 2*np.degrees(np.arcsin(lon_miroir/(2*Rayon_crossbar)))
    nombre_miroirs = np.floor(2*demiangle_soustendu_crossbars/angle_soustendu_miroirs).astype(int)
    return positions_spaciales_miroirs, Rayon_crossbar, demiangle_soustendu_crossbars, angle_soustendu_miroirs, nombre_miroirs

# Calculez si les 4 points résultants sont coplanaires
def Sont_coplanaires(Am, Bm, Cm, Dm, tol=1e-8):
    AB = Bm - Am
    AC = Cm - Am
    AD = Dm - Am
    volumen = np.dot(AB, np.cross(AC, AD))
    return abs(volumen) < tol

# Calculer le vecteur normal et le centre des plans résultant de la discrétisation du réflecteur
def Vecteur_normal_plan(Am, Bm, Cm, Dm):
    v1 = Bm - Am
    v2 = Dm - Am
    Centre_m = (0.25)*(Am+Bm+Cm+Dm)
    normal = np.cross(v1, v2)
    normal_unitaire = normal / np.linalg.norm(normal)
    return Centre_m, normal_unitaire

# Modifiez les axes de position des points résultants qui forment la structure du réflecteur
def Swap_axes(M_tile, ordre=(0, 2, 1)):
    M_tile_modifie = []
    for points in M_tile:
        nouveaux_points = [p[list(ordre)] for p in points]
        M_tile_modifie.append(nouveaux_points)
    return M_tile_modifie

# Routine pour calculer la distribution des miroirs
def Determiner_distribution_miroirs(positions_spaciales_miroirs, Rayon_crossbar, demiangle_soustendu_crossbars, angle_soustendu_miroirs, nombre_miroirs):    
    e = positions_spaciales_miroirs
    Rm = Rayon_crossbar
    ha = demiangle_soustendu_crossbars
    ma = angle_soustendu_miroirs
    nm = nombre_miroirs
    Uhc = np.array([1, 0, 0])
    Vhc = np.array([0, 0, 1])
    M_tile = []
    k = 0
    for i in range(len(nm)):
        if e[i,2] < e[i + 1,2]:
            mac = np.linspace(-ha[i], ha[i], nm[i]+1)
            for j in range(nm[i]):
                Dm = np.zeros(3)
                Dm[0] = Rm[i] * (np.cos(np.radians(mac[j])) * Uhc[0] + np.sin(np.radians(mac[j])) * Vhc[0])
                Dm[1] = e[i,1] + Rm[i] * (np.cos(np.radians(mac[j])) * Uhc[1] + np.sin(np.radians(mac[j])) * Vhc[1])
                Dm[2] = Rm[i] * (np.cos(np.radians(mac[j])) * Uhc[2] + np.sin(np.radians(mac[j])) * Vhc[2])
                Cm = np.zeros(3)
                angle = mac[j] + ma[i]
                Cm[0] = Rm[i] * (np.cos(np.radians(angle)) * Uhc[0] + np.sin(np.radians(angle)) * Vhc[0])
                Cm[1] = e[i,1] + Rm[i] * (np.cos(np.radians(angle)) * Uhc[1] + np.sin(np.radians(angle)) * Vhc[1])
                Cm[2] = Rm[i] * (np.cos(np.radians(angle)) * Uhc[2] + np.sin(np.radians(angle)) * Vhc[2])
                m = -(Dm[0] - Cm[0]) / (Dm[2] - Cm[2])
                def eqn_1(fy):
                    return Rm[i+1] * np.sin(fy) - Dm[2] - m * (Rm[i + 1] * np.cos(fy) - Dm[0])
                fy_initial_guess = 0.0
                angle_rad = fsolve(eqn_1, fy_initial_guess)
                angle_deg = np.degrees(angle_rad)
                Am = None
                if abs(angle_deg) < ha[i + 1]:
                    Am = np.zeros(3)
                    Am[0] = Rm[i+1] * np.cos(angle_rad[0])
                    Am[1] = e[i+1,1]
                    Am[2] = Rm[i+1] * np.sin(angle_rad[0])
                if Am is None:
                    continue
                Bm = Am + (Cm - Dm)
                M_tile.append([Am, Bm, Cm, Dm])
                k += 1
        else:
            break
    for i in range(len(nm)-1, -1, -1):
        if e[i,2] < e[i-1,2]:
            mac = np.linspace(-ha[i], ha[i], nm[i] + 1)
            for j in range(nm[i]):
                Am = np.zeros(3)
                Am[0] = Rm[i] * (np.cos(np.radians(mac[j])) * Uhc[0] + np.sin(np.radians(mac[j])) * Vhc[0])
                Am[1] = e[i][1] + Rm[i] * (np.cos(np.radians(mac[j])) * Uhc[1] + np.sin(np.radians(mac[j])) * Vhc[1])
                Am[2] = Rm[i] * (np.cos(np.radians(mac[j])) * Uhc[2] + np.sin(np.radians(mac[j])) * Vhc[2])
                angle = mac[j] + ma[i]
                Bm = np.zeros(3)
                Bm[0] = Rm[i] * (np.cos(np.radians(angle)) * Uhc[0] + np.sin(np.radians(angle)) * Vhc[0])
                Bm[1] = e[i][1] + Rm[i] * (np.cos(np.radians(angle)) * Uhc[1] + np.sin(np.radians(angle)) * Vhc[1])
                Bm[2] = Rm[i] * (np.cos(np.radians(angle)) * Uhc[2] + np.sin(np.radians(angle)) * Vhc[2])
                m = -(Am[0] - Bm[0]) / (Am[2] - Bm[2])
                def eqn_2(fy):
                    return Rm[i-1] * np.sin(fy) - Am[2] - m * (Rm[i-1] * np.cos(fy) - Am[0])
                fy_initial_guess = 0.0
                angle_rad = fsolve(eqn_2, fy_initial_guess)
                angle_deg = np.degrees(angle_rad)
                Dm = np.zeros(3)
                Dm[0] = Rm[i-1] * np.cos(angle_rad[0])
                Dm[1] = e[i-1,1]
                Dm[2] = Rm[i-1] * np.sin(angle_rad[0])
                Cm = Dm + (Bm - Am)
                M_tile.append([Am, Bm, Cm, Dm])
                k += 1
        else:
            break
        M_tile_modifie = Swap_axes(M_tile)
    return M_tile_modifie

# Fonction pour définir les matrices de plans de réflexion
def Calculer_plans_de_simulation_raytracing(M_tile, lon_miroir):
    Matrice_Point_C = []
    Matrice_L_xy = []
    Matrice_vec_normal = []
    for points in M_tile:
        Am, Bm, Cm, Dm = points
        Centre_m, normal_unitaire = Vecteur_normal_plan(Am, Bm, Cm, Dm)
        Matrice_Point_C.append(Centre_m)
        Matrice_L_xy.append([lon_miroir, lon_miroir])
        Matrice_vec_normal.append(normal_unitaire)
    return Matrice_Point_C, Matrice_L_xy, Matrice_vec_normal

# Fonction permettant de calculer la rotation du réflecteur en fonction du jour de l'année
def faire_pivoter_les_plans_autour_de_y(M_tile, angle_rad, f_y):
    rotation = R.from_rotvec(angle_rad * np.array([0, 1, 0]))
    M_tile_rotate = []
    for quad in M_tile:
        quad_rotate = []
        for point in quad:
            # Translation vers l'origine focale
            point_translaté = point - np.array([0, 0, f_y])
            # Application de la rotation
            point_tourné = rotation.apply(point_translaté)
            # Retour à la position initiale
            point_final = point_tourné + np.array([0, 0, f_y])
            quad_rotate.append(point_final)
        M_tile_rotate.append(quad_rotate)
    return M_tile_rotate

# Calculer tous les paramètres de la parabole du réflecteur de Scheffler
def Calculer_parabole_réfléchissante_adéquate(foyer_y, A_approx, n_crossbars_real, gap_crossbars, n_year, lon_miroir):
    Parametres_parabole_equinnoxe, Parametres_ligne_equinnoxe, Parametres_ellipse_equinnoxe, Parametres_crossbars_equinnoxe = Determiner_Parabole_equinnoxe(foyer_y, A_approx, n_crossbars_real, gap_crossbars)
    alpha_year = alpha_solar(n_year)
    
    # Calculer la parabole pour l'hiver
    n_hiver = 355
    Parametres_parabole_hiver, Parametres_ligne_hiver, Parametres_ellipse_hiver = Determiner_Parabole_saison(n_hiver, Parametres_parabole_equinnoxe, Parametres_ligne_equinnoxe, Parametres_ellipse_equinnoxe)
    positions_spaciales_miroirs, Rayon_crossbar, demiangle_soustendu_crossbars, angle_soustendu_miroirs, nombre_miroirs_hiver = Determiner_nombre_miroirs(Parametres_parabole_hiver, Parametres_ligne_hiver, Parametres_ellipse_hiver, Parametres_parabole_equinnoxe, lon_miroir)
    
    # Calculer la parabole pour l'été
    n_ete = 172
    Parametres_parabole_ete, Parametres_ligne_ete, Parametres_ellipse_ete = Determiner_Parabole_saison(n_ete, Parametres_parabole_equinnoxe, Parametres_ligne_equinnoxe, Parametres_ellipse_equinnoxe)
    positions_spaciales_miroirs, Rayon_crossbar, demiangle_soustendu_crossbars, angle_soustendu_miroirs, nombre_miroirs_ete = Determiner_nombre_miroirs(Parametres_parabole_ete, Parametres_ligne_ete, Parametres_ellipse_ete, Parametres_parabole_equinnoxe, lon_miroir)
    
    # Calculer la parabole pour le jour sélectionné
    Parametres_parabole_saison, Parametres_ligne_saison, Parametres_ellipse_saison = Determiner_Parabole_saison(n_year, Parametres_parabole_equinnoxe, Parametres_ligne_equinnoxe, Parametres_ellipse_equinnoxe)
    positions_spaciales_miroirs, Rayon_crossbar, demiangle_soustendu_crossbars, angle_soustendu_miroirs, nombre_miroirs = Determiner_nombre_miroirs(Parametres_parabole_saison, Parametres_ligne_saison, Parametres_ellipse_saison, Parametres_parabole_equinnoxe, lon_miroir)
    
    # Déterminer le nombre de miroirs à utiliser
    nombre_miroirs_hiver = np.array(nombre_miroirs_hiver)
    nombre_miroirs_ete = np.array(nombre_miroirs_ete)
    len_hiver = len(nombre_miroirs_hiver)
    len_ete = len(nombre_miroirs_ete)
    max_len = max(len_hiver, len_ete)
    if len_hiver < max_len:
        nombre_miroirs_hiver = np.pad(nombre_miroirs_hiver, (0, max_len - len_hiver), mode='constant')
    if len_ete < max_len:
        nombre_miroirs_ete = np.pad(nombre_miroirs_ete, (0, max_len - len_ete), mode='constant')
    nombre_miroirs = np.minimum(nombre_miroirs_hiver, nombre_miroirs_ete)
    
    # Calculer les plans qui font partie du réflecteur de Scheffler
    M_tile = Determiner_distribution_miroirs(positions_spaciales_miroirs, Rayon_crossbar, demiangle_soustendu_crossbars, angle_soustendu_miroirs, nombre_miroirs)
    M_tile = faire_pivoter_les_plans_autour_de_y(M_tile, alpha_year, foyer_y)
    Matrice_Point_C, Matrice_L_xy, Matrice_vec_normal= Calculer_plans_de_simulation_raytracing(M_tile, lon_miroir)
    return Matrice_Point_C, Matrice_L_xy, Matrice_vec_normal, M_tile, Parametres_ligne_saison


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================
# ================================================== FONCTION_MAIN =====================================================
# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


if __name__ == '__main__':
    foyer_y = 2 #m
    A_approx = 16 #m^2
    n_crossbars_real = 8 #impair
    gap_crossbars = 0.5 #m
    n_year = 80
    alpha_year = alpha_solar(n_year)
    lon_miroir = 0.25
    #----------------------
    Longueur_four = 1.4
    Largeur_four = 1.4
    Hauteur_four = 0.6
    Longueur_creuse = 0.8
    Largeur_creuse = 0.25
    Centre_spatial = np.array([-Longueur_four/2,0,foyer_y])
    z_reference = foyer_y-Hauteur_four/2
    #----------------------
    z_0_rayons = 18
    deviation_rayons = 5e-3 
    #----------------------
    n_refraction_materiel = 1.7
    step_1 = (0.15/np.sqrt(2))/2
    #-----------------------------------------------------------------------------------------------------------------
    Matrice_Point_C_ad = np.array([[Centre_spatial[0]+Longueur_four/2-step_1-Longueur_four*5,Centre_spatial[1],Centre_spatial[2]-Largeur_creuse/2 + step_1],
                                [Centre_spatial[0]+Longueur_four/2-0.1-Longueur_four*5,Centre_spatial[1],Centre_spatial[2]+Hauteur_four/2-0.05],
                                [Centre_spatial[0]-Longueur_four/2+0.1-Longueur_four*5,Centre_spatial[1],Centre_spatial[2]+Hauteur_four/2-0.05]])
    Matrice_L_xy_ad = np.array([[0.15,Largeur_four],
                            [0.3,Largeur_four],
                            [0.3,Largeur_four]])
    Matrice_vec_normal_ad = np.array([[1,0,2],
                                    [1,0,3],
                                    [1,0,-3]])
    #-----------------------------------------------------------------------------------------------------------------
    Matrice_Point_C_special_ad = np.array([])
    Matrice_L_xy_special_ad = np.array([])
    Matrice_vec_normal_special_ad = np.array([])
    Matrice_region_creuse_special_ad = np.array([])
    #-----------------------------------------------------------------------------------------------------------------
    Matrice_Point_C_refraction_ad = np.array([])
    Matrice_L_xy_refraction_ad = np.array([])
    Matrice_vec_normal_refraction_ad = np.array([])
    Matrice_n_refraction_ad = np.array([])
    #----------------------
    parametres_creation_simulation = [Centre_spatial,Longueur_four,Largeur_four,Hauteur_four,foyer_y,A_approx,n_crossbars_real,gap_crossbars,n_year,lon_miroir,z_0_rayons,Longueur_creuse,Largeur_creuse,n_refraction_materiel,deviation_rayons,alpha_year,foyer_y]
    matrices_creation_simulation = [Matrice_Point_C_ad,Matrice_L_xy_ad,Matrice_vec_normal_ad,Matrice_Point_C_special_ad,Matrice_L_xy_special_ad,Matrice_vec_normal_special_ad,Matrice_region_creuse_special_ad,Matrice_Point_C_refraction_ad,Matrice_L_xy_refraction_ad,Matrice_vec_normal_refraction_ad,Matrice_n_refraction_ad]

    angulo_elev = 0
    angulo_azim = 90
    x_lim = [-2, 5.5]
    y_lim = [-3, 3]
    z_lim = [0.5, 5.5]
    
    num_rayos = 200000
    num_max_rebonds = 8
    num_rayons_graph = 2000
    num_max_rayons_KDE = num_rayos
    
    start_time = time.time()
    Matrice_plans_normaux, Matrice_plans_speciaux,Matrice_plans_refraction,Matrice_plans_normaux_parabole,Matrice_rayons,n_normales,n_speciaux,n_refraction,n_normales_reflecteur, Points_graph_parabole = Creer_simulation(num_rayos,parametres_creation_simulation,matrices_creation_simulation)
    trajectoires_rayons, Matrice_rayons = Evolution_rayons(Matrice_plans_normaux,Matrice_plans_speciaux,Matrice_plans_refraction,Matrice_rayons,Matrice_plans_normaux_parabole,n_normales,n_speciaux,n_refraction,n_normales_reflecteur,num_rayos,num_max_rebonds)
    Intersections_plaque = Determiner_intersection_plaque(trajectoires_rayons,num_rayos,num_max_rebonds,z_reference)
    creation_simulation_time = time.time() - start_time
    print(f"Temps de création de la simulation : {creation_simulation_time:.6f} secondes")

    
    start_time = time.time()
    Graphique_general(Matrice_plans_normaux,Matrice_plans_speciaux,Matrice_plans_normaux_parabole,n_normales,n_speciaux,n_normales_reflecteur,num_max_rebonds,num_rayons_graph,trajectoires_rayons,angulo_elev,angulo_azim,Matrice_rayons,Points_graph_parabole,x_lim,y_lim,z_lim)
    graph_1_time = time.time() - start_time
    print(f"Temps du premier graphique : {graph_1_time:.6f} secondes")
    
    start_time = time.time()
    Graphique_points_collision(Intersections_plaque,num_max_rayons_KDE,Matrice_plans_normaux[4,:])
    graph_2_time = time.time() - start_time
    print(f"Temps du deuxième graphique : {graph_2_time:.6f} secondes")