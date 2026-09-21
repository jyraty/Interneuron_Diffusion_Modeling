import numpy as np
from scipy.optimize import minimize_scalar
from numpy.polynomial import Polynomial




def polynome_3(x1, y1, x2, y2, x3, y3, x4, y4):

    A = np.array([
        [x1**3, x1**2, x1, 1],
        [x2**3, x2**2, x2, 1],
        [x3**3, x3**2, x3, 1],
        [x4**3, x4**2, x4, 1]
    ], dtype=float)

    Y = np.array([y1, y2, y3, y4], dtype=float)

    a, b, c, d = np.linalg.solve(A, Y)

    return a, b, c, d

def polynome_4(x1, y1, x2, y2, x3, y3, x4, y4, x5,y5):

    A = np.array([
        [x1**4, x1**3, x1**2, x1, 1],
        [x2**4, x2**3, x2**2, x2, 1],
        [x3**4, x3**3, x3**2, x3, 1],
        [x4**4, x4**3, x4**2, x4, 1],
        [x5**4, x5**3, x5**2, x5, 1]
    ], dtype=float)

    Y = np.array([y1, y2, y3, y4,y5], dtype=float)

    a, b, c, d,e = np.linalg.solve(A, Y)

    return a, b, c, d,e





def distance_point_courbeh(x1, y1):
    """
    Distance minimale entre le point (x1, y1)
    et la courbe y = a*x^3 + b*x^2 + c*x + d.

    Retourne :
        distance, x_courbe, y_courbe
    """

    def f(x):
        return polyh(x)

    def distance2(x):
        return (x - x1)**2 + (f(x) - y1)**2

    # Recherche numerique de la distance minimale
    resultat = minimize_scalar(distance2)

    x_c = resultat.x
    y_c = f(x_c)

    distance = (distance2(x_c))**0.5

    return distance, x_c, y_c

def distance_point_courbel(x1, y1):
    """
    Distance minimale entre le point (x1, y1)
    et la courbe y = a*x^3 + b*x^2 + c*x + d.

    Retourne :
        distance, x_courbe, y_courbe
    """

    def f(x):
        return polyl(x)

    def distance2(x):
        return (x - x1)**2 + (f(x) - y1)**2

    # Recherche numerique de la distance minimale
    resultat = minimize_scalar(distance2)

    x_c = resultat.x
    y_c = f(x_c)

    distance = (distance2(x_c))**0.5

    return distance, x_c, y_c

def distance_point_courbe4(x1, y1, a, b, c, d,e):
    """
    Distance minimale entre le point (x1, y1)
    et la courbe y = a*x^3 + b*x^2 + c*x + d.

    Retourne :
        distance, x_courbe, y_courbe
    """

    def f(x):
        return a*x**4 + b*x**3 + c*x**2 + d*x+e

    def distance2(x):
        return (x - x1)**2 + (f(x) - y1)**2

    # Recherche numerique de la distance minimale
    resultat = minimize_scalar(distance2)

    x_c = resultat.x
    y_c = f(x_c)

    distance = (distance2(x_c))**0.5

    return distance, x_c, y_c



nom_fichier = "traj_t_x_y_vx_vy.txt"
nom_out='traj_t_x_y_vx_vy_distri.txt'
nr_to_avg=144000
x=0.0
y=0.0
ay=350.0/1200.
ax=400.0/800.

x1=243.0
y1=150.0
x2=115.0
y2=240.0
x3=393.0
y3=269.0
x4=327.0
y4=399.0


dist=0.0
a=0.0
b=0.0
c=0.0
d=0.0
e=0.0

xx = np.array([167, 190 ,218, 251, 291, 333, 364])
yy = np.array([94, 136,  169, 197, 219, 236, 244])

xxh= np.array([ 111, 142 , 175, 208, 242, 281, 313])
yyh= np.array([ 233, 268,  297, 322, 348, 373, 391])


polyh = Polynomial.fit(xxh, yyh, deg=2)
polyl = Polynomial.fit(xx, yy, deg=3)




# 4. Convertir en coefficients standards pour un affichage classique

coefficients = polyl.convert().coef
print("\nCoefficients standardiss :")
print(coefficients)

#(a,b,c,d)=polynome_3(360*ax, 370*ay, 447.0*ax , 553.0*ay, 555.0*ax, 685.0*ay, 682.0*ax,778.0*ay)
#print('a b c d', a, b, c, d)

#(a,b,c,d,e)=polynome_4(360*ax, 370*ay, 447.0*ax , 553.0*ay, 241.0,178.0,555.0*ax, 685.0*ay, 682.0*ax,778.0*ay)




with open(nom_fichier, "r", encoding="utf-8") as f_read:
    with open(nom_out, "w", encoding="utf-8") as f_avg:    
        lines=f_read.readlines()
        #print(" nlines "+str(len(lines)))
        
        maxpoint=int(len(lines))
        minpoint=maxpoint-nr_to_avg
        if (minpoint < 0):
            minpoint=0
        il=minpoint
        
        while il < maxpoint-1:    
            parts = lines[il].split()
            il+=1
             

            x=float(parts[1])
            y=float(parts[2])
            print(" il ",il,x,y)
            in_zone = 1
            ymin=y1+(x-x1)*(y2-y1)/(1.*(x2-x1))
            if (y < ymin):
               in_zone= 0
            ymax=y3+(x-x3)*(y4-y3)/(1.*(x4-x3))
            if (y > ymax):
                in_zone=0

            if (in_zone==1):
                #dist= distance_point_courbe(x, y, a, b, c, d)
                disth= distance_point_courbeh(x, y)
                distl= distance_point_courbel(x, y)
                dist=(disth[0]+0*distl[0])/1.0
                print(f'x y d {x} {y} {dist}')
                ligne = f"{x:9.5f} \t{y:9.5f} \t  {dist:9.5f}\n"
                f_avg.write(ligne)



