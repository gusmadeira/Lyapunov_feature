#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from math import *
import numpy as np
import os
import fnmatch

#Element in nd array aux closest to the scalar value a0
def find_nearest(aux, a0):
    return aux[np.abs(aux - a0).argmin()]

#Função que modula o angulo entre 0 e interval
def angle_mod(angle, interval = 2*np.pi):
  if (angle > interval): angle = angle % interval
  if (angle < 0): angle = angle + interval
  return angle

def find(pattern, path):
  result = []
  for root, dirs, files in os.walk(path):
    for name in files:
      if fnmatch.fnmatch(name, pattern):
        result.append(os.path.join(root, name))
  return result

def xv_to_aei(G,m_0,m,xyz):
  x = xyz[0]
  y = xyz[1]
  z = xyz[2]
  vx = xyz[3]
  vy = xyz[4]
  vz = xyz[5]
  
  hx = y*vz-z*vy
  hy = z*vx-x*vz
  hz = x*vy-y*vx
  h2 = hx**2+hy**2+hz**2
  r2 = x**2+y**2+z**2
  v2 = vx**2+vy**2+vz**2
  rv = x*vx  +  y*vy  +  z*vz
  
  # semieixo maior
  a_aux = 2./(r2)**(.5) - v2/((m+m_0)*G)
  a = 1./a_aux
  
  # excentricidade
  e_aux = h2/((m+m_0)*G*a)
  e = (1.-e_aux)**(.5)
  
  #e2
  gm = G*(m+m_0)
  s = h2 / gm
  temp = 1.e0  +  s * (v2 / gm  -  2.e0 / r2**.5)
  e2 = sqrt(temp)
  #print (e,e2)
  
  # inclinação
  cosi = hz/h2**(.5)
  i = np.arccos(cosi) #rad
  
  # longitude do nodo
  if (abs(cosi) < 1):
    omega = atan2(hx,-hy)
    omega = angle_mod(omega)
  else:
    omega = 0
  
  # anomalia verdadeira
  cosf = (a*(1-e**2)/r2**.5-1)/e
  if (abs(cosf) > 1): cosf = 1
  f = np.arccos(cosf)
  
  # anomalia verdadeira + argumento do pericentro
  if (i != 0):
    sin_f_plus_g = z/((r2**.5)*np.sin(i))
    f_plus_g = np.arcsin(sin_f_plus_g)
  else:
    f_plus_g = f
  
  # argumento do pericentro
  g = f_plus_g - f
  g = angle_mod(g)
  
  # anomalia excêntrica
  cosE = (a-r2**.5)/(a*e)
  if (abs(cosE) > 1): cosE = 1
  E = np.arccos(cosE)
  
  # anomalia média
  if (rv < 0): E = 2*np.pi-E
  M = E - e*np.sin(E)
  M = angle_mod(M)
  
  aei = [a,e,i,g,omega,M]
  return (aei)
  
def aei_to_xv(G,m_0,m,aei):
  #Semieixo maior
  a = aei[0]
  #Excentricidade
  e = aei[1]
  #Inclinação(rad)
  i = aei[2]*np.pi/180.
  #Argumento do Pericentro(rad)
  g = aei[3]*np.pi/180.
  #Nodo(rad)
  omega = aei[4]*np.pi/180.
  #Anomalia Média(rad)
  M = aei[5]*np.pi/180.
  
  #Calculando movimento médio
  mu = G*(m_0 + m)
  n = np.sqrt(mu/(a**3))

  #Metodo de Newton-Raphson para calcular a nomalia excentrica (E)
  E = M
  E0 = 0
  ERR = M

  while (ERR>1.0e-14):
    E0 = E
    E = E0 - (E0-e*np.sin(E0)-M)/(1.0-e*np.cos(E0))
    ERR = abs(E-E0)
  #print (M, E)

  #Calculando a anomalia verdadeira(f)

  f = 2*np.arctan2(np.sqrt(1.0 + e)*np.sin(E/2.0), np.sqrt(1.0 - e)*np.cos(E/2.0))

  #Calculando a distância radial (eq. 2.20)e
  r = (a*(1 - e**2))/(1 + e*np.cos(f))

  #Calculando dr/dt (eq. 2.31)
  dr_dt = (n*a/np.sqrt(1 - e**2))*e*np.sin(f)

  #Matriz P1 (eq. 2.119)
  P1 = np.array([[np.cos(g), -np.sin(g), 0],
  [np.sin(g), np.cos(g), 0],
  [0, 0, 1]])

  #Matriz P2 (eq. 2.119)
  P2 = np.array([[1, 0, 0],
  [0, np.cos(i), -np.sin(i)],
  [0, np.sin(i), np.cos(i)]])

  #Matriz P3 (eq. 2.120)
  P3 = np.array([[np.cos(omega), -np.sin(omega), 0],
  [np.sin(omega), np.cos(omega), 0],
  [0, 0, 1]])

  #-----Calculando a posição do corpo (eq. 2.122)
  A = np.array([[r*np.cos(f)],[r*np.sin(f)],[0]])
  Pos = np.matmul(np.matmul(np.matmul(P3,P2),P1),A)

  #-----Calculando a velocidade do corpo
  B = np.array([[-n*a*np.sin(f)/np.sqrt(1 - e**2)],[n*a*(e + np.cos(f))/np.sqrt(1 - e**2)],[0]])
  Vel = np.matmul(np.matmul(np.matmul(P3,P2),P1),B)
  
  xyz = np.concatenate((Pos, Vel)).transpose()
  return (xyz)

def aei_to_xv2(G,m_0,m,aei):
  #Semieixo maior
  a = aei[0]
  #Excentricidade
  e = aei[1]
  #Inclinação(rad)
  i = aei[2]
  #Argumento do Pericentro(rad)
  g = aei[3]
  #Nodo(rad)
  omega = aei[4]
  #anomalia verdadeira(f)
  f = aei[5]
  
  #Calculando movimento médio
  mu = G*(m_0 + m)
  n = np.sqrt(mu/(a**3))

  #Calculando a distância radial (eq. 2.20)e
  r = (a*(1 - e**2))/(1 + e*np.cos(f))

  #Calculando dr/dt (eq. 2.31)
  #dr_dt = (n*a/np.sqrt(1 - e**2))*e*np.sin(f)

  #Matriz P1 (eq. 2.119)
  P1 = np.array([[np.cos(g), -np.sin(g), 0],
  [np.sin(g), np.cos(g), 0],
  [0, 0, 1]])

  #Matriz P2 (eq. 2.119)
  P2 = np.array([[1, 0, 0],
  [0, np.cos(i), -np.sin(i)],
  [0, np.sin(i), np.cos(i)]])

  #Matriz P3 (eq. 2.120)
  P3 = np.array([[np.cos(omega), -np.sin(omega), 0],
  [np.sin(omega), np.cos(omega), 0],
  [0, 0, 1]])

  #-----Calculando a posição do corpo (eq. 2.122)
  A = np.array([[r*np.cos(f)],[r*np.sin(f)],[0]])
  Pos = np.matmul(np.matmul(np.matmul(P3,P2),P1),A)

  #-----Calculando a velocidade do corpo
  B = np.array([[-n*a*np.sin(f)/np.sqrt(1 - e**2)],[n*a*(e + np.cos(f))/np.sqrt(1 - e**2)],[0]])
  Vel = np.matmul(np.matmul(np.matmul(P3,P2),P1),B)
  
  xyz = np.concatenate((Pos, Vel)).transpose()
  return (xyz)