# -*- coding: utf-8 -*-
import multiprocessing as mp
import numpy as np
from scipy.integrate import solve_ivp
import os
import pathlib
from itertools import product
import sys
from numpy.random import default_rng
import time

# -------------------------------------------------------------------
# Carregar parâmetros do input.ini
# -------------------------------------------------------------------
parameter = {}
try:
    with open('input.ini', 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=')
                parameter[key.strip()] = value.strip()
except FileNotFoundError:
    print("Aviso: input.ini nao encontrado. Usando padroes.")

alpha=float(parameter.get('alpha', 1e-3)) 
Trot=float(parameter.get('Trot', 10.)) 
save_xy=int(parameter.get('save_xy', 0)) 
central=int(parameter.get('central', 1)) 
distribution=int(parameter.get('distribution', 1)) 

############ Values (S.I.) ##############
G=6.6743e-11 
M1=float(parameter.get('M1', 1e20)) 
M2=-alpha*M1 
MT=M1+M2 
p=float(parameter.get('p', 1e3))
R=(3.*MT/(4.*np.pi*p))**(1./3.)
wk=np.sqrt(G*MT/R**3.)
Trot=Trot*60.*60. 
w=2.*np.pi/Trot
rejec=float(parameter.get('rejec', 100.)) 
J2=float(parameter.get('J2', 0.0)) 
C22=float(parameter.get('C22', 0.0)) 
############ Values (normalized) ##############
Gn=1.0
mu=M2/MT 
Rn=1. 
asinc=((G*M1/w**2.)**(1./3.))/R
lbd=w/wk
x1,y1=-mu,0.
x2,y2=Rn-mu,0.

#if mu<0 and alpha>0:
#    alpha = -alpha
#elif mu>0 and alpha<0:
#    alpha = -alpha

############ Initial conditions (time) ##############
Norb=int(parameter.get('Norb', 1000)) 
Tend=Norb*(2.*np.pi) 
time_steps_per_orbit = 100 
time=np.linspace(0,Tend,int(time_steps_per_orbit*Norb)) 

############ Initial conditions (space) ##############
a_init=float(parameter.get('a_init', 2.0))
a_end=float(parameter.get('a_end', 4.0))
e_init=float(parameter.get('e_init', 1e-4))
e_end=float(parameter.get('e_end', 0.5))
varpi=float(parameter.get('varpi', 0.0))
varpi = varpi *np.pi/180.
Npart=int(parameter.get('Npart', 10)) 

if distribution==1:
    a_vec=np.linspace(a_init,a_end+1e-6,Npart) 
    e_vec=np.linspace(e_init,e_end,Npart) 
elif distribution==2:
    a_vec = np.random.uniform(a_init, a_end, Npart)
    e_vec = np.random.uniform(e_init, e_end, Npart)
# -------------------------------------------------------------------
# Funções de movimento
# -------------------------------------------------------------------
def r_to_b1(XV):    
    x,y,vx,vy=XV
    x_n=x-x1
    y_n=y-y1
    r_n=np.sqrt(x_n**2.+y_n**2.)
    return [x_n,y_n,r_n]

def r_to_b2(XV):    
    x,y,vx,vy=XV
    x_n=x-x2
    y_n=y-y2
    r_n=np.sqrt(x_n**2.+y_n**2.)
    return [x_n,y_n,r_n]

def force_anomaly(t,XVl):    
    x,y,vx,vy=XVl
    xn1,yn1,r1=r_to_b1(XVl)
    xn2,yn2,r2=r_to_b2(XVl)
    Fx=xn1/r1**3.+alpha*xn2/r2**3.
    Fy=yn1/r1**3.+alpha*yn2/r2**3.
    return[Fx,Fy]    

def force_ellipsoid(t,XVl):
# IMPLEMENTAR AQUI O ELIPSOIDE    
    x,y,vx,vy=XVl
    xn1,yn1,r1=r_to_b1(XVl)
    xn2,yn2,r2=r_to_b2(XVl)
    Fx=xn1/r1**3.+alpha*xn2/r2**3.
    Fy=yn1/r1**3.+alpha*yn2/r2**3.
    return[Fx,Fy]    

def force(t,XVl):    
    if central ==1:
        Fx,Fy=force_anomaly(t,XVl)
    if central ==2:
        Fx,Fy=force_ellipsoid(t,XVl)
    return[Fx,Fy]    

def eqmotion(t, XVl):                   
    x,y,vx,vy=XVl
    Fx,Fy=force(t,XVl)   
    vxdot=(2.0*lbd*vy)+(lbd*lbd*x)-Fx   
    vydot=-(2.0*lbd*vx)+(lbd*lbd*y)-Fy  
    xdot=vx
    ydot=vy  
    return [xdot,ydot,vxdot,vydot]  

# -------------------------------------------------------------------
# Funções Lyapunov
# -------------------------------------------------------------------
def jacobian_force_components(XVl):
    x,y,vx,vy=XVl
    xn1,yn1,r1=r_to_b1(XVl)
    xn2,yn2,r2=r_to_b2(XVl)
    
    if r1 == 0 or r2 == 0:
        return 0, 0, 0, 0
        
    r1_3 = r1**3.0
    r1_5 = r1**5.0
    r2_3 = r2**3.0
    r2_5 = r2**5.0

    dFx_dx = (1./r1_3 - 3.*xn1**2./r1_5) + alpha * (1./r2_3 - 3.*xn2**2./r2_5)
    dFx_dy = (-3.*xn1*yn1/r1_5) + alpha * (-3.*xn2*yn2/r2_5)
    dFy_dx = dFx_dy 
    dFy_dy = (1./r1_3 - 3.*yn1**2./r1_5) + alpha * (1./r2_3 - 3.*yn2**2./r2_5)
    
    # IMPLEMENTAR O ELIPSOIDE
    
    return dFx_dx, dFx_dy, dFy_dx, dFy_dy

def eqmotion_with_lyap(t, Y):
    XVl = Y[0:4]
    w = Y[4:8]
    d_XVl = eqmotion(t, XVl)
    dFx_dx, dFx_dy, dFy_dx, dFy_dy = jacobian_force_components(XVl)
    J31 = lbd*lbd - dFx_dx
    J32 = -dFx_dy
    J41 = -dFy_dx
    J42 = lbd*lbd - dFy_dy
    d_wx = w[2] 
    d_wy = w[3] 
    d_wvx = J31*w[0] + J32*w[1] + (2.0*lbd)*w[3]
    d_wvy = J41*w[0] + J42*w[1] - (2.0*lbd)*w[2]
    d_w = [d_wx, d_wy, d_wvx, d_wvy]
    return np.concatenate((d_XVl, d_w))

# -------------------------------------------------------------------
# Conversões
# -------------------------------------------------------------------
def angle_mod(angle, interval = 2*np.pi):
  if (angle > interval): angle = angle % interval
  if (angle < 0): angle = angle + interval
  return angle

def rotacional_para_inercial(t, x, y, z, vx, vy, vz, n):
    cos_nt = np.cos(n * t)
    sin_nt = np.sin(n * t)
    X = cos_nt * x - sin_nt * y
    Y = sin_nt * x + cos_nt * y
    Z = z
    VX = cos_nt * (vx - n * y) - sin_nt * (vy + n * x)
    VY = sin_nt * (vx - n * y) + cos_nt * (vy + n * x)
    VZ = vz
    return X, Y, Z, VX, VY, VZ

def inercial_para_rotacional(t, X, Y, Z, VX, VY, VZ, n):
    cos_nt = np.cos(n * t)
    sin_nt = np.sin(n * t)
    x = cos_nt * X + sin_nt * Y
    y = - sin_nt * X + cos_nt * Y
    z = Z
    vx = cos_nt * VX + sin_nt * VY + n * y
    vy = - sin_nt * VX + cos_nt * VY - n * x
    vz = VZ
    return x, y, z, vx, vy, vz


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
  e2 = np.sqrt(temp)
  #print (e,e2)
  
  # inclinação
  cosi = hz/h2**(.5)
  i = np.arccos(cosi) #rad
  
  # longitude do nodo
  if (abs(cosi) < 1):
    omega = np.atan2(hx,-hy)
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

# -------------------------------------------------------------------
# Eventos
# -------------------------------------------------------------------

def col1(t, Y):
    x,y,vx,vy = Y[0:4] 
    z = 0.*x
    vz = 0.*vx
    X,Y,Z,VX,VY,VZ = rotacional_para_inercial(t, x, y, z, vx, vy, vz, lbd)
    r = np.sqrt(X**2.+Y**2.)
    hz = VY*X-Y*VX
    a = (2./r - (VX**2.+VY**2.)/(Gn*(1.+mu)))**(-1.)
    e = 1 - hz*hz/(Gn*(1.+mu)*a)
    e = np.sqrt(e) if e >= 0 else -1.0 
    q = a * (1-e)
    return q-Rn

def col2(t, Y):
    x,y,vx,vy = Y[0:4] 
    r = np.sqrt(x**2.+y**2.)
    return r-Rn

def parabolic(t, Y):
    x,y,vx,vy = Y[0:4] 
    z = 0.*x
    vz = 0.*vx
#   adicionar aqui a rotina de conversao    
    X,Y,Z,VX,VY,VZ = rotacional_para_inercial(t, x, y, z, vx, vy, vz, lbd)
    r = np.sqrt(X**2.+Y**2.)
    C =  (VX**2.+VY**2.)/2. - Gn*(1.+mu)/r
#    r = np.sqrt(X**2.+Y**2.)
#    hz = VY*X-Y*VX
#    a = (2./r - (VX**2.+VY**2.)/(Gn*(1.+mu)))**(-1.)
#    e = 1 - hz*hz/(Gn*(1.+mu)*a)
#    e = np.sqrt(e) if e >= 0 else -1.0 
    return C-0.0

def ejecao(t, Y):
    x,y,vx,vy = Y[0:4]
    r=np.sqrt(x**2.+y**2.)  
    return r-rejec

col1.terminal = True
col2.terminal = True
parabolic.terminal = True
ejecao.terminal = True


# -------------------------------------------------------------------
# Integrador
# -------------------------------------------------------------------

def safe_norm(vec):
    vec = np.asarray(vec, dtype=float)
    if not np.all(np.isfinite(vec)):
        return np.nan
    scale = np.max(np.abs(vec))
    if scale == 0:
        return 0.0
    return scale * np.sqrt(np.sum((vec/scale)**2))

def orbita(XV0, time):
    # -----------------------------
    # Configuração do Lyapunov
    # -----------------------------
    w0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w0_norm = safe_norm(w0)

    # tamanho do bloco de renormalização
    # podes ajustar; 5 a 20 órbitas costuma funcionar bem
    block_size = 10.0 * 2.0 * np.pi

    # estado inicial completo
    Y_current = np.concatenate((np.asarray(XV0, dtype=float), w0))

    t0 = float(time[0])
    tfinal = float(time[-1])

    # acumuladores
    log_sum = 0.0
    t_global = []
    y_global = []
    lyap_global = []

    event_triggered = False
    event_time = None
    event_state = None
    event_index = None

    current_time = t0

    while current_time < tfinal:
        next_time = min(current_time + block_size, tfinal)

        # t_eval apenas dentro do bloco atual
        mask_block = (time >= current_time) & (time <= next_time)
        t_eval_block = time[mask_block]

        # garantir pelo menos o ponto final do bloco
        if t_eval_block.size == 0 or t_eval_block[-1] < next_time:
            t_eval_block = np.unique(np.append(t_eval_block, next_time))

        sol = solve_ivp(
            eqmotion_with_lyap,
            [current_time, next_time],
            Y_current,
            t_eval=t_eval_block,
            method='RK45',
            rtol=1e-8,
            atol=1e-8,
            events=[col1, ejecao, col2, parabolic]
        )

        # guardar trajetória do bloco
        t_block = sol.t
        y_block = sol.y.T  # shape (npts, 8)

        # construir lyap_t bloco a bloco
        for idx in range(len(t_block)):
            t_now = t_block[idx]
            w_now = y_block[idx, 4:8]
            w_now_norm = safe_norm(w_now)

            if t_now <= 0 or not np.isfinite(w_now_norm) or w_now_norm <= 0:
                lyap_now = np.nan
            else:
                # contribuição acumulada até o início do bloco + crescimento dentro do bloco
                lyap_now = (log_sum + np.log(w_now_norm / w0_norm)) / t_now

            t_global.append(t_now)
            y_global.append(y_block[idx])
            lyap_global.append(lyap_now)

        # checar se houve evento terminal
        for ievt, tev in enumerate(sol.t_events):
            if len(tev) > 0:
                event_triggered = True
                event_time = tev[0]
                event_state = sol.y_events[ievt][0]
                event_index = ievt
                break

        # se houve evento, encerra
        if event_triggered:
            break

        # renormalização do vetor variacional no fim do bloco
        Y_end = sol.y[:, -1].copy()
        w_end = Y_end[4:8]
        w_end_norm = safe_norm(w_end)

        if (not np.isfinite(w_end_norm)) or (w_end_norm <= 0):
            # Lyapunov inválido daqui em diante
            XV_end = Y_end[0:4]
            Y_current = np.concatenate((XV_end, w0))
        else:
            log_sum += np.log(w_end_norm / w0_norm)
            Y_end[4:8] = w_end / w_end_norm * w0_norm
            Y_current = Y_end

        current_time = next_time

    # converter acumulados para array
    t = np.array(t_global, dtype=float)
    Yall = np.array(y_global, dtype=float)
    lyap_t = np.array(lyap_global, dtype=float)

    x = Yall[:, 0]
    y = Yall[:, 1]
    vx = Yall[:, 2]
    vy = Yall[:, 3]

    # valor final do expoente de Lyapunov
    lyap_final = lyap_t[-1] if len(lyap_t) > 0 and np.isfinite(lyap_t[-1]) else np.nan

    # identificar tipo de evento
    col = 0
    collision_string = None

    if event_triggered:
        xev, yev, vxev, vyev = event_state[0:4]

        # ievt = 0 -> col1
        # ievt = 1 -> ejecao
        # ievt = 2 -> col2
        # ievt = 3 -> parabolic
        if event_index in [0, 2]:
            col = 1
        elif event_index in [1, 3]:
            col = 2

        collision_string = "{} {} {} {} {} {} {} {}\n".format(
            XV0[0], XV0[3], col, event_time, xev, yev, vxev, vyev
        )

    # estado final para af, ef
    X, Y, Z, VX, VY, VZ = rotacional_para_inercial(
        t[-1], x[-1], y[-1], 0.0, vx[-1], vy[-1], 0.0, lbd
    )

    r_fin = safe_norm([X, Y, Z])
    v_fin = safe_norm([VX, VY, VZ])
    h_fin = safe_norm(np.cross([X, Y, Z], [VX, VY, VZ], axis=0))

    if (not np.isfinite(r_fin)) or r_fin == 0:
        af = np.nan
        ef = np.nan
    elif (not np.isfinite(v_fin)) or v_fin == 0:
        af = r_fin
        ef = 1.0
    else:
        try:
            energy = (v_fin**2.) / 2. - 1. / r_fin
            if not np.isfinite(energy):
                af = np.nan
                ef = np.nan
            elif energy == 0:
                af = np.inf
                ef = 1.0
            else:
                af = -1. / (2. * energy)
                ef_squared = 1. + 2. * energy * h_fin**2.
                ef = np.sqrt(ef_squared) if ef_squared >= 0 else np.nan
        except Exception:
            af = np.nan
            ef = np.nan

    full_trajectory_data = np.vstack((t, x, y, vx, vy, lyap_t)).T

    return t[-1], col, af, ef, lyap_final, full_trajectory_data, collision_string

# -------------------------------------------------------------------
# SIMULACAO UNITARIA (LOGICA CORRIGIDA)
# -------------------------------------------------------------------
def run_simulation(indices):
    i, j = indices 
    a = a_vec[i]
    e = e_vec[j]
    if distribution==2:
        rng = default_rng()
        a = rng.uniform(a_init, a_end)
        e = rng.uniform(e_init, e_end)
#        a = np.random.uniform(a_init, a_end, 1)[0]
#        e = np.random.uniform(e_init, e_end, 1)[0]
    delta_a = 100.0  # ou np.nan
    delta_e = 1.0  # ou np.nan
    varpi = default_rng().uniform(0.0, 2.*np.pi)
    XYZ = aei_to_xv2(Gn,1.0, mu, [a,e,0.,varpi,0.,0.])
    X0, Y0, Z0, VX0, VY0, VZ0 = XYZ[0]
    x0, y0, z0, vx0, vy0, vz0 = inercial_para_rotacional(0.,X0, Y0, Z0, VX0, VY0, VZ0, lbd)    
    
    # Protecao n
    try:
        n = np.sqrt(Gn*(1.+alpha)/a**3.0) if a>0 else 0
    except:
        n = 0 

    particle_string = None
    collision_string = None
    print_string = ""

    # Logica de seguranca e colisao NaN
    try:
        if np.sqrt(x0**2.+y0**2.) > Rn and n > 0:
            XV0 = [x0, y0, vx0, vy0]
            
            tf, col, af, ef, lyap_final, full_trajectory, col_str = orbita(XV0, time)
            collision_string = col_str
            
            t = full_trajectory[:,0]
            x = full_trajectory[:,1]
            y = full_trajectory[:,2]
            vx = full_trajectory[:,3]
            vy = full_trajectory[:,4]
            z = x *0.
            vz = vx * 0.
            X, Y, Z, VX, VY, VZ = rotacional_para_inercial(t, x, y, z, vx, vy, vz, lbd)

            a_evol = []
            e_evol = []
#            print(a_evol)
            # Laço para calcular os elementos orbitais linha por linha
            for k in range(len(t)):
                xyz_k = [X[k], Y[k], Z[k], VX[k], VY[k], VZ[k]]
                aei_k = xv_to_aei(Gn, 1.0, mu, xyz_k)
                a_evol.append(aei_k[0])
                e_evol.append(aei_k[1])
            delta_a = max(a_evol)-min(a_evol)
            delta_e = max(e_evol)-min(e_evol)
            
            # --- SALVAR TRAJETORIA (OPCIONAL) ---
            if save_xy == 1:
                traj_filename = "trajectories/traj_a_{:.6f}_e_{:.6f}.txt".format(a, e)
                header = "t x y vx vy lyapunov_t"
                np.savetxt(traj_filename, full_trajectory, header=header, fmt="%.10e")

            # --- CORRECAO: Se colidiu/ejetou, Lyapunov eh NaN ---
            if col != 0:
                lyap_final = np.nan

        else:
            # Colisao no t=0
            tf = 0.
            col = 1
            af = a; ef = e
            lyap_final = np.nan # Invalido
    
    except Exception as ex:
        # Crash numerico = Colisao
        tf = 0.
        col = 1
        af = a; ef = e
        delta_a = 100.0
        delta_e = 1.0
        lyap_final = np.nan
        print_string = f"Erro em a={a:.4f} e={e:.4f}: {ex}"

    # Formatacao segura
    particle_string = "{:.6f} {:.6f} {:.4f} {} {:.6f} {:.6f} {:.6e} {:.6f} {:.6e} {:.6e}\n".format(
        a, e, varpi*180./np.pi, tf, int(col), af, ef, 1./lyap_final, delta_a, delta_e
    )
    
    # Reduzindo prints para nao lotar log do cluster
    if print_string: print(print_string)
        
    return particle_string, collision_string, ""



# -------------------------------------------------------------------
# Bloco Principal com CHECKPOINT (Resume de onde parou)
# -------------------------------------------------------------------
if __name__ == "__main__":

    # Configurações de arquivos
    file_part = "particles.txt"
    file_col = "colisao.txt"
    traj_dir = "trajectories"

    # Se salvar trajetorias, cria a pasta
    if save_xy == 1:
        pathlib.Path(traj_dir).mkdir(parents=True, exist_ok=True)
        print(f"Salvando trajetorias em ./{traj_dir}/")

    # --- PASSO 1: Descobrir o que ja foi feito ---
    processed_pairs = set()
    mode = 'w' # Modo padrao: write (sobrescrever)
    
    if os.path.exists(file_part):
        print("Arquivo particles.txt encontrado. Verificando progresso anterior...")
        mode = 'a' # Mudar para append (adicionar ao final)
        
        with open(file_part, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                try:
                    parts = line.split()
                    # Lemos 'a' e 'e' e arredondamos para 6 casas (mesma formatação da saida)
                    # para garantir que a comparação funcione
                    a_done = round(float(parts[0]), 6)
                    e_done = round(float(parts[1]), 6)
                    processed_pairs.add((a_done, e_done))
                except:
                    pass # Ignora linhas corrompidas
        
        print(f"Encontradas {len(processed_pairs)} particulas ja processadas.")

    # --- PASSO 2: Preparar arquivos ---
    # Se for 'w' (começar do zero), escreve o cabeçalho.
    # Se for 'a' (resumir), não escreve cabeçalho de novo.
#    if mode == 'w':
#        with open(file_part, "w", encoding='utf-8') as s:
#            s.write("# a e tf col af ef lyapunov_final\n")
#        with open(file_col, 'w', encoding='utf-8') as f:
#            pass # Limpa arquivo de colisão

    # --- PASSO 3: Filtrar o Grid ---
    full_indices_grid = list(product(range(Npart), range(Npart)))
    tasks_to_run = []

    for idx in full_indices_grid:
        i, j = idx
        a_val = round(a_vec[i], 6)
        e_val = round(e_vec[j], 6)
        
        # Se esse par (a,e) NÃO estiver no set de processados, adiciona na fila
        if (a_val, e_val) not in processed_pairs:
            tasks_to_run.append(idx)

    # --- PASSO 4: Executar ---
    try:
        num_cpus = int(os.environ['PBS_NP'])
    except KeyError:
        num_cpus = mp.cpu_count()

    num_cpus = 15

    total_tasks = len(tasks_to_run)
    
    if total_tasks == 0:
        print("Todas as particulas ja foram simuladas! Nada a fazer.")
    else:
        print(f"Total no grid: {len(full_indices_grid)}. Restantes: {total_tasks}.")
        print(f"Iniciando simulacao em {num_cpus} processos...")
        
        with open(file_part, "a", encoding='utf-8') as f_part, \
             open(file_col, "a", encoding='utf-8') as f_col, \
             mp.Pool(processes=num_cpus) as pool:
            
            # Nota: Usamos tasks_to_run em vez de indices_grid
            results_iterator = pool.imap_unordered(run_simulation, tasks_to_run)
            
            count = 0
            for result in results_iterator:
                p_str, c_str, _ = result
                if p_str: f_part.write(p_str)
                if c_str: f_col.write(c_str)
                
                # Forçar escrita no disco a cada resultado (flush) para garantir
                # que se cair a luz agora, a linha esteja salva.
                f_part.flush()
                f_col.flush()
                
                count += 1
                if count % num_cpus == 0: # Feedback mais frequente para teste
                    print(f"Progresso atual: {count}/{total_tasks}")
        
        print(f"Simulacao concluida!")
