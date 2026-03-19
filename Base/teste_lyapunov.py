# -*- coding: utf-8 -*-
import multiprocessing as mp
import numpy as np
from scipy.integrate import solve_ivp
import os
import pathlib
from itertools import product
import sys

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

############ Values (normalized) ##############
Gn=1.0
mu=M2/MT 
Rn=1. 
asinc=((G*M1/w**2.)**(1./3.))/R
lbd=w/wk
x1,y1=-mu,0.
x2,y2=Rn-mu,0.

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
Npart=int(parameter.get('Npart', 10)) 
a_vec=np.linspace(a_init,a_end+1e-6,Npart) 
e_vec=np.linspace(e_init,e_end,Npart) 

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

def force(t,XVl):    
    x,y,vx,vy=XVl
    xn1,yn1,r1=r_to_b1(XVl)
    xn2,yn2,r2=r_to_b2(XVl)
    Fx=xn1/r1**3.+alpha*xn2/r2**3.
    Fy=yn1/r1**3.+alpha*yn2/r2**3.
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
# Eventos
# -------------------------------------------------------------------
def col1(t, Y):
    x,y,vx,vy = Y[0:4] 
    xn1,yn1,r1=r_to_b1([x,y,vx,vy])
    return r1-Rn

def ejecao(t, Y):
    x,y,vx,vy = Y[0:4]
    r=np.sqrt(x**2.+y**2.)  
    return r-rejec

col1.terminal = True
ejecao.terminal = True

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

# -------------------------------------------------------------------
# Integrador
# -------------------------------------------------------------------
def orbita(XV0, time):
    w0 = np.array([1.0, 0.0, 0.0, 0.0])
    w0_norm = np.linalg.norm(w0)
    Y0 = np.concatenate((XV0, w0)) 

    sol = solve_ivp(eqmotion_with_lyap, [time[0], time[-1]], Y0, t_eval=time, method='RK45', rtol=1e-8, atol=1e-8, events=[col1, ejecao])

    t = sol.t
    x,y,vx,vy = sol.y[0:4]
    w_vec = sol.y[4:8]     

    w_norm = np.linalg.norm(w_vec, axis=0)
    lyap_t = np.zeros_like(t)
    
    if len(t) > 1:
        non_zero_t = t[1:] > 0
        safe_w_norm = w_norm[1:][non_zero_t]
        safe_t = t[1:][non_zero_t]
        if safe_w_norm.size > 0:
             lyap_t[1:][non_zero_t] = (1.0 / safe_t) * np.log(safe_w_norm / w0_norm)
    
    lyap_final = lyap_t[-1]
    
    tcol,tejecao = sol.t_events
    Ycol,Yejecao = sol.t_events 

    col=0
    collision_string = None
    
    if len(tcol)>0:
        col=1
        Yc = Ycol[0] 
        # Usando .format para seguranca
        collision_string = "{} {} {} {} {} {} {} {}\n".format(
            XV0[0], XV0[3], col, tcol[0], Yc[0], Yc[1], Yc[2], Yc[3]
        )
    elif len(tejecao)>0:
        col=2
        Ye = Yejecao[0]
        collision_string = "{} {} {} {} {} {} {} {}\n".format(
            XV0[0], XV0[3], col, tejecao[0], Ye[0], Ye[1], Ye[2], Ye[3]
        )
        
    X,Y,Z,VX,VY,VZ = rotacional_para_inercial(t[-1], x[-1], y[-1], 0, vx[-1], vy[-1], 0, lbd)
    r_fin = np.linalg.norm([X,Y,Z])
    v_fin = np.linalg.norm([VX,VY,VZ])
    h_fin = np.linalg.norm(np.cross([X,Y,Z], [VX,VY,VZ], axis=0))
    
    if r_fin == 0: af=0; ef=0
    elif v_fin == 0: af=r_fin; ef=1.0 
    else:
        try:
            energy = (v_fin**2.) / 2. - 1./r_fin 
            if energy == 0: af=np.inf; ef=1.0
            else:
                 af = -1. / (2. * energy)
                 ef_squared = 1 + 2 * energy * h_fin**2.
                 ef = np.sqrt(ef_squared) if ef_squared >= 0 else -1.0 
        except Exception:
            af=0; ef=-1 
    
    full_trajectory_data = np.vstack((t, x, y, vx, vy, lyap_t)).T

    return t[-1], col, af, ef, lyap_final, full_trajectory_data, collision_string

# -------------------------------------------------------------------
# SIMULACAO UNITARIA (LOGICA CORRIGIDA)
# -------------------------------------------------------------------
def run_simulation(indices):
    i, j = indices 
    a = a_vec[i]
    e = e_vec[j]
    x0 = a*(1.+e)
    
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
        if x0 > Rn and n > 0:
            vy0 = n*a*np.sqrt((1.-e)/(1.+e)) - lbd*x0
            XV0 = [x0, 0., 0., vy0]
            
            tf, col, af, ef, lyap_final, full_trajectory, col_str = orbita(XV0, time)
            collision_string = col_str
            
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
        lyap_final = np.nan
        print_string = f"Erro em a={a:.4f} e={e:.4f}: {ex}"

    # Formatacao segura
    particle_string = "{:.6f} {:.6f} {:.4f} {} {:.6f} {:.6f} {:.6e}\n".format(
        a, e, tf, int(col), af, ef, lyap_final
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
    if mode == 'w':
        with open(file_part, "w", encoding='utf-8') as s:
            s.write("# a e tf col af ef lyapunov_final\n")
        with open(file_col, 'w', encoding='utf-8') as f:
            pass # Limpa arquivo de colisão

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
                if count % 10 == 0: # Feedback mais frequente para teste
                    print(f"Progresso atual: {count}/{total_tasks} (Total acumulado: {len(processed_pairs)+count})")
        
        print(f"Simulacao concluida!")