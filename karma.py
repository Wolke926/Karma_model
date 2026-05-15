import numpy as np
import math
import random
import os
import numba
from numba import cuda
import time
import argparse
import matplotlib.pyplot as plt
import time


@cuda.jit
def applyBC(N,M,field):
    i, j = cuda.grid(2)

    if i < N and j < M:
        if i == 0:
            field[i, j] = field[N-2, j]

        elif i == N-1:
            field[i, j] = field[1, j]

        if j == 0:
            field[i, j] = field[i, 1]

        elif j == M-1:
            field[i, j] = field[i, M-2]

@cuda.jit
def init_phi(N,M,dx,phi,posArr,W):
    i,j = cuda.grid(2)
    if i > 0 and i < N-1:
        if j > 0 and j < M-1:

            x_p = j*dx
            xpos = posArr[i]

            phi[i, j] = - math.tanh((x_p - xpos) / (math.sqrt(2.0) * W))

            if phi[i, j] > 1.0:
                phi[i, j] = 1.0
            elif phi[i, j] < -1.0:
                phi[i, j] = -1.0

@cuda.jit()
def init_c(N,M,c,phi,ke,c0,noise_amp):
    i,j = cuda.grid(2)
    if i > 0 and i < N-1:
        if j > 0 and j < M-1:
            phi_ = phi[i, j]

            c_ = c0 * ( (1 - phi_)/2.0 + ke * (1 + phi_)/2.0 )

            c[i, j] = c_

@cuda.jit(device =True)      
def gibbsL(X,T,R,VM):
    Y  =  1 - X 
    gibbsliq_Cr =  15483.015 + 146.059775 * T - 26.908 * T * math.log(T) + 1.89435E-3*T**2 - 1.47721E-6*T**3 + 139250/T + 237.615E-23*T**7
    gibbsliq_Co =  15395.278 + 124.434078*T-25.0861*T*math.log(T) - 2.654739E-3*T**2 - 0.17348E-6*T**3+ 72527/T - 219.801E-23*T**7

    liq_L_0 = -12538.69 + 2.8471 * T
    liq_L_1 = -6416.82 + 1.1284 * T
    liq_L_2 = 4589.87
    
    id = R * T * (X * math.log(X) + Y * math.log(Y))+ (gibbsliq_Co * X + gibbsliq_Cr * Y)
    term_ex = X * Y *(liq_L_0 + liq_L_1 * (X-Y) + liq_L_2 * (X-Y)**2 )

    return (id + term_ex)/VM

@cuda.jit(device =True)
def gibbsS(X,T,R,VM): 
    Y = 1-X
    gibbsBCC_Cr =  -8856.94 + 157.48 * T - 26.908 * T * math.log(T) + 1.89435E-3*T**2 - 1.47721E-6*T**3 + 139250/T
    gibbsBCC_Co =  3248.241 + 132.65221*T - 25.0861*T*math.log(T) - 2.654739E-3*T**2 - 0.17348E-6*T**3 + 72527/T

    BCC_L_0 = -6204.65 +2.3246 * T
    BCC_L_1 = - 4427.40 - 2.2830 * T

    id = R * T * (X * math.log(X) + Y * math.log(Y))+ (gibbsBCC_Co * X + gibbsBCC_Cr * Y)
    term_ex = X * Y *(BCC_L_0 + BCC_L_1 * (X-Y) )

    return (id + term_ex)/VM

@cuda.jit(device =True)
def DgibbsL(X,T,R,VM):
    Y  =  1 - X
 
    gibbsliq_Cr =  15483.015 + 146.059775 * T - 26.908 * T * math.log(T) + 1.89435E-3*T**2 - 1.47721E-6*T**3 + 139250/T + 237.615E-23*T**7
    gibbsliq_Co =  15395.278 + 124.434078*T-25.0861*T*math.log(T) - 2.654739E-3*T**2 - 0.17348E-6*T**3+ 72527/T - 219.801E-23*T**7

    liq_L_0 = -12538.69 + 2.8471 * T
    liq_L_1 = -6416.82 + 1.1284 * T
    liq_L_2 = 4589.87

    D_id = gibbsliq_Co - gibbsliq_Cr + R * T * (math.log(X) - math.log(Y))
    term_ex = X * Y *(liq_L_0 + liq_L_1 * (X-Y) + liq_L_2 * (X-Y)**2 )
    D_ex = (1 - 2*X)*(liq_L_0 + liq_L_1 * (X-Y) + liq_L_2 * (X-Y)**2 )+ X*Y*(2*liq_L_1 + 4*liq_L_2*(X*2-1))

    return (D_id + D_ex) / VM

@cuda.jit(device =True)
def DgibbsS(X,T,R,VM):
    Y = 1-X
    gibbsBCC_Cr =  -8856.94 + 157.48 * T - 26.908 * T * math.log(T) + 1.89435E-3*T**2 - 1.47721E-6*T**3 + 139250/T
    gibbsBCC_Co =  3248.241 + 132.65221*T - 25.0861*T*math.log(T) - 2.654739E-3*T**2 - 0.17348E-6*T**3 + 72527/T

    BCC_L_0 = -6204.65 +2.3246 * T
    BCC_L_1 = - 4427.40 - 2.2830 * T

    D_id = gibbsBCC_Co - gibbsBCC_Cr + R * T * (math.log(X) - math.log(Y))
    D_ex = (1 - 2*X)*(BCC_L_0 + BCC_L_1 * (X-Y)) + X*Y*(2*BCC_L_1 )

    return (D_id + D_ex) / VM


def DgibbsL_dT(X,T,R,VM):
    Y = 1.0 - X

    dG_liq_Cr_dT = (
        146.059775
        - 26.908 * (math.log(T) + 1.0)
        + 2.0 * 1.89435e-3 * T
        - 3.0 * 1.47721e-6 * T**2
        - 139250.0 / T**2
        + 7.0 * 237.615e-23 * T**6
    )
    liq_L_0 = -12538.69 + 2.8471 * T
    liq_L_1 = -6416.82 + 1.1284 * T
    liq_L_2 = 4589.87
    dG_liq_Co_dT = (124.434078- 25.0861 * (math.log(T) + 1.0)- 2.0 * 2.654739e-3 * T- 3.0 * 0.17348e-6 * T**2
        - 72527.0 / T**2 - 7.0 * 219.801e-23 * T**6)

    DidDT = R * (X * math.log(X) + Y * math.log(Y))
    DexDT = X*Y*(2.8471 + 1.1284* (X-Y))

    return (DidDT + X*dG_liq_Co_dT + Y*dG_liq_Cr_dT + DexDT) / VM
    
  
def DgibbsS_dT(X,T,R,VM):
    Y = 1-X
    gibbsBCC_Cr =  -8856.94 + 157.48 * T - 26.908 * T * math.log(T) + 1.89435E-3*T**2 - 1.47721E-6*T**3 + 139250/T
    gibbsBCC_Co =  3248.241 + 132.65221*T - 25.0861*T*math.log(T) - 2.654739E-3*T**2 - 0.17348E-6*T**3 + 72527/T

    dG_BCC_Cr_dT = (
        157.48
        - 26.908 * (math.log(T) + 1.0)
        + 2.0 * 1.89435e-3 * T
        - 3.0 * 1.47721e-6 * T**2
        - 139250.0 / T**2
    )

    dG_BCC_Co_dT = (
        132.65221
        - 25.0861 * (math.log(T) + 1.0)
        - 2.0 * 2.654739e-3 * T
        - 3.0 * 0.17348e-6 * T**2
        - 72527.0 / T**2
    )

    BCC_L_0 = -6204.65 +2.3246 * T
    BCC_L_1 = - 4427.40 - 2.2830 * T

    # ideal mixing entropy term
    d_id_dT = ( R * (X * math.log(X) + Y * math.log(Y)) + X * dG_BCC_Co_dT+ Y * dG_BCC_Cr_dT)

    d_ex_dT = X * Y * (2.3246+(- 2.2830)* (X-Y))

    return (d_id_dT + d_ex_dT) / VM
    

@cuda.jit(device =True)
def g_func(phi):
    return 15/8 * (phi - 2/3 * phi**3 + 1/5 * phi**5)

@cuda.jit(device =True)
def d_g_func(phi):
    res = 1.875 - phi * phi * (3.75 - 1.875 * phi * phi)
    return res

@cuda.jit(device =True)
def q_func(A,phi,qs):

    y = 0.5 * (1.0 - phi)
    q_old = A*y - (A - 1.0)*y*y

    q = qs + (1.0 - qs)*q_old

    return q

@cuda.jit()
def solveC(N,M,phi,c,c_next,TEMP,tau_0,eps_s,eps_k,S, W0, dx, dt,ke, GIBBSTHOMSON, mu_k0,a01,A,Dl,Tm,h0,R,Vs,VM,qs):
    i,j = cuda.grid(2)
    if i < 1 or i >= N - 1 :
        return
    else:
        if j < 1 or j >= M-1:
            return
        else:
            phi_c = phi[i,j]
            phi_l  = phi[i-1,j]
            phi_r  = phi[i+1,j]
            phi_b  = phi[i,j-1]
            phi_t  = phi[i,j+1]
            phi_lb = phi[i-1,j-1]
            phi_lt = phi[i-1,j+1]
            phi_rb = phi[i+1,j-1]       
            phi_rt = phi[i+1,j+1] 

            c_c    = c[i,j]
            c_l  = c[i-1,j]
            c_r  = c[i+1,j]
            c_b  = c[i,j-1]
            c_t  = c[i,j+1]
            c_lb = c[i-1,j-1]
            c_lt = c[i-1,j+1]
            c_rb = c[i+1,j-1]       
            c_rt = c[i+1,j+1] 

            T_c = TEMP[i,j]
            T_l  = TEMP[i-1,j]
            T_r  = TEMP[i+1,j]
            T_b  = TEMP[i,j-1]
            T_t  = TEMP[i,j+1]
            T_lb = TEMP[i-1,j-1]
            T_lt = TEMP[i-1,j+1]
            T_rb = TEMP[i+1,j-1]       
            T_rt = TEMP[i+1,j+1] 
            
            
            beta_c = (
                     (1+g_func(phi_c))/(2*h0)*DgibbsS(c_c,T_c,R,VM) 
                        + (1-g_func(phi_c))/(2*h0)*DgibbsL(c_c,T_c,R,VM) 
                    )
            beta_l = (
                     (1+g_func(phi_l))/(2*h0)*DgibbsS(c_l,T_l,R,VM) 
                        + (1-g_func(phi_l))/(2*h0)*DgibbsL(c_l,T_l,R,VM)
                    )
            
            beta_r = ((1+g_func(phi_r))/(2*h0)*DgibbsS(c_r,T_r,R,VM) 
                        + (1-g_func(phi_r))/(2*h0)*DgibbsL(c_r,T_r,R,VM) )
            
            beta_b =  (
                        (1+g_func(phi_b))/(2*h0)*DgibbsS(c_b,T_b,R,VM) 
                        + (1-g_func(phi_b))/(2*h0)*DgibbsL(c_b,T_b,R,VM) 
                        ) 
            beta_t =  (
                (1+g_func(phi_t))/(2*h0)*DgibbsS(c_t,T_t,R,VM) 
                        + (1-g_func(phi_t))/(2*h0)*DgibbsL(c_t,T_t,R,VM) 
            )
            beta_lb =  (
                        (1+g_func(phi_lb))/(2*h0)*DgibbsS(c_lb,T_lb,R,VM) 
                        + (1-g_func(phi_lb))/(2*h0)*DgibbsL(c_lb,T_lb,R,VM) 
                    )
            beta_rb = ((1+g_func(phi_rb))/(2*h0)*DgibbsS(c_rb,T_rb,R,VM) 
                        + (1-g_func(phi_rb))/(2*h0)*DgibbsL(c_rb,T_rb,R,VM) 
                    )
            beta_lt = ((1+g_func(phi_lt))/(2*h0)*DgibbsS(c_lt,T_lt,R,VM) 
                        + (1-g_func(phi_lt))/(2*h0)*DgibbsL(c_lt,T_lt,R,VM) 
                        )
            beta_rt = ((1+g_func(phi_rt))/(2*h0)*DgibbsS(c_rt,T_rt,R,VM) 
                        + (1-g_func(phi_rt))/(2*h0)*DgibbsL(c_rt,T_rt,R,VM) 
                     )

            alpha_c  = Dl * c_c * (1-c_c) * q_func(A,phi_c,qs)
            alpha_l  = Dl * c_l * (1-c_l) * q_func(A,phi_l,qs)
            alpha_r  = Dl * c_r * (1-c_r) * q_func(A,phi_r,qs)
            alpha_b  = Dl * c_b * (1-c_b) * q_func(A,phi_b,qs)
            alpha_t  = Dl * c_t * (1-c_t) * q_func(A,phi_t,qs)
            alpha_lb = Dl * c_lb * (1-c_lb)* q_func(A,phi_lb,qs)
            alpha_lt = Dl * c_lt * (1-c_lt)* q_func(A,phi_lt,qs)   
            alpha_rb = Dl * c_rb * (1-c_rb)* q_func(A,phi_rb,qs)                              
            alpha_rt = Dl * c_rt* (1-c_rt) * q_func(A,phi_rt,qs)   

            alpha_hrt = 0.25 * ( alpha_c + alpha_r + alpha_t + alpha_rt ) 
            alpha_hrb = 0.25 * ( alpha_c + alpha_r + alpha_b + alpha_rb ) 
            alpha_hlt = 0.25 * ( alpha_c + alpha_l + alpha_t + alpha_lt )
            alpha_hlb = 0.25 * ( alpha_c + alpha_l + alpha_b + alpha_lb )  
            
            flx_hl = 0.25 * (alpha_c + alpha_l + alpha_hlt + alpha_hlb) * (beta_l - beta_c)
            flx_hr = 0.25 * (alpha_c + alpha_r + alpha_hrt + alpha_hrb) * (beta_r - beta_c)
            flx_hb = 0.25 * (alpha_c + alpha_b + alpha_hlb + alpha_hrb) * (beta_b - beta_c)        
            flx_ht = 0.25 * (alpha_c + alpha_t + alpha_hlt + alpha_hrt) * (beta_t - beta_c)        
  
            flx_hlt = alpha_hlt * (beta_lt - beta_c)        
            flx_hlb = alpha_hlb * (beta_lb - beta_c)
            flx_hrt = alpha_hrt * (beta_rt - beta_c)            
            flx_hrb = alpha_hrb * (beta_rb - beta_c)  

            div01 = (2.0/3.0) * (flx_hl + flx_hr + flx_hb + flx_ht)
            div10 = (1.0/6.0) * (flx_hlt + flx_hlb + flx_hrt + flx_hrb) 
            dcdt = (div01+div10)/(dx*dx)

            c_next[i,j] = c_c + dt * dcdt
            if c_next[i,j] < 0.0:
                c_next[i,j] = 1e-8
            elif c_next[i,j] > 1.0:
                c_next[i,j] = 1-1e-8

@cuda.jit()
def solvePhi(N,M,phi,phi_next,c,TEMP,tau_0,eps_s,eps_k,S, W0, dx, dt,ke, GIBBSTHOMSON, mu_k0,a01,Tm,W,R,Vs,VM,h):
    i,j = cuda.grid(2)
    if i < 1 or i >= N - 1 :
        return
    else:
        if j < 1 or j >= M-1:
            return
        else:
            phi_c = phi[i,j]
            c_c    = c[i,j]
            T = TEMP[i,j]
            phi_l  = phi[i-1,j]
            phi_r  = phi[i+1,j]
            phi_b  = phi[i,j-1]
            phi_t  = phi[i,j+1]
            phi_lb = phi[i-1,j-1]
            phi_lt = phi[i-1,j+1]
            phi_rb = phi[i+1,j-1]       
            phi_rt = phi[i+1,j+1] 

            g_phi   = g_func(phi_c)
            d_g_phi = d_g_func(phi_c)

            phix = ( phi_r - phi_l + 0.25 * (phi_rt + phi_rb - phi_lt - phi_lb) ) / (3.*dx) #(gradx,grady) = gradphi
            phiy = ( phi_t - phi_b + 0.25 * (phi_rt + phi_lt - phi_rb - phi_lb) ) / (3.*dx)
            
            G10 = ( phi_r  - phi_l  ) * ( phi_r  - phi_l  ) + ( phi_t  - phi_b  ) * ( phi_t  - phi_b  )
            G01 = ( phi_rt - phi_lb ) * ( phi_rt - phi_lb ) + ( phi_rb - phi_lt ) * ( phi_rb - phi_lt )       
            
            #NormG2=phx*phx+phy*phy
            NormG2 = ( 4. * G10 + G01 ) /( 24. * dx * dx ) # |gradphi|^2
            lap_phi = ( 4.0 * (phi_r + phi_l + phi_t + phi_b) + phi_rt + phi_rb + phi_lt + phi_lb - 20.0*phi_c) / (6.0*dx*dx)      

            if (abs(NormG2)>1.e-15):# //where｜grad phi｜> 0,is interface
            
                dphix=phix #phx*cAlpha[pha]+phy*sAlpha[pha] 
                dphiy=phiy #-phx*sAlpha[pha]+phy*cAlpha[pha] 
                
                phxx =      ( (phi_rt - 2.0 * phi_t + phi_lt )   + 
                       10.0 * (phi_r  - 2.0 * phi_c + phi_l  )   +
                              (phi_rb - 2.0 * phi_b + phi_lb ) ) / (12.*dx*dx)   #2nd deriv             
                phyy =      ( (phi_rt - 2.0 * phi_r + phi_rb )   + 
                       10.0 * (phi_t  - 2.0 * phi_c + phi_b  )   +
                              (phi_lt - 2.0 * phi_l + phi_lb ) ) / (12.*dx*dx)                               
                phxy = ( phi_rt - phi_lt - phi_rb + phi_lb ) / (4.*dx*dx) 
                
                dphixx= phxx #phyy*sAlpha2[pha]+phxy*s2Alpha[pha]+phxx*cAlpha2[pha] 
                dphiyy= phyy # phyy*cAlpha2[pha]-phxy*s2Alpha[pha]+phxx*sAlpha2[pha] 
                dphixy= phxy #phyy*cAlphasAlpha[pha]+phxy*c2Alpha[pha]-phxx*cAlphasAlpha[pha] 
				
                NormG4 = NormG2 * NormG2
                thx = (dphix*dphixy-dphiy*dphixx)/NormG2 #dtheta_dx, theta = tan-1(dphiy_dphix)
                thy = (dphix*dphiyy-dphiy*dphixy)/NormG2
                c4 = 1.0 - 8.0*dphix*dphix*dphiy*dphiy / NormG4                       #  cos 4 theta
                s4 = 4.0*(dphix*dphix*dphix*dphiy - dphiy*dphiy*dphiy*dphix) / NormG4  # sin 4theta
                
                
                anis= (c4*(2 +eps_s*c4)* lap_phi                               #  (1/eps) (a^2-1) lap(phi)
                       - 8.0*s4*(1.0 + eps_s*c4)*(thx*dphix + thy*dphiy)                #  (1/eps) (2a ap) (phix thx + phiy thy)
                      - 16.0*(c4 + eps_s*(c4*c4 - s4*s4))*(thy*dphix - thx*dphiy)  )        #  (1/eps) (app a + ap^2) (phix thy - phiy thx)  
                anis *= eps_s 
                    
            else:
                
                anis=0
                c4=0
            
            a_k = 1.0 + eps_k * c4
            a_s = 1.0 + eps_s * c4
            tau = tau_0 * a_s * a_s / a_k

            

            part1 =  S*W0*S*W0 * (lap_phi + anis)
            part2= phi_c * (1.0 - phi_c * phi_c)
            part3 = - 1/(2*h)*d_g_func(phi_c)* gibbsS(c_c,T,R,VM)
            part4 = 1/(2*h)*d_g_func(phi_c)* gibbsL(c_c,T,R,VM)
            dphi_dt = part1 + part2 + part3 + part4
         
            dphi_dt = dphi_dt/tau
            phi_next[i,j] = phi_c + dt * dphi_dt 


        
@cuda.jit()
def updateAll(N,M,phi,phi_next,c,c_next):
    i,j = cuda.grid(2)
    if 1 <= i <= N-2 and 1 <= j <= M-2:
        phi[i, j] = phi_next[i, j]
        c[i, j] = c_next[i, j]
        


@cuda.jit
def PullBack(N, M,phi, c, c0):
    i,j = cuda.grid(2)

    if i < N:
        if j < (M - 1):
            phi[i, j] = phi[i, j + 1]
            c[i, j] = c[i, j + 1]
           
            # right boundary: new liquid
        phi[i, M - 1] = -1.0
        c[i, M - 1] = c0


@cuda.jit()
def Tarrupdate(TEMP,N,M,Ts,G,Vs,xoffs,t,dx,initXpos):
    i,j = cuda.grid(2)
    if i < N and i >= 0:
        if j < M and j >= 0:
            TEMP[i, j] = Ts + G * (j * dx - initXpos * dx + xoffs - Vs * t)


@cuda.jit
def check_pullback(phi, flag, thresh_ind, N):
    i = cuda.grid(1)

    if i < N:
        if phi[i, thresh_ind] > 0.1:
            cuda.atomic.max(flag, 0, 1) #flag = 1, need pullback,flag[0] = max(flag[0], 1)

def main():
    parser = argparse.ArgumentParser(description = "karma model")
    parser.add_argument("--G",type = float,help = "temp grad")
    parser.add_argument("--Vs",type = float,help = "pulling rate")
    
    args = parser.parse_args()
    G = args.G
    Vs = args.Vs
    S = 1
    A = 1
    foldername = f"{G}_{Vs}_{A}_{S}"
    filename = "dendrite.csv"
  
    os.makedirs(foldername, exist_ok=True)
    N = 2000
    M = 2000

    xoffs = 0
    R = 8.314
    VM = (0.6* 7.19 + 0.4 * 8.9)*10**(-6)
    c0 = 0.4

    ke = 0.84
    Ts = 1751.0
    Tm = 1820

    Rt = 0.6
    
    mapsize = np.array([N,M],dtype='int32')

    initXpos = 0.6* M

    #coefs
    Dl =  3.647994e-9 #m^2/s
    Ds = 1.813807e-11
    qs = Ds/Dl
    GIBBSTHOMSON = 3.47e-7 #Km

    W0 = 1e-9 # 1nm
    dx = 0.6* S *W0 # 4e-9
    W = S * W0 #width
    eps_k =  0.13
    eps_s = 0.018
    n_fold = 4
    mu_k0 = 0.3
    tau_0 =(S*W0)*(S*W0)/(GIBBSTHOMSON*mu_k0) #1/(K_phi * h) 

    dt = 0.25 * tau_0 * Rt * (dx/(S*W0))**2 #dt = 0.25*(S*W0)*(S*W0)/(GIBBSTHOMSON*mu_k0)*Rt*(dx/(S*W0))*(dx/(S*W0)) 
    
    a01 = 2*math.sqrt(2)/3
    
    h0 = R * Tm/VM

    phi_h = np.ones((N,M),dtype = np.float64)
    c_h = np.ones((N,M),dtype = np.float64) * c0
    phi = cuda.to_device(phi_h)
    c = cuda.to_device(c_h)

    TEMP_h = np.zeros((N,M),dtype=np.float64)
    for i in range(N):
        for j in range(M):
            TEMP_h[i,j] = Ts + G*(j*dx - initXpos*dx)

    TEMP = cuda.to_device(TEMP_h)
    posArr_h = initXpos*dx + (np.random.rand(N) - 0.5) * dx
    posArr = cuda.to_device(posArr_h)

    phi_next = cuda.to_device(np.zeros((N, M),dtype = np.float64))
    c_next = cuda.to_device(np.zeros((N, M),dtype = np.float64))
    n_steps = 2000000000
    n_save = 500000
    
    threadsperblock=(16, 16)
    blockspergrid_x = math.ceil(N/threadsperblock[0])
    blockspergrid_y = math.ceil(M/threadsperblock[1])
    blockspergrid = (blockspergrid_x,blockspergrid_y)
    init_phi[blockspergrid,threadsperblock](N,M,dx,phi,posArr,W)
    applyBC[blockspergrid,threadsperblock](N,M,phi)
    init_c[blockspergrid,threadsperblock](N,M,c,phi,ke,c0,0.0)
    applyBC[blockspergrid,threadsperblock](N,M,c)

    flag_h = np.zeros(1, dtype=np.int32)
    flag_d = cuda.to_device(flag_h)

    xoffs = 0.0
    thresh = initXpos
    thresh_ind = int(math.floor(thresh))
    print(f"------- Simulation Start ( Karma Model )-------")
    print(f"G= {G}, R = {Vs}, dx = {dx},dt = {dt},Ds = {Ds},Dl = {Dl}")
    dh_f = Tm * (DgibbsS_dT(1e-8,Tm,R,VM) - DgibbsL_dT(1e-8,Tm,R,VM))
    gamma_0 = (GIBBSTHOMSON * dh_f) /Tm
    h = gamma_0/( a01 * W)
    with open(f"{foldername}/{filename}","a") as filestream:
        filestream.write(f"G= {G}, R = {Vs}, dx = {dx},dt = {dt},A = {A}\n")
    for iteration in range(n_steps):
        cuda.synchronize()

        solvePhi[blockspergrid,threadsperblock] (N,M,phi,phi_next,c,TEMP,tau_0,eps_s,eps_k,S, W0, dx, dt,ke, GIBBSTHOMSON, mu_k0,a01,Tm,W,R,Vs,VM,h)
        
        cuda.synchronize()
        solveC[blockspergrid,threadsperblock] (N,M,phi,c,c_next,TEMP,tau_0,eps_s,eps_k,S, W0, dx, dt,ke, GIBBSTHOMSON, mu_k0,a01,A,Dl,Tm,h0,R,Vs,VM,qs)
        
        cuda.synchronize()
        updateAll[blockspergrid,threadsperblock](N,M,phi,phi_next,c,c_next)
        applyBC[blockspergrid,threadsperblock](N,M,phi)
        applyBC[blockspergrid,threadsperblock](N,M,c)

        flag_h[0] = 0
        flag_d.copy_to_device(flag_h)

        # check phi[:, thresh_ind]
        threads_1d = 256
        blocks_1d = math.ceil(N / threads_1d)

        check_pullback[blocks_1d, threads_1d](phi, flag_d, thresh_ind, N)

        # only copy one integer back
        flag_d.copy_to_host(flag_h)

        if flag_h[0] == 1:
            PullBack[blockspergrid,threadsperblock](N, M, phi, c, c0)
            xoffs += dx

        t = dt * iteration
        Tarrupdate[blockspergrid,threadsperblock](TEMP,N,M,Ts,G,Vs,xoffs,t,dx,initXpos)
        
        if iteration % n_save == 0:
            phi.copy_to_host(phi_h)
            c.copy_to_host(c_h)
            TEMP.copy_to_host(TEMP_h)

            phi_int = 0
            for i in range(1,N-1):
                phiinterface_ind = np.argmin(np.abs(phi_h[i,:])) #return the min index position
                if phiinterface_ind > phi_int:
                    phi_int = phiinterface_ind

            T_int = TEMP_h[5,phi_int]

            with open(f"{foldername}/{filename}","a") as filestream:
                filestream.write(f"{iteration},{T_int},{xoffs}\n")
            print(f"Step = {iteration}, Temp_interface = {T_int}, Xoffs_grids = {xoffs/dx}, Growth_dist = {xoffs} ")
            np.savetxt(f"{foldername}/phi_{iteration}.csv",phi_h,delimiter=",")
            np.savetxt(f"{foldername}/c_{iteration}.csv",c_h,delimiter=",")
            np.savetxt(f"{foldername}/T_{iteration}.csv",TEMP_h,delimiter=",")
            
            filepath = os.path.join(foldername, f"iteration_{iteration}.png")

            plt.figure(figsize=(10,4))

            plt.subplot(1,2,1)
            plt.title("phi")
            plt.imshow(phi_h, cmap='jet', vmin=-1, vmax=1)
            plt.colorbar()

            plt.subplot(1,2,2)
            plt.title("c_h")
            plt.imshow(c_h, cmap='viridis', vmin=0.2, vmax=0.6)
            plt.colorbar()
            plt.savefig(filepath)
            plt.close()


if __name__ == "__main__":
    main()