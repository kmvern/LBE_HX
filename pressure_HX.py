#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:44:08 2020

@author: kelleyverner
"""


import math
import numpy as np
import pandas as pd
import tube_geometry

# Below are the measurements for standard tube sizing

tube_OD_inch = [0.25, 0.375, 0.5, 0.625, 0.75,
                0.875, 1, 1.125, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
                3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25]

tube_name = ['T-' + name for name in map(str, tube_OD_inch)]

tube_OD = [0.00635, 0.009525, 0.0127, 0.015875, 0.01905, 0.022225,
           0.0254, 0.028575, 0.03175, 0.0381, 0.04445, 0.0508, 0.05715,
           0.0635, 0.06985, 0.0762, 0.08255, 0.0889, 0.09525, 0.1016,
           0.10795, 0.1143, 0.12065, 0.127, 0.13335]


tube_thickness = [0.009652, 0.008636, 0.00762, 0.0072136, 0.0065786, 0.0060452,
                  0.005588, 0.0051562, 0.004572, 0.004191, 0.0037592, 0.0034036, 0.003048,
                  0.0027686, 0.002413, 0.0021082, 0.0018288, 0.001651, 0.0014732, 0.0012446,
                  0.0010668, 0.000889, 0.0008128, 0.0007112, 0.000635, 0.0005588]

gauge = [00, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
         12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
         23, 24]

# inner tube dimensions must be calculated

# Below are standard pipe sizes.

pipe_name = [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25,
             1.5, 2, 2.5, 3, 3.5, 4, 5, 6]

pipe_name = ['P-' + name for name in map(str, pipe_name)]

pipe_OD = [0.0103, 0.0137, 0.0172, 0.0213, 0.0267, 0.0334, 0.0422,
           0.0483, 0.0603, 0.073, 0.0889, 0.1016, 0.1143, 0.1413, 0.1683]

pipe_ID = [0.0078, 0.01038, 0.01388, 0.01708, 0.02248, 0.02786, 0.03666,
           0.04276, 0.05476, 0.0669, 0.0828, 0.0955, 0.1082, 0.13448, 0.16148]

#Make a dataframe of all the diameters to use. 

org_diameters = pd.DataFrame([tube_OD_inch, tube_OD]).T


Water, LBE, Total = tube_geometry.inner_dia(tube_OD, tube_name, tube_thickness, gauge,
                                   pipe_name, pipe_ID, pipe_OD)

Final = tube_geometry.gas_gap_calcs(Water, LBE).rename(columns = {'OD_gas':'OD_w', 'ID_gas':'ID_w', 'GasGap':'WaterGap'})
 
 
#This is the final collection of the diameters and velocity
last = Final.loc[Final['ID_w']> Final['OD_Pb']]
last = last.loc[last['ID_Pb']>0]
last = last.loc[last['ID_w']>0]

def area_calcs(ID_pb, OD_pb, ID_w, OD_w):
    T_1 = (OD_pb - ID_pb)/2
    T_2 = (OD_w - ID_w)/2
    A_pb = (math.pi*(ID_pb/2)**2)
    A_water = math.pi*((ID_w/2)**2-(OD_pb/2)**2)
    Dh_w = ID_w - OD_pb
    Dc = (OD_pb+OD_pb)/2

    return T_1, T_2, A_pb, A_water, Dc, Dh_w

#Temperatures and geometry calcs for Q calculations
T_1, T_2, Area_pb, A_water, Dc, Dh_w = area_calcs(last['ID_Pb'], 
                last['OD_Pb'], last['ID_w'], last['OD_w'])

areas = (pd.concat([T_1, T_2, Area_pb, A_water, Dc, Dh_w], 
                   axis=1)).rename(columns={0:'T1', 1:'T2', 'ID_Pb':'A_Pb',
                                            2:'A_water', 'OD_Pb':'Dc', 3:'Dh_w'})
last = pd.concat([last, areas], axis = 1)


#find delta T
LBE_MFR = 26.7 #kg/s
T_hout = 200+273 #K
T_cin = 20 #C
Q_req = 100E3 #W
delta_Tcold = 10 #K
delta_Thot, T_hin, T_cin, T_cout = tube_geometry.delta_T(LBE_MFR, T_hout, Q_req, T_cin, delta_Tcold)

print('The required temperature decrease is approximatly', delta_Thot)

T_avg = (T_hin + T_hout)/2

#Properties of the LBE
rho_hot, cp_hot, k_hot, DV_hot, KV_hot = tube_geometry.thermo_prop(T_avg)

#H2O properties 
rho_cold = 999.45 #kg/cm^3
cp_cold = 4175.9 #specific heat water J/kg K
k_cold = 0.59941 #thermal conductivity
DV_cold = 0.0010042 #dynamic viscosity Pa*s
KV_cold = DV_cold/rho_cold

#required water mass flow rate

w_MFR = Q_req/(cp_cold*delta_Tcold)
print('The required mass flow rate of water is ', w_MFR)

def MFR(density, area, velocity):
    MFR = density*area*velocity 
    return MFR
def velocity(density, AWater, mfr):
    velocity = mfr/(density*AWater)
    return velocity
def T_Log(T_hotin, T_hotout, T_coldin, T_coldout):
    ΔT_1=T_hotin-T_coldout
    ΔT_2=T_hotout-T_coldin
    ΔT_log=(ΔT_1-ΔT_2)/np.log((ΔT_1)/(ΔT_2))
    
    return ΔT_log


LBE_V = 1 #m/s
max_h2oV = 5 #m/s

#w_MFR = MFR(rho_cold, last['A_water'], 5)
last['H2O_V'] = velocity(rho_cold, last['A_water'], w_MFR)
last['MFR_Pb']= MFR(rho_hot, last['A_Pb'], LBE_V)#pd.DataFrame([MFR(rho_hot, Area_pb, LBE_V)], index=['LBE_mass_flow']).T
#last = pd.concat([last, H2O_V, MFR], axis = 1).rename(columns={0:'W_velocity'})

#heat capacity 
last['C_LBE'] = cp_hot*last['MFR_Pb']#.rename(columns={'LBE_mass_flow':'CpLBE'})
last['C_w'] = cp_cold*w_MFR

#Calculated cold out temp and ideal Q    
last['T_cout'] = ((last['C_LBE']/last['C_w'])*(T_hin-T_hout)+T_cin)
last['LMTD'] = T_Log(T_hin, T_hout, T_cin, last['T_cout'])
last['Q_hot'] = (last['C_LBE']*(T_hin - T_hout))
last = last[last['Q_hot'] < 105000][last['Q_hot']>90000]
last = last.reset_index(drop=True)


#Reynolds number
Rey_LBE = LBE_V*(last['ID_Pb']/KV_hot)
Rey_w = last['H2O_V']*(last['Dh_w']/KV_cold)

#Prandtl number 
Pr_LBE = cp_hot*DV_hot/k_hot
Pr_w = cp_cold*DV_cold/k_cold

#Peclet number
Pe_LBE = Rey_LBE * Pr_LBE

#Nusselt
Nu_LBE = 5 + 0.025*Pe_LBE**0.8
Nu_w = 0.023 * (Rey_w**0.8)*(Pr_w**0.33)

#heat transfer W/m^2 K) 
h_hot = Nu_LBE*(k_hot/last['ID_Pb'])
h_cold = Nu_w*(k_cold/last['Dh_w'])

#pipe thermal conductivities W/m K
k_316 = 15 

last['U']=1/(1/h_hot +last['T1']/k_316 +1/h_cold)

#LMTD method



#Dc = pd.DataFrame(last['Dc'].reset_index(drop=True))

last['Ac_LMTD'] = last['Q_hot']/(last['U']*last['LMTD'])
last['Lc_LMTD'] = last['Ac_LMTD']/(math.pi*last['Dc'])
#C_w = rho_cold*A_water*H2O_V*cp_cold


last['one_M'] = ((1*last['Dc']*math.pi)*last['U']*last['LMTD'])
QLMTD_max = max(last['one_M'])

#e-NTU method


Min=[]
Max= []
Eff = []


last['C_min'] = last['C_LBE'].where(last['C_LBE']<last['C_w'], last['C_w'])
last['C_max'] = last['C_LBE'].where(last['C_LBE']>last['C_w'], last['C_w'])

    
for index, row in last.iterrows():
    if row['C_min'] == row['C_LBE']:
        e = ((T_hin-T_hout)/(T_hin-T_cin))
    else:
        e = (row['T_cout']-T_cin)/(T_hin-T_cin)
    Eff.append(e)
last['e'] = pd.DataFrame(np.concatenate([Eff]))
  
last['C_ratio'] = last['C_min']/last['C_max']

#values for e-ntu
data_calor = pd.DataFrame(np.array([[0, 0, 0, 0, 0, 0], 
     [0.25, 0.221, 0.216, 0.21, 0.206, 0.205],
     [0.5, 0.393, 0.378, 0.362, 0.35, 0.348],
     [0.75, 0.528, 0.502, 0.477, 0.457, 0.452],
     [1, 0.632, 0.598, 0.565, 0.538, 0.532],
     [1.25, 0.713, 0.675, 0.635, 0.603, 0.595],
     [1.5, 0.777, 0.735, 0.691, 0.655, 0.645]]),
    index = [0, 0.25, 0.5, 0.75, 1, 1.25,
    1.5], columns = ['NTU', 0, 0.25, 0.5, 0.7, 0.75])
index_calor = [0, 0.25, 0.5, 0.7, 0.75]

data_Ubend = pd.DataFrame(np.array([[0, 0, 0, 0, 0, 0], 
     [0.25, 0.221, 0.215, 0.209, 0.204, 0.198],
     [0.5, 0.393, 0.375, 0.357, 0.340, 0.324],
     [0.75, 0.528, 0.494, 0.463, 0.434, 0.407],
     [1.00, 0.632, 0.584, 0.540, 0.500, 0.463],
     [1.25, 0.714, 0.652, 0.597, 0.546, 0.500],
     [1.5, 0.777, 0.705, 0.639, 0.579, 0.526]]),
    index = [0.0, 0.25, 0.5, 0.75, 1, 1.25,
    1.5], columns = ['NTU', 0.0, 0.25, 0.5, 0.75, 1.00])
index_ubend = [0.0, 0.25, 0.5, 0.75, 1.00]
#NTU calculations

NTU =[]
for ti, tu in zip(last['C_ratio'], last['e']):
    
    NTU.append(tube_geometry.interp_NTU(tu, ti, data_calor, index_calor))
last['NTU_e'] = pd.DataFrame(np.concatenate([NTU]))

last['Ac_NTU'] = (last['NTU_e']*last['C_min'])/last['U']
last['Lc_NTU'] = last['Ac_NTU']/(math.pi*last['Dc'])



#one meter calcs
'''
last['one_M_NTU']=((1*last['Dc']*math.pi)*last['U'])/last['C_min']

e_calc = []
for tie, tue in zip(C_ratio[0], one_M_NTU):
    e_calc.append(tube_geometry.interp_e(tue, tie, data_calor, index_calor))
   
e_calc_new = pd.DataFrame(np.concatenate([e_calc]))
Q_NTU_one = e_calc_new*C_min*(T_hin-last['T_cout'])#.rename(columns ={0:'Q_NTU'})
Q_NTU_one = Q_NTU_one.rename(columns = {0:'Q_NTU'})
QNTU_max = max(Q_NTU_one['Q_NTU'])

'''

#P = ((((Rey_w*DV_cold)/last['Dh_w'])**2)/(2*9.81))*v_w*(0.046/(Rey_w**0.2))*(last['Lc_LMTD']/(last['Dh_w']/4))
last['P_water'] = (0.046/(Rey_w**0.2))*(last['Lc_LMTD']/(last['Dh_w']))*((rho_cold*(last['H2O_V']**2))/2)
last['HL_w'] = (0.046/(Rey_w**0.2))*(last['Lc_LMTD']/last['Dh_w'])*((last['H2O_V']**2)/(2*9.81))
last['HL_LBE'] = (0.046/(Rey_LBE**0.2))*(last['Lc_LMTD']/last['ID_Pb'])*((1**2)/(2*9.81))
last['P_LBE'] = (0.046/(Rey_LBE**0.2))*(last['Lc_LMTD']/(last['ID_Pb']))*((rho_hot*(LBE_V**2))/2)
'''
HL_oneM = (0.046/(Rey_LBE.reset_index(drop=True)**0.2))*(1/last['ID_Pb'].reset_index(drop=True))*((1**2)/(2*9.81))
HL_oneM_w = (0.046/Rey_w.reset_index(drop=True)**0.2)*(1/last['Dh_w'].reset_index(drop=True))*((last['W_velocity'].reset_index(drop=True)**2)/(2*9.81))
'''
last['Q_real'] = last['Ac_LMTD']*last['LMTD']*last['U']


last = last[last['HL_LBE'] < 9][last['H2O_V'] < 5][last['A_water']<0.0005]
last = last[last['P_water'] == min(last['P_water'])]

print(last['Lc_NTU'])
print(last['Lc_LMTD'])
'''
lengths_and_dia = lengths_and_dia[lengths_and_dia.L_LMTD == min(lengths_and_dia['L_LMTD'])]


onemeter = pd.concat([last.reset_index(drop=True), one_M, Q_NTU_one, HL_oneM, HL_oneM_w], axis = 1).rename(columns=
                                                                                      {1:'Head_loss_w'})
onemeter = onemeter[onemeter.Head_loss_w < 9][onemeter.LBE_mass_flow < 25]
onemeter = onemeter[onemeter.Q_NTU == max(onemeter['Q_NTU'])]
#onemeter = onemeter[2003:2004] 
lengths_and_dia.to_csv(r'original_HX.csv')
onemeter.to_csv(r'onemeter.csv')


#for U-tube calculations
R = (T_cin - T_cout)/(T_hout - T_hin)
S = (T_hout- T_hin)/(T_cin - T_hin)
Fg = (((R**2 + 1)**0.5)*np.log((1-S)/(1-R*S)))/((R-1)*np.log(2-S*(R+1-(R**2+1)**0.5)/(2-S*(R+1+np.sqrt(R**2+1)))))


U_Ubend=pd.DataFrame(1/(1/h_hot +last['T1']/k_316 +1/h_cold))
U_Ac_LMTD = Q_hot.reset_index(drop=True)/(U_Ubend.reset_index(drop=True)*LMTD*Fg.reset_index(drop=True))
U_Lc_LMTD = U_Ac_LMTD[0]/(math.pi*Dc['Dc'])
Ubend = pd.concat([last.reset_index(drop=True), U_Ac_LMTD, U_Lc_LMTD], axis =1).rename(columns = {0:'Q_U', 1:'Ac_U', 2:'L_U'})
Ubend = U_Ac_LMTD*U_Ubend.reset_index(drop=True)*LMTD*Fg.reset_index(drop=True)
'''


