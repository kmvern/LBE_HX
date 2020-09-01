#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 16:41:02 2020

@author: kelleyverner
"""
import tube_geometry
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

#Radial temperature profiles along calorimeter

design = pd.read_csv('original_HX.csv')

#geometry
ID_pb = design['ID_Pb']
OD_pb = design['OD_Pb']
ID_gas = design['ID_gas']
OD_gas = design['OD_gas']
ID_w = design['ID_w']
OD_w = design['ID_w']-2*design['T_w']
MFR = design['MFR_Pb']
length = design['Lc_LMTD']

#temperatures
T_hout = 200+273 #K
T_roomtemp = 20 #C
Q = design['Q_real']
delta_Tcold = 10 #K


#heat transfer W/m^2 K
h_Pb = 7302.596466
h_water = 5748.32

#pipe thermal conductivities W/m K
k_316 = 15 
k_He = 0.189
k_Ar = 0.0335

T_1, T_2, A_pb, A_water, Dc, Dh_w = tube_geometry.area_calcs(ID_pb, OD_pb, ID_gas, OD_gas, ID_w)

delta_Thot, T_hin, T_cin, T_cout = tube_geometry.delta_T(MFR, T_hout, Q, T_roomtemp, delta_Tcold)
T_max = T_hin
T_min = T_cout
def pipe_resist(ID, OD, k):
    ro = OD/2
    ri = ID/2
    A_heat = 2*math.pi*k
    
    R_pipe = (np.log(ro/ri))/A_heat
    
    return R_pipe
    
def convec_resist(h, diameter):
    r = diameter/2
    Area = 2*math.pi*r
    
    R_conv = 1/(h*Area)
    
    return R_conv
#heat transfer for air around it
    

#convective resistances
Pb_conv = convec_resist(h_Pb, ID_pb)
print(float(Pb_conv), 'K/W')
H2O_conv = convec_resist(h_water, OD_gas)
print(float(H2O_conv), 'K/W')
#Outer_conv = 
#pipe/tube resistances K/W

R_Pb_pipe = pipe_resist(ID_pb, OD_pb, k_316)
R_gas = pipe_resist(OD_pb, ID_gas, k_He)
R_gas_pipe = pipe_resist(ID_gas, OD_gas, k_316)
R_w_pipe = pipe_resist(ID_w, OD_w, k_316)


R_total = Pb_conv + H2O_conv + R_Pb_pipe + R_gas + R_gas_pipe #+ R_w_pipe

Q_pipe = (T_max - T_min)/R_total #Watts/m

def delta_t(OD, ID, k, T_hin, Q_pipe):
    data = []
    d = []
    num = float(OD-ID)/5
    for i in np.linspace(float(ID), float(OD), num=10):
        R = pipe_resist(float(ID), i, k)
        delta = T_hin - (Q_pipe*R)
        data.append(delta)
        d.append(i/2)
        
    data = pd.DataFrame(data)
    d = pd.DataFrame(d)
    temp_stop = delta
    return data, d, temp_stop

#max temp to Pb wall
max_PbID = Q_pipe*Pb_conv
delta_Pb = pd.DataFrame(np.linspace(T_max, (T_max-max_PbID), num =10))
dia_Pb = pd.DataFrame(np.linspace(0, float(ID_pb), num =10))

#Pb pipe
Pb_temp = Q_pipe*R_Pb_pipe
delta_LBE_pipe, dia_pb_pipe, out_tmp1 = delta_t(OD_pb, ID_pb, k_316,  T_max-max_PbID, Q_pipe)

#gas gap
gas_DT = Q_pipe*R_gas
delta_gas, dia_gas, out_tmp2 = delta_t(ID_gas, OD_pb, k_He,  out_tmp1, Q_pipe)

#gas pipe 
gas_pipe_DT = Q_pipe*R_gas_pipe
delta_gas_pipe, dia_gp, out_tmp3 = delta_t(OD_gas, ID_gas, k_316,  out_tmp2, Q_pipe)

#water inner temp to T_min
H2O_DT = Q_pipe*H2O_conv
delta_water = pd.DataFrame(np.linspace(out_tmp3, (out_tmp3-H2O_DT), num =10))
dia_water = pd.DataFrame(np.linspace(float(OD_gas), float(ID_w), num =10))

print(T_max, T_max - max_PbID, out_tmp1, out_tmp2, out_tmp3, out_tmp3-H2O_DT)

temp_change = pd.concat([delta_Pb, delta_LBE_pipe, delta_gas, delta_gas_pipe, delta_water]).reset_index(drop=True)
distance = pd.concat([dia_Pb/2, dia_pb_pipe, dia_gas, dia_gp, dia_water/2]).reset_index(drop = True)
    
plt.plot(distance[0],temp_change[0])
plt.ylabel('Temperature (K)')
plt.xlabel('Radius (m)')
plt.axvline(float(ID_pb)/2, 0, 500 ,label = 'ID_Pb', color='b', linestyle = '--', linewidth = 1)
plt.axvline(float(OD_pb)/2, 0, 500, label = 'OD_Pb', color='g', linestyle = '--', linewidth = 1)
plt.axvline(float(ID_gas)/2, 0, 500, label = 'ID_gas', color = 'r', linestyle = '--', linewidth = 1)
plt.axvline(float(OD_gas)/2, 0, 500, label = 'OD_gas', color = 'y', linestyle = '--', linewidth = 1)
plt.axvline(float(ID_w)/2, 0, 500, label = 'ID_w', color = 'c', linestyle = '--', linewidth = 1)
plt.legend()
plt.axis([0.0,0.039, 280, 500])
plt.title('Radial temperature profile (He gap)')
pd.DataFrame([temp_change[0], distance[0]]).T


