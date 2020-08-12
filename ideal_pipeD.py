import math
import numpy as np
import pandas as pd

data = []
gapgas = []
outer = []
LBE_o =[]
gas_inner = []
outer_gauge = []
OD2_all = []
OD2_name = []
OD2_try = []
data_in = []




tube_OD2 = [0.009525
, 0.0127
, 0.015875
, 0.01905
, 0.022225
, 0.0254
, 0.028575
, 0.03175
, 0.0381
, 0.04445
, 0.0508
, 0.05715
, 0.0635
, 0.06985
, 0.0762
, 0.08255
, 0.0889
, 0.09525
, 0.1016
, 0.10795
, 0.1143
, 0.12065
, 0.127
, 0.13335]

tube_OD1 = [0.00635
, 0.009525
, 0.0127
, 0.015875
, 0.01905
, 0.022225
, 0.0254
, 0.028575
, 0.03175
, 0.0381
, 0.04445
, 0.0508
, 0.05715
, 0.0635
, 0.06985
, 0.0762
, 0.08255
, 0.0889
, 0.09525
, 0.1016
, 0.10795
, 0.1143
, 0.12065
, 0.127]

pipe_OD1 = [0.0078,
0.01038,
0.01388,
0.01708,
0.02248,
0.02786,
0.03666,
0.04276,
0.05476,
0.0669,
0.0828,
0.0955,
0.1082,
0.13448]

pipe_OD2 = [0.0103, 0.0137,
0.0172,
0.0213,
0.0267,
0.0334,
0.0422,
0.0483,
0.0603,
0.073,
0.0889,
0.1016,
0.1143,
0.1413,
0.1683]

pipe_ID2 = [0.0078,
0.01038,
0.01388,
0.01708,
0.02248,
0.02786,
0.03666,
0.04276,
0.05476,
0.0669,
0.0828,
0.0955,
0.1082,
0.13448,
0.16148
]

OD1 = [0.00635,
0.009525
, 0.0127
, 0.015875
, 0.01905
, 0.022225
, 0.0254
, 0.028575
, 0.03175
, 0.0381
, 0.04445
, 0.0508
, 0.05715
, 0.0635
, 0.06985
, 0.0762
, 0.08255
, 0.0889
, 0.09525
, 0.1016
, 0.10795
, 0.1143
, 0.12065
, 0.127, 
0.0103,
0.0137,
0.0172,
0.0213,
0.0267,
0.0334,
0.0422,
0.0483,
0.0603,
0.073,
0.0889,
0.1016,
0.1143,
0.1413]


T =  [0.009652, 0.008636, 0.00762, 0.0072136, 0.0065786, 0.0060452, 0.005588, 0.0051562, 
0.004572, 0.004191, 0.0037592, 0.0034036, 0.003048, 0.0027686, 0.002413, 0.0021082, 0.0018288, 
0.001651, 0.0014732, 0.0012446, 0.0010668, 0.000889, 0.0008128, 0.0007112, 0.000635, 0.0005588]
'''
, 
0.00166, 0.00166,
0.00211,
0.00211,
0.00277,
0.00277,
0.00277,
0.00277,
0.00305,
0.00305,
0.00305,
0.00305,
0.00341,
0.00341]
'''
name = ['tube']*24 + ['pipe']*14
gauge = [00, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

gauge2= [10.3,
13.7,
17.2,
21.3,
26.7,
33.4,
42.2,
48.3,
60.3,
73,
88.9,
101.6,
114.3,
141.3,
168.3]



df = pd.DataFrame()

'''
The 'tube_OD2' is the outer diameter of the tubes for the gas gap pipe.
The thickness of each gauge is subtracted from each tube_OD2 to get all inner 
tube diameters (ID2) possible. These are the same as what's in the Excel sheet.
'''
for i in tube_OD2:
    for t, g in zip(T, gauge):
        ID2 = i - 2*t
        
        data.append(ID2)
        outer.append(g)
        OD2_name.append(i)
    #results = pd.concat(data), axis = 0)
    #print(data)

'''
ID_all combines the ID2 diameters with the known innder diameters of schedule 10 pipes.
'''
ID_all = pd.DataFrame(data + pipe_ID2)
Outer_all = pd.DataFrame(OD2_name + pipe_OD2)
Gauge_all = pd.DataFrame(outer + gauge2)


'''
OD1 are all the possible outer diameters of the LBE pipe that must be subtracted and halved 
with the ID2 from above to calculate the possible gas gap. 
'''
for j in OD1:
    gap = (ID_all - j)/2
    gapgas.append(gap)
    
    #to make the array correct
    length = np.ones(len(ID_all))
    dia = j*length

    LBE_o.append(dia)
    gas_inner.append(ID_all)
    #gas_outer.append(OD)
    OD2_all.append(Gauge_all)
    OD2_try.append(Outer_all)

for k in tube_OD1:
    for q in T:
        ID1 = k - q 
        data_in.append(ID1)



outer_LBE = pd.DataFrame(np.concatenate(gas_inner), columns = ['ID2'])
inner_name = pd.DataFrame({'DO1':np.concatenate(LBE_o)})
gasgaptry = pd.DataFrame(np.concatenate(gapgas), columns = ['gap'])
#print(inner_name, gasgaptry)
outer_gas = pd.DataFrame(np.concatenate(OD2_all), columns = ['Gauge'])
outer_name = pd.DataFrame(np.concatenate(OD2_try), columns = ['OD2'])

PB_inner = outer_LBE.drop([0])
print(PB_inner)



size = pd.concat([outer_name, outer_LBE, outer_gas, inner_name, gasgaptry],  axis=1)

final_gap = size['gap']
new_size = size[size['gap']>0.0002]
final_gap = new_size[new_size['gap']<0.00026]
print(final_gap)
'''

def delta_T(MFR, T_hout, Q_required, T_roomtemp, delta_Tcold):
    cp_hot = 164.8-3.94E-2*T_hout+1.25E-5*T_hout**2-4.56E5*T_hout**-2
    delta_Thot = Q_req/(cp_hot*LBE_MFR)
    
    #final temps
    T_hin = T_hout + delta_Thot
    T_cin = T_roomtemp+273 #K

    T_cout = T_cin + delta_Tcold

    

    return delta_Thot, T_hin, T_cin, T_cout

def thermo_prop(T):
    #calculations of thermophysical props
    #LBE
    rho_hot = -1.2046*T + 10989 #kg/cm^3
    cp_hot = 164.8-3.94E-2*T+1.25E-5*T**2-4.56E5*T**-2
    k_hot = 3.284 + 1.612E-2*T - 2.305E-6*T**2 #thermal conductivity
    DV_hot = 4.94E-4*math.exp(754.1/T) #dynamic viscosity
    KV_hot = DV_hot/rho_hot
    return rho_hot, cp_hot, k_hot, DV_hot, KV_hot

def area_calcs(ID_pb, OD_pb, ID_gas, OD_gas, ID_w, gas):
    T_1 = OD_pb - ID_pb
    T_2 = OD_gas - ID_gas
    A_pb = math.pi*(ID_pb/2)**2
    A_water = math.pi*((ID_w/2)**2-(OD_gas/2)**2)
    Dh_pb = ID_pb
    Dh_w = ID_w - OD_gas
    Dc = (ID_gas+OD_pb)/2
    return T_1, T_2, A_pb, A_water, Dh_pb, Dh_w, Dc

flowV_hot = 1 #m/s
#mass flow rate LBE



#find delta T
LBE_MFR = 26.7 #kg/s
T_hout = 200+273 #K
T_cin = 20 #C
Q_req = 100E3 #W
delta_Tcold = 10 #K
delta_Thot, T_hin, T_cin, T_cout = delta_T(LBE_MFR, T_hout, Q_req, T_cin, delta_Tcold)

print('The required temperature decrease is approximatly', delta_Thot)

T_avg = (T_hin + T_hout)/2

rho_hot, cp_hot, k_hot, DV_hot, KV_hot = thermo_prop(T_avg)

#H2O
rho_cold = 1000 #kg/cm^3
cp_cold = 4180 #specific heat water J/kg K
k_cold = 0.60694 #thermal conductivity
DV_cold = 0.00089313 #dynamic viscosity
KV_cold = DV_cold/rho_cold

#required water mass flow rate

w_MFR = Q_req/(cp_cold*delta_Tcold)
print('The required mass flow rate of water is ', w_MFR)



#required flow area and diameter for perfect system
def ideal_pipeD(density, MFR, velocity):
    A = MFR/(density * velocity)
    D = math.sqrt((4*A)/math.pi)
    return D

Ideal_D = ideal_pipeD(rho_hot, LBE_MFR, flowV_hot)
#actual mass flow rates from equation 2-a
'''
