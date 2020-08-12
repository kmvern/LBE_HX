import math
import numpy as np
import pandas as pd

# Below are the measurements for standard tube sizing

tube_ID  = []
gauge_name = []
OD_all = []
OD_name = []
gap = []
OD_Pb = []
ID_gas = []
Pb_data = []
Pb_inner = []
name = []
Gas_d1 = []
Pb_d1 = []
H2O_name = []
W_gap = []
combo = []
tube_OD_inch = [0.25, 0.375, 0.5, 0.625, 0.75,
                0.875, 1, 1.125, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75,
                3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25]

tube_name = ['T-' + name for name in map(str, tube_OD_inch)]

tube_name_gas = tube_name[1:]

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

tube_OD_gas = tube_OD[1:]

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


#calculate the inner diameters of all the tube options with gauges
'''
The 'tube_OD2' is the outer diameter of the tubes for the gas gap pipe.
The thickness of each gauge is subtracted from each tube_OD2 to get all inner 
tube diameters (ID2) possible. These are the same as what's in the Excel sheet.
'''
for OD, n in zip(tube_OD, tube_name):
    for t, g in zip(tube_thickness, gauge):
        ID = OD - 2*t
        
        tube_ID.append(ID)
        gauge_name.append(g)
        OD_all.append(OD)
        OD_name.append(n)

'''
ID_all combines the ID2 diameters with the known inner diameters of schedule 10 pipes.
'''

Gas = pd.DataFrame([OD_name + pipe_name, gauge_name + pipe_name, tube_ID+pipe_ID, OD_all+pipe_OD], 
            index = ['Gas Name','Gauge', 'ID_gas', 'OD_gas']).T
LBE = pd.DataFrame([OD_name + pipe_name, gauge_name + pipe_name, tube_ID+pipe_ID, OD_all+pipe_OD], index = ['Name','Gauge','ID_Pb', 'OD_Pb']).T


Gas = Gas[Gas.OD_gas != 0.00635].reset_index()
LBE = LBE[LBE.OD_Pb != 0.13335].reset_index()

df = pd.DataFrame([tube_OD + pipe_OD, tube_name+pipe_name]).T

for i, k  in zip(df[0], df[1]):
    gas_gap = (Gas['ID_gas'] - i)/2
    gap.append(gas_gap)
    length = np.ones(len(Gas['ID_gas']))
    dia = i * length
    x = [k]*len(Gas['ID_gas'])
    name.append(x)
    OD_Pb.append(dia)
    ID_gas.append(Gas)
    Pb_data.append(LBE)

gap_calc = pd.DataFrame(np.concatenate(gap), columns = ['GasGap'])
OD_lbe = pd.DataFrame(np.concatenate(OD_Pb),columns = ['OD_LBE'])
name_pb = pd.DataFrame(np.concatenate(name), columns = ['LBE Name'])
ID_info = pd.concat(ID_gas, ignore_index = True)
Pb_info = pd.concat(Pb_data, ignore_index = True)

Final = pd.concat([ID_info, gap_calc, OD_lbe, name_pb], axis = 1)

#gas gap options
Final = Final[Final.GasGap >= 0.0002]
Final = Final[Final.GasGap <= 0.00026]

Total = pd.DataFrame([OD_name+pipe_name, OD_all+pipe_OD, tube_ID+pipe_ID, gauge_name + pipe_name],
                     index =  ['Name', 'Outer', 'Inner', 'Gauge']).T
print(Final)
#gas gap names



'''
OD1 are all the possible outer diameters of the LBE pipe that must be subtracted and halved 
with the ID2 from above to calculate the possible gas gap. 

for OD in Outer_all:
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
'''

def area_calcs(ID_pb, OD_pb, ID_gas, OD_gas, ID_w):
    for H2O_ID in ID_w:
        H2O_ID - OD_gas
    T_1 = (OD_pb - ID_pb)/2
    T_2 = (OD_gas - ID_gas)/2
    A_pb = math.pi*(ID_pb/2)**2
    #A_water = math.pi*((ID_w/2)**2-(OD_gas/2)**2)
    Dh_pb = ID_pb
    #Dh_w = ID_w - OD_gas
    Dc = (ID_gas+OD_pb)/2
    print(A_pb)
    return T_1, T_2, A_pb, Dh_pb, Dc
H2O_d = []

#print(Final['ID_gas']
for H2O_ID, w_name in zip(Total['Inner'], Total['Name']):
    W_gap.append((H2O_ID - Final['OD_gas'])/2)
    l = [w_name]*len(Final['OD_gas'])
    w = H2O_ID*np.ones(len(Final['OD_gas']))
    H2O_name.append(l)
    H2O_d.append(w)
    combo.append(Final)

h2o = pd.DataFrame([np.concatenate(W_gap), np.concatenate(H2O_name), np.concatenate(H2O_d)], index = ['T_w', 'W name', 'ID_w']).T
 
last = pd.concat([h2o, pd.concat(combo, ignore_index=True)], axis = 1)

last = last.loc[last['ID_w']> last['OD_gas']]


for m, t in zip(Final['OD_LBE'], Final['LBE Name']):
    Pb_d = Total.loc[Total['Outer']==m]
    Pb_d1.append(Pb_d.loc[Total['Name']==t])
Pb_df = pd.concat(Pb_d1)
    #print(Gas_d)
#print(Total.loc[Total['Name']=='T-1.125'])

#gas gap names
for m1, t1 in zip(Final['Gas Name'], Final['Gauge']):
    Gas_d = Total.loc[Total['Name']==m1]
    Gas_d1.append(Gas_d.loc[Total['Gauge']==t1])

    #print(Gas_d)
Gas_df = pd.concat(Gas_d1)
print(last)
#area_calcs((Pb_df)['Inner'].values, (Pb_df)['Outer'].values, Final['ID_gas'].values, Final['OD_gas'].values, Total['Inner'].values)

#Testing