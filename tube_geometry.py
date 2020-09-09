import pandas as pd
import numpy as np
import math

#calculate the inner diameters of all the tube options with gauges
'''
The 'tube_OD2' is the outer diameter of the tubes for the gas gap pipe.
The thickness of each gauge is subtracted from each tube_OD2 to get all inner 
tube diameters (ID2) possible. These are the same as what's in the Excel sheet.
'''
def inner_dia(tube_OD, tube_name, tube_thickness, gauge, pipe_name, pipe_ID, pipe_OD):
    tube_ID = []
    gauge_name = []
    OD_all = []
    OD_name = []

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

    Gas = pd.DataFrame([OD_name + pipe_name, gauge_name + pipe_name, OD_all+pipe_OD, tube_ID+pipe_ID], 
                index = ['Gas Name','Gas Gauge', 'OD_gas', 'ID_gas']).T
    LBE = pd.DataFrame([OD_name + pipe_name, gauge_name + pipe_name, tube_ID+pipe_ID, 
                        OD_all+pipe_OD], index = ['Pb Name','Pb Gauge','ID_Pb', 'OD_Pb']).T
    Total = pd.DataFrame([OD_name+pipe_name, OD_all+pipe_OD, tube_ID+pipe_ID, gauge_name + pipe_name],
                     index =  ['Name', 'Outer', 'Inner', 'Gauge']).T
    return Gas, LBE, Total

#calculates all possible combinations of the gas gap 
def gas_gap_calcs(Gas, LBE):
    gap = []
    name = []
    OD_Pb = []
    ID_gas = []
    Pb_data = []

    for i, k, j  in zip(LBE['OD_Pb'], LBE['Pb Name'], LBE['ID_Pb']):
        gas_gap = (Gas['ID_gas'] - i)/2
        gap.append(gas_gap)
        length = np.ones(len(Gas['ID_gas']))
        dia = i * length
        x = [k]*len(Gas['ID_gas'])
        name.append(x)
        OD_Pb.append(dia)
        ID_gas.append(Gas)
        Pb_data.append([j]*len(Gas['ID_gas']))

    gap_calc = pd.DataFrame(np.concatenate(gap), columns = ['GasGap'])
    OD_lbe = pd.DataFrame(np.concatenate(OD_Pb),columns = ['OD_Pb'])
    name_pb = pd.DataFrame(np.concatenate(name), columns = ['LBE Name'])
    ID_info = pd.concat(ID_gas, ignore_index = True)
    Pb_info = pd.DataFrame(np.concatenate(Pb_data), columns = ['ID_Pb'])

    Final = pd.concat([ID_info, name_pb, OD_lbe, Pb_info, gap_calc], axis = 1)
    return Final 

#calculates the area and dimentions of the gas gap, water, and LBE
def area_calcs(ID_pb, OD_pb, ID_gas, OD_gas, ID_w):
    T_1 = (OD_pb - ID_pb)/2
    T_2 = (OD_gas - ID_gas)/2
    A_pb = math.pi*(ID_pb/2)**2
    A_water = math.pi*((ID_w/2)**2-(OD_gas/2)**2)
    Dh_w = ID_w - OD_gas
    Dc = (ID_gas+OD_pb)/2

    return T_1, T_2, A_pb, A_water, Dc, Dh_w

#calculates the water diameters and dimensions
def water_dia(Total, Final):
    H2O_d = []
    W_gap = []
    H2O_name = []
    combo = []

    for H2O_ID, w_name in zip(Total['Inner'], Total['Name']):
        W_gap.append((H2O_ID - Final['OD_gas'])/2)
        l = [w_name]*len(Final['OD_gas'])
        w = H2O_ID*np.ones(len(Final['OD_gas']))
        H2O_name.append(l)
        H2O_d.append(w)
        combo.append(Final)

    h2o = pd.DataFrame([np.concatenate(W_gap), np.concatenate(H2O_name), 
                        np.concatenate(H2O_d)], index = ['Thick_w', 'W name', 'ID_w']).T
    last = pd.concat([h2o, pd.concat(combo, ignore_index=True)], axis = 1)
    return last

def delta_T(MFR, T_hout, Q_required, T_roomtemp, delta_Tcold):
    cp_hot = 164.8-3.94E-2*T_hout+1.25E-5*T_hout**2-4.56E5*T_hout**-2
    delta_Thot = Q_required/(cp_hot*MFR)

    #final temps
    T_hin = T_hout + delta_Thot
    T_cin = T_roomtemp+273 #K

    T_cout = T_cin + delta_Tcold



    return delta_Thot, T_hin, T_cin, T_cout

#calculates the LBE properties
def thermo_prop(T):
    #calculations of thermophysical props
    #LBE
    rho_hot = -1.2046*T + 10989 #kg/cm^3
    cp_hot = 164.8-3.94E-2*T+1.25E-5*T**2-4.56E5*T**-2
    k_hot = 3.284 + 1.612E-2*T - 2.305E-6*T**2 #thermal conductivity
    DV_hot = 4.94E-4*math.exp(754.1/T) #dynamic viscosity
    KV_hot = DV_hot/rho_hot
    return rho_hot, cp_hot, k_hot, DV_hot, KV_hot

#interpolates the value of NTU from a given effectiveness
def interp_NTU(e_new, C_new, data, index):    
    lower_c = max([t for t in index if t < C_new])


    higher_c = min([t for t in index if t > C_new])

    lower_e1 = max([t for t in data[lower_c] if t < e_new])

    higher_e1 = min([t for t in data[lower_c] if t > e_new])

    lower_e2 = max([t for t in data[higher_c] if t < e_new])
    higher_e2 = min([t for t in data[higher_c] if t > e_new])

    #calc the interpolated e's
    x1 = [lower_e1, lower_e2]
    x2 = [higher_e1, higher_e2]
    C = [lower_c, higher_c]
    e_new_low = np.interp(C_new, C, x1)
    e_new_high = np.interp(C_new, C, x2)


    #calc interpolated NTU
    lower_NTU = float(data['NTU'][data[lower_c] == lower_e2])
    higher_NTU = float(data['NTU'][data[higher_c] == higher_e2])

    x3 = [lower_NTU, higher_NTU]
    e = [e_new_low, e_new_high]
    NTU = np.interp(e_new, e, x3)

    return NTU

#interpolates the effectiveness from a given NTU
def interp_e(NTU_new, C_new, data, index):    
    lower_c = max([t for t in index if t < C_new])


    higher_c = min([t for t in index if t > C_new])


    lower_NTU = max([t for t in data['NTU'] if t < NTU_new])

    higher_NTU = min([t for t in data['NTU'] if t > NTU_new])

    lower_e1 = data[lower_c][lower_NTU]
    lower_e2 = data[higher_c][lower_NTU]
    higher_e1 = data[lower_c][higher_NTU]
    higher_e2= data[higher_c][higher_NTU]
    C = [lower_c, higher_c]
    x1 =[lower_e1, lower_e2]
    e_new_low = np.interp(C_new, C, x1)
    e_new_high = np.interp(C_new,  C, [higher_e1, higher_e2])

    x2 = [lower_NTU, higher_NTU]
    e = [e_new_low, e_new_high]

    e_new = np.interp(NTU_new, x2, e)

    return e_new