#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:51:29 2020

@author: kelleyverner
"""


#for U-tube calculations
R = (T_cin - T_cout)/(T_hout - T_hin)
S = (T_hout- T_hin)/(T_cin - T_hin)
Fg = (((R**2 + 1)**0.5)*np.log((1-S)/(1-R*S)))/((R-1)*np.log(2-S*(R+1-(R**2+1)**0.5)/(2-S*(R+1+np.sqrt(R**2+1)))))


U_Ubend=pd.DataFrame(1/(1/h_hot +last['T1']/k_316 +1/h_cold))
U_Ac_LMTD = Q_hot.reset_index(drop=True)/(U_Ubend.reset_index(drop=True)*LMTD*Fg.reset_index(drop=True))
U_Lc_LMTD = U_Ac_LMTD[0]/(math.pi*Dc['Dc'])
Ubend = pd.concat([last.reset_index(drop=True), U_Ac_LMTD, U_Lc_LMTD], axis =1).rename(columns = {0:'Q_U', 1:'Ac_U', 2:'L_U'})
Ubend = U_Ac_LMTD*U_Ubend.reset_index(drop=True)*LMTD*Fg.reset_index(drop=True)