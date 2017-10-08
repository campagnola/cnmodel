# -*- encoding: utf-8 -*-
from ._db import add_table_data
"""
Ion channel density tables
All of the ion channel densities for the models implemented in cnmodel
are (or should be) stated here, and should not be modified in the
cnmodel code itself. 

"""

add_table_data('RM03_channels', row_key='field', col_key='cell_type', 
               species='guineapig', data=u"""

This table describes the ion channel densities (and voltage shifts if necessary)
for different cell types in the original Rothman Manis 2003 model.
Data from Table 1, except for "octopus" cells, which is modified (see note 3)

-----------------------------------------------------------------------------------------------------------------------------------
                    bushy-II      bushy-II-I    tstellate     tstellate-t   bushy-I-II    octopus
                                                                                                   
RM03_name           II            II-I          I-c           I-t           I-II          II-o  [4]
soma_na_gbar        1000. [1]     1000. [1]     1000. [1]     1000. [1]     1000. [2]     1000. [3]
soma_kht_gbar       150.0 [1]     150.0 [1]     150.0 [1]     80.0  [1]     150.0 [2]     150.0 [3] 
soma_klt_gbar       200.0 [1]     35.0  [1]     0.0   [1]     0.0   [1]     20.0  [2]     1000. [3] 
soma_ka_gbar        0.0   [1]     0.0   [1]     0.0   [1]     65.0  [1]     0.0   [2]     0.0   [3]
soma_ih_gbar        20.0  [1]     3.5   [1]     0.5   [1]     0.5   [1]     2.0   [2]     30.0  [3]
soma_leak_gbar      2.0   [1]     2.0   [1]     2.0   [1]     2.0   [1]     2.0   [2]     2.0   [3]
soma_leak_erev      -65   [1]     -65   [1]     -65   [1]     -65   [1]     -65   [2]     -65   [3]
soma_na_type        nacn  [1]     nacn  [1]     nacn  [1]     nacn  [1]     nacn  [2]     nacn  [3]
soma_ih_type        ihvcn [1]     ihvcn [1]     ihvcn [1]     ihvcn [1]     ihvcn [2]     ihvcn [3]
soma_Cap            12.0  [1]     12.0  [1]     12.0  [1]     12.0  [1]     12.0  [2]     25.0  [3]
soma_e_k            -84   [1]     -84   [1]     -84   [1]     -84   [2]     -84   [2]     -84   [2] 
soma_e_na           50.   [1]     50.   [1]     50.   [1]     50.   [2]     50.   [2]     50.   [2] 
soma_ih_eh          -43   [1]     -43   [1]     -43   [1]     -43   [2]     -43   [2]     -43   [2] 

-----------------------------------------------------------------------------------------------------------------------------------

[1] Rothman and Manis, 2003
    Age "adult", Temperature=22C
    Units are nS.

[2] Rothman and manis, 2003, model I-II
    Some low-voltage K current, based on observations of
    a single spike near threshold and regular firing for higher
    currents (Xie and Manis, 2017)
    
[3] Derived from Rothman and Manis, 2003, model II
    Large amounts of low-voltage K current, and elevated HCN. Conductances
    based on Rothman and Manis, 2003; concept from Cao and Oertel

[4] Designation for elevated LTK and Ih for octopus cells

""")

add_table_data('XM13_channels', row_key='field', col_key='cell_type', 
               species='mouse', data=u"""

This table describes the ion channel densities (and voltage shifts if necessary)
for different cell types based on the Xie and Manis 2013 models for mouse.

-----------------------------------------------------------------------------------------------------------------------------------
                    bushy-II      bushy-II-I    tstellate    bushy-II-I
                                                                    
XM13_name           II            II-I          I-c          I-II      
soma_na_gbar        1000. [1]     1000. [1]     3000. [1]    1000. [2] 
soma_kht_gbar       58.0  [1]     58.0  [1]     500.0 [1]    150.0 [2] 
soma_klt_gbar       80.0  [1]     14.0  [1]     0.0   [1]    20.0  [2] 
soma_ka_gbar        0.0   [1]     0.0   [1]     0.0   [1]    0.0   [2] 
soma_ih_gbar        30.0  [1]     30.0  [1]     18.0  [1]    2.0   [2] 
soma_leak_gbar      2.0   [1]     2.0   [1]     8.0   [1]    2.0   [2] 
soma_leak_erev      -65   [1]     -65   [1]     -65   [1]    -65   [2] 
soma_na_type        nacn  [1]     nacn  [1]     nacn  [1]    nacn  [2] 
soma_ih_type        ihvcn [1]     ihvcn [1]     ihvcn [1]    ihvcn [2] 
soma_Cap            26.0  [1]     26.0  [1]     25.0  [1]    26.0  [2] 
soma_na_vshift      4.3   [1]     4.3   [1]     4.3   [1]    4.3   [1]
soma_e_k            -84   [1]     -84   [1]     -84   [1]    -84   [2] 
soma_e_na           50.   [1]     50.   [1]     50.   [1]    50.   [2] 
soma_ih_eh          -43   [1]     -43   [1]     -43   [1]    -43   [2] 

-----------------------------------------------------------------------------------------------------------------------------------

[1] Rothman and Manis, 2003
    Age "adult", Temperature=22C
    Units are nS.

[2] Rothman and manis, 2003, model I-II
    Some low-voltage K current, based on observations of
    a single spike near threshold and regular firing for higher
    currents (Xie and Manis, 2017)
    
[3] Derived from Rothman and Manis, 2003, model II
    Large amounts of low-voltage K current, and elevated HCN. Conductances
    based on Rothman and Manis, 2003; concept from Cao and Oertel

[4] Designation for elevated LTK and Ih for octopus cells

""")


add_table_data('POK_channels', row_key='field', col_key='cell_type', 
               species='rat', data=u"""

This table describes the ion channel densities and voltage shifts for rat DCN pyramidal cells,
from Kanold and Manis, 2001

The table includes 2 additiona variants

-----------------------------------------------------------------------------------------------------------------------------------
                  pyramidal   
                           
soma_napyr_gbar   350.0  [1]      
soma_nap_gbar     0.
soma_kdpyr_gbar   80.0   [1]
soma_kcnq_gbar    0.
soma_kif_gbar     150.0  [1]
soma_kis_gbar     40.0   [1]
soma_ihpyr_gbar   2.8    [1]     
soma_leak_gbar    2.8    [1]
soma_leak_erev    -62.0  [1]
soma_e_na         50.    [1]
soma_e_k          -81.5  [1]
soma_e_h          -43.0  [1]
soma_natype       napyr
soma_Cap          12     [1]


-----------------------------------------------------------------------------------------------------------------------------------

[1] Kanold and Manis, 1999, 2001, 2005
    Age P11-14, Temperature=22C
    Units are nS.

""")

add_table_data('CW_channels', row_key='field', col_key='cell_type', 
               species='mouse', data=u"""

This table describes the ion channel densities and voltage shifts
for a mouse carthweel cell model.
Ad-hoc model, based on a Purkinje cell model.


-----------------------------------------------------------------------------------------------------------------------------------
                   cartwheel   
                           
soma_narsg_gbar    500.0  [1]     
soma_bkpkj_gbar    2.0
soma_kpkj_gbar     100.   [1]
soma_kpkj2_gbar    50.
soma_kpkjslow_gbar 150    [1]
soma_kpksk_gbar    25.0   [1]
soma_lkpkj_gbar    5.0    [1]     
soma_hpkj_gbar     5.0    [1]
soma_e_na          50.    [1]
soma_e_k           -80.0  [1]
soma_hpkj_eh       -43.0  [1]
soma_lkpkj_e       -65.0  [1]
soma_e_ca          50.
soma_na_type       narsg
soma_pcabar        0.00015 [1]
soma_Dia           18  

-----------------------------------------------------------------------------------------------------------------------------------

[1] Raman

""")

add_table_data('TV_channels', row_key='field', col_key='cell_type', 
               species='mouse', data=u"""

This table describes the ion channel densities and voltage shifts
for a mouse tuberculoventral cell model.
Ad-hoc model, based on the t-stellate cell model, but adjusted
to match the data from Kuo and Trussell.

-----------------------------------------------------------------------------------------------------------------------------------
                  pyramidal   
                           
soma_napyr_gbar   350.0  [1]      
soma_nap_gbar     0.
soma_kdpyr_gbar   80.0   [1]
soma_kcnq_gbar    0.
soma_kif_gbar     150.0  [1]
soma_kis_gbar     40.0   [1]
soma_ihpyr_gbar   2.8    [1]     
soma_leak_gbar    2.8    [1]
soma_leak_erev    -62.0  [1]
soma_e_na         50.    [1]
soma_e_k          -81.5  [1]
soma_e_h          -43.0  [1]
soma_na_type      napyr
soma_Cap          12     [1]


-----------------------------------------------------------------------------------------------------------------------------------

[1] Kuo-Trussell.


""")

add_table_data('sgc_channels', row_key='field', col_key='cell_type', 
               species='mouse', data=u"""

This table describes the ion channel densities (and voltage shifts if necessary)
for different cell types in the original Rothman Manis 2003 model.
Data from Table 1, except for "octopus" cells, which is modified (see note 3)

-----------------------------------------------------------------------------------------------------------------------------------
                    sgc-a         sgc-bm   
                                        
sgc_name            a             bm     
soma_na_gbar        350.  [1]     350.  [1]
soma_kht_gbar       58.0  [1]     58.0  [1]
soma_klt_gbar       80.0  [1]     80.0  [1]
soma_ihap_gbar      3.0   [1]     0.0   [1]
soma_ihap_eh        -41.0 [1]     -41.0 [1]
soma_ihbm_gbar      0.0   [1]     3.0   [1]
soma_ihbm_eh        -41.0 [1]     -41.0 [1]
soma_leak_gbar      2.0   [1]     2.0   [1]
soma_leak_erev      -65   [1]     -65   [1]
soma_na_type        jsrna [1]     jsrna [1]
soma_Cap            12.0  [1]     12.0  [1]
soma_e_k            -84   [1]     -84   [1]
soma_e_na           50.   [1]     50.   [1]

-----------------------------------------------------------------------------------------------------------------------------------

[1] Liu, Manis, Davis, 2003
    Age "P10" (cultured SGC cells), Temperature=22C; code translates to 34.
    Units are nS.

""")

add_table_data('sgcgp_channels', row_key='field', col_key='cell_type', 
               species='guineapig', data=u"""

This table describes the ion channel densities (and voltage shifts if necessary)
for different cell types in the original Rothman Manis 2003 model.
Data from Table 1, except for "octopus" cells, which is modified (see note 3)

-----------------------------------------------------------------------------------------------------------------------------------
                    sgc-a         sgc-bm   
                                        
sgc_name            a             bm     
soma_na_gbar        1000. [1]     1000. [1]
soma_kht_gbar       150.0 [1]     150.0 [1]
soma_klt_gbar       200.0 [1]     200.0 [1]
soma_ihap_gbar      3.0   [1]     0.0   [1]
soma_ihap_eh        -41.0 [1]     -41.0 [1]
soma_ihbm_gbar      0.0   [1]     3.0   [1]
soma_ihbm_eh        -41.0 [1]     -41.0 [1]
soma_leak_gbar      2.0   [1]     2.0   [1]
soma_leak_erev      -65   [1]     -65   [1]
soma_na_type        jsrna [1]     jsrna [1]
soma_Cap            12.0  [1]     12.0  [1]
soma_e_k            -84   [1]     -84   [1]
soma_e_na           50.   [1]     50.   [1]

-----------------------------------------------------------------------------------------------------------------------------------

[1] Liu, Manis, Davis, 2003
    Age "P10" (cultured SGC cells), Temperature=22C; code translates to 34.
    Units are nS.
    Data are translated to guinea pig...

""")


