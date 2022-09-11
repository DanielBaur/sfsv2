



############################################
### imports
############################################


import subprocess
import numpy as np
import json
import scipy.integrate as integrate
import scipy.constants as constants
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.stats import rv_continuous
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import wimprates
import os
from random import randrange
import time
from datetime import timedelta





############################################
### variable input
############################################


# paths and files: NEST installation
abspath_nest_installation = os.getenv('ABSPATH_NEST_INSTALLATION')
abspath_nest_installation_install = abspath_nest_installation +"install/"
abspath_nest_installation_build = abspath_nest_installation +"build/"
abspath_nest_installation_nest = abspath_nest_installation +"nest/"
abspath_nest_installation_nest_include_detectors = abspath_nest_installation_nest +"include/Detectors/"
abspathfile_nest_installation_execNEST_cpp = abspath_nest_installation_nest +"src/execNEST.cpp"
abspathfile_nest_installation_execNEST_bin = abspath_nest_installation_install +"bin/execNEST"


# The following dictionary resembles an exemplary 'detector_dict'
exemplary_darwin_baseline_detector_dict = {
    # primary scintillation (S1) parameters
    "g1"                : 0.12,                     # phd per S1 phot at dtCntr (not phe), divide out 2-PE effect,                          JN: 0.119, LUX_Run03: 0.1170 (0.117+/-0.003 WS,0.115+/-0.005 D-D,0.115+/-0.005 CH3T,0.119+/-0.001 LUXSim), XENON10: 0.073
    "sPEres"            : 0.37,                     # single phe (=PE=photoelectrons) resolution (Gaussian assumed),                        JN: 0.38, LUX_Run03: 0.37 (arXiv:1910.04211.), XENON10: 0.58
    "sPEthr"            : 0.35,                     # POD threshold in phe, usually used IN PLACE of sPEeff,                                JN: 0.35, LUX_Run03: (0.3 * 1.173) / 0.915 (arXiv:1910.04211.), XENON10: 0.35
    "sPEeff"            : 0.9,                      # actual efficiency, can be used in lieu of POD threshold, units: fractional,           JN: 0.90, LUX_Run03: 1.00 (arXiv:1910.04211), XENON10: 1.00
    "noiseBaseline[0]"  : 0.0,                      # baseline noise mean in PE (Gaussian),                                                 JN: 0.0, LUX_Run03: 0.00 (arXiv:1910.04211 says -0.01), XENON10: 0.0
    "noiseBaseline[1]"  : 0.0,                      # baseline noise width in PE (Gaussian),                                                JN: 0.0, LUX_Run03: 0.08 (arXiv:1910.04211), XENON10: 0.0
    "noiseBaseline[2]"  : 0.0,                      # baseline noise mean in e- (for grid wires),                                           JN: none, LUX_Run03: 0.0, XENON10: 0.0
    "noiseBaseline[3]"  : 0.0,                      # baseline noise width in e- (for grid wires),                                          JN: none, LUX_Run03: 0.0, XENON10: 0.0
    "P_dphe"            : 0.2,                      # chance 1 photon makes 2 phe instead of 1 in Hamamatsu PMT, units: fractional,         JN: 0.22, LUX_Run03: 0.173 (arXiv:1910.04211), XENON10: 0.2
    "coinWind"          : 100,                      # S1 coincidence window in ns,                                                          JN: 100, LUX_Run03: 100 (1310.8214), XENON10: 100
    "coinLevel"         : 3,                        # how many PMTs have to fire for an S1 to count,                                        JN: 3, LUX_Run03: 2 (1512.03506), XENON10: 2
    "numPMTs"           : 494,                      # for coincidence calculation,                                                          JN: 494, LUX_Run03: 119 (122 minus 3 off), XENON10: 89
    "OldW13eV"          : "true",                   # default true, which means use "classic" W instead of Baudis / EXO's,                  JN: none, LUX_Run03: "true", XENON10: "true"
    "noiseLinear[0]"    : 0.0e-2,                   # S1->S1 Gaussian-smeared with noiseL[0]*S1, units: fraction NOT %!                     JN: none, LUX_Run03: 0.0e-2 (1910.04211 p.12, to match 1610.02076 Fig. 8.), XENON10: 3e-2
    "noiseLinear[1]"    : 0.0e-2,                   # S2->S2 Gaussian-smeared with noiseL[1]*S2, units: fraction NOT %!                     JN: none, LUX_Run03: 0.0e-2 (1910.04211 p.12, to match 1610.02076 Fig. 8.), XENON10: 3e-2
    # ionization and secondary scintillation (S2) parameters
    "g1_gas"            : 0.1,                      # phd per S2 photon in gas, used to get SE size, units: phd per e-,                     JN: 0.102, LUX_Run03: 0.1 (0.1 in 1910.04211), XENON10: 0.0655
    "s2Fano"            : 3.6,                      # Fano-like fudge factor for SE width, dimensionless,                                   JN: 3.61, LUX_Run03: 3.6 (3.7 in 1910.04211; this matches 1608.05381 better), XENON10: 3.61
    "s2_thr"            : 100,                      # the S2 threshold in phe or PE, *not* phd. Affects NR most,                            JN: 100.0, LUX_Run03: (150.0 * 1.173) / 0.915 (65-194 pe in 1608.05381), XENON10: 300.0
    "E_gas"             : 10.0,                     # field in kV/cm between liquid/gas border and anode,                                   JN: 10.85, LUX_Run03: 6.25 (6.55 in 1910.04211), XENON10: 12.0
    "eLife_us"          : 5000.0,                   # the drift electron mean lifetime in micro-seconds,                                    JN: 1600.0, LUX_Run03: 800.0 (p.44 of James Verbus PhD thesis Brown), XENON10: 2200.0
    # thermodynamic properties
#    "inGas"             : "false",                 # (duh),                                                                               JN: "false", LUX_Run03: commented out, XENON10: "false"
    "T_Kelvin"          : 175.0,                    # for liquid drift speed calculation, temperature in Kelvin,                            JN: 175.0, LUX_Run03: 173.0 (1910.04211), XENON10: 177.0
    "p_bar"             : 2.0,                      # gas pressure in units of bars, it controls S2 size,                                   JN: 2.0, LUX_Run03: 1.57 (1910.04211), XENON10: 2.14
    # data analysis parameters and geometry
    "dtCntr"            : 822.0,                    # center of detector for S1 corrections, in usec.,                                      JN: 822.0, LUX_Run03: 160.0 (p.61 Dobi thesis UMD, 159 in 1708.02566), XENON10: 40.0
    "dt_min"            : 75.8,                     # minimum. Top of detector fiducial volume, units: microseconds,                        JN: 75.8, LUX_Run03: 38.0 (1608.05381), XENON10: 20.0
    "dt_max"            : 1536.5,                   # maximum. Bottom of detector fiducial volume, units: microseconds,                     JN: 1536.5, LUX_Run03: 305.0 (1608.05381), XENON10: 60.0
    "radius"            : 1300.0,                   # millimeters (fiducial rad), units: millimeters,                                       JN: 1300., LUX_Run03: 200.0 (1512.03506), XENON10: 50.0
    "radmax"            : 1350.0,                   # actual physical geo. limit, units: millimeters,                                       JN: 1350., LUX_Run03: 235.0 (1910.04211), XENON10: 50.0
    "TopDrift"          : 3005.0,                   # top of drif volume in mm not cm or us, i.e., this *is* where dt=0, z=0mm is cathode,  JN: 3005.0, LUX_Run03: 544.95 (544.95 in 1910.04211), XENON10: 150.0
    "anode"             : 3012.5,                   # the level of the anode grid-wire plane in mm,                                         JN: 3012.5, LUX_Run03: 549.2 (1910.04211 and 549 in 1708.02566), XENON10: 152.5
    "gate"              : 3000.0,                   # mm. this is where the E-field changes (higher),                                       JN: 3000.0, LUX_Run03: 539.2 (1910.04211 and 539 in 1708.02566), XENON10: 147.5
    "cathode"           : 250.0,                    # mm. defines point below which events are gamma-X                                      JN: 250, LUX_Run03: 55.90 (55.9-56 in 1910.04211,1708.02566), XENON10: 1.00
    # 2D (xy) position reconstruction
    "PosResExp"         : 0.015,                    # exp increase in pos recon res at hi r, units: 1/mm,                                   JN: 0.015, LUX_Run03: 0.015 (arXiv:1710.02752 indirectly), XENON10: 0.015
    "PosResBase"        : 30.,                      # baseline unc in mm, see NEST.cpp for usage,                                           JN: 30.0, LUX_Run03: 70.8364 ((1710.02752 indirectly), XEONON10: 70.8364
}


translate_parameter_to_latex_dict = { # latex symbol, written quantity, latex unit
    "g1"             : [r"$g_1$",                   r"g1 parameter",                   r"$\mathrm{\frac{phd}{photon}}$"],
    "g1_gas"         : [r"$g_1^{\mathrm{gas}}$",    r"jfk",                            r"$\mathrm{\frac{phd}{photon}}$"],
    "E_gas"          : [r"$E_{\mathrm{gas}}$",      r"amplification field strength",   r"$\frac{\mathrm{V}}{\mathrm{cm}}$"],
    "eLife_us"       : [r"$\tau_{e^{-}}$",          r"electron life-time",             r"$\mathrm{\upmu s}$"],
    "e_drift_v_cm"   : [r"$E_{\mathrm{drift}}$",    r"drift field strength",           r"$\frac{\mathrm{V}}{\mathrm{cm}}$"],
}





############################################
### general definitions
############################################



# This function is used to retrieve a Python3 dictionary stored as a .json file.
def get_dict_from_json(input_pathstring_json_file):
    with open(input_pathstring_json_file, "r") as json_input_file:
        ret_dict = json.load(json_input_file)
    return ret_dict


# This function is used to save a Python3 dictionary as a .json file.
def write_dict_to_json(output_pathstring_json_file, save_dict):
    with open(output_pathstring_json_file, "w") as json_output_file:
        json.dump(save_dict, json_output_file, indent=4)
    return


def compute_array_sum(array_list):
    array_sum = np.zeros_like(array_list[0])
    for k in range(len(array_sum)):
        for array in array_list:
            array_sum[k] += array[k]
    array_sum = [float(jfk) for jfk in array_sum]
    return array_sum


def bin_centers_from_interval(
    interval,
    n_bins,
):
    binwidth = (interval[1]-interval[0])/n_bins
    bin_edges = np.linspace(interval[0], interval[1], n_bins+1, True)
    return [be +0.5*binwidth for be in bin_edges[:-1]]


def convert_detector_header_into_detector_dict(
    abspathfile_detector_header,
):

    """
    This function is used to extract the detector parameters from a detector header file and to write them into a detector dictionary.
    """

    # initialization
    line_list = []
    detector_dict = {}
    parameter_list = [
        "g1", "sPEres", "sPEthr", "sPEeff", "noiseBaseline[0]", "noiseBaseline[1]", "noiseBaseline[2]",
        "noiseBaseline[3]", "P_dphe", "coinWind", "coinLevel", "numPMTs", "OldW13eV", "noiseLinear[0]",
        "noiseLinear[1]", "g1_gas", "s2Fano", "s2_thr", "E_gas", "eLife_us", "T_Kelvin", "p_bar", "dtCntr",
        "dt_min", "dt_max", "radius", "radmax", "TopDrift", "anode", "gate", "cathode", "PosResExp", "PosResBase"]

    # reading the lines of the input file
    with open(abspathfile_detector_header, 'r') as inputfile:
        for line in inputfile:
            line_list.append(line)

    # extracting the detector parameters into 
    for line in line_list:
        if "=" in line and any([param in line for param in parameter_list]) and ";" in line:
            line_mod = line
            for k in range(0,50):
                line_mod = line_mod.replace("  "," ")
            line_mod_list = list(line_mod.split(" "))
            extracted_parameter = line_mod_list[1]
            extracted_value = line_mod_list[-1][:-2] if extracted_parameter=="OldW13eV" else float(line_mod_list[-1][:-2])
            detector_dict.update({extracted_parameter:extracted_value})

    return detector_dict


def compute_g2_from_detector_configuration(
    detector_dict, # dict, detector configuration
):

    """
    This function is used to compute the g2 value of a specific detector configuration.
    The comments on the official repo regarding the g2 calculation are outdated.
    The code below was based on the NESTv2.3.9 computation of g2.
    """

    # initialization
    EPS_GAS = 1.00126
    EPS_LIQ = 1.85 # instead of 1.96 as in old NEST versions
    NEST_AVO = 6.0221409e+23
    molarMass = 131.293
    alpha = 0.137
    beta = 4.70e-18
    gamma = 0
    bara = detector_dict["p_bar"]
    gasGap = detector_dict["anode"] -detector_dict["TopDrift"]
    E_liq = detector_dict["E_gas"]/(EPS_LIQ/EPS_GAS)
    T_Kelvin = detector_dict["T_Kelvin"]

    # computing 'ExtEff'
    em1 = 8.807528626640e4 -2.026247730928e3*T_Kelvin +1.747197309338e1*T_Kelvin**2 -6.692362929271e-2*T_Kelvin**3 +9.607626262594e-5*T_Kelvin**4
    em2 = 5.074800229635e5 -1.460168019275e4*T_Kelvin +1.680089978382e2*T_Kelvin**2 -9.663183204468e-1*T_Kelvin**3 +2.778229721617e-3*T_Kelvin**4 -3.194249083426e-6*T_Kelvin**5
    em3 = -4.659269964120e6 +1.366555237249e5*T_Kelvin -1.602830617076e3*T_Kelvin**2 +9.397480411915e-0*T_Kelvin**3 -2.754232523872e-2*T_Kelvin**4 +3.228101180588e-5*T_Kelvin**5
    ExtEff = 1 - em1 *np.exp(-em2 *(E_liq**em3))
    if ExtEff > 1:
        ExtEff = 1
    elif ExtEff < 0:
        ExtEff = 0

    # computing 'SE'
    VaporP_bar = 10**(4.0519 - 667.16 / T_Kelvin)
    p_Pa = bara *10**5
    RidealGas = 8.31446261815324 # NEST.hh, Joules/mole/Kelvin
    RealGasA = 0.4250 # NEST.hh, m^6*Pa/mol^2 or m^4*N/mol^2
    RealGasB = 5.105e-5 # NEST.hh, m^3/mol
    rho = 1.0 / ((RidealGas*T_Kelvin)**3 / (p_Pa*(RidealGas*T_Kelvin)**2 +RealGasA*p_Pa*p_Pa) +RealGasB) # Van der Waals equation, mol/m^3
    rho = rho *molarMass * 1e-6
    elYield = (alpha*detector_dict["E_gas"]*1e3 -beta*(NEST_AVO*rho/molarMass))*gasGap*0.1
    SE = elYield *detector_dict["g1_gas"] *0.5 # I have no idea where this factor of 0.5 comes from... I added it such that the function outputs the same value as the NEST onscreen output
    #print(ExtEff, SE)

    # computing 'g2'
    g2 = ExtEff*SE
    return g2


def compute_drift_velocity_from_detector_configuration(
    detector_dict, # dict, detector configuration
    drift_field_v_cm, # float, drift velocity in V/cm
):

    """
    This function is used to compute the drift velocity of a specific detector configuration in mm/usec.
    The code below is based on the NESTv2.3.9 computation of according to the function 'NESTcalc::GetDriftVelocity_Liquid' (lines 2258 to 2408).
    """

    # initialization
    fn = "compute_drift_velocity_from_detector_configuration"
    Kelvin = detector_dict["T_Kelvin"] # extracting LXe temperatire in K
    eField = float(drift_field_v_cm) # extracting drift field in V/cm
    assert (Kelvin >= 100 and Kelvin <= 230), f"{fn}: temperature out of range"

    speed = 0.0

    polyExp = [
        [-3.1046, 27.037, -2.1668, 193.27, -4.8024, 646.04, 9.2471], # 100K
        [-2.7394, 22.760, -1.7775, 222.72, -5.0836, 724.98, 8.7189], # 120K
        [-2.3646, 164.91, -1.6984, 21.473, -4.4752, 1202.2, 7.9744], # 140K
        [-1.8097, 235.65, -1.7621, 36.855, -3.5925, 1356.2, 6.7865], # 155K
        [-1.5000, 37.021, -1.1430, 6.4590, -4.0337, 855.43, 5.4238], # 157K
        [-1.4939, 47.879, 0.12608, 8.9095, -1.3480, 1310.9, 2.7598], # 163K
        [-1.5389, 26.602, -.44589, 196.08, -1.1516, 1810.8, 2.8912], # 165K
        [-1.5000, 28.510, -.21948, 183.49, -1.4320, 1652.9, 2.884], # 167K
        [-1.1781, 49.072, -1.3008, 3438.4, -.14817, 312.12, 2.8049], # 184K
        [1.2466, 85.975, -.88005, 918.57, -3.0085, 27.568, 2.3823], # 200K
        [334.60, 37.556, 0.92211, 345.27, -338.00, 37.346, 1.9834],] # 230K

    Temperatures = [100., 120., 140., 155., 157., 163., 165., 167., 184., 200., 230.]


    if (Kelvin >= Temperatures[0] and Kelvin < Temperatures[1]):
        i = 0
    elif (Kelvin >= Temperatures[1] and Kelvin < Temperatures[2]):
        i = 1
    elif (Kelvin >= Temperatures[2] and Kelvin < Temperatures[3]):
        i = 2
    elif (Kelvin >= Temperatures[3] and Kelvin < Temperatures[4]):
        i = 3
    elif (Kelvin >= Temperatures[4] and Kelvin < Temperatures[5]):
        i = 4
    elif (Kelvin >= Temperatures[5] and Kelvin < Temperatures[6]):
        i = 5
    elif (Kelvin >= Temperatures[6] and Kelvin < Temperatures[7]):
        i = 6
    elif (Kelvin >= Temperatures[7] and Kelvin < Temperatures[8]):
        i = 7
    elif (Kelvin >= Temperatures[8] and Kelvin < Temperatures[9]):
        i = 8
    elif (Kelvin >= Temperatures[9] and Kelvin <= Temperatures[10]):
        i = 9

    j = i + 1;

    Ti = Temperatures[i]
    Tf = Temperatures[j]

    vi = polyExp[i][0] * np.exp(-eField / polyExp[i][1]) +polyExp[i][2] * np.exp(-eField / polyExp[i][3]) +polyExp[i][4] * np.exp(-eField / polyExp[i][5]) + polyExp[i][6]
    vf = polyExp[j][0] * np.exp(-eField / polyExp[j][1]) +polyExp[j][2] * np.exp(-eField / polyExp[j][3]) +polyExp[j][4] * np.exp(-eField / polyExp[j][5]) + polyExp[j][6]

    if math.isclose(Kelvin, Ti, rel_tol=1e-7):
        return vi
    if math.isclose(Kelvin, Tf, rel_tol=1e-7):
        return vf

    if (vf < vi):
        offset = (np.sqrt((Tf * (vf - vi) - Ti * (vf - vi) - 4.) * (vf - vi)) +np.sqrt(Tf - Ti) * (vf + vi)) /(2. * np.sqrt(Tf - Ti))
        slope = -(np.sqrt(Tf - Ti) *np.sqrt((Tf * (vf - vi) - Ti * (vf - vi) - 4.) * (vf - vi)) -(Tf + Ti) * (vf - vi)) /(2. * (vf - vi))
        speed = 1. / (Kelvin - slope) + offset
    else:
        slope = (vf - vi) / (Tf - Ti);
        speed = slope * (Kelvin - Ti) + vi;

    if speed <= 0.:
        speed = 0.1
        raise Exception(f"drift velocity of {speed} mm/usec is less than zero")

    return speed


def adjust_detector_drift_time_parameters(
    detector_dict,
    drift_field_v_cm):

    """
    This function is used to adjust the "dtCntr", "dt_min", and "dt_max" parameters in a 'detector_dict' for the respective drift field.
    """

    new_detector_dict = detector_dict.copy()
    drift_velocity_mm_usec = compute_drift_velocity_from_detector_configuration(detector_dict,drift_field_v_cm)
    max_drift_time_usec = (detector_dict['TopDrift']-detector_dict['cathode'])/drift_velocity_mm_usec
    dt_min = 0.1*max_drift_time_usec
    dt_max = 0.9*max_drift_time_usec
    dtCntr = 0.5*max_drift_time_usec
    new_detector_dict.update({
        "dtCntr" : float(f"{dtCntr:.1f}"),
        "dt_max" : float(f"{dt_max:.1f}"),
        "dt_min" : float(f"{dt_min:.1f}"),})
    return new_detector_dict


def calc_active_xenon_mass_of_detector_dict_t(detector_dict):

    """
    This function is used to compute the active xenon mass of a NEST detector dict in metric tonnes.
    """

    Kelvin = detector_dict["T_Kelvin"]
    lxe_density_t_m3 =  2.90 # This function to compute 'lxe_density_t_m3' below is taken from NEST v2.3.9
    lxe_density_t_m3 = 2.9970938084691329e+02 *np.exp(-8.2598864714323525e-02 * Kelvin) -1.8801286589442915e+06 * np.exp(-   ((Kelvin - 4.0820251276172212e+02) /2.7863170223154846e+01)**2   ) -5.4964506351743057e+03 * np.exp(-   ((Kelvin - 6.3688597345042672e+02) /1.1225818853661815e+02)**2   ) +8.3450538370682614e+02 * np.exp(-   ((Kelvin + 4.8840568924597342e+01) /7.3804147172071107e+03)**2   ) -8.3086310405942265e+02
    tpc_height_m = detector_dict["TopDrift"]/1000
    tpc_radius_m = detector_dict["radmax"]/1000
    active_volume_m3 = tpc_radius_m**2 *math.pi *tpc_height_m
    active_xenon_mass_t = lxe_density_t_m3 *active_volume_m3
    return active_xenon_mass_t





############################################
### spectra computation
############################################


color_s1_default = "#008f0e" # green as S1 color in miscfic TPC scheme
colro_s2_default = "#e39700" # orange as S2 color in miscfig TPC scheme
color_wimps_default = '#004A9B' # ALUFR logo blue
color_nrs_default = "#06bcd4" # blue as xenon liquid color in miscfig TPC scheme
color_ers_default = '#C1002A' # ALUFR logo red
color_hep_default = "pink"
color_pep_default = "cyan"
color_pp_default = "orange"
color_cno_default = "blue"
color_be7_default = "green"
color_b8_default = "red"
color_atm_default = "purple"
color_dsnb_default =  "brown"
color_radiogenic_neutrons_default = "gray"
color_nunubetabeta_default = "olive"


xenon_isotopic_composition = {
    "124" : {
        "m_u" : 123.905893, # atom mas in atomic mass units u
        "abundance" : 0.00095, # isotopic abundance, not in percent
    },
    "126" : {
        "m_u" : 125.904274,
        "abundance" : 0.00089,
    },
    "128" : {
        "m_u" : 127.9035313,
        "abundance" : 0.01910,
    },
    "129" : {
        "m_u" : 128.9047794,
        "abundance" : 0.26401,
    },
    "130" : {
        "m_u" : 129.9035080,
        "abundance" : 0.04071,
    },
    "131" : {
        "m_u" : 130.9050824,
        "abundance" : 0.21232,
    },
    "132" : {
        "m_u" : 131.9041535,
        "abundance" : 0.26909,
    },
    "134" : {
        "m_u" : 133.9053945,
        "abundance" : 0.10436,
    },
    "136" : {
        "m_u" : 135.907219,
        "abundance" : 0.08857,
    },
}


def convert_grabbed_csv_to_ndarray(
    abspathfile_grabbed_csv,
):
    """
    This function is used to convert a .csv file generated with 'WebPlotDigitizer' ('https://automeris.io/WebPlotDigitizer/', accessed: 24th July 2022) into a numpy structured array.
    """
    data_tuple_list = []
    with open(abspathfile_grabbed_csv, 'r') as input_file:
        for k, line in enumerate(input_file):
            line_list = list(line.split(","))
            x = line_list[0]
            y = line_list[1]
            data_tuple_list.append((x, y))
    dtype = np.dtype([
        ("x_data", np.float64),
        ("y_data", np.float64),])
    ndarray = np.array(data_tuple_list, dtype)
    return ndarray


def give_spectrum(
    # entering -1 yields the default values. (They are different for ER- and NR-events, so they are not directly
    # used here). The user recieves a notification about the values used.
    interaction_type ="ER", # ER or NR
    bkg_array=[], # For ER: [nunubetabeta, pp_7Be]
                # For NR: [nu_sum, nu_8B, nu_hep, nu_DSNB, nu_atm]
    bin_size = -1, # in keV
    min_energy = -1, # in keV
    max_energy = -1, # in keV
    flag = "midpoint", # specifies the method for integrating. Choose "midpoint" or "integrate".
    y_t = 1e5, # runtime of the detector [y] * mass of the detector [t]. Is ignored when num_events > 0.
    field_drift = 200, #in V/cm
    num_events = -1, # if bigger than zero, y_t will be redefined to gurantee the given number of events.
                     # Otherwise, the value is ignored.
    file_prefix = "/home/catharina/Desktop/praktikum_freiburg_2022/resources/", #Path to spectrums saved as numpy arrays.
    verbose=True
):
#Sanity check: plotting energy_bins and spectrum.
#spectrum_dict =give_spectrum(interaction_type="ER",bkg_array=[1,0],num_events=1e6)
#energy_bins =spectrum_dict["E_min[keV]"]
#spectrum =  spectrum_dict["numEvts"]
#plt.xlabel("Energy [keV]")
#plt.ylabel("Number of events")
#plt.xscale("log")
#plt.yscale("log")
#plt.scatter(energy_bins, spectrum, s=5)

    NR_spectrum_name_list=["NR_spectrum_nu_sum", "NR_spectrum_nu_8B",
                       "NR_spectrum_nu_hep", "NR_spectrum_nu_DSNB", "NR_spectrum_nu_atm"]
    ER_spectrum_name_list=["ER_spectrum_nunubetabeta", "ER_spectrum_pp_7Be"]

    spectrum_raw =[]
    function_name = "give_spectrum"
    if verbose: print(f"{function_name} running.")

    assert interaction_type in ["NR", "ER"], "invalid interaction type, choose 'NR' or 'ER' (default)"
    if interaction_type=="NR":
        # Default options
        if min_energy<0:
            min_energy = 0.424
            if verbose: print("Default lower energy-bound for NR-events used, min_energy = 0.424 keV")
        if max_energy<0:
            max_energy = 11.818
            if verbose: print("Default upper energy-bound for NR-events used, max_energy = 11.818 keV")
        if len(bkg_array)==0:
            bkg_array=[1,0,0,0,0]
            if verbose: print("Default NR-background used. The background consists of nu_sum only.")
            if verbose: print("If you want to use a different background, enter for bkg_array = [nu_sum, nu_8B, nu_hep, nu_DSNB, nu_atm]")
        if bin_size<0:
            bin_size = 0.05
            if verbose: print(f"Default bin_size={bin_size} keV for NR-events used.")

        assert len(bkg_array)==5, "invalid number of parameters for NR-background, required 5, given "+str(len(bkg_array))

        for file_name in NR_spectrum_name_list:
            # loading the spectrum from a saved numpy-array.
            spectrum_raw.append(np.load(file_prefix+file_name+".npy"))

            # finding the minimum energy and the maximum energy allowed for later asserting
            # the validity of given min_energy, max_energy.
            if len(spectrum_raw)==1:
                min_energy_allowed=spectrum_raw[-1]["Energy [keV]"][0]
                max_energy_allowed=spectrum_raw[-1]["Energy [keV]"][-1]

            if spectrum_raw[-1]["Energy [keV]"][0]<min_energy_allowed:
                min_energy_allowed = spectrum_raw[-1]["Energy [keV]"][0]
            if spectrum_raw[-1]["Energy [keV]"][-1]>max_energy_allowed:
                max_energy_allowed = spectrum_raw[-1]["Energy [keV]"][-1]

    else:
        #Default options
        if min_energy<0:
            min_energy = 1
            if verbose: print("Default lower energy-bound for ER-events used, min_energy = 1 keV")
        if max_energy<0:
            max_energy = 192
            if verbose: print("Default upper energy-bound for ER-events used, max_energy = 192 keV")
        if len(bkg_array)==0:
            bkg_array=[1,1]
            if verbose: print("Default ER-background used. It contains nunubetabeta and pp_7Be without scaling them further.")
            if verbose: print("If you want to use a different background, enter for bkg_array = [nunubetabeta, pp_7Be]")
        if bin_size<0:
            bin_size = 1
            if verbose: print(f"Default bin_size={bin_size} keV for ER-events used.")

        assert len(bkg_array)==2, "invalid number of parameters for NR-background, required 2, given "+str(len(bkg_array))

        for file_name in ER_spectrum_name_list:
            # loading the spectrum from a saved numpy-array.
            spectrum_raw.append(np.load(file_prefix+file_name+".npy"))

            # finding the minimum energy and the maximum energy allowed for later asserting
            # the validity of given min_energy, max_energy.
            if len(spectrum_raw)==1:
                min_energy_allowed=spectrum_raw[-1]["Energy [keV]"][0]
                max_energy_allowed=spectrum_raw[-1]["Energy [keV]"][-1]
            if spectrum_raw[-1]["Energy [keV]"][0]<min_energy_allowed:
                min_energy_allowed = spectrum_raw[-1]["Energy [keV]"][0]
            if spectrum_raw[-1]["Energy [keV]"][-1]>max_energy_allowed:
                max_energy_allowed = spectrum_raw[-1]["Energy [keV]"][-1]

    #print(min_energy_allowed, max_energy_allowed)

    # Checking the soundness of given min_energy and max_energy.
    assert min_energy<max_energy, "Invalid energy bounds. "
    assert min_energy>min_energy_allowed, "Minimum energy too small, smallest allowed value "+str(min_energy_allowed)
    assert max_energy<max_energy_allowed, "Maximum energy too big, biggest allowed value "+str(max_energy_allowed)

    # creating energy-bins with width bin_size.
    # bins are centered around the values in the energy_bins array.
    energy_bins =[min_energy+bin_size/2]
    spectrum =[]
    while energy_bins[-1]+bin_size*3/2<=max_energy:
        energy_bins.append(energy_bins[-1]+bin_size)


    if verbose: print(f"Actual upper and lower energy bounds (change depends on bin_size): {energy_bins[0]-bin_size/2}, {energy_bins[-1]+bin_size/2}")

    if num_events>0:
        total_events = 0
        for i in range(len(bkg_array)):
            total_events+= bkg_array[i]* integrate.quad(lambda x:
                    np.interp(x, spectrum_raw[i]["Energy [keV]"],
                              spectrum_raw[i]["Rate [evts/y/t/keV]"]),energy_bins[0]-bin_size/2, energy_bins[-1]+bin_size/2)[0]
        y_t = num_events/total_events

    assert flag in ["midpoint", "integrate"], "Invalid flag, choose 'midpoint' or 'integrate'."

    if flag=="midpoint":
    # Simply takes the midpoint of the bin and plugs it into the rate-function.
        for e in energy_bins:
            rate=0
            for i in range(len(bkg_array)):
                rate+= bkg_array[i] * np.interp(e, spectrum_raw[i]["Energy [keV]"],
                                                spectrum_raw[i]["Rate [evts/y/t/keV]"])
            rate *= bin_size*y_t
            spectrum.append(int(rate+0.5)) # round to the next integer value

    if flag=="integrate":
    # Integrates the rate-function over the whole bin. Computationally more expensive.
        for e in energy_bins:
            rate=0
            for i in range(len(bkg_array)):
                rate+= bkg_array[i] * integrate.quad(lambda x:
                    np.interp(x, spectrum_raw[i]["Energy [keV]"],
                              spectrum_raw[i]["Rate [evts/y/t/keV]"]), e-bin_size/2, e+bin_size/2)[0]
            rate *= y_t
            spectrum.append(int(rate+0.5)) # round to the next integer value

    if verbose: print(str(sum(spectrum))+ " "+interaction_type+"-events generated.")
    if verbose: print(f"number of bins: {len(energy_bins)}")

    spectrum_dict = {
            "numEvts" : spectrum,
            "type_interaction" : interaction_type,
            "E_min[keV]" : energy_bins,
            "E_max[keV]" : energy_bins,
            "field_drift[V/cm]" : field_drift,
            "x,y,z-position[mm]" : "-1",
            "seed" : 0
        }

    return spectrum_dict


def calculate_wimp_induced_nuclear_recoil_rate_dru(
    # main parameters
    e_nr_kev, # float, nuclear recoil energy in keV
    mass_wimp_gev = 40.0, # float, hypothetical WIMP mass in GeV/c^2
    cross_section_wimp_proton_cm2 = 1e-47, # float, hypothetical WIMP-proton cross-section in cm^2
    energy_threshold_kev = 2.0, # float, minimum nuclear recoil energy in keV_nr detectable with detector (required for the integration of the Maxwellian velocity distribution)
    wimp_dark_matter_mass_density_gevcm3 = 0.3, # float, dark matter energy density in GeV/c^2/cm^3
    velocity_escape_kmps = 544.0, # float, galactic escape velocity in km/s
    velocity_circular_kmps = 220, # float, earth's circular velocity in km/s
    mass_target_nucleus_u = 130.9050824, # float, mass of target neucleus in amu
    mass_proton_u = 1.00727647, # float, proton mass in amu
    mass_number_target_nucleus = 131, # int, atomic mass number of target nucleus
    # flags
    flag_verbose = False,
):

    """
    This functions is used to calculate the WIMP recoil spectrum based on the standard Lewin Smith ansatz.
    """

    # conversion to SI units
    mass_target_nucleus = mass_target_nucleus_u *1.66053906660 *10**(-27) # conversion to kg
    mass_proton = mass_proton_u *1.66053906660 *10**(-27) # conversion to kg
    mass_wimp = (mass_wimp_gev *10**9 *1.60218 *10**(-19))/((3*10**8)**2) # conversion to kg
    energy_threshold = energy_threshold_kev *1000 *1.60218 *10**(-19) # conversion to Joule
    e_nr_j = e_nr_kev *1000 *1.60218 *10**(-19) # conversion to Joule
    cross_section_wimp_proton = cross_section_wimp_proton_cm2 *(1/10000) # conversion to m^2
    wimp_dark_matter_mass_density = wimp_dark_matter_mass_density_gevcm3 *(1000000) # conversion to GeV/m^3
    velocity_escape = velocity_escape_kmps *1000 # conversion to m/s
    velocity_circular = velocity_circular_kmps *1000 # conversion to m/s

    # derived quantities
    number_of_target_nuclei = 1000/mass_target_nucleus # number of target nuclei within 1 tonne of liquid xenon
    redmass_wimp_proton = (mass_proton *mass_wimp)/(mass_proton +mass_wimp)
    redmass_wimp_nucleus = (mass_wimp *mass_target_nucleus)/(mass_wimp +mass_target_nucleus)
    velocity_min = np.sqrt((mass_target_nucleus *energy_threshold)/(2 *(redmass_wimp_nucleus**2)))
    velocity_min = np.sqrt((mass_target_nucleus *e_nr_j)/(2 *(redmass_wimp_nucleus**2)))

    # integrated spin-independent WIMP nucleus cross-section
    cross_section_integrated_wimp_nucleus_spin_independent = mass_number_target_nucleus**2 *(redmass_wimp_nucleus/redmass_wimp_proton)**2 *cross_section_wimp_proton

    # defining the nuclear form factor (Helm model: https://iopscience.iop.org/article/10.1088/0253-6102/55/6/21/pdf)
    def nuclear_form_factor(energy_nuclear_recoil):
        # recoil energy -> momentum transfer
        p = np.sqrt(2*mass_target_nucleus *(energy_nuclear_recoil*1000 *1.60218 *10**(-19)))
        q = p/(1.0545718*10**(-34))
        r_n = 1.2*mass_number_target_nucleus**(1/3)*(1/10**(15)) # nuclear radius in m
        # calculating substeps
        qr_n = q*r_n
        s = 1*(1/10**(15)) # skin thickness in m
        qs = q*s
        #a = 3*(np.sin(math.radians(qr_n))-qr_n*np.cos(math.radians(qr_n)))
        a = 3*(np.sin(qr_n)-qr_n*np.cos(qr_n))
        b = (qr_n)**3
        c = np.exp(-((q*s)**2/2))
        return a/b *c

    # defining the Maxwellian WIMP velocity distribution
    def wimp_velocity_distribution(
        v, # float, velocity in km/s
        v_circ = velocity_circular):
        v_0 = np.sqrt(2/3) *v_circ
        j = 4 *np.pi *v**2
        f = np.sqrt((np.pi *v_0**2))**3
        exp = np.exp(-((v**2)/(v_0**2)))
        return (j/f) *exp

    # differential WIMP-nucleus cross-section
    def cross_section_differential_wimp_nucleus(v, energy_nuclear_recoil):
        return (mass_target_nucleus *cross_section_integrated_wimp_nucleus_spin_independent *nuclear_form_factor(energy_nuclear_recoil=energy_nuclear_recoil)**2) /(2 *redmass_wimp_nucleus**2 *v**2)

    # integrating the product of Maxwellian velocity distribution and differential WIMP-nucleus corss-section
    def integrand_function(v, energy_nuclear_recoil):
        return wimp_velocity_distribution(v) *v *cross_section_differential_wimp_nucleus(v, energy_nuclear_recoil=energy_nuclear_recoil)
    # NOTE: You might ask why there is a factor v**3 instead of just v in the integrand function.
    # The reason is that you are integrating over f(v) in three spatial dimensions.
    # Hence (using spherical coordinates) you also pick up a factor of r**2 *sin(theta).
    # Integrating sin(theta) over dtheta and dphi gives you a factor of 4*np.pi and then you still have to additionally integrate r**2.

    # coputing the final product
    scaling_factor = (60 *60 *24 *365) *(1/((1/1000) *1.60218 *10**(+19))) # conversion from events/s/J into events/ (t x y x keV)
    differential_recoil_rate_dru = \
        scaling_factor \
        *number_of_target_nuclei \
        *(wimp_dark_matter_mass_density/mass_wimp_gev) \
        *quad(
            integrand_function,
            velocity_min,
            velocity_escape +velocity_circular,
            args=(e_nr_kev)
        )[0]

    return differential_recoil_rate_dru


def maxwellian_velocity_distribution(
    v, # float, velocity in m/s
    v_0, # float, mean velocity in m/s
):
    """
    This function resembles a Maxwellian distribution function.
    """
    j = 4 *np.pi *v**2
    f = np.sqrt((np.pi *v_0**2))**3
    exp = np.exp(-((v**2)/(v_0**2)))
    return (j/f) *exp


def calculate_nuclear_form_factor_helm_approximation(
    nuclear_recoil_energy_j,
    target_nucleus_mass_kg,
    target_nucleus_mass_number,
):
    """
    This function is used to calculate the Helm approximation of the nuclear form factor.
    See: https://iopscience.iop.org/article/10.1088/0253-6102/55/6/21/pdf
    """
    # recoil energy -> momentum transfer
    p = np.sqrt(2*target_nucleus_mass_kg *nuclear_recoil_energy_j)
    q = p/(1.0545718*10**(-34))
    r_n = 1.2*target_nucleus_mass_number**(1/3)*(1/10**(15)) # nuclear radius in m
    # calculating substeps
    qr_n = q*r_n
    s = 1*(1/10**(15)) # skin thickness in m
    qs = q*s
    #a = 3*(np.sin(math.radians(qr_n))-qr_n*np.cos(math.radians(qr_n)))
    a = 3*(np.sin(qr_n)-qr_n*np.cos(qr_n))
    b = (qr_n)**3
    c = np.exp(-((q*s)**2/2))
    return a/b *c


def calculate_differential_spin_independent_wimp_nucleus_cross_section(
    v, # float, relative WIMP-nucleus velocity in m/s
    nuclear_recoil_energy_j, # float, resulting nuclear recoil energy in Joule
    target_nucleus_mass_kg, # float
    target_nucleus_mass_number, # int
    reduced_mass_wimp_nucleus_kg, # float
    reduced_mass_wimp_proton_kg, # float
    wimp_proton_cross_section_m2, # float
):
    """
    This function resembles the energy-dependent differential spin-independent WIMP-nucleus cross-section.
    """
    fraction = target_nucleus_mass_kg/(2 *reduced_mass_wimp_nucleus_kg**2 *v**2)
    integrated_spin_independent_wimp_nucleon_cross_section = target_nucleus_mass_number**2 *(reduced_mass_wimp_nucleus_kg/reduced_mass_wimp_proton_kg)**2 *wimp_proton_cross_section_m2
    helm_form_factor = calculate_nuclear_form_factor_helm_approximation(
        nuclear_recoil_energy_j = nuclear_recoil_energy_j,
        target_nucleus_mass_kg = target_nucleus_mass_kg,
        target_nucleus_mass_number = target_nucleus_mass_number)
    return  fraction *integrated_spin_independent_wimp_nucleon_cross_section *helm_form_factor**2


def wimp_recoil_rate_integrand_function(
    v,
    nuclear_recoil_energy_j,
    wimp_proton_cross_section_m2,
    target_nucleus_mass_kg,
    target_nucleus_mass_number,
    reduced_mass_wimp_nucleus_kg,
    reduced_mass_wimp_proton_kg,
    earth_circular_velocity_mps):
    """
    This function resembles the integrand of XXX.
    """
    f = maxwellian_velocity_distribution(
        v = v,
        v_0 = np.sqrt(2/3)*earth_circular_velocity_mps)
    del_sigma_del_enr = calculate_differential_spin_independent_wimp_nucleus_cross_section(
        v = v,
        nuclear_recoil_energy_j = nuclear_recoil_energy_j,
        target_nucleus_mass_kg = target_nucleus_mass_kg,
        target_nucleus_mass_number = target_nucleus_mass_number,
        reduced_mass_wimp_nucleus_kg = reduced_mass_wimp_nucleus_kg,
        reduced_mass_wimp_proton_kg = reduced_mass_wimp_proton_kg,
        wimp_proton_cross_section_m2 = wimp_proton_cross_section_m2)
    return f *v *del_sigma_del_enr


def calculate_wimp_induced_nuclear_recoil_rate_events_t_y_kev(
    # main parameters
    nuclear_recoil_energy_kev, # float, nuclear recoil energy in keV
    wimp_mass_gev, # float, hypothetical WIMP mass in GeV/c^2
    wimp_proton_cross_section_cm2, # float, hypothetical WIMP-proton cross-section in cm^2
    # detector material
    target_nucleus_mass_u = 130.9050824, # float, mass of target neucleus in amu
    target_nucleus_mass_number = 131, # int, atomic mass number of target nucleus
    # model parameters
    dark_matter_energy_density_gev_cm3 = 0.3, # float, dark matter energy density in GeV/c^2/cm^3
    milky_way_escape_velocity_kmps = 544.0, # float, galactic escape velocity in km/s
    earth_circular_velocity_kmps = 220, # float, earth's circular velocity in km/s
    # flags
    flag_verbose = False,):

    """
    This function is used to calculate the differential WIMP recoil spectrum based on the standard Lewin Smith ansatz.
    """

    # conversion to SI units
    nuclear_recoil_energy_j = nuclear_recoil_energy_kev *1000 *1.60218 *10**(-19)
    target_nucleus_mass_kg = target_nucleus_mass_u *1.66053906660 *10**(-27) # conversion to kg
    proton_mass_kg = 1.00727647 *1.66053906660 *10**(-27) # conversion to kg
    wimp_mass_kg = (wimp_mass_gev *10**9 *1.60218 *10**(-19))/((3*10**8)**2) # conversion to kg
    wimp_proton_cross_section_m2 = wimp_proton_cross_section_cm2 *(1/10000) # conversion to m^2
    dark_matter_energy_density_gev_m3 = dark_matter_energy_density_gev_cm3 *(1000000) # conversion to GeV/m^3
    earth_circular_velocity_mps = earth_circular_velocity_kmps *1000 # conversion to m/s
    milky_way_escape_velocity_mps = earth_circular_velocity_kmps *1000 # conversion to m/s

    # calculating derived quantities
    target_nuclei_in_one_tonne = 1000/target_nucleus_mass_kg
    reduced_mass_wimp_proton_kg = (proton_mass_kg *wimp_mass_kg)/(proton_mass_kg +wimp_mass_kg)
    reduced_mass_wimp_nucleus_kg = (wimp_mass_kg *target_nucleus_mass_kg)/(wimp_mass_kg +target_nucleus_mass_kg)
    unit_scaling_factor = (60 *60 *24 *365) *(1/((1/1000) *1.60218 *10**(+19))) # conversion from events/s/J into events/y/keV
    v_min = np.sqrt((target_nucleus_mass_kg *nuclear_recoil_energy_j)/(2 *(reduced_mass_wimp_nucleus_kg**2)))
    #v_min = 0
    v_max = milky_way_escape_velocity_mps +earth_circular_velocity_mps

    # coputing the final product
    factor = target_nuclei_in_one_tonne *dark_matter_energy_density_gev_m3 *(1/wimp_mass_gev) *unit_scaling_factor
    integral = quad(
        wimp_recoil_rate_integrand_function,
        v_min,
        v_max,
        args = (
            nuclear_recoil_energy_j,
            wimp_proton_cross_section_m2,
            target_nucleus_mass_kg,
            target_nucleus_mass_number,
            reduced_mass_wimp_nucleus_kg,
            reduced_mass_wimp_proton_kg,
            earth_circular_velocity_mps,))[0]
    return factor *integral


def calculate_wimp_induced_nuclear_recoil_rate_in_natural_xenon_events_t_y_kev(
    # main parameters
    nuclear_recoil_energy_kev, # float, nuclear recoil energy in keV
    wimp_mass_gev, # float, hypothetical WIMP mass in GeV/c^2
    wimp_proton_cross_section_cm2, # float, hypothetical WIMP-proton cross-section in cm^2
    # detector material
    xenon_isotopic_composition_dict = xenon_isotopic_composition, # dict, isotopic composition of natural xenon
    # model parameters
    dark_matter_energy_density_gev_cm3 = 0.3, # float, dark matter energy density in GeV/c^2/cm^3
    milky_way_escape_velocity_kmps = 544.0, # float, galactic escape velocity in km/s
    earth_circular_velocity_kmps = 220, # float, earth's circular velocity in km/s
    # flags
    flag_verbose = False,):

    """
    This function is used to calculate the differential WIMP recoil spectrum based on the standard Lewin Smith ansatz in natural xenon.
    """

    diffrate_events_t_y_kev = 0
    for a_str in [*xenon_isotopic_composition_dict]:
        diffrate_events_t_y_kev += xenon_isotopic_composition_dict[a_str]["abundance"] *calculate_wimp_induced_nuclear_recoil_rate_events_t_y_kev(
            nuclear_recoil_energy_kev = nuclear_recoil_energy_kev,
            wimp_mass_gev = wimp_mass_gev,
            wimp_proton_cross_section_cm2 = wimp_proton_cross_section_cm2,
            target_nucleus_mass_u = xenon_isotopic_composition_dict[a_str]["m_u"],
            target_nucleus_mass_number = int(a_str),
            dark_matter_energy_density_gev_cm3 = dark_matter_energy_density_gev_cm3,
            milky_way_escape_velocity_kmps = milky_way_escape_velocity_kmps,
            earth_circular_velocity_kmps = earth_circular_velocity_kmps,
            flag_verbose = flag_verbose,) 

    return diffrate_events_t_y_kev


def calculate_nunubetabeta_er_rate_events_t_y_kev_alt(
    electronic_recoil_energy_kev_ee, # electronic recoil energy in keV_ee
    abundance_xe136 = 0.08857, # abundance of xe136, default is natural abundance, not in percent
):

    """
    Gives dN/dT for the two neutrino double beta decay of 136 Xe as estimated by the Primakoff-Rosen Appromximation.
    For details see: https://www.physik.uzh.ch/groups/groupbaudis/aspera09/wiki/doku.php?id=simulation:0v2b:2nbb_analytic (accessed: 30th July 2022)
    """

    # definitions
    electron_mass_kev = 510.998950
    Q_kev = 2.4578e3 # Q-value in keV
    Q_em = Q_kev/electron_mass_kev
    xe136_isotopic_mass_kg = 135.907219 *1.66053906660 *10**(-27)
    xe136_half_life_s = 2.165 *10**21 *(365*24*60*60)
    if electronic_recoil_energy_kev_ee > Q_kev:
        return 0

    # Primakoff-Rosen approximation
    def Primakoff_Rosen_approximation(recoil_energy_kev):
        T = recoil_energy_kev/electron_mass_kev
        Q = Q_kev/electron_mass_kev
        return T *(Q-T)**5 *(1 +2*T +(4/3)*T**2 +(1/3)*T**3 +(1/30)*T**4)

    # computing Primakoff-Rosen approximation PDF
    normalization_constant = quad(Primakoff_Rosen_approximation, 0, Q_kev)[0]
    primakoff_rosen_approximation  = (1/normalization_constant) *Primakoff_Rosen_approximation(electronic_recoil_energy_kev_ee)

    # computing rate in events / (t x y x keV)
    xe136_atoms_per_metric_tonne = 1000 *abundance_xe136 /xe136_isotopic_mass_kg
    xe136_activity_per_metric_tonne = xe136_atoms_per_metric_tonne *(np.log(2)/xe136_half_life_s)
    xe136_decays_per_metric_tonne_per_year = xe136_activity_per_metric_tonne *(60*60*24*365)
    return xe136_decays_per_metric_tonne_per_year *primakoff_rosen_approximation









###  WIMPs  #######################################
# This function is used to generate a WIMP recoil spectrum.
def gen_wimp_recoil_spectrum(
    # main parameters
    mass_wimp_gev = 40.0, # in GeV/c^2
    cross_section_wimp_proton_cm2 = 1e-47, # in cm^2
    mass_detector_target_t = 40, # in tonnes
    time_exposure_y = 5, # in years
    e_nr_kev = None, # default is 'None'
    # parameters
    energy_threshold_kev = 2, # in keV_nr
    wimp_dark_matter_mass_density_gevcm3 = 0.3, # in GeV/c^2/cm^3
    velocity_escape_kmps = 544, # in km/s
    velocity_circular_kmps = 220, # in km/s
    mass_target_nucleus_u = 130.9050824, # in amu
    mass_proton_u = 1.00727647, # in amu
    mass_number_target_nucleus = 131,
    # output parameters
    energy_nuclear_recoil_min = 0, # in keV_nr
    energy_nuclear_recoil_max = 60, # in keV_nr
    number_of_bins_or_samples = 120,
    # flags
    flag_output = "histogram"
):

    ### model calculation
    # conversion to SI units
    mass_target_nucleus = mass_target_nucleus_u *1.66053906660 *10**(-27) # conversion to kg
    mass_proton = mass_proton_u *1.66053906660 *10**(-27) # conversion to kg
    mass_detector_target = mass_detector_target_t *1000 # conversion to kg
    time_exposure = time_exposure_y *365 *24 *60 *60 # conversion to s
    mass_wimp = (mass_wimp_gev *10**9 *1.60218 *10**(-19))/((3*10**8)**2) # conversion to kg
    energy_threshold = energy_threshold_kev *1000 *1.60218 *10**(-19) # conversion to Joule
    cross_section_wimp_proton = cross_section_wimp_proton_cm2 *(1/10000) # conversion to m^2
    wimp_dark_matter_mass_density = wimp_dark_matter_mass_density_gevcm3 *(1000000) # conversion to GeV/m^3
    velocity_escape = velocity_escape_kmps *1000 # conversion to m/s
    velocity_circular = velocity_circular_kmps *1000 # conversion to m/s
    # derived quantities
    number_of_target_nuclei = mass_detector_target/mass_target_nucleus
    redmass_wimp_proton = (mass_proton *mass_wimp)/(mass_proton +mass_wimp)
    redmass_wimp_nucleus = (mass_wimp *mass_target_nucleus)/(mass_wimp +mass_target_nucleus)
    velocity_min = np.sqrt((mass_target_nucleus *energy_threshold)/(2 *(redmass_wimp_nucleus**2)))
    # integrated spin-independent WIMP nucleus cross-section
    cross_section_integrated_wimp_nucleus_spin_independent = mass_number_target_nucleus**2 *(redmass_wimp_nucleus/redmass_wimp_proton)**2 *cross_section_wimp_proton
    # nuclear form factors (Helm model: https://iopscience.iop.org/article/10.1088/0253-6102/55/6/21/pdf)
    def nuclear_form_factor(energy_nuclear_recoil):
        # recoil energy -> momentum transfer
        p = np.sqrt(2*mass_target_nucleus *(energy_nuclear_recoil*1000 *1.60218 *10**(-19)))
        q = p/(1.0545718*10**(-34))
        r_n = 1.2*mass_number_target_nucleus**(1/3)*(1/10**(15)) # nuclear radius in m
        # calculating substeps
        qr_n = q*r_n
        s = (1/10**(15)) # skin thickness in m
        qs = q*s
        #a = 3*(np.sin(math.radians(qr_n))-qr_n*np.cos(math.radians(qr_n)))
        a = 3*(np.sin(qr_n)-qr_n*np.cos(qr_n))
        b = (qr_n)**3
        c = np.exp(-((q*s)**2/2))
        return a/b *c
    # wimp velocity distribution
    def wimp_velocity_distribution(v):
        v_0 = np.sqrt(2/3) *velocity_circular
        j = 4 *np.pi *v**2
        f = np.sqrt((np.pi *v_0**2))**3
        exp = np.exp(-((v**2)/(v_0**2)))
        return (j/f) *exp
    # differential WIMP-nucleus cross-section
    def cross_section_differential_wimp_nucleus(v, energy_nuclear_recoil):
        return (mass_target_nucleus *cross_section_integrated_wimp_nucleus_spin_independent *nuclear_form_factor(energy_nuclear_recoil=energy_nuclear_recoil)**2) /(2 *redmass_wimp_nucleus**2 *v**2)
    # integrand function
    # NOTE: You might ask why there is a factor v**3 instead of just v in the integrand function.
    # The reason is that you are integrating over f(v) in three spatial dimensions.
    # Hence (using spherical coordinates) you also pick up a factor of r**2 *sin(theta).
    # Integrating sin(theta) over dtheta and dphi gives you a factor of 4*np.pi and then you still have to additionally integrate r**2.
    def integrand_function(v, energy_nuclear_recoil):
        return wimp_velocity_distribution(v) *v *cross_section_differential_wimp_nucleus(v, energy_nuclear_recoil=energy_nuclear_recoil)
    # differential recoil rate in DRU (i.e. events/kg/d/keV)
    def differential_recoil_rate_dru(energy_nuclear_recoil):
        scaling_factor = (1/(mass_detector_target)) *(60 *60 *24) *(1/((1/1000) *1.60218 *10**(+19))) # conversion from events/s/J into events/kg/d/keV
        return scaling_factor *number_of_target_nuclei *(wimp_dark_matter_mass_density/mass_wimp_gev) *quad(integrand_function, velocity_min, velocity_escape +velocity_circular, args=(energy_nuclear_recoil))[0]
    # differential recoil rate adapted to detector settings (i.e. events/detector_mass/exposure_time/keV)
    def differential_recoil_rate_det(energy_nuclear_recoil):
        scaling_factor = time_exposure *(1/((1/1000) *1.60218 *10**(+19))) # conversion from events/s/J into events/kg/d/keV
        return scaling_factor *number_of_target_nuclei *(wimp_dark_matter_mass_density/mass_wimp_gev) *quad(integrand_function, velocity_min, velocity_escape +velocity_circular, args=(energy_nuclear_recoil))[0]

    ### generating output
    # returning absolute rates by integrating the differential energy spectrum; i.e. energy bin centers, absolute counts per energy bin
    if flag_output == "histogram":
        binwidth = (energy_nuclear_recoil_max -energy_nuclear_recoil_min)/number_of_bins_or_samples
        energy_bin_centers = np.linspace(start=energy_nuclear_recoil_min+0.5*binwidth, stop=energy_nuclear_recoil_max-0.5*binwidth, num=number_of_bins_or_samples, endpoint=True)
        counts_per_energy_bin = np.zeros_like(energy_bin_centers)
        for i in range(len(energy_bin_centers)):
            counts_per_energy_bin[i] = quad(differential_recoil_rate_det, energy_bin_centers[i]-0.5*binwidth, energy_bin_centers[i]+0.5*binwidth)[0]
        return energy_bin_centers, counts_per_energy_bin
    # returning the differential recoil spectrum in DRU (events/kg/d/keV)
    elif flag_output == "rate":
        energy_nuclear_recoil_list = np.linspace(start=energy_nuclear_recoil_min, stop=energy_nuclear_recoil_max, num=number_of_bins_or_samples, endpoint=True)
        diff_rate_list = np.zeros_like(energy_nuclear_recoil_list)
        for i in range(len(energy_nuclear_recoil_list)):
            diff_rate_list[i] = differential_recoil_rate_dru(energy_nuclear_recoil=energy_nuclear_recoil_list[i])
        return energy_nuclear_recoil_list, diff_rate_list
    # returning a single value of the differential recoil spectrum in DRU
    elif flag_output == "single_dru_value" and e_nr_kev != None:
        return differential_recoil_rate_dru(energy_nuclear_recoil=e_nr_kev)
    else:
        print("invalid input: 'flag_output'")
        return


def diffratefunct(
    e_nr_kev,
    a,
    abundance,
    target_mass_u,
    wimp_mass_gev,
    cross_section_cm2,
):

    """
    This function resembles the 'diffratefunct' function translated from a C++ script into Python to compute the differential WIMP rate in events per t x y x keV.
    """
    
    N0 = 6.02214199e26 # Avogadro Constant
    c = 299792458.0 # vacuum speed of light [m/s]
    mp = 0.9382728 # mass of the proton [GeV/c^2]
    u = 0.93149401 # Atomic mass unit [GeV/c^2]
    # WIMP halo constants
    v_0 = 220000.0 # real mean velocity of DM Maxewellian distribution [m/s]
    v_esc = 544000.0 # real escape velocity of DM [m/s]
    v_E = 232000.0 # real mean velocity of Earth [m/s]
    rho_DM = 0.3 # local DM density [GeV/cm^3]
    # conversion factors
    fmtoGeV = 1.0/0.197327 # conversion from fm to 1/GeV
    keVtoGeV = 1e-6 # conversion from keV to GeV
    daytos = 60.0*60.0*24.0 # conversion from day to seconds
    stoday = 1.0/daytos # conversion from s to day
    mtocm = 1e2 # conversion from m to cm
    cm2topbarn = 1e36 # conversion from cm^2 to picobarn
    pbarntocm2 = 1e-36 # conversion from picobarn to cm^2
    epsilon = 1e-10
    # nuclear constants for form factor
    a0 = 0.52 #fm
    s = 0.9  #fm
    
    # reduced masses
    m_wimp = wimp_mass_gev
    sig = cross_section_cm2
    munucwimp =m_wimp*u*target_mass_u/(m_wimp+u*target_mass_u)
    mupwimp = m_wimp*mp/(m_wimp+mp)

    # mean kinetic energy of WIMP
    E0 = 0.5*1.e6*m_wimp*(v_0/c)**2 #   // 1/2 mv^2 and 1e6 is conversion from GeV to keV	
    # kinematic factor
    r = 4.0*m_wimp*u*target_mass_u / (m_wimp+u*target_mass_u)**2	

    # Formfactor -----------------------------------------------------------------
    # variables
    c0 = 1.23 * a**(1/3) - 0.6 # fm
    rn = np.sqrt(c0**2 +7.0/3.0 *math.pi**2 *a0*2 -5.0*s**2)
    # FF definition
    F = 3.0*(np.sin(np.sqrt(2.0*u*target_mass_u*e_nr_kev*keVtoGeV)*rn*fmtoGeV)-(np.sqrt(2.0*u*target_mass_u*e_nr_kev*keVtoGeV)*rn*fmtoGeV) *np.cos(np.sqrt(2.0*u*target_mass_u*e_nr_kev*keVtoGeV)*rn*fmtoGeV))/   (np.sqrt(2.0*u*target_mass_u*e_nr_kev*keVtoGeV)*rn*fmtoGeV)**3*np.exp(-1.0/2.0*  (np.sqrt(2.0*u*target_mass_u*e_nr_kev*keVtoGeV)*s*fmtoGeV)**2)		

    # Velocity integral -----------------------------------------------------------
    # minimum velocity to generate recoil Er=e_nr_kev
    vmin = np.sqrt(e_nr_kev/(E0*r))*v_0
    # k-factor for normaization
    k1_k0 = 1.0/(math.erf(v_esc/v_0)-2.0/np.sqrt(math.pi)*v_esc/v_0*np.exp(-v_esc**2/v_0**2))

    # velocity integral
    # -- separation in the different energy bins see Savage et al, JCAP  04 (2009) 010, or Sebastian's PhD thesis
    # -- if the standard L&S approach should be used, the first if clause should be changed to "if (0.0<vmin && vmin<=(v_esc+v_E))"
    #    and evergything else should be removed
    if vmin > 0 and vmin <= (v_esc+v_E):
        velocityint = (np.sqrt(math.pi)/4.0*v_0/v_E*(math.erf((vmin+v_E)/v_0)-math.erf((vmin-v_E)/v_0))-np.exp(-(v_esc/v_0)**2))
    else:
        velocityint = 0
        #raise Exception(f"vim = {vmin}")

    # uncorrected Rate
    R0  =2.0/np.sqrt(math.pi)*N0/target_mass_u*rho_DM/m_wimp*pbarntocm2*a**2*munucwimp**2/mupwimp**2*v_0*mtocm*daytos*sig*cm2topbarn

    # differential rate
    diffrateval = R0*abundance*k1_k0*(1.0/(E0*r))*velocityint *F**2

    return diffrateval


def give_flat_spectrum(
    # entering -1 yields the default values. (They are different for ER- and NR-events, so they are not directly
    # used here). The user recieves a notification about the values used if verbose=True.
    interaction_type ="ER", # ER or NR
    bin_size = -1, # in keV
    min_energy = -1, # in keV
    max_energy = -1, # in keV
    field_drift = 200, #in V/cm
    num_events = 1e6, # Total number of events. Number of events per bin depends on this parameter
    verbose=True
):
    function_name = "give_flat_spectrum"

    if verbose: print(f"{function_name} running.")
    assert interaction_type in ["NR", "ER"], "invalid interaction type, choose 'NR' or 'ER' (default)"

    if interaction_type=="NR":
        # Default options
        if min_energy<0:
            min_energy = 0.424
            if verbose: print("Default lower energy-bound for NR-events used, min_energy = 0.424 keV")
        if max_energy<0:
            max_energy = 11.818
            if verbose: print("Default upper energy-bound for NR-events used, max_energy = 11.818 keV")
        if bin_size<0:
            bin_size = 0.05
            if verbose: print(f"Default bin_size={bin_size} keV for NR-events used.")
    else:
        #Default options
        if min_energy<0:
            min_energy = 1
            if verbose: print("Default lower energy-bound for ER-events used, min_energy = 1 keV")
        if max_energy<0:
            max_energy = 192
            if verbose: print("Default upper energy-bound for ER-events used, max_energy = 192 keV")
        if bin_size<0:
            bin_size = 1
            if verbose: print(f"Default bin_size={bin_size} keV for ER-events used.")

    assert min_energy<max_energy, "Invalid energy bounds. "

    # creating energy-bins with width bin_size.
    # bins are centered around the values in the energy_bins array.
    energy_bins =[min_energy+bin_size/2]
    spectrum =[]
    while energy_bins[-1]+bin_size*3/2<=max_energy:
        energy_bins.append(energy_bins[-1]+bin_size)

    if verbose: print(f"Actual upper and lower energy bounds (change depends on bin_size): {energy_bins[0]-bin_size/2}, {energy_bins[-1]+bin_size/2}")

    events_per_bin = int(num_events/len(energy_bins)+0.5)

    assert events_per_bin>0, "Please enter for num_events a value >= {len(energy_bins)} to avoid empty bins. (NEST hates that and will come to haunt you if you don't.)"

    spectrum=np.ones(len(energy_bins), dtype = int)*events_per_bin

    if verbose:
        print(f"number of bins: {len(energy_bins)}")
        print(f"Events per bin: {events_per_bin}")
        print(f"total number of events: {len(energy_bins)*events_per_bin}")

    spectrum_dict = {
            "numEvts" : spectrum,
            "type_interaction" : interaction_type,
            "E_min[keV]" : energy_bins,
            "E_max[keV]" : energy_bins,
            "field_drift[V/cm]" : field_drift,
            "x,y,z-position[mm]" : "-1",
            "seed" : 0
        }

    return spectrum_dict


"""

###################### EXAMPLE FOR TESTING ##################################

import matplotlib.pyplot as plt

spectrum_dict =give_flat_spectrum(interaction_type="NR",num_events=1e6)
energy_bins =spectrum_dict["E_min[keV]"]
spectrum =  spectrum_dict["numEvts"]
plt.xlabel("Energy [keV]")
plt.ylabel("Number of events")
plt.xscale("log")
plt.yscale("log")
plt.scatter(energy_bins, spectrum, s=5)


"""


spectrum_dict_default_dict = {
    "nr_wimps_nat_xe"                           : {
        "latex_label"                           : r"WIMPs",
        "color"                                 : color_wimps_default,
        "linestyle"                             : "-",
        "linewidth"                             : 2,
        "zorder"                                : 2,
        "differential_rate_computation"         : calculate_wimp_induced_nuclear_recoil_rate_in_natural_xenon_events_t_y_kev,
        "differential_rate_parameters"          : {
            "wimp_mass_gev"                     : 25,
            "wimp_proton_cross_section_cm2"     : 1e-47,
            "xenon_isotopic_composition_dict"   : xenon_isotopic_composition,
            "dark_matter_energy_density_gev_cm3": 0.3,
            "milky_way_escape_velocity_kmps"    : 544.0,
            "earth_circular_velocity_kmps"      : 220,
            "flag_verbose"                      : False,
        },
    },
    "nr_wimps_wimprates"                        : {
        "latex_label"                           : r"WIMPs",
        "color"                                 : color_wimps_default,
        "linestyle"                             : "-",
        "linewidth"                             : 2,
        "zorder"                                : 2,
        "differential_rate_computation"         : wimprates.rate_wimp_std,
        "differential_rate_parameters"          : {
            "mw"                                : 25,
            "sigma_nucleon"                     : 1e-47,
        },
    },
    "nr_atm"		                            : {
        "latex_label"                           : r"atm",
        "color"                                 : color_atm_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_dsnb"		                            : {
        "latex_label"                           : r"DSNB",
        "color"                                 : color_dsnb_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_b8"		                                : {
        "latex_label"                           : r"$^{8}\mathrm{B}$",
        "color"                                 : color_b8_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_hep"		                            : {
        "latex_label"                           : r"hep",
        "color"                                 : color_hep_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_neutrons_baseline"                      : {
        "latex_label"                           : r"radiogenic neutrons (baseline)",
        "color"                                 : color_radiogenic_neutrons_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
    "nr_neutrons_less_cryostat"                 : {
        "latex_label"                           : r"radiogenic neutrons (better cryostat)",
        "color"                                 : color_radiogenic_neutrons_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
    "nr_neutrons_less_ptfe_pmt"                 : {
        "latex_label"                           : r"radiogenic neutrons (better PTFE and PMTs)",
        "color"                                 : color_radiogenic_neutrons_default,
        "linestyle"                             : "--",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
    "nr_neutrons_34t"                           : {
        "latex_label"                           : r"radiogenic neutrons (34t FV)",
        "color"                                 : color_radiogenic_neutrons_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
    "nr_neutrons_28t"                           : {
        "latex_label"                           : r"radiogenic neutrons (28t FV)",
        "color"                                 : color_radiogenic_neutrons_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
    "nr_neutrons_20t"                           : {
        "latex_label"                           : r"radiogenic neutrons (20t FV)",
        "color"                                 : color_radiogenic_neutrons_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
    "er_pp"		                                : {
        "latex_label"                           : r"pp",
        "color"                                 : color_pp_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "er_be7_384"	                            : {
        "latex_label"                           : r"$^{7}\mathrm{Be}\,(384\,\mathrm{keV})$",
        "color"                                 : color_be7_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "er_be7_861"	                            : {
        "latex_label"                           : r"$^{7}\mathrm{Be}\,(861\,\mathrm{keV})$",
        "color"                                 : color_be7_default,
        "linestyle"                             : "--",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "er_o15"		                            : {
        "latex_label"                           : r"$^{15}\mathrm{O}$",
        "color"                                 : color_cno_default,
        "linestyle"                             : "--",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "er_n13"		                            : {
        "latex_label"                           : r"$^{13}\mathrm{N}$",
        "color"                                 : color_cno_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "er_pep"		                            : {
        "latex_label"                           : r"pep",
        "color"                                 : color_pep_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "er_nunubetabeta"	                        : {
        "latex_label"                           : r"$\nu\nu\beta\beta$",
        "color"                                 : color_nunubetabeta_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : calculate_nunubetabeta_er_rate_events_t_y_kev_alt,
        "differential_rate_parameters"          : {
            "abundance_xe136"                   : 0.08857,
        },
    },
    "er_rn222"                                  : { # assuming 0.1 uBq/kg
        "latex_label"                           : r"naked $^{214}\mathrm{Pb}$ betas ($0.1\,\mathrm{\frac{\upmu Bq}{kg}}$ of $^{222}\mathrm{Rn}$)",
        "color"                                 : color_ers_default,
        "linestyle"                             : "--",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "mc_output",
    },
}


# adding combination profiles
spectrum_dict_default_dict.update({
    "er_be7"		                            : {
        "latex_label"                           : r"$^{7}\mathrm{Be}$",
        "color"                                 : color_be7_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"               : ["er_be7_384", "er_be7_861"],
    },
    "er_cno"		                            : {
        "latex_label"                           : r"CNO",
        "color"                                 : color_cno_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"               : ["er_o15", "er_n13"],
    },
    "nr_cno"		                            : {
        "latex_label"                           : r"CNO",
        "color"                                 : color_cno_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"               : ["nr_o15", "nr_n13", "nr_f17"],
    },
    "nr_be7"		                            : {
        "latex_label"                           : r"$^{7}\mathrm{Be}$",
        "color"                                 : color_be7_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"              : ["nr_be7_384", "nr_be7_861"],
    },
    # This is the combined NR background model used for the SFS study
    "combined_nr_background"                             : {
        "latex_label"                           : r"combined NR background",
        "color"                                 : color_nrs_default,
        "linestyle"                             : "-",
        "linewidth"                             : 2,
        "zorder"                                : 2,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"              : ["nr_atm", "nr_hep", "nr_b8", "nr_dsnb", "nr_neutrons_baseline"],
    },
    # This is the combined ER background model used for the SFS study
    "combined_er_background"                    : {
        "latex_label"                           : r"combined ER background",
        "color"                                 : color_ers_default,
        "linestyle"                             : "-",
        "linewidth"                             : 2,
        "zorder"                                : 2,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"              : ["er_be7_384", "er_be7_861", "er_nunubetabeta", "er_pp", "er_rn222"],
    },
})


def give_differential_rate_for_mc_output(
    recoil_energy_kev,
    xp,
    fp,
    left,
    right,
):
    if recoil_energy_kev < xp[0]:
        return left
    elif recoil_energy_kev > xp[-1]:
        return right
    else:
        index = np.argmin(np.abs(np.array(xp)-recoil_energy_kev))
        return fp[index]


def give_spectrum_dict(
    spectrum_name,                          # string, name of the spectrum to be sampled from, corresponding to one of the keys of 'spectrum_dict_default_dict'
    recoil_energy_kev_list,                 # list of two floats, resembling the energy interval within which events are generated
    # 
    abspath_spectra_files,                  # abspath, directory where the .csv files of the spectra are found
    exposure_t_y                            = 30*5, # float, fiducial exposure of the experiment
    num_events                              = [42, "exposure_rounded", "exposure_poisson"][2], # number of generated events, giving an integer will generate exactly that many events, giving 'exposure' will generate events according to 'exposure_t_y' (rounded to the next integer value), giving 'exposure_poisson' will generate an random integer number of events corresponding to a sample drawn from a Poissonian distribution with the parameter corresponding to the exposure of the experiment
    # nest parameters
    seed                                    = 0, # integer between 0 and 10000000, or "randomint" to generate a random integer between 0 and 10000000
    drift_field_v_cm                        = 200,
    xyz_pos_mm                              = "-1",
    # flags
    flag_spectrum_type                      = ["differential", "integral"][0],
    flag_verbose                            = False,
    flag_return_non_integer_events          = False,
    flag_inhibit_scaling                    = False,
    flag_number_of_output_spectrum_dicts    = 1, # positive non-zero int, number of generated output dictionaries, only relevant if 'flag_spectrum_type'=='integral', if equals 1 a dictionary is returned, if greater than 1 a list of dictionaries is returned
    # keywords
    spectrum_dict_default_values            = spectrum_dict_default_dict, # default 'spectrum_dict' values
    differential_rate_parameters            = {} # additional keyword argument values overwriting those from 'spectrum_dict_default_values'
):

    """
    This function is used to generate a 'spectrum_dict' based on one of the templates in 'spectrum_dict_default_dict'.
    If 'flag_spectrum_type' == "differential" the 'spectrum_dict' resembles the differential rate (in events per tonne x year x keV, computed for every 'recoil_energy_kev' value).
    If 'flag_spectrum_type' == "integral" the 'spectrum_dict' resembles the integrated number of events per aequidistant recoil energy bin (bin centers are given by the 'recoil_energy_kev' values).
    An obtained 'integral'-type 'spectrum_dict' would typically be fed into NEST for further processing.
    """

    # initializing
    fn = "give_spectrum_dict"
    if flag_verbose: print(f"{fn}: initializing '{fn}'")

    # instantiating the 'spectrum_dict'
    if flag_verbose: print(f"{fn}: instantiating the 'spectrum_dict'")
    spectrum_dict = spectrum_dict_default_values[spectrum_name].copy()
    spectrum_dict.update({
        "recoil_energy_kev_list"        : list(recoil_energy_kev_list),
        "exposure_t_y"                  : exposure_t_y,
        "num_events"                    : num_events,
        "seed"                          : seed,
        "field_drift[V/cm]"             : drift_field_v_cm,
        "x,y,z-position[mm]"            : xyz_pos_mm,
        "flag_verbose"                  : flag_verbose,
        "flag_spectrum_type"            : flag_spectrum_type,
        "differential_rate_parameters"  : differential_rate_parameters,
    })

    # looping over the constituent spectra and defining the combined spectrum differential rate function
    if flag_verbose: print(f"{fn}: defining the combined differential rate function")
    binwidth_kev = recoil_energy_kev_list[1]-recoil_energy_kev_list[0]
    recoil_energy_kev_min = recoil_energy_kev_list[0] -0.5*binwidth_kev
    recoil_energy_kev_max = recoil_energy_kev_list[-1] +0.5*binwidth_kev
    bin_edges_kev = np.linspace(start=recoil_energy_kev_min, stop=recoil_energy_kev_max, num=len(recoil_energy_kev_list)+1, endpoint=True)
    x_combined_differential_rate_spectrum = np.linspace(start=recoil_energy_kev_min, stop=recoil_energy_kev_max, num=3001, endpoint=True)
    if spectrum_dict["differential_rate_computation"] == "spectrum_sum":
        constituent_spectra_list = spectrum_dict["constituent_spectra_list"]
        flag_spectrum_sum = True
    else:
        constituent_spectra_list = [spectrum_name]
        flag_spectrum_sum = False
    constituent_spectra_differential_rate_list_list = []
    for constituent_spectrum in constituent_spectra_list:
        if spectrum_dict_default_values[constituent_spectrum]["differential_rate_computation"] == "interpolation_from_file":
            digitized_spectrum_ndarray = convert_grabbed_csv_to_ndarray(abspath_spectra_files +constituent_spectrum +".csv")
            differential_rate_function = np.interp
            differential_rate_param_dict = {"xp" : digitized_spectrum_ndarray["x_data"], "fp" : digitized_spectrum_ndarray["y_data"], "left" : digitized_spectrum_ndarray["y_data"][0], "right" : 0}
        elif spectrum_dict_default_values[constituent_spectrum]["differential_rate_computation"] == "mc_output":
            digitized_spectrum_ndarray = convert_grabbed_csv_to_ndarray(abspath_spectra_files +constituent_spectrum +".csv")
            differential_rate_function = give_differential_rate_for_mc_output
            differential_rate_param_dict = {"xp" : digitized_spectrum_ndarray["x_data"], "fp" : digitized_spectrum_ndarray["y_data"], "left" : digitized_spectrum_ndarray["y_data"][0], "right" : 0}
        elif callable(spectrum_dict_default_values[constituent_spectrum]["differential_rate_computation"]):
            differential_rate_function = spectrum_dict_default_values[constituent_spectrum]["differential_rate_computation"]
            differential_rate_param_dict = spectrum_dict["differential_rate_parameters"]
        constituent_spectrum_differential_rate_list = [differential_rate_function(x, **differential_rate_param_dict) for x in x_combined_differential_rate_spectrum]
        constituent_spectra_differential_rate_list_list.append(constituent_spectrum_differential_rate_list)
    y_combined_differential_rate_spectrum = compute_array_sum(constituent_spectra_differential_rate_list_list)
    def differential_spectrum_rate_events_t_y_kev(recoil_energy_kev):
        differential_rate_events_t_y_kev = np.interp(recoil_energy_kev, xp=x_combined_differential_rate_spectrum, fp=y_combined_differential_rate_spectrum, right=0)
        return differential_rate_events_t_y_kev

    # case: returning differential 'spectrum_dict'
    if  flag_spectrum_type == "differential":
        if flag_verbose: print(f"{fn}: returning 'differential'-type 'spectrum_dict'")
        differential_recoil_rate_events_t_y_kev = [differential_spectrum_rate_events_t_y_kev(e) for e in recoil_energy_kev_list]
        spectrum_dict.update({
            "differential_recoil_rate_events_t_y_kev"       : list(differential_recoil_rate_events_t_y_kev),
        })
        if callable(spectrum_dict["differential_rate_computation"]): # functions cannot be JSON-serialized
            spectrum_dict.pop("differential_rate_computation")


    # case: returning integral 'spectrum_dict'
    elif flag_spectrum_type == "integral":
        if flag_verbose: print(f"{fn}: returning 'integral'-type 'spectrum_dict'")
        integral_spectrum_dict_list = []

        # determining the expected number of events given the experiment's exposure
        if flag_verbose: print(f"\tdetermining the expected number of events given the experiment's exposure")
        expected_number_of_events_float_per_energy_bin = [exposure_t_y *integrate.quad(
            differential_spectrum_rate_events_t_y_kev,
            bc -0.5*binwidth_kev,
            bc +0.5*binwidth_kev,
        )[0] for bc in recoil_energy_kev_list]
        expected_number_of_events_float = np.sum(expected_number_of_events_float_per_energy_bin)
        expected_number_of_events_rounded = round(expected_number_of_events_float)
        expected_number_of_events_poisson = np.random.default_rng(seed=seed).poisson(expected_number_of_events_float, 1)[0]
        if flag_verbose: print(f"\t---> 'float': {expected_number_of_events_float}")
        if flag_verbose: print(f"\t---> 'rounded': {expected_number_of_events_rounded}")
        if flag_verbose: print(f"\t---> 'poisson': {expected_number_of_events_poisson}")
        if type(num_events)==int:
            n_samples = num_events
        elif num_events=="exposure_rounded":
            n_samples = expected_number_of_events_rounded
        elif num_events=="exposure_poisson":
            n_samples = expected_number_of_events_poisson

        # looping over 'flag_number_of_output_spectrum_dicts'
        for k in range(flag_number_of_output_spectrum_dicts):
            if (flag_verbose and flag_number_of_output_spectrum_dicts != 1) : print(f"\tk={k}/{flag_number_of_output_spectrum_dicts-1}")

            # drawing the samples from the custom spectrum pdf according to the specified number of events 'num_events'
            if (k != 0 and num_events=="exposure_poisson") : n_samples = np.random.default_rng(seed=randrange(10000001)).poisson(expected_number_of_events_float, 1)[0]
            if flag_verbose: print(f"\tdrawing {n_samples} samples from the custom spectrum pdf")
            samples = generate_samples_from_discrete_pdf(
                random_variable_values = recoil_energy_kev_list,
                pdf_values = expected_number_of_events_float_per_energy_bin,
                nos = n_samples,
                seed = seed,)
    #        if flag_verbose: print(f"\tsamples: {samples}")

            # histogramming the drawn samples according to the specified bin centers 'recoil_energy_kev_list'
            if flag_verbose: print(f"\thistogramming the drawn samples")
            hist, hist_bin_edges = np.histogram(a=samples, bins=np.linspace(start=recoil_energy_kev_min, stop=recoil_energy_kev_max, num=len(recoil_energy_kev_list)+1, endpoint=True))

            # assembling the 'integral'-type 'spectrum_dict'
            if flag_verbose: print(f"\tupdating the 'integral'-type 'spectrum_dict'")
            interaction_type = list(spectrum_name.split("_"))[0].upper() if flag_spectrum_sum == True else list(spectrum_name.split("_"))[1].upper()
            interaction_type = "ER" if "er" in list(spectrum_name.split("_")) else "NR"
            spectrum_dict.update({
                "numEvts"               : [int(entry) for entry in hist],
                "type_interaction"      : interaction_type,
                "E_min[keV]"            : list(recoil_energy_kev_list),
                "E_max[keV]"            : list(recoil_energy_kev_list),
                "field_drift[V/cm]"     : str(drift_field_v_cm),
                "x,y,z-position[mm]"    : str(xyz_pos_mm),
                "seed"                  : str(seed),})
            integral_spectrum_dict_list.append(spectrum_dict.copy()) # if '.copy()' is not appended, then all appended 'spectrum_dict's will always be updated

    # finishing
    if flag_verbose: print(f"{fn}: finished")
    for jfk in range(len(integral_spectrum_dict_list)):
        if callable(integral_spectrum_dict_list[jfk]["differential_rate_computation"]): # functions cannot be JSON-serialized
            integral_spectrum_dict_list[jfk].pop("differential_rate_computation")
    if flag_number_of_output_spectrum_dicts == 1:
        return integral_spectrum_dict_list[0]
    else:
        return integral_spectrum_dict_list

#flag_number_of_output_spectrum_dicts












def gen_spectrum_plot(
    spectra_list, # list of 'spectra_dict' keys, e.g., ["nr_wimps", "nr_atm", "nr_dsnb"]
    abspath_spectra_files,
    differential_rate_parameters = {},
    # plot parameters
    plot_fontsize_axis_label = 11,
    plot_figure_size_x_inch = 5.670,
    plot_aspect_ratio = 9/16,
    plot_log_y_axis = False,
    plot_log_x_axis = False,
    plot_xlim = [],
    plot_ylim = [],
    plot_x_axis_units = ["kev", "kev_nr", "kev_ee"][0],
    plot_legend = True,
    plot_legend_bbox_to_anchor = [0.45, 0.63, 0.25, 0.25],
    plot_legend_labelspacing = 0.5,
    plot_legend_fontsize = 9,
    # flags
    flag_output_abspath_list = [],
    flag_output_filename = "spectrum_plot.png",
    flag_shade_wimp_eroi = [],
    flag_verbose = False,
):

    """
    This function is used to generate a plot with all spectra specified in 'spectra_list'.
    The type of the 0th element in 'spectra_list' determines the type of spectrum this function generates:
        type(spectra_list[0])==str ---> compute the differential rate spectrum for each entry
        type(spectra_list[0])=='spectrum_dict' ---> display the actual spectrum histogram fed into 'execNEST' (as generated with 'give_spectrum_dict')
    """

    # initialization
    fn = "gen_spectrum_plot"
    if flag_verbose: print(f"{fn}: initializing")

    # setting up the canvas
    if flag_verbose: print(f"{fn}: setting up canvas and axes")
    fig = plt.figure(
        figsize = [plot_figure_size_x_inch, plot_figure_size_x_inch*plot_aspect_ratio],
        dpi = 150,
        constrained_layout = True) 

    # axes
    ax1 = fig.add_subplot()
    if plot_x_axis_units == "kev":
        ax1.set_xlabel(r"recoil energy, $E$ / $\mathrm{keV}$", fontsize=plot_fontsize_axis_label)
        ax1.set_ylabel(r"differential event rate $\frac{\mathrm{d}R}{\mathrm{d}E}$ / $\mathrm{\frac{events}{t\times y\times keV}}$", fontsize=plot_fontsize_axis_label)
    elif plot_x_axis_units == "kev_nr":
        ax1.set_xlabel(r"nuclear recoil energy, $E_{\mathrm{nr}}$ / $\mathrm{keV}_{\mathrm{nr}}$", fontsize=plot_fontsize_axis_label)
        ax1.set_ylabel(r"differential event rate $\frac{\mathrm{d}R}{\mathrm{d}E_{\mathrm{nr}}}$ / $\mathrm{\frac{events}{t\times y\times keV}}$", fontsize=plot_fontsize_axis_label)
    elif plot_x_axis_units == "kev_ee":
        ax1.set_xlabel(r"electronic recoil energy, $E_{\mathrm{ee}}$ / $\mathrm{keV}_{\mathrm{ee}}$", fontsize=plot_fontsize_axis_label)
        ax1.set_ylabel(r"differential event rate, $\frac{\mathrm{d}R}{\mathrm{d}E_{\mathrm{ee}}}$ / $\mathrm{\frac{events}{t\times y\times keV}}$", fontsize=plot_fontsize_axis_label)
    elif mode=="spectrum_dict":
        ax1.set_ylabel(r"events per energy bin", fontsize=plot_fontsize_axis_label)
    if plot_xlim != []: ax1.set_xlim(plot_xlim)
    if plot_ylim != []: ax1.set_ylim(plot_ylim)
    if plot_log_y_axis: ax1.set_yscale('log')
    if plot_log_x_axis: ax1.set_xscale('log')

    # looping over all specified spectra
    for spectrum in spectra_list:

        # case: generating differential spectrum only from specified string
        if type(spectrum) == str:
            x_data_size = 4801
            if plot_log_x_axis:
                if plot_xlim==[]:
                    x_data_recoil_energy_kev = np.logspace(start=-2, stop=+2, num=x_data_size, endpoint=True)
                else:
                    exp_lower = int(list(f"{plot_xlim[0]:.2E}".split("E"))[-1])-1
                    exp_upper = int(list(f"{plot_xlim[1]:.2E}".split("E"))[-1])+1
                    x_data_recoil_energy_kev = np.logspace(start=exp_lower, stop=exp_upper, num=x_data_size, endpoint=True)
            else:
                if plot_xlim==[]:
                    x_data_recoil_energy_kev = np.linspace(start=0.01, stop=100.01, num=x_data_size, endpoint=True)
                else:
                    x_data_recoil_energy_kev =  np.linspace(start=plot_xlim[0], stop=plot_xlim[1], num=x_data_size, endpoint=True)
            spectrum_dict = give_spectrum_dict(
                spectrum_name = spectrum,
                recoil_energy_kev_list = x_data_recoil_energy_kev,
                abspath_spectra_files = abspath_spectra_files,
                differential_rate_parameters = differential_rate_parameters)
            plot_x_data = spectrum_dict["recoil_energy_kev_list"]
            plot_y_data = spectrum_dict["differential_recoil_rate_events_t_y_kev"]

        # case: retrieving information from differential 'spectrum_dict'
        elif type(spectrum) == dict:
            if spectrum["flag_spectrum_type"] == "differential":
                spectrum_dict = spectrum.copy()
                plot_x_data = spectrum_dict["recoil_energy_kev_list"]
                plot_y_data = spectrum_dict["differential_recoil_rate_events_t_y_kev"]

            # case: retrieving information from integral 'spectrum_dict'
            elif spectrum["flag_spectrum_type"] == "integral":
                ax1.set_ylabel(r"integral number of events per energy bin", fontsize=plot_fontsize_axis_label)
                spectrum_dict = spectrum.copy()
                plot_x_data = spectrum_dict["E_min[keV]"]
                binwidth = plot_x_data[1] -plot_x_data[0]
                plot_x_data = [plot_x_data[k//2] for k in range(len(plot_x_data+plot_x_data))]
                plot_x_data = [plot_x_data[k]-0.5*binwidth if k%2==0 else plot_x_data[k]+0.5*binwidth for k in range(len(plot_x_data))]
                plot_x_data = [plot_x_data[0]] +plot_x_data +[plot_x_data[-1]]
                plot_y_data = list(spectrum_dict["numEvts"])
                plot_y_data = [plot_y_data[k//2] for k in range(len(plot_y_data+plot_y_data))]
                plot_y_data = [0] +plot_y_data +[0]
                # plotting the histogram bar lines
                for k, x in enumerate(plot_x_data):
                    ax1.plot(
                        [plot_x_data[k],plot_x_data[k]],
                        [0,plot_y_data[k]],
                        linestyle = spectrum_dict["linestyle"],
                        linewidth = spectrum_dict["linewidth"]/4,
                        zorder = spectrum_dict["zorder"]-1,
                        color = spectrum_dict["color"],)

        # plotting the current spectrum
        ax1.plot(
            plot_x_data,
            plot_y_data,
            label = spectrum_dict["latex_label"],
            linestyle = spectrum_dict["linestyle"],
            linewidth = spectrum_dict["linewidth"],
            zorder = spectrum_dict["zorder"],
            color = spectrum_dict["color"],)

    # shading the WIMP EROI
    if flag_shade_wimp_eroi != []:
        ax1.axvspan(
            flag_shade_wimp_eroi[0],
            flag_shade_wimp_eroi[1],
            alpha = 0.2,
            linewidth = 0,
            color = "grey",
            zorder = -1)

    # legend
    if plot_legend : ax1.legend(
        loc = "center",
        labelspacing = plot_legend_labelspacing,
        fontsize = plot_legend_fontsize,
        bbox_to_anchor = plot_legend_bbox_to_anchor,
        bbox_transform = ax1.transAxes,)

    # saving
    plt.show()
    for abspath in flag_output_abspath_list:
        fig.savefig(abspath +flag_output_filename)





############################################
### Executing NEST
############################################


def convert_detector_dict_into_detector_header(
    detector_dict, # dict, dictionary containing the detector parameters
    detector_name, # string, filename of the output detector header file
    abspath_output_list = [], # list, directories into which the output detector header file is saved
    flag_verbose = False, # bool, flag indicating whether verbose output is being printed
):

    """
    This function is used to save a .hh detector header file based on the input 'detector_dict'.
    """

    # initialization
    fn = "convert_detector_dict_into_detector_header"
    detector_name = list(detector_name.split("."))[0]
    line_list = []
    if flag_verbose: print(f"\n{fn}: initializing")
    if flag_verbose: print(f"\tdetector_name: {detector_name}")


    # adding the initial lines
    if flag_verbose: print(f"{fn}: adding initial lines to 'line_list'.")
    line_list = line_list +[
        "#ifndef " +detector_name +"_hh",
        "#define " +detector_name +"_hh 1",
        "",
        '#include "VDetector.hh"',
        "",
        "using namespace std;",
        "",
        "class " +detector_name +" : public VDetector {",
        "    public:",
        "        " +detector_name +"() {",
        "",
        "            Initialization();",
        "        };",
        "        ~" +detector_name +"() override = default;",
        "",
        "        void Initialization() override {",
    ]

    # filling 'line_list' with detector parameters
    if flag_verbose: print(f"{fn}: adding 'detector_dict' parameters to 'line_list'.")
    for k, key in enumerate([*detector_dict]):
        line_list.append("            " +key.ljust(20) +" =   " +str(detector_dict[key]) +";")

    # adding the final lines
    if flag_verbose: print(f"{fn}: adding final lines to 'line_list'.")
    line_list = line_list +[
        "        };",
        "    };",
        "#endif",
    ]

    # writing all lines into output .hh file
    if flag_verbose: print(f"{fn}: writing 'line_list' into header files.")
    for abspath in abspath_output_list:
        with open(abspath +detector_name +".hh", 'w') as outputfile:
            for k, line in enumerate(line_list):
                outputfile.write(line +"\n")
        if flag_verbose: print(f"\tsaved: {abspath +detector_name +'.hh'}")

    return


def make_clean_reinstall(
    abspath_build = abspath_nest_installation_build,
    flag_verbose = False,
):

    """
    This function is used to make a clean re-install of NEST.
    This is required, e.g., everytime the source code is modified.
    """

    # initialization
    fn = "make_clean_reinstall"
    if flag_verbose : print(f"\n{fn}: initialization")

    # defining the commands to be executed
    if flag_verbose : print(f"{fn}: defining shell commands in 'cmd_list'")
    cmd_list = [
        {
            "cmd" : "ls -la", # Keep this seemingly irrelevant first command in the list. If an error occurrs, then the program will raise an error instead of potentially damaging anything by executing 'make clean'.
            "stderr_default" : "", # Giving a string as default means the 'stderr' output has to exactly match the given string.
            "stdout_default" : ["total"], # Giving a list as default means that every list element has to appear in at least one line of the 'stdout' output.
        },
        {
            "cmd" : "make clean",
            "stderr_default" : "",
            "stdout_default" : "",
        },
        {
            "cmd" : "make",
            "stderr_default" : ["warning"],
            "stdout_default" : ["[100%]", "Built target", "Building"],
        },
        {
            "cmd" : "make install",
            "stderr_default" : "",
            "stdout_default" : ["[100%]", "Built target", "Install configuration", "Up-to-date"],
        },
    ]

    # looping over all entries in 'cmd_list' and executing all commands specified therein
    if flag_verbose : print(f"{fn}: looping over all entries in 'cmd_list'")
    for k, cmd_dict in enumerate(cmd_list):

        # executing the command
        cmd_string = "(cd " +abspath_build +" && " +cmd_dict["cmd"] +")" # utilizing subshells, see https://unix.stackexchange.com/questions/13802/execute-a-specific-command-in-a-given-directory-without-cding-to-it (accessed: 17th July 2022)
        if flag_verbose : print(f"{fn}: executing '$ {cmd_string}'")
        cmd_return = subprocess.run(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = cmd_return.stdout.decode('utf-8')
        stderr = cmd_return.stderr.decode('utf-8')

        # storing the output
        cmd_dict.update({"stdout" : stdout})
        cmd_dict.update({"stderr" : stderr})
        #print(f"\nstdout:\n {stdout}")
        #print(f"\nstderr:\n {stderr}")

        # sanity checking the output of the executed command
        for l, default in enumerate(["stderr", "stdout"]):
            if cmd_dict[default +"_default"]==False:
                if flag_verbose : print(f"\tno validity check for {default}: 'cmd_dict[{default}_default]'=False")
                continue
            elif type(cmd_dict[default +"_default"])==str:
                if cmd_dict[default +"_default"]==cmd_dict[default]:
                    if flag_verbose : print(f"\tpositive validity check for {default}: 'cmd_dict[{default}]' == '{cmd_dict[default +'_default']}'")
                    continue
                else:
                    exception_string = f"ERROR in {fn} executing {cmd_string}" +f"\n\tcmd_dict['{default}_default']!=cmd_dict['{default}']" +f"\n\t{default} = '{cmd_dict[default]}'" +f"\n\t{default}_default = '{cmd_dict[default +'_default']}'"
                    raise Exception(exception_string)
            elif type(cmd_dict[default +"_default"])==list:
                output_list = list(cmd_dict[default].split("\n"))
                for check_string in cmd_dict[default +"_default"]:
                    if any([check_string in line for line in output_list]):
                        if flag_verbose : print(f"\tpositive validity check for {default}: one 'cmd_dict[{default}]' line contains '{check_string}'")
                        continue
                    else:
                        exception_string = f"ERROR in {fn} executing {cmd_string}" +f"\n\tcheck string not in any line of cmd_dict['{default}']" +f"\n\tcheck string = '{check_string}'" +f"\n\tcmd_dict['{default}']-list = '{output_list}'"
                        raise Exception(exception_string)

    # end of program
    if flag_verbose : print(f"{fn}: successfully executed all entries in 'cmd_list'")
    return


def install_detector_header_file(
    abspathfile_new_detector_hh,
    abspathfile_nest_execNEST_cpp = abspathfile_nest_installation_execNEST_cpp,
    abspath_nest_detectors = abspath_nest_installation_nest_include_detectors,
    flag_clean_reinstall = False,
    flag_verbose = False,
):

    """
    This function is used to implement the specified new detector header file 'abspathfile_new_detector_hh' into the current NEST installation.
    """

    # initialization
    detector_name = list(abspathfile_new_detector_hh[:-3].split("/"))[-1]
    fn = "install_detector_header_file"
    if flag_verbose : print(f"\n{fn}: initialization")

    # copying the detector files
    if flag_verbose : print(f"{fn}: copying detector header into NEST installation")
    cmd_string = f"cp {abspathfile_new_detector_hh} {abspath_nest_detectors}{detector_name}.hh"
    cmd_return = subprocess.run(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = cmd_return.stdout.decode('utf-8')
    stderr = cmd_return.stderr.decode('utf-8')
    #print(f"stdout: '{stdout}'")
    #print(f"stderr: '{stderr}'")

    # reading the 'execNEST.cpp' file
    if flag_verbose : print(f"{fn}: reading '{abspathfile_nest_execNEST_cpp}'")
    line_list = []
    with open(abspathfile_nest_execNEST_cpp, 'r') as inputfile:
        for line in inputfile:
            line_list.append(line)

    # modifying the extracted lines
    if flag_verbose : print(f"{fn}: modifying '{abspathfile_nest_execNEST_cpp}'")
    for k, line in enumerate(line_list):
        if ("include" in line and "etector" in line and ".hh" in line) or ("include" in line and "LUX" in line and ".hh" in line):
            new_line = f'#include "{detector_name}.hh"\n'
            line_list[k] = new_line
            if flag_verbose : print(f"\tinserted '{new_line[:-1]}'")
        elif "auto* detector = new " in line:
            new_line = f"  auto* detector = new {detector_name}();\n"
            line_list[k] = new_line
            if flag_verbose : print(f"\tinserted '{new_line[:-1]}'")
        elif 'cerr << "You are currently using the' in line:
            new_line = f'    cerr << "You are currently using the {detector_name} detector." << endl\n'
            line_list[k] = new_line
            if flag_verbose : print(f"\tinserted '{new_line[:-1]}'")
        else:
            continue

    # write new file
    if flag_verbose : print(f"{fn}: saving modified version of 'execNEST_cpp'")
    with open(abspathfile_nest_execNEST_cpp, 'w') as outputfile:
        for line in line_list:
            outputfile.write(line)

    # performing a clean re-install
    if flag_clean_reinstall:
        if flag_verbose : print(f"{fn}: performing clean reinstall")
        make_clean_reinstall(flag_verbose=flag_verbose)

    return


def execNEST(
    spectrum_dict, # dict, dictionary resembling the input spectrum to be simulated by NEST
    baseline_drift_field_v_cm, # float, baseline electrical drift field, this needs to be given since parameters of the NEST detector header file (.hh) depend on the drift field (e.g., 'dtCntr')
    baseline_detector_dict, # dict, 'detector_dict' of the DARWIN baseline detector
    detector_dict = {}, # dict, resembling the detector to be installed, no new detector is installed if empty
    detector_name = "", # string, name of the detector, only required when not referring to an existing file
    abspath_list_detector_dict_json_output = [], # list of strings, list of abspaths into which the detector_dict.json files are saved
    abspathfile_execNEST_binary = abspathfile_nest_installation_execNEST_bin, # string, abspathfile of the 'execNEST' executiable generated in the 'install' NEST folder
    flag_verbose = False, # bool, flag indicating whether the print-statements are being printed
    flag_print_stdout_and_stderr = False, # bool, flag indicating whether the 'stdout' and 'stderr' values returned by executing the 'execNEST' C++ executable are being printed
    flag_min_selection_fraction = 0.05, # float, minimum number of events with non-negative S1 or S2 signal
    flag_sign_flip = [False,True][1], # string, how to handle negative-flagged NEST output
    flag_event_selection = ["remove_-1e-6_events"][0], # string, "remove_-1e-6_events" removes all events with -1e-6 values
):

    """
    This function is used to execute the 'execNEST' C++ executable from the NEST installation.
    Returns the NEST output in the form of a numpy structured array.
    """

    ### initializing
    fn = "execNEST" # name of this function, required for the print statements
    if flag_verbose: print(f"\n{fn}: initializing")
    execNEST_output_tuple_list = []
    cmd_list = []
    debug_list = []

    ### detector adaptation
    if detector_dict == {} and float(baseline_drift_field_v_cm)==float(spectrum_dict["field_drift[V/cm]"]):
        if flag_verbose: print(f"{fn}: no detector specified --> running with the pre-installed detector")
        pass
    else:
#        if type(detector_dict)==str:
#            if detector_dict.endswith(".hh"):
#                detector_name = list(detector_dict[:-3].split("/"))[-1]
#                if flag_verbose: print(f"{fn}: specified detector '{detector_name}' as .hh-file: {detector_dict}")
#                new_detector_hh_abspathfile = detector_dict
#            elif detector_dict.endswith(".json"):
#                detector_name = list(detector_dict[:-5].split("/"))[-1]
#                if flag_verbose: print(f"{fn}: specified detector '{detector_name}' as .json-file: {detector_dict}")
#                if flag_verbose: print(f"{fn}: updating baseline detector: {abspathfile_baseline_detector_json}")
#                new_detector_dict = baseline_detector_dict.copy()
#                new_detector_dict.update(get_dict_from_json(detector_dict))
#                convert_detector_dict_into_detector_header()
        if type(detector_dict)==dict:
            if detector_name=="" : raise Exception(f"ERROR: You did not specify a detector name!")
            if flag_verbose: print(f"{fn}: specified detector '{detector_name}'as dictionary: {detector_dict}")
            if flag_verbose: print(f"{fn}: updating baseline detector")
            new_detector_dict = baseline_detector_dict.copy()
            new_detector_dict.update(detector_dict)
            new_detector_dict = adjust_detector_drift_time_parameters(detector_dict=new_detector_dict, drift_field_v_cm=spectrum_dict["field_drift[V/cm]"])
            #print(new_detector_dict)
            convert_detector_dict_into_detector_header(
                detector_dict = new_detector_dict,
                abspath_output_list = abspath_list_detector_dict_json_output +[abspath_nest_installation_nest_include_detectors],
                detector_name = detector_name,
                flag_verbose = flag_verbose,
            )
        text_add="\n" if detector_dict != {} else ""
        if flag_verbose: print(text_add +f"{fn}: installing new detector header file")
        install_detector_header_file(
            abspathfile_new_detector_hh = abspath_nest_installation_nest_include_detectors +detector_name +".hh",
            flag_clean_reinstall = True,
            flag_verbose = flag_verbose)

    ### executing the 'execNEST' executable to simulate the input spectrum
    text_add="\n" if detector_dict != {} else ""
    if flag_verbose: print(text_add +f"{fn}: compiling the 'execNEST' command strings")
    if spectrum_dict["type_interaction"] in ["ER", "NR", "gamma", "beta"]:
        # non-necessary default values
        default_seed = 0
        seed = default_seed
        if "seed" in [*spectrum_dict]:
            if spectrum_dict["seed"] != 0:
                seed = spectrum_dict["seed"]
        default_xyz = "-1"
        xyz = default_xyz
        if "x,y,z-position[mm]" in [*spectrum_dict]:
            if spectrum_dict["x,y,z-position[mm]"] not in [0, "", default_xyz]:
                xyz = spectrum_dict["x,y,z-position[mm]"]
        # case: spectrum consists of single energy interval (typically used for quick tests)
        if type(spectrum_dict["numEvts"]) in [str,int]:
            cmd_string = " ".join([
                abspathfile_execNEST_binary,
                str(spectrum_dict["numEvts"]),
                str(spectrum_dict["type_interaction"]),
                str(spectrum_dict["E_min[keV]"]),
                str(spectrum_dict["E_max[keV]"]),
                str(spectrum_dict["field_drift[V/cm]"]),
                str(xyz),
                str(seed)])
            cmd_list.append(cmd_string)
        # case: spectrum consists of multiple energy intervals (typically used for actual spectra processing)
        elif hasattr(spectrum_dict["numEvts"], "__len__"):
            # checking validity of input 'spectrum_dict'
            if len(spectrum_dict["numEvts"])==len(spectrum_dict["E_min[keV]"])==len(spectrum_dict["E_max[keV]"]):
                if flag_verbose: print(f"\tinput 'spectrum_dict' appears valid")
            else:
                raise Exception(f"ERROR: len(spectrum_dict['numEvts'])==len(spectrum_dict['E_min[keV]'])==len(spectrum_dict['E_max[keV]'])")
            # looping over all resulting 'cmd_strings'
            for k, num in enumerate(spectrum_dict["numEvts"]):
                if float(spectrum_dict["numEvts"][k]) > 0: # sometimes you automatically generate spectrum histograms with empty bins
                    cmd_string = " ".join([
                        abspathfile_execNEST_binary,
                        str(spectrum_dict["numEvts"][k]),
                        str(spectrum_dict["type_interaction"]),
                        str(spectrum_dict["E_min[keV]"][k]),
                        str(spectrum_dict["E_max[keV]"][k]),
                        str(spectrum_dict["field_drift[V/cm]"]),
                        str(xyz),
                        str(seed)])
                    cmd_list.append(cmd_string)

    ### looping over all commands and executing them
    if flag_verbose: print(f"{fn}: executing the 'execNEST' command strings")
    for k, cmd_string in enumerate(cmd_list):

        # executing the 'execNEST' C++ executable
        cmd_print_string = "execNEST " +" ".join(list(cmd_string.split(" "))[1:])
        if flag_verbose: print(f"{fn}: executing '$ {cmd_print_string}'")
        cmd_return = subprocess.run(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = cmd_return.stdout.decode('utf-8')
        stderr = cmd_return.stderr.decode('utf-8')
        if flag_print_stdout_and_stderr:
            print(f"{fn}: 'stdout':")
            print(stdout)
            print(f"{fn}: 'stderr':")
            print(stderr)
        debug_dict = {
            "stderr" : stderr,
            "cmd_string" : cmd_string,}

        # checking whether the specified detector was implemented correctly
        if detector_dict != {}:
            if flag_verbose: print(f"\tchecking implemented detector")
            implemented_detector_list = []
            for line in stderr.split("\n"):
                line_list = list(line.split(" "))
                if "ou" in line and "are" in line and "using" in line and "detector" in line:
                    implemented_detector_list.append(line_list[-2])
                else:
                    continue
            if len(implemented_detector_list) != 1:
                raise Exception(f"ERROR: More than one potential detector implementations were found: {implemented_detector_list}. Maybe check your detector string search criteria.")
            else:
                implemented_detector = implemented_detector_list[0]
                if implemented_detector != detector_name:
                    raise Exception(f"ERROR: The implemented detector 'implemented_detector' does not match the one you specified 'detector_name'. It appears an error ocurred during installation.")
            if flag_verbose: print(f"\timplemented detector = '{implemented_detector}' = '{detector_name}' = specified detector")

        # searching for the NEST output header line
        header_line_list = []
        if flag_verbose: print(f"\tsearching for the output header line")
        for line in stdout.split("\n"):
            line_list = list(line.split("\t"))
            if "Nph" in line_list:
                header_line_list.append(line)
            else:
                continue
        if len(header_line_list) != 1:
            raise Exception(f"ERROR: More than one potential header line was found in the 'execNEST' C++ executable output were found: {header_line_list}. Maybe check your header line string search criteria.")
        else:
            header_line = header_line_list[0]
            header_line_split = list(header_line.split("\t"))
            if flag_verbose: print(f"\theader I: {header_line_split[:7]}")
            if flag_verbose: print(f"\theader II: {header_line_split[7:]}")
            dtype_list = []
            for header in header_line_split:
                if header in ["X,Y,Z [mm]"]:
                    dtype_list.append((header, np.unicode_, 16))
                else:
                    dtype_list.append((header, np.float64))
            execNEST_dtype = np.dtype(dtype_list)

        # extracting the NEST output
        if flag_verbose: print(f"{fn}: writing the NEST output into tuple list")
        this_nest_run_tuple_list = []
        for line in stdout.split("\n"):
            line_list = list(line.split("\t"))
            if len(line_list) == 12 and "Nph" not in line:
                if flag_sign_flip == True:
                    execNEST_output_sign_flipped_list = [np.sqrt((float(val))**2) if "," not in val else val for val in line_list]
                    execNEST_output_tuple = tuple(execNEST_output_sign_flipped_list)
                else:
                    execNEST_output_tuple = tuple(line_list)
                this_nest_run_tuple_list.append(execNEST_output_tuple)
            else:
                continue
        execNEST_output_tuple_list += this_nest_run_tuple_list
        num = int(list(cmd_string.split(" "))[1])
        e_min = float(list(cmd_string.split(" "))[3])
        e_max = float(list(cmd_string.split(" "))[3])
        if len(this_nest_run_tuple_list) != num: raise Exception(f"This NEST run yielded {len(this_nest_run_tuple_list)} events instead of the specified {num} at E_min={e_min} and E_max={max}.")

    # casting the 'execNEST_output_tuple_list' into a ndarray
    if flag_verbose: print(f"{fn}: casting 'execNEST_output_tuple_list' into numpy ndarray")
    execNEST_output_ndarray = np.array(execNEST_output_tuple_list, execNEST_dtype)

    # removing negative-flagged events from output
    if flag_event_selection == "remove_-1e-6_events":
        if flag_verbose: print(f"{fn}: removing -1e-6-flagged events")
        len_total = len(execNEST_output_ndarray)
        execNEST_output_ndarray = execNEST_output_ndarray[
            ((execNEST_output_ndarray["S1_3Dcor [phd]"] < -1e-06) |
            (execNEST_output_ndarray["S1_3Dcor [phd]"] > 1e-06)) &
            ((execNEST_output_ndarray["S2_3Dcorr [phd]"] < -1e-06) |
            (execNEST_output_ndarray["S2_3Dcorr [phd]"] > 1e-06))
        ]
        len_selected = len(execNEST_output_ndarray)
        if flag_verbose: print(f"\tselected {len_selected} out of {len_total} events")
        selection_fraction = len_selected/len_total
        if selection_fraction < flag_min_selection_fraction:
            raise Exception(f"ERROR: 'selection_fraction' = {selection_fraction} < {flag_min_selection_fraction} = 'flag_min_selection_fraction'")

    return execNEST_output_ndarray


def convert_from_s2_vs_s1_to_s2_over_s1_vs_s1_over_g1(
    s1_array,
    s2_array,
    detector_dict,
):
    s2_over_s1_array  = [s2_array[k]/s1_array[k] for k in range(len(s1_array))]
    s1_over_g1_array = [s1/float(detector_dict["g1"]) for s1 in s1_array]
    return s1_over_g1_array, s2_over_s1_array


def energy_contour_line_in_s2_over_s1(
    recoil_energy_kev_ee,
    detector_dict,
    n_samples = 100
):
    """
    This function is used to compute two arrays resembling an energy contour line plottable in a signature plot (E = W(s1/g1 +s2/g2)).
    """
    w_old = 13.6*0.001 # conversion to kev
    g1 = detector_dict["g1"]
    g2 = compute_g2_from_detector_configuration(detector_dict)
    energy_contour_s1 = list(np.linspace(start=0.0001, stop=recoil_energy_kev_ee/w_old*g1, endpoint=True, num=n_samples))
    energy_contour_s2 = [(recoil_energy_kev_ee/w_old-s1/g1)*g2 for s1 in energy_contour_s1]
    return energy_contour_s1, energy_contour_s2


def gen_signature_plot(
    signature_dict_list, # list of signature ndarrays to be plotted onto the canvas
    detector_dict,
    # plot parameters
    plot_fontsize_axis_label = 11,
    plot_figure_size_x_inch = 5.670,
    plot_aspect_ratio = 9/16,
    plot_log_y_axis = False,
    plot_log_x_axis = False,
    plot_xlim = [],
    plot_ylim = [],
    plot_axes_units = ["cs2_vs_cs1", "cs2_over_cs1_vs_cs1_over_g1"][1],
    plot_legend = False,
    plot_legend_bbox_to_anchor = [0.45, 0.63, 0.25, 0.25],
    plot_legend_labelspacing = 0.5,
    plot_legend_fontsize = 9,
    plot_energy_contours = [],
    plot_text_dict_list = [],
    plot_discrimination_line_dict = {},
    # flags
    flag_output_abspath_list = [],
    flag_output_filename = "signature_plot.png",
    flag_profile = ["default"][0],
    flag_verbose = False,
):

    """
    This function is used to generate a plot displaying all signatures specified in 'signature_list'.
    """

    # initialization
    fn = "gen_signature_plot"
    if flag_verbose: print(f"{fn}: initializing")

    # canvas
    if flag_verbose: print(f"{fn}: setting up canvas and axes")
    fig = plt.figure(
        figsize = [plot_figure_size_x_inch, plot_figure_size_x_inch*plot_aspect_ratio],
        dpi = 150,
        constrained_layout = True) 

    # axes
    ax1 = fig.add_subplot()
    if plot_log_y_axis: ax1.set_yscale('log')
    if plot_log_x_axis: ax1.set_xscale('log')
    if plot_xlim != [] : ax1.set_xlim(plot_xlim)
    if plot_ylim != [] : ax1.set_ylim(plot_ylim)
    if plot_axes_units == "cs2_over_cs1_vs_cs1_over_g1":
        ax1.set_xlabel(r"$\frac{\mathrm{c}S_1}{g_1}$ / $\text{number of primary photons}$", fontsize=plot_fontsize_axis_label)
        ax1.set_ylabel(r"$\frac{\mathrm{c}S_2}{\mathrm{c}S_1}$ / $\mathrm{\frac{phd}{phd}}$", fontsize=plot_fontsize_axis_label)
    elif plot_axes_units == "cs2_vs_cs1":
        ax1.set_xlabel(r"$\mathrm{c}S_1$ / $\mathrm{phd}$", fontsize=plot_fontsize_axis_label)
        ax1.set_ylabel(r"$\mathrm{c}S_2$ / $\mathrm{phd}$", fontsize=plot_fontsize_axis_label)

    # defining defbbault scatter format
    default_scatter_format_dict = {
        "alpha" : 1,
        "zorder" : 1,
        "marker" : "o", # markerstyle, see: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
        "linewidths" : 0.0,
        "s" : 2,
        "edgecolors" : "black",
        "facecolors" : "red",
        "linestyles" : "-",}

    # plotting
    if flag_verbose: print(f"{fn}: plotting")
    for signature_dict in signature_dict_list:
        if flag_verbose: print(f"{fn}: plotting {signature_dict['label']}")

        # selecting the data to be plotted
        data = signature_dict["signature_ndarray"]
        if plot_axes_units == "cs2_over_cs1_vs_cs1_over_g1":
            plot_x_data, plot_y_data = convert_from_s2_vs_s1_to_s2_over_s1_vs_s1_over_g1(data["S1_3Dcor [phd]"], data["S2_3Dcorr [phd]"], detector_dict)
        elif plot_axes_units == "cs2_vs_cs1":
            plot_x_data = data["S1_3Dcor [phd]"]
            plot_y_data = data["S2_3Dcorr [phd]"]

        # formatting the current signature
        format_dict = default_scatter_format_dict.copy()
        for key in [k for k in [*signature_dict] if k not in ["signature_ndarray", "latex_label"]]:
            format_dict.update({key : signature_dict[key]})

        # plotting the current signature
        ax1.scatter( plot_x_data, plot_y_data, **format_dict)

    # plotting the energy contour lines
    for recoil_energy_kev_ee in plot_energy_contours:
        plot_energy_contour_x, plot_energy_contour_y = energy_contour_line_in_s2_over_s1(recoil_energy_kev_ee, detector_dict)
        if plot_axes_units == "cs2_over_cs1_vs_cs1_over_g1":
            plot_energy_contour_x, plot_energy_contour_y = convert_from_s2_vs_s1_to_s2_over_s1_vs_s1_over_g1(plot_energy_contour_x, plot_energy_contour_y, detector_dict)
        ax1.plot(
            plot_energy_contour_x,
            plot_energy_contour_y,
            color = "black",
            linewidth = 0.5,
            linestyle = "-",
            zorder = 5,)
        index = np.argmin(np.abs(np.array(plot_energy_contour_y)-plot_ylim[0]))
        ax1.text(
            x = (ax1.transAxes + ax1.transData.inverted()).inverted().transform([plot_energy_contour_x[index],1])[0],
            y = 0.024,
            s = r"$" +f"{recoil_energy_kev_ee:.1f}" +"\,\mathrm{keV}_{\mathrm{ee}}$",
            transform = ax1.transAxes,
            fontsize = 8,
            color = "black",
            horizontalalignment = "left",
            verticalalignment = "bottom",)

    # plot discrimination line
    dl_x_data = plot_discrimination_line_dict["dl_x_data_s1_over_g1"] if plot_axes_units=="cs2_over_cs1_vs_cs1_over_g1" else plot_discrimination_line_dict["dl_x_data_s1"]
    dl_y_data = plot_discrimination_line_dict["dl_y_data_s2_over_s1"] if plot_axes_units=="cs2_over_cs1_vs_cs1_over_g1" else plot_discrimination_line_dict["dl_y_data_s2"]
    ax1.plot(
        dl_x_data,
        dl_y_data,
        linestyle = "-",
        linewidth = 0.5,
        color = "black",)

#        "nr_acceptance": float(nr_acceptance),
#        "er_rejection": float(er_rejection),


    # text annotations
    if flag_verbose: print(f"{fn}: text annotations")
    default_text_format_dict = {
        "horizontalalignment" : "center",
        "verticalalignment"   : "center",
        "zorder"              : 1,
        "color"               : "black",
        "fontsize"            : 11,
        "transform"           : ax1.transAxes}
    for text_dict in plot_text_dict_list:
        text_annotation_dict = default_text_format_dict.copy()
        text_annotation_dict.update(text_dict)
        ax1.text(**text_annotation_dict)

    # legend
    if flag_verbose: print(f"{fn}: legend")
    if plot_legend : ax1.legend(
        loc = "center",
        labelspacing = plot_legend_labelspacing,
        fontsize = plot_legend_fontsize,
        bbox_to_anchor = plot_legend_bbox_to_anchor,
        bbox_transform = ax1.transAxes,)

    # saving
    if flag_verbose: print(f"{fn}: saving")
    plt.show()
    for abspath in flag_output_abspath_list:
        fig.savefig(abspath +flag_output_filename)
    return







############################################
### ER/NR discrimination
############################################


def calc_er_nr_discrimination_line(
    er_spectrum,
    nr_spectrum,
    detector_dict,
    min_energy, #in keV
    max_energy, # in keV
    bin_number, #number of energy-bins
    nr_acceptance = 0.30, # portion of the nr-spectrum below the discrimination line
    approx_depth = 10,# only integers allowed
    # upper bound for errors in the discrimination line is
    # ~ 2**(-approx_depth-1) * (max(nr_spectrum[S2]/nr_spectrum[S1]) - min(nr_spectrum[S2]/nr_spectrum[S1]))
    verbose =True,
    num_interp_steps = 10, # used for interpolating the discr-line to project it into S1-vs.-S2-space
    ):
    #generate energy bins of size energy_bin_size and an array of the bin edges (energy_bin_edges).
    #Bins are centerd around the entries in the energy_bins array.

    if verbose: print("calc_er_nr_discrimination_line running.")
    w = 13.6
    g1 = detector_dict["g1"]
    g2 = compute_g2_from_detector_configuration(detector_dict)
    bin_edges = np.linspace(min_energy, max_energy, bin_number+1)

    dtype = np.dtype([("S2/S1", np.float64)]) #dtype for nr_energies and er_energies numpy arrays

    total_er_remaining = 0
    total_er = 0
    total_nr_below_discr_line = 0
    total_nr = 0

    #x-data and y-data for the discrimination line
    dl_x_data=[] # cS1/g1 [ph]
    dl_y_data=[] # cS1/cS2
    dl_S1_data = [] # cS1 [phd]
    dl_S2_data = [] # cS2 [phd]

    #array containing reconstructed energies for ER- and NR-events (but only accurate for ER)
    energy_er=w*(er_spectrum["S1_3Dcor [phd]"]/g1 +er_spectrum["S2_3Dcorr [phd]"]/g2)/1000

    energy_nr=w*(nr_spectrum["S1_3Dcor [phd]"]/g1 +nr_spectrum["S2_3Dcorr [phd]"]/g2)/1000

    nr_below_dl=np.array([])
    nr_below_dl.dtype = nr_spectrum.dtype
    for bin_index in range(bin_number):

        #filter the ER- and NR-events that lie in the current energy-bin.
        er_bin_data =er_spectrum[(energy_er> bin_edges[bin_index])&(energy_er<bin_edges[bin_index+1])]

        nr_bin_data =nr_spectrum[(energy_nr> bin_edges[bin_index])&(energy_nr<bin_edges[bin_index+1])]


        total_er += len(er_bin_data)
        total_nr += len(nr_bin_data)

        total_nr_events = len(nr_bin_data["S2_3Dcorr [phd]"])
        threshold_nr_events = total_nr_events*nr_acceptance

        # calculates s2/s1 for every event in the bin
        nr_y = nr_bin_data["S2_3Dcorr [phd]"]/nr_bin_data["S1_3Dcor [phd]"]
        er_y = er_bin_data["S2_3Dcorr [phd]"]/er_bin_data["S1_3Dcor [phd]"]
        nr_y.dtype =dtype
        er_y.dtype = dtype

        if len(nr_y)==0:
            #if verbose:
                #print(f"energy bin [{bin_edges[bin_index]}, {bin_edges[bin_index+1]}] is empty.")
                #print(len(er_bin_data), len(nr_bin_data))
            continue

        lower_bound = min(nr_y["S2/S1"])
        upper_bound = max(nr_y["S2/S1"])
        guess = binary_search(lower_bound, upper_bound,
                                 lambda x: len(nr_y[ nr_y["S2/S1"]<=x]),threshold_nr_events, 0, approx_depth)
        total_er_remaining += len(er_y[er_y["S2/S1"]<= guess])


        total_nr_below_discr_line += len(nr_y[ nr_y["S2/S1"]<=guess])

        nr_below_dl = np.append(nr_below_dl, nr_bin_data[nr_y["S2/S1"]<=guess])
        #append guess and bin

        dl_x_data.append(1000*bin_edges[bin_index]/w/(1+g1/g2*guess)) #bin_edges are in keV, formula
                                                                 # E = w(S1/g1 + S2/g2) is in eV
        dl_x_data.append(1000*bin_edges[bin_index+1]/w/(1+g1/g2*guess))

        dl_interp_x = np.linspace(dl_x_data[-2], dl_x_data[-1],num_interp_steps)
        for dl_x in dl_interp_x:
            x_S1 = dl_x*g1
            y_S2 = guess * x_S1
            dl_S1_data.append(x_S1)
            dl_S2_data.append(y_S2)

        dl_y_data.append(guess)
        dl_y_data.append(guess)

    er_rejection = 1-(total_er_remaining/total_er)
    if verbose:
        print("Total number of ER-events:",total_er)
        print("Total number of NR-events:",total_nr)
        print("Input NR-acceptance:", nr_acceptance)
        print("Actual NR-acceptance:", total_nr_below_discr_line/total_nr)
        print("Ratio of ER-Events left below the discrimination line:",er_rejection)

    output_dict = {
        "dl_x_data_s1_over_g1": list(dl_x_data),
        "dl_y_data_s2_over_s1": list(dl_y_data),
        "dl_x_data_s1": list(dl_S1_data),
        "dl_y_data_s2": list(dl_S2_data),
        "nr_acceptance": float(nr_acceptance),
        "er_rejection": float(er_rejection),
        "er_rejection_uncertainty": np.sqrt(2)*(np.sqrt(total_er)/total_er), # with Poissontian uncertainties approximated as np.sqrt, since statistics sufficiently high
#        "nr_below_dl": list(nr_below_dl),
    }

    return output_dict


def binary_search(lower_bound, upper_bound, eval_func, threshold,current_depth, max_depth):
    # function finds recursively a good guess for eval_func(guess) = threshold.
    # eval_func must be monotonically increasing.

    assert upper_bound>=lower_bound, "upper_bound should be greater than lower_bound."
    assert max_depth == int(max_depth), "max_depth should be an integer."

    guess = (upper_bound+lower_bound)/2
    # if maximum depth is reached, stop searching and return current best guess.
    if current_depth>= max_depth:
           return guess

    if eval_func(guess)>threshold:
        # now guess is new upper bound, lower_bound stays
        return binary_search(lower_bound, guess,  eval_func, threshold, current_depth+1, max_depth)

    elif eval_func(guess)<threshold:
        # now guess ist new lower bound, upper_bound stays
        return binary_search( guess, upper_bound, eval_func, threshold, current_depth+1, max_depth)

    elif eval_func(guess) == threshold:
        # This will propably never happen, but whatever.
        # if the current guess fits the threshold, stop.
        return guess

"""
# detector parameters
g1=0.12
g1_gas = 0.1
g2 = 20 #rough estimate
w=13.6 #eV


nr_spectrum = sfs.execNEST(
    spectrum_dict = give_flat_spectrum(interaction_type="NR", num_events =1e6, min_energy =0.5,
                                       max_energy= 10,verbose=False),
    detector_dict = {
        "g1" : g1,
        "g1_gas" : g1_gas,
    },
    detector_name="random_test_detector2",
    flag_verbose = False,
)

er_spectrum = sfs.execNEST(
    spectrum_dict = give_flat_spectrum(interaction_type="ER", num_events =1e6, min_energy = 1,
                                       max_energy=50, verbose=False),
    detector_dict = {
        "g1" : g1,
        "g1_gas" : g1_gas,
    },
    detector_name="random_test_detector2",
    flag_verbose = False,
)

dl_dict = calc_er_nr_discrimination_line(er_spectrum_corr, nr_spectrum_corr, g1, g2,
    w, min_energy=1,max_energy=10,bin_size=0.2, nr_acceptance=0.3)


# plotting NR- and ER-events, NR-events below the discrimination line, and the discrimination line
dl_x_data=dl_dict["dl_x_data_s1_over_g1"]
dl_y_data=dl_dict["dl_y_data_s2_over_s1"]
nr_below_dl = dl_dict["nr_below_dl"]

alpha = 0.05
s=1
plt.figure(figsize=(10,10))
plt.yscale("log")
plt.xlabel("cS1/g1 [phd]")
plt.ylabel("cS2/cS1")
plt.xlim([0,400])
plt.scatter(nr_spectrum_corr["S1_3Dcor [phd]"]/g1, nr_spectrum_corr["S2_3Dcorr [phd]"]/nr_spectrum_corr["S1_3Dcor [phd]"],
            alpha=alpha,s=s, label="NR")
plt.scatter(er_spectrum_corr["S1_3Dcor [phd]"]/g1, er_spectrum_corr["S2_3Dcorr [phd]"]/er_spectrum_corr["S1_3Dcor [phd]"],
            alpha=alpha,s=s, label="ER")
plt.scatter(nr_below_dl["S1_3Dcor [phd]"]/g1, nr_below_dl["S2_3Dcorr [phd]"]/nr_below_dl["S1_3Dcor [phd]"],
            alpha=alpha,s=s, label="NR below dl")
plt.plot(dl_x_data, dl_y_data, label="discriminiation line", c="black")
plt.legend()


"""


def er_nr_discrimination_line_scan(
        baseline_detector_dict,
        baseline_drift_field_v_cm,
        parameter_name,
        parameter_value_list,
        detector_name,
        er_spectrum_dict,
        nr_spectrum_dict,
        calc_er_nr_discrimination_line_kwargs_dict, # dictionary with keyword parameters passed on to 'calc_er_nr_discrimination_line'
        flag_abspath_list_discrimination_lines_output = [],
        flag_verbose = [False, True][0],
    ):

    """
    This function is used to automatically compute the ER/NR discrimination for a list of parameter values.
    NOTE: so far a drift field sweep is not possible.
    returns:    'discrimination_line_dict_dict' is a dictionary with 
    """

    # initialization
    fn = "er_nr_discrimination_line_loop"
    discrimination_line_scan_dict = {}
    if flag_verbose : print(f"\n{fn}: initializing")
    start_time = time.time()

    # looping over the specified parameter space
    if flag_verbose : print(f"{fn}: looping over parameter space")
    for k, parameter_value in enumerate(parameter_value_list):
        if flag_verbose : print(f"{fn}: 'parameter_name' = '{parameter_name}', 'parameter_value'='{parameter_value}'")

        # adapting the detector
        current_detector_dict = baseline_detector_dict.copy()
        if parameter_name != "e_drift_v_cm":
            current_detector_dict.update({parameter_name : parameter_value})
            current_er_spectrum_dict = er_spectrum_dict
            current_nr_spectrum_dict = nr_spectrum_dict
        else:
            current_er_spectrum_dict = er_spectrum_dict
            current_er_spectrum_dict["field_drift[V/cm]"] = parameter_value
            current_nr_spectrum_dict = nr_spectrum_dict
            current_nr_spectrum_dict["field_drift[V/cm]"] = parameter_value

        # executing 'execNEST'
        if flag_verbose : print(f"{fn}: executing 'execNEST'")
        er_spectrum_ndarray = execNEST(
            spectrum_dict = current_er_spectrum_dict,
            baseline_detector_dict = baseline_detector_dict,
            baseline_drift_field_v_cm = baseline_drift_field_v_cm,
            detector_dict = current_detector_dict,
            detector_name = detector_name +"__" +parameter_name +"__" +str(parameter_value).replace(".","_"),
            abspath_list_detector_dict_json_output = [],
            flag_verbose = flag_verbose,
            flag_print_stdout_and_stderr = False,)
        nr_spectrum_ndarray = execNEST(
            spectrum_dict = current_nr_spectrum_dict,
            baseline_drift_field_v_cm = baseline_drift_field_v_cm,
            baseline_detector_dict = baseline_detector_dict,
            detector_dict = {},
            detector_name = detector_name +"__" +parameter_name +"__" +str(parameter_value).replace(".","_"),
            abspath_list_detector_dict_json_output = [],
            flag_verbose = flag_verbose,
            flag_print_stdout_and_stderr = False,)

        # computing the 'discrimination_line_dict'
        if flag_verbose : print(f"{fn}: computing 'discrimination_line_dict'")
        discrimination_line_dict = calc_er_nr_discrimination_line(
            er_spectrum = er_spectrum_ndarray,
            nr_spectrum = nr_spectrum_ndarray,
            detector_dict = current_detector_dict,
            verbose = flag_verbose,
            **calc_er_nr_discrimination_line_kwargs_dict)
        discrimination_line_scan_dict.update({ detector_name +"__" +parameter_name +"__" +str(parameter_value).replace(".","_") : discrimination_line_dict })

    # finishing
    end_time = time.time()
    elapsed_time_s = end_time -start_time
    td = timedelta(seconds=elapsed_time_s)
    if flag_verbose : print(f"{fn}: processed {len(parameter_value_list)} discrimination lines within {td} h")
    return discrimination_line_scan_dict


def gen_discrimination_line_scan_plot(
    # required input
    discrimination_line_scan_dict, # dict, as generated with 'er_nr_discrimination_line_scan()' or multiplie summarized instances thereof
    primary_parameter_name, # string, an individual discrimination line scan is plotted for each value of this parameter
    secondary_parameter_name, # string, this parameter defines the x-axis of the plot
    # predefined input
    output_abspathstring_list = [], # list, list of abspathstrings according to which the generated plot is saved
    parameter_translation_dict = translate_parameter_to_latex_dict,
    # plot style parameters
    plot_fontsize_axis_label = 11,
    plot_figure_size_x_inch = 5.670,
    plot_aspect_ratio = 9/16,
    plot_log_y_axis = False,
    plot_log_x_axis = False,
    plot_xlim = [],
    plot_ylim = [],
    plot_legend = True,
    plot_legend_bbox_to_anchor = [0.45, 0.63, 0.25, 0.25],
    plot_legend_labelspacing = 0.5,
    plot_legend_fontsize = 9,
    plot_legend_invert_order = [False,True][0],
    plot_text_dict_list = [],
    plot_marker_linewidth = 0.9,
    plot_marker_size = 5,
    plot_cmap = ["viridis", "plasma", "gist_rainbow", "brg", "YlGnBu"][0],
    # flags
    flag_verbose = [False,True][0],):

    """
    This function is used to plot the various ER rejections inferred with 'er_nr_discrimination_line_scan()'.
    """

    # initializing
    fn = "gen_discrimination_line_scan_plot"
    if flag_verbose : print(f"\n{fn}: initializing")
    primary_parameter_latex_string = primary_parameter_name.replace("_", "\_")
    secondary_parameter_latex_string = secondary_parameter_name.replace("_", "\_")

    # parameter value retrieval
    primary_parameter_values = []
    secondary_parameter_values = []
    detector_list = []
    nr_acceptance_list = []
    for detector_name in [*discrimination_line_scan_dict]:
        detector_name_list = list(detector_name.split("__"))
        detector_list.append(detector_name_list[0])
        nr_acceptance_list.append(discrimination_line_scan_dict[detector_name]["nr_acceptance"])
        primary_parameter_index = detector_name_list.index(primary_parameter_name)
        secondary_parameter_index = detector_name_list.index(secondary_parameter_name)
        primary_parameter_value = detector_name_list[primary_parameter_index+1]
        secondary_parameter_value = detector_name_list[secondary_parameter_index+1]
        primary_parameter_values.append(primary_parameter_value)
        secondary_parameter_values.append(secondary_parameter_value)
    detector_list = list(set(detector_list))
    assert len(detector_list)==1, f"ERROR multiple detector names found in 'discrimination_line_scan_dict': {detector_list}"
    nr_acceptance_list = list(set(nr_acceptance_list))
    assert len(nr_acceptance_list)==1, f"ERROR multiple NR acceptances found in 'discrimination_line_scan_dict': {nr_acceptance_list}"
    detector_name = detector_list[0]
    nr_acceptance = nr_acceptance_list[0]
    primary_parameter_values_strings = sorted(list(set(primary_parameter_values)))
    secondary_parameter_values_strings = sorted(list(set(secondary_parameter_values)))
    primary_parameter_values_floats = [float(valstring.replace("_", ".")) for valstring in primary_parameter_values_strings]
    secondary_parameter_values_floats = [float(valstring.replace("_", ".")) for valstring in secondary_parameter_values_strings]

    # canvas
    if flag_verbose: print(f"{fn}: setting up canvas")
    fig = plt.figure(
        figsize = [plot_figure_size_x_inch, plot_figure_size_x_inch*plot_aspect_ratio],
        dpi = 150,
        constrained_layout = True) 

    # axes
    if flag_verbose: print(f"{fn}: setting up axes")
    ax1 = fig.add_subplot()
    ax1.set_xlabel(parameter_translation_dict[secondary_parameter_name][1] +", " +parameter_translation_dict[secondary_parameter_name][0] +" / " +parameter_translation_dict[secondary_parameter_name][2], fontsize=plot_fontsize_axis_label)
    ax1.set_ylabel(r"ER rejection at $" +f"{100*nr_acceptance:.0f}" +r"\,\%$ NR acceptance / $\%$", fontsize=plot_fontsize_axis_label)
    if plot_xlim != []: ax1.set_xlim(plot_xlim)
    if plot_ylim != []: ax1.set_ylim(plot_ylim)
    if plot_log_y_axis: ax1.set_yscale('log')
    if plot_log_x_axis: ax1.set_xscale('log')

    # plotting
    for k, (primary_parameter_value_string, primary_parameter_value_float) in enumerate(zip(primary_parameter_values_strings, primary_parameter_values_floats)):
        # scatter plot color
        cmap = mpl.cm.get_cmap(plot_cmap)
        color_float_index = list(np.linspace(start=0, stop=1, num=len(primary_parameter_values_strings), endpoint=False))[k]
        color = mpl.colors.to_hex(cmap(color_float_index), keep_alpha=True)
        # scatter plot format
        default_scatter_format_dict = {
            "alpha"           : 1,
            "zorder"          : 1,
            "marker"          : "o", # markerstyle, see: https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
            "linewidths"      : plot_marker_linewidth,
            "s"               : plot_marker_size**2, # due to weird size scaling this needs to be squared to match the errorbar function
            "edgecolors"      : color,
            "facecolors"      : "white",
            "linestyles"      : "-",}
        default_errorbar_format_dict = {
            "alpha"           : 1,
            "zorder"          : 1,
            "marker"          : "o",
            "markersize"      : plot_marker_size,
            "markerfacecolor" : "white",
            "markeredgewidth" : plot_marker_linewidth,
            "markeredgecolor" : color,
            "linestyle"       : "",
            "fmt"             : '',
            "ecolor"          : color,
            "elinewidth"      : plot_marker_linewidth,
            "capsize"         : 1.8,
            "barsabove"       : True,
            "capthick"        : plot_marker_linewidth,}
        scatter_format_dict = default_scatter_format_dict.copy()
        #scatter_format_dict.update(plot_scatter_format_dict)
        errorbar_format_dict = default_errorbar_format_dict.copy()
        #errorbar_format_dict.update(plot_errorbar_format_dict)
        # scatter plotting
        ax1.errorbar(
            x = secondary_parameter_values_floats,
            y = [discrimination_line_scan_dict[detector_name +"__" +primary_parameter_name +"__" +primary_parameter_value_string +"__" +secondary_parameter_name +"__" +secondary_parameter_value_string]["er_rejection"]*100 for secondary_parameter_value_string in secondary_parameter_values_strings],
            yerr = [discrimination_line_scan_dict[detector_name +"__" +primary_parameter_name +"__" +primary_parameter_value_string +"__" +secondary_parameter_name +"__" +secondary_parameter_value_string]["er_rejection_uncertainty"]*100 for secondary_parameter_value_string in secondary_parameter_values_strings],
            label = parameter_translation_dict[primary_parameter_name][0] +r"$=" +f"{primary_parameter_value_float}" +r"\,$" +parameter_translation_dict[primary_parameter_name][2],
            **errorbar_format_dict,)
        ax1.scatter(
            secondary_parameter_values_floats,
            [discrimination_line_scan_dict[detector_name +"__" +primary_parameter_name +"__" +primary_parameter_value_string +"__" +secondary_parameter_name +"__" +secondary_parameter_value_string]["er_rejection"]*100 for secondary_parameter_value_string in secondary_parameter_values_strings],
            #label = parameter_translation_dict[primary_parameter_name][0] +r"$=" +f"{primary_parameter_value_float}" +r"\,$" +parameter_translation_dict[primary_parameter_name][2],
            **scatter_format_dict,)

    # legend
    handles, labels = ax1.get_legend_handles_labels()
    if plot_legend_invert_order:
        handles = handles[::-1]
        labels = labels[::-1]
    if plot_legend : ax1.legend(
        handles,
        labels,
        loc = "center",
        labelspacing = plot_legend_labelspacing,
        fontsize = plot_legend_fontsize,
        bbox_to_anchor = plot_legend_bbox_to_anchor,
        bbox_transform = ax1.transAxes,)

    # finishing
    if flag_verbose : print(f"{fn}: finishing")
    plt.show()
    for abspathstring in output_abspathstring_list:
        fig.savefig(abspathstring)
    if flag_verbose : print(f"{fn}: finished")
    return





############################################
### likelihood stuff helper functions
############################################


def reduce_nest_signature_to_eroi(
    sim_ndarray,            # ndarry, outpur from execNEST
    detector_dict,          # 
    eroi                    = [], # array containing start and end of the energy region of interest in keV, will be ignored if eroi = []
    s1_selection_window     = [], # array containing start and end of the s1 selection window, will be ignored if s1_selection_window = []
    s2_selection_window     = [], # array containing start and end of the s2 selection window, will be ignored if s2_selection_window = []
):

    """
    This function is used to select the signature data that lies within the WIMP EROI.
    credits: Catharina Hock
    """

    g1 = detector_dict["g1"]
    g2 = compute_g2_from_detector_configuration(detector_dict)
    w = 13.6 # eV

    if eroi != []:
        sim_ndarray = sim_ndarray[
            ( eroi[0] < w/1000*(sim_ndarray["S1_3Dcor [phd]"]/g1 + sim_ndarray["S2_3Dcorr [phd]"]/g2) ) &
            ( w/1000 *(sim_ndarray["S1_3Dcor [phd]"] /g1 + sim_ndarray["S2_3Dcorr [phd]"] /g2) < eroi[1] ) &
            ( sim_ndarray["S1_3Dcor [phd]"] > 0 ) &
            ( sim_ndarray["S2_3Dcorr [phd]"] > 0 )
        ]

    if s1_selection_window != []:
        sim_ndarray = sim_ndarray[
            ( sim_ndarray["S1_3Dcor [phd]"] > s1_selection_window[0] ) &
            ( sim_ndarray["S1_3Dcor [phd]"] < s1_selection_window[1] )
        ]

    if s2_selection_window != []:
        sim_ndarray = sim_ndarray[
            ( sim_ndarray["S2_3Dcorr [phd]"] > s2_selection_window[0] ) &
            ( sim_ndarray["S2_3Dcorr [phd]"] < s2_selection_window[1] )
        ]

    return sim_ndarray


def infer_two_dimensional_pdf_value_for_observation(
    observation, # two-element list, observation in x- and y-coordinates
    bin_edges_x, # list, bin edges of the x-axis, monotonously INncreasing (!), right edge included in bin, left side excluded (exept for 0-th bin))
    bin_edges_y, # list, bin edges of the y-axis, monotonously DEcreasing (!), upper (i.e., left) edge included in bin, lower (i.e., right) side excluded (exept for -1-th bin))
    pdf, # list of lists resembling the two-dimensional PDF, pdf[0] returns the PDF values of the uppermost row, pdf[-1][-1] returns the PDF value of the bottom righthand corner
):

    """
    This function is used to determine the PDF value for a given observation in two observables.
    The PDF is resembled by two arrays of bin edges 'bin_edges_x' and 'bin_edges_y' and a two-dimensional array representing the PDF values 'pdf'.
    Note that 'bin_edges_y' is monotonically decreasing and 'bin_edges_x' is monotonically increasing, such that they resemble a two dimensional histogram plot.
    This function is exemplarily used in 'sfs.calculate_wimp_parameter_exclusion_curve_dict()'.
    """

    # computing the 'x_index'
    if observation[0] < bin_edges_x[0] or observation[0] > bin_edges_x[-1]:
        raise Exception(f"observation {observation} outside outermost x-axis bin edges [{bin_edges_x[0]},{bin_edges_x[-1]}]")
    elif observation[0] == bin_edges_x[0]:
        x_index = 0
    else:
        red_bin_edges_x = np.array(bin_edges_x[1:]) # neglecting 0-th bin-edge
        difference_betweeen_bin_edge_x_and_observation_x = np.array(red_bin_edges_x -observation[0])
        difference_betweeen_bin_edge_x_and_observation_x = np.array([np.inf if dval <0 else dval for dval in difference_betweeen_bin_edge_x_and_observation_x])
        x_index = np.argmin(difference_betweeen_bin_edge_x_and_observation_x)
    # computing the 'y_index'
    if observation[1] > bin_edges_y[0] or observation[1] < bin_edges_y[-1]:
        raise Exception(f"observation {observation} outside outermost y-axis bin edges [{bin_edges_y[0]},{bin_edges_y[-1]}]")
    elif observation[1] == bin_edges_y[-1]:
        y_index = len(bin_edges_y)-2 # -2 instead of -1, because y_index corresponds to last y-bin which is indexed len(bin_edges_y)-2
    else:
        red_bin_edges_y = np.array(bin_edges_y[:-1]) # neglecting 0-th bin-edge
        difference_betweeen_bin_edge_y_and_observation_y = np.array(red_bin_edges_y -observation[1])
        difference_betweeen_bin_edge_y_and_observation_y = np.array([np.inf if dval <0 else dval for dval in difference_betweeen_bin_edge_y_and_observation_y])
        y_index = np.argmin(difference_betweeen_bin_edge_y_and_observation_y)
    # returning the desired pdf value
    #print(x_index)
    #print(y_index)
    #print(pdf[y_index][x_index])
    return pdf[y_index][x_index]


def generate_samples_from_discrete_pdf(
    random_variable_values,   # list of floats, representing the possible random variable values
    pdf_values,               # list of floats, representing the discrete probabilities of the random variable values
    nos,                      # int, number of samples to be generated
    seed                      = randrange(10000001), # int, random seed for drawing samples
):

    """
    This function is used to draw 'nos'-many samples from a discrete probability density function.
    """

    cumulative_pdf_values = [np.sum(pdf_values[:k])for k in range(len((pdf_values))+1)]
    #print(f"'cumulative_pdf_values' : {cumulative_pdf_values})
    random_uniform_samples_within_cumulative_pdf_values_edges = np.random.default_rng(seed=seed).uniform(0, cumulative_pdf_values[-1],nos)
    random_uniform_samples_index_list = []
    for rus in random_uniform_samples_within_cumulative_pdf_values_edges:
        if rus==0:
            index = 0
        else:
            red_bin_edges = cumulative_pdf_values[1:] # neglecting 0-th bin-edge
            difference_betweeen_bin_edge_and_rus = np.array(red_bin_edges -rus)
            difference_betweeen_bin_edge_and_rus = np.array([np.inf if dval <0 else dval for dval in difference_betweeen_bin_edge_and_rus])
            index = np.argmin(difference_betweeen_bin_edge_and_rus)
        random_uniform_samples_index_list.append(index)
    samples = list([float(random_variable_values[index]) for index in random_uniform_samples_index_list])
    return samples


def generate_two_dimensional_pdf_from_ndarray(
    ndarray, # ndarray, data from which the PDF is generated
    x_axis_key, # string, ndarray[x_axis_key] resembles the x-axis values of the data according to which the ndarray is binned
    y_axis_key, # 
    x_axis_bin_edges, # list of floats, bin edges along the x-axis according to which the PDF is computed
    y_axis_bin_edges,): # 

    """
    This function is used to compute the two-dimensional PDF for a ndarray within a grid of x- and y-bin edges.
    Note: 'x_axis_bin_edges' and 'y_axis_bin_edges' resemble a two-dimensional grid layed out on top of the ndarray --> The x bin edges are ascending while the y bin edges are descending
    Note: For every xy-bin all events within the semi-open interval ]x_{i},x_{i+1}] are summed up --> the lowest bin edges, i.e., x_axis_bin_edges[0] and x_axis_bin_edges[-1] is exluded.
    """

    # computing the reduced ndarray
    reduced_ndarray = ndarray[
        (ndarray[y_axis_key] <= y_axis_bin_edges[0]) &
        (ndarray[y_axis_key] > y_axis_bin_edges[-1]) &
        (ndarray[x_axis_key] <= x_axis_bin_edges[-1]) &
        (ndarray[x_axis_key] > x_axis_bin_edges[0]) ]
    total_number_of_events_within_outer_bin_edges = len(reduced_ndarray)

    # looping over the rows and columns of the data binning...
    pdf_array = []
    for row_index, y_axis_bin_edge in enumerate(y_axis_bin_edges[:-1]):
        pdf_row = []
        for column_index, x_axis_bin_edge in enumerate(x_axis_bin_edges[1:]):

            # ... and determining the data falling into the current bin
            data_within_current_bin = reduced_ndarray[
                (reduced_ndarray[y_axis_key] <= y_axis_bin_edges[row_index] ) &
                (reduced_ndarray[y_axis_key] > y_axis_bin_edges[row_index+1] ) &
                (reduced_ndarray[x_axis_key] <= x_axis_bin_edges[column_index+1] ) &
                (reduced_ndarray[x_axis_key] > x_axis_bin_edges[column_index] ) ]

            # filling and returning the 'pdf_array' with the bin-wise inferred PDF values
            pdf_val = len(data_within_current_bin)/total_number_of_events_within_outer_bin_edges
            pdf_row.append(pdf_val)
        pdf_array.append(pdf_row)
    return pdf_array


def compute_expected_number_of_events_within_eroi(
    spectrum_name,                  # 
    detector_dict,                  # 
    recoil_energy_kev_list,         # 
    abspath_spectra_files,          # 
    exposure_t_y,                   # 
    drift_field_v_cm,               # 
    xyz_pos_mm,                     # 
    spectrum_dict_default_values,   # 
    differential_rate_parameters,   # 
    number_of_simulated_signatures  = 100, # int, number of NEST-simulated signatures
    selection_window_recoil_energy  = [], # 
    selection_window_s1             = [], # 
    selection_window_s2             = [], # 
    flag_verbose                    = False, # flag indicating whether or not text output will be printed onto the screen
):

    """
    This function is used to compute the expected number of events within the WIMP EROI by simulating Poisson-fluctuating datasets, selecting the events falling within the EROI and averaging over all simulations.
    """

    # initialization
    fn = "compute_expected_number_of_events_within_eroi"
    number_of_events_within_eroi_list = []
    if flag_verbose : print(f"\n{fn}: initializing '{fn}'")

    # generating the 'spectrum_dict's
    if flag_verbose : print(f"\t\tgenerating the 'spectrum_dict's")
    spectrum_dict_list = give_spectrum_dict(
        spectrum_name                           = spectrum_name,
        recoil_energy_kev_list                  = recoil_energy_kev_list,
        abspath_spectra_files                   = abspath_spectra_files,
        exposure_t_y                            = exposure_t_y,
        num_events                              = "exposure_poisson",
        seed                                    = randrange(10000001),
        drift_field_v_cm                        = drift_field_v_cm,
        xyz_pos_mm                              = xyz_pos_mm,
        flag_spectrum_type                      = "integral",
        flag_verbose                            = False,
        flag_return_non_integer_events          = False,
        flag_inhibit_scaling                    = False,
        flag_number_of_output_spectrum_dicts    = number_of_simulated_signatures,
        spectrum_dict_default_values            = spectrum_dict_default_values,
        differential_rate_parameters            = differential_rate_parameters,)

    # simulating 'number_of_simulated_signatures'-many signatures
    if flag_verbose : print(f"{fn}: simulating '{number_of_simulated_signatures}'-many signatures")
    for k in range(len(spectrum_dict_list)):
        if flag_verbose : print(f"\tk={k}/{number_of_simulated_signatures}")

        # simulating the detector signature with NEST
        if flag_verbose : print(f"\t\tsimulating the detector signature with NEST")
        signature_data = execNEST(
            spectrum_dict               = spectrum_dict_list[k],
            baseline_drift_field_v_cm   = drift_field_v_cm,
            baseline_detector_dict      = detector_dict,
            detector_dict = {},)

        # selecting the events falling into the EROI
        if flag_verbose : print(f"\t\tselecting the events falling into the EROI")
        reduced_signature_data = reduce_nest_signature_to_eroi(
            sim_ndarray             = signature_data,
            detector_dict           = detector_dict,
            eroi                    = selection_window_recoil_energy,
            s1_selection_window     = selection_window_s1,
            s2_selection_window     = selection_window_s2,)
        number_of_events_within_eroi = len(reduced_signature_data)
        number_of_events_within_eroi_list.append(number_of_events_within_eroi)

    # computing the expectation value as the arithmetic mean of all extracted numbers
    if flag_verbose : print(f"{fn}: computing expectation value as arithmetic mean")
    #print(f"'number_of_events_within_eroi_list' = {number_of_events_within_eroi_list}")
    number_of_events_within_eroi_expectation_value = np.mean(number_of_events_within_eroi_list)
    if flag_verbose : print(f"\t-----> {number_of_events_within_eroi_expectation_value:.2f}")

    return number_of_events_within_eroi_expectation_value





############################################
### likelihood stuff main function
############################################


def calculate_wimp_parameter_exclusion_curve(
    # physical detector parameters
    detector__drift_field_v_cm,                                      # electrical drift field strength of the detector in V/cm
    detector__nest_parameter_dict,                                   # dict, NEST detector parameters
    detector__runtime_y,                                             # float, detector live time in years
    detector__fiducial_mass_t,                                       # float, fiducial mass of the detector in metric tonnes
    detector__detector_name,                                         # string, detector name string
    # physical spectrum parameters
    spectrum__default_spectrum_profiles,                             # dict, default spectrum parameters
    spectrum__resources,                                             # string, abspath to spectrum resources (e.g., .csv files)
    spectrum__er_background_model,                                   # string, ER background model
    spectrum__nr_background_model,                                   # string, NR background model
    spectrum__wimp_model,                                            # string, WIMP model
    # NEST settings
    
    # simulation setting
    simulation__er_spectrum_energy_simulation_window_kev,            # two-tuple, ER simulation window (due to leakage from lower and higher energies it is not sufficient to just simulate for the WIMP EROI)
    simulation__nr_spectrum_energy_simulation_window_kev,            # two-tuple, NR simulation window (due to leakage from lower and higher energies it is not sufficient to just simulate for the WIMP EROI)
    simulation__number_of_spectrum_energy_bins,                      # int, number of energy bins into which the simulated spectra are histogrammed
    simulation__number_of_upper_limit_simulations_per_wimp_mass,     # int, number of upper limits the median upper limit is computed from
    simulation__number_of_pdf_calculation_events,                    # int, number of events simulated per PDF computation (i.e., one for ER and NR each plus one for each WIMP mass investigated)
    simulation__number_of_samples_for_expectation_value_computation, # int, number of physical ER and NR datasets that are used to compute the expected number of ER and NR events within the WIMP EROI
    # limit calculation parameters
    limit__er_eroi_kev,                                              # two-tuple, WIMP EROI in keV_ee
    limit__nr_eroi_kev,                                              # two-tuple, WIMP EROI in keV_nr
    limit__wimp_mass_gev_list                                        = list(np.geomspace(start=10, stop=500, num=5, endpoint=True)),     # list of WIMP masses in GeV the upper exclusion limit in \sigma is computed for
    limit__number_of_cs1_bins                                        = 20, # int, number of bins in cS1 based on which the spectrum PDF is computed for the likelihood function
    limit__number_of_cs2_bins                                        = 20, # int, number of bins in cS2 based on which the spectrum PDF is computed for the likelihood function
    # flags
    flag_verbose                                                     = [False, True, "high-level-only"][2],                              # flag indicating the output this function is printing onto the screen
    flag_load_er_and_nr_signatures_for_pdf_calculation               = [False, True][0],                                                 # flag indicating whether or not the ER and NR signature are loaded instead of being computed (mainly relevant for testing)
    flag_plot_pdfs                                                   = [False, True, "er_nr_only"][2],                                   # flag indicating whether or not the inferred pdfs are being plotted
):

    """
    This function is used to calculate a WIMP parameter exclusion curve.
    """

    # initializing
    fn = "cwpec"
    if flag_verbose : print(f"\n{fn}: initializing 'calculate_wimp_parameter_exclusion_curve' ({fn})")
    flag_verbose_low_level = True if flag_verbose == True else False
    start_time = time.time()

    # implementing the detector
    """
    Within this section the specified detector will be implemented into NEST.
    It will not be modified beyond this point.
    """
    if flag_verbose : print(f"{fn}: implementing the detector")
    if [False,True][1]:
        #calc_active_xenon_mass_of_detector_dict_t(detector_dict)
        if flag_verbose : print(f"\tupdating drift velocity-dependent parameters")
        install_detector_dict = detector__nest_parameter_dict.copy()
        install_detector_dict = adjust_detector_drift_time_parameters(detector_dict=install_detector_dict, drift_field_v_cm=detector__drift_field_v_cm)
        if flag_verbose : print(f"\tsaving detector header file")
        convert_detector_dict_into_detector_header(
            detector_dict = install_detector_dict,
            abspath_output_list = [abspath_nest_installation_nest_include_detectors],
            detector_name = detector__detector_name,
            flag_verbose = flag_verbose_low_level,)
        if flag_verbose : print(f"\tinstalling detector header file")
        install_detector_header_file(
            abspathfile_new_detector_hh = abspath_nest_installation_nest_include_detectors +detector__detector_name +".hh",
            flag_clean_reinstall = True,
            flag_verbose = flag_verbose_low_level)
    ct = time.time()-start_time
    ctd = timedelta(seconds=ct)
    if flag_verbose : print(f"\t\tfinished within {ctd} h'")


    # a priori calculations
    """
    Within the following sections some quantities will be calculated that will be utilized throughout the exclusion curve calculation.
    This information will be stored and updated within the 'spectrum_components_dict', which will also be returned ad the end of the function call.
    """
    if flag_verbose : print(f"{fn}: a priori calculations:")


    # a priori calculations: initial definitions
    if flag_verbose : print(f"\tinitial definitions")
    spectrum_components_dict = {
        "cs1_bin_edges"                                     : [], # list of floats, cS1 bin edges of the binned observable space
        "cs2_bin_edges"                                     : [], # list of floats, cS2 bin edges of the binned observable space, note: will be descending in order
        "er_background" : {
            "spectral_pdf"                                  : [], # list of list of floats, resembling the pdf of the ER background for the binned observable space, will be computed below
            "number_of_expected_events_within_eroi"         : 0,  # float, number of expected ER background events within the whole binned observable space, will be computed below
            "recoil_energy_kev_list"                        : bin_centers_from_interval(simulation__er_spectrum_energy_simulation_window_kev, simulation__number_of_spectrum_energy_bins), # list of floats, energy values of the energy bin centers for which the ER events will be simulated
        },
        "nr_background" : {
            "spectral_pdf"                                  : {},
            "number_of_expected_events_within_eroi"         : 0,
            "recoil_energy_kev_list"                        : bin_centers_from_interval(simulation__nr_spectrum_energy_simulation_window_kev, simulation__number_of_spectrum_energy_bins),
        },
        "wimps" : {
            "spectral_pdf"                                  : {},
            "number_of_expected_events_within_eroi"         : 0,
            "recoil_energy_kev_list"                        : bin_centers_from_interval(simulation__nr_spectrum_energy_simulation_window_kev, simulation__number_of_spectrum_energy_bins),
        },}


    # a priori calculations: observable space binning
    if flag_verbose : print(f"\tcalculating bin edges of the binned cS1-cS2 observable space")
    if [False,True][1]:
        # ER and NR background 'spectrum_dict's
        integral_spectra_dict = {}
        for spectrum_string in ["er_background", "nr_background"]:
            if flag_verbose : print(f"\t\tcalculating 'integral_spectrum_dict' for '{spectrum_string}'")
            integral_spectrum_dict = give_spectrum_dict(
                spectrum_name                  = spectrum__er_background_model if spectrum_string=="er_background" else spectrum__nr_background_model,
                recoil_energy_kev_list         = spectrum_components_dict[spectrum_string]["recoil_energy_kev_list"],
                abspath_spectra_files          = spectrum__resources,
                exposure_t_y                   = detector__fiducial_mass_t *detector__runtime_y,
                num_events                     = simulation__number_of_pdf_calculation_events,
                # nest parameters
                seed                           = 1,
                drift_field_v_cm               = detector__drift_field_v_cm,
                xyz_pos_mm                     = "-1",
                # flags
                flag_spectrum_type             = ["differential", "integral"][1],
                flag_verbose                   = flag_verbose_low_level,
                # keywords
                spectrum_dict_default_values   = spectrum__default_spectrum_profiles,
                differential_rate_parameters   = {})
            integral_spectra_dict.update({spectrum_string : integral_spectrum_dict})
        # ER and NR background signatures
        if flag_verbose : print(f"\t\tgenerating ER and NR background signatures signatures by executing NEST")
        if flag_load_er_and_nr_signatures_for_pdf_calculation:
            er_spectrum_signature = np.load("/home/daniel/Desktop/arbeitsstuff/sfs/github_repo_v2/signatures/example__signature__darwin_baseline_detector__er_background_high_stat.npy")
            nr_spectrum_signature = np.load("/home/daniel/Desktop/arbeitsstuff/sfs/github_repo_v2/signatures/example__signature__darwin_baseline_detector__nr_background_high_stat.npy")
        else:
            er_spectrum_signature = execNEST(
                spectrum_dict = integral_spectra_dict["er_background"],
                baseline_detector_dict = install_detector_dict, # NOTE: replace with 'install_detector_dict' once finished with testing
                baseline_drift_field_v_cm = detector__drift_field_v_cm,
                detector_dict = {},
                detector_name = detector__detector_name,
                abspath_list_detector_dict_json_output = [],
                flag_verbose = flag_verbose_low_level,
                flag_print_stdout_and_stderr = False,)
            nr_spectrum_signature = execNEST(
                spectrum_dict = integral_spectra_dict["nr_background"],
                baseline_detector_dict = install_detector_dict, # NOTE: replace with 'install_detector_dict' once finished with testing
                baseline_drift_field_v_cm = detector__drift_field_v_cm,
                detector_dict = {},
                detector_name = detector__detector_name,
                abspath_list_detector_dict_json_output = [],
                flag_verbose = flag_verbose_low_level,
                flag_print_stdout_and_stderr = False,)
        # ER and NR background signature reduction
        if flag_verbose : print(f"\t\treducing ER and NR background signatures to WIMP EROI")
        er_spectrum_signature = reduce_nest_signature_to_eroi(
            sim_ndarray = er_spectrum_signature,
            eroi = limit__er_eroi_kev,
            detector_dict = detector__nest_parameter_dict)
        nr_spectrum_signature = reduce_nest_signature_to_eroi(
            sim_ndarray = nr_spectrum_signature,
            eroi = limit__er_eroi_kev,
            detector_dict = detector__nest_parameter_dict)
        # determining the observable space bin edges
        if flag_verbose : print(f"\t\tdetermining the cS1 and cS2 observable space bin edges")
        spectral_pdf_bin_edges_cs1 = list(np.linspace(
            start = np.min([np.min(er_spectrum_signature["S1_3Dcor [phd]"]),np.min(nr_spectrum_signature["S1_3Dcor [phd]"])]),
            stop = np.max([np.max(er_spectrum_signature["S1_3Dcor [phd]"]),np.max(nr_spectrum_signature["S1_3Dcor [phd]"])]),
            num = limit__number_of_cs1_bins,
            endpoint = True))
        spectral_pdf_bin_edges_cs2 = list(np.geomspace(
            start = np.min([np.min(er_spectrum_signature["S2_3Dcorr [phd]"]),np.min(nr_spectrum_signature["S2_3Dcorr [phd]"])]),
            stop = np.max([np.max(er_spectrum_signature["S2_3Dcorr [phd]"]),np.max(nr_spectrum_signature["S2_3Dcorr [phd]"])]),
            num = limit__number_of_cs2_bins,
            endpoint = True))[::-1] # note, that the cS2 bins start with the highest bin edge first
        # updating the 'spectrum_components_dict'
        if flag_verbose : print(f"\t\tupdating the 'spectrum_components_dict'")
        spectrum_components_dict.update({
            "cs1_bin_edges" : spectral_pdf_bin_edges_cs1,
            "cs2_bin_edges" : spectral_pdf_bin_edges_cs2,}) # note, that the cS2 bins start with the highest bin edge first
        ct = time.time()-ct
        ctd = timedelta(seconds=ct)
        if flag_verbose : print(f"\t\tfinished within {ctd} h'")


    # a priori calculations: spectral PDFs for ER and NR background
    if flag_verbose : print(f"\tcalculating the PDFs of the ER and NR backgrounds within the binned observable space")
    if [False,True][1]:
        # calculating the PDFs
        if flag_verbose : print(f"\t\tcalculating the ER PDF")
        er_pdf_array = generate_two_dimensional_pdf_from_ndarray(
            ndarray = er_spectrum_signature,
            x_axis_key = "S1_3Dcor [phd]",
            y_axis_key = "S2_3Dcorr [phd]",
            x_axis_bin_edges = spectral_pdf_bin_edges_cs1,
            y_axis_bin_edges = spectral_pdf_bin_edges_cs2,)
        if flag_verbose : print(f"\t\tcalculating the NR PDF")
        nr_pdf_array = generate_two_dimensional_pdf_from_ndarray(
            ndarray = nr_spectrum_signature,
            x_axis_key = "S1_3Dcor [phd]",
            y_axis_key = "S2_3Dcorr [phd]",
            x_axis_bin_edges = spectral_pdf_bin_edges_cs1,
            y_axis_bin_edges = spectral_pdf_bin_edges_cs2,)
        # updating the 'spectrum_components_dict'
        if flag_verbose : print(f"\t\tupdating the 'spectrum_components_dict'")
        spectrum_components_dict["er_background"]["spectral_pdf"] = er_pdf_array
        spectrum_components_dict["nr_background"]["spectral_pdf"] = nr_pdf_array
        # deleting the high-statistics ER and NR signatures
        if flag_verbose : print(f"\t\tdeleting the high-statistics ER and NR signatures")
        del(er_spectrum_signature)
        del(nr_spectrum_signature)
        # plotting the inferred PDFs
        if flag_plot_pdfs:
            for spectrum_string in ["er_background", "nr_background"]:
                if flag_verbose : print(f"\t\tplotting spectral PDF for '{spectrum_string}'")
                x_list = []
                y_list = []
                weights_list = []
                for k, bin_edge_y_top in enumerate(spectrum_components_dict["cs2_bin_edges"][:-1]):
                    for l, bin_edge_x_left in enumerate(spectrum_components_dict["cs1_bin_edges"][:-1]):
                        x_list.append(spectrum_components_dict["cs1_bin_edges"][l] +0.5*(spectrum_components_dict["cs1_bin_edges"][l+1]-spectrum_components_dict["cs1_bin_edges"][l]))
                        y_list.append(spectrum_components_dict["cs2_bin_edges"][k] -0.5*(spectrum_components_dict["cs2_bin_edges"][k]-spectrum_components_dict["cs2_bin_edges"][k+1]))
                        weights_list.append(spectrum_components_dict[spectrum_string]["spectral_pdf"][k][l])
                fig = plt.figure(
                    figsize = [5.670, 5.670*9/16],
                    dpi = 150,
                    constrained_layout = True)
                ax1 = fig.add_subplot()
                hist = ax1.hist2d(
                    x = x_list,
                    y = y_list,
                    bins = [spectrum_components_dict["cs1_bin_edges"], spectrum_components_dict["cs2_bin_edges"][::-1]],
                    weights = weights_list,
                    cmap = "YlGnBu",
                    cmin = 0.00000000001,)
                if spectrum_components_dict["cs2_bin_edges"][0]-spectrum_components_dict["cs2_bin_edges"][1] != spectrum_components_dict["cs2_bin_edges"][1]-spectrum_components_dict["cs2_bin_edges"][2]: ax1.set_yscale('log')
                ax1.set_xlabel(r"$cS_1$ / $\mathrm{phd}$", fontsize=11)
                ax1.set_ylabel(r"$cS_2$ / $\mathrm{phd}$", fontsize=11)
                plt.show()
        ct = time.time()-ct
        ctd = timedelta(seconds=ct)
        if flag_verbose : print(f"\t\tfinished within {ctd} h'")


    # a priori calculations: ER and NR expectation values
    if flag_verbose : print(f"\tcalculating the expected number of ER and NR background events within the binned observable space")
    if [False,True][1]:
        # calculating the ER and NR events expected within the WIMP EROI
        if flag_verbose : print(f"\t\tcalculating the number of ER background events expected within the WIMP EROI")
        number_of_expected_er_background_events_within_wimp_eroi = compute_expected_number_of_events_within_eroi(
            spectrum_name                   = spectrum__er_background_model,
            detector_dict                   = install_detector_dict,
            recoil_energy_kev_list          = spectrum_components_dict["er_background"]["recoil_energy_kev_list"],
            abspath_spectra_files           = spectrum__resources,
            exposure_t_y                    = detector__runtime_y*detector__fiducial_mass_t,
            drift_field_v_cm                = detector__drift_field_v_cm,
            xyz_pos_mm                      = "-1",
            spectrum_dict_default_values    = spectrum__default_spectrum_profiles,
            differential_rate_parameters    = {},
            number_of_simulated_signatures  = simulation__number_of_samples_for_expectation_value_computation,
            selection_window_recoil_energy  = limit__er_eroi_kev,
            selection_window_s1             = [],
            selection_window_s2             = [],
            flag_verbose                    = flag_verbose_low_level,)
        if flag_verbose : print(f"\t\tcalculating the number of NR background events expected within the WIMP EROI")
        number_of_expected_nr_background_events_within_wimp_eroi = compute_expected_number_of_events_within_eroi(
            spectrum_name                   = spectrum__nr_background_model,
            detector_dict                   = install_detector_dict,
            recoil_energy_kev_list          = spectrum_components_dict["nr_background"]["recoil_energy_kev_list"],
            abspath_spectra_files           = spectrum__resources,
            exposure_t_y                    = detector__runtime_y*detector__fiducial_mass_t,
            drift_field_v_cm                = detector__drift_field_v_cm,
            xyz_pos_mm                      = "-1",
            spectrum_dict_default_values    = spectrum__default_spectrum_profiles,
            differential_rate_parameters    = {},
            number_of_simulated_signatures  = simulation__number_of_samples_for_expectation_value_computation,
            selection_window_recoil_energy  = limit__er_eroi_kev,
            selection_window_s1             = [],
            selection_window_s2             = [],
            flag_verbose                    = flag_verbose_low_level,)
        # updating the 'spectrum_components_dict'
        if flag_verbose : print(f"\t\tupdating the 'spectrum_components_dict'")
        spectrum_components_dict["er_background"]["number_of_expected_events_within_eroi"] = number_of_expected_er_background_events_within_wimp_eroi
        spectrum_components_dict["nr_background"]["number_of_expected_events_within_eroi"] = number_of_expected_nr_background_events_within_wimp_eroi
        # finishing
        ct = time.time()-ct
        ctd = timedelta(seconds=ct)
        if flag_verbose : print(f"\t\tfinished within {ctd} h'")


    # looping over the specified WIMP masses
    for k, wimp_mass_gev in enumerate(limit__wimp_mass_gev_list):
        if flag_verbose : print(f"{fn}: starting WIMP mass loop with k={k}/{len(limit__wimp_mass_gev_list)} for {wimp_mass_gev:.2f} GeV")


        # calculating the spectral PDF of the WIMP spectrum
        if flag_verbose : print(f"\tcalculating the PDF of the WIMP spectrum within the binned observable space")
        """
        Note that the WIMP PDF does not depend on the spin-independent WIMP-nucleon cross-section.
        However, its spectral shape (i.e., in cS1-cS2 ovservable space) depends on the WIMP mass.
        Hence we need to calculate it for every investigated WIMP mass
        """
        if [False,True][1]:
            # generating the high-statistics signature data
            if flag_verbose : print(f"\t\tcalculating 'integral_spectrum_dict' for the high-statistics WIMP spectrum")
            wimps_integral_spectrum_dict = give_spectrum_dict(
                spectrum_name                  = spectrum__wimp_model,
                recoil_energy_kev_list         = spectrum_components_dict["wimps"]["recoil_energy_kev_list"],
                abspath_spectra_files          = spectrum__resources,
                exposure_t_y                   = detector__fiducial_mass_t *detector__runtime_y,
                num_events                     = simulation__number_of_pdf_calculation_events,
                # nest parameters
                seed                           = 1,
                drift_field_v_cm               = detector__drift_field_v_cm,
                xyz_pos_mm                     = "-1",
                # flags
                flag_spectrum_type             = ["differential", "integral"][1],
                flag_verbose                   = flag_verbose_low_level,
                # keywords
                spectrum_dict_default_values   = spectrum__default_spectrum_profiles,
                differential_rate_parameters   = {
                    "mw"                       : wimp_mass_gev, # GeV
                    "sigma_nucleon"            : 1e-45, # cm^2
                })
            # generating the high-statistics WIMP signature
            if flag_verbose : print(f"\t\tgenerating the high-statistics WIMP signature by executing NEST")
            wimp_spectrum_signature = execNEST(
                spectrum_dict                           = wimps_integral_spectrum_dict,
                baseline_detector_dict                  = install_detector_dict, # NOTE: replace with 'install_detector_dict' once finished with testing
                baseline_drift_field_v_cm               = detector__drift_field_v_cm,
                detector_dict                           = {},
                detector_name                           = detector__detector_name,
                abspath_list_detector_dict_json_output  = [],
                flag_verbose                            = flag_verbose_low_level,
                flag_print_stdout_and_stderr            = False,)
            wimp_spectrum_signature = reduce_nest_signature_to_eroi(
                sim_ndarray     = wimp_spectrum_signature,
                eroi            = limit__er_eroi_kev,
                detector_dict   = install_detector_dict)
            # calculating the PDF
            if flag_verbose : print(f"\t\tcalculating the PDF")
            wimps_pdf_array = generate_two_dimensional_pdf_from_ndarray(
                ndarray = wimp_spectrum_signature,
                x_axis_key = "S1_3Dcor [phd]",
                y_axis_key = "S2_3Dcorr [phd]",
                x_axis_bin_edges = spectral_pdf_bin_edges_cs1,
                y_axis_bin_edges = spectral_pdf_bin_edges_cs2,)
            # updating the 'spectrum_components_dict'
            if flag_verbose : print(f"\t\tupdating the 'spectrum_components_dict'")
            spectrum_components_dict["wimps"]["spectral_pdf"] = wimps_pdf_array
            # deleting the high-statistics ER and NR signatures
            if flag_verbose : print(f"\t\tdeleting the high-statistics WIMP signature")
            del(wimp_spectrum_signature)
            # plotting the inferred PDF
            if flag_plot_pdfs:
                spectrum_string = "wimps"
                if flag_verbose : print(f"\t\tplotting spectral PDF for '{spectrum_string}'")
                x_list = []
                y_list = []
                weights_list = []
                for k, bin_edge_y_top in enumerate(spectrum_components_dict["cs2_bin_edges"][:-1]):
                    for l, bin_edge_x_left in enumerate(spectrum_components_dict["cs1_bin_edges"][:-1]):
                        x_list.append(spectrum_components_dict["cs1_bin_edges"][l] +0.5*(spectrum_components_dict["cs1_bin_edges"][l+1]-spectrum_components_dict["cs1_bin_edges"][l]))
                        y_list.append(spectrum_components_dict["cs2_bin_edges"][k] -0.5*(spectrum_components_dict["cs2_bin_edges"][k]-spectrum_components_dict["cs2_bin_edges"][k+1]))
                        weights_list.append(spectrum_components_dict[spectrum_string]["spectral_pdf"][k][l])
                fig = plt.figure(
                    figsize = [5.670, 5.670*9/16],
                    dpi = 150,
                    constrained_layout = True)
                ax1 = fig.add_subplot()
                hist = ax1.hist2d(
                    x = x_list,
                    y = y_list,
                    bins = [spectrum_components_dict["cs1_bin_edges"], spectrum_components_dict["cs2_bin_edges"][::-1]],
                    weights = weights_list,
                    cmap = "YlGnBu",
                    cmin = 0.00000000001,)
                if spectrum_components_dict["cs2_bin_edges"][0]-spectrum_components_dict["cs2_bin_edges"][1] != spectrum_components_dict["cs2_bin_edges"][1]-spectrum_components_dict["cs2_bin_edges"][2]: ax1.set_yscale('log')
                ax1.set_xlabel(r"$cS_1$ / $\mathrm{phd}$", fontsize=11)
                ax1.set_ylabel(r"$cS_2$ / $\mathrm{phd}$", fontsize=11)
                plt.show()
            # finishing this substep
            ct = time.time()-ct
            ctd = timedelta(seconds=ct)
            if flag_verbose : print(f"\t\tfinished within {ctd} h'")


        # calculating the expected number of WIMP events within the binned observable space
        if flag_verbose : print(f"\tcalculating the expected number of WIMP events within the binned observable space")
        """
        Note, that the WIMP PDF does not depend on the spin-independent WIMP-nucleon cross-section.
        Instead, the expectation value depends on both the spin-independent WIMP-nucleon cross-section and the WIMP mass.
        For every WIMP mass we calculate the expectation value corresponding to a 10**(-45) cm^2 WIMP.
        Since the expectation value depends linearly on the cross-section we will just scale this value accordingly.
        """
        if [False,True][1]:
            # calculating the WIMP events expected within the WIMP EROI
            if flag_verbose : print(f"\t\tcalculating the number of WIMP events expected within the WIMP EROI")
            number_of_expected_wimp_events_within_wimp_eroi = compute_expected_number_of_events_within_eroi(
                spectrum_name                   = spectrum__wimp_model,
                detector_dict                   = install_detector_dict,
                recoil_energy_kev_list          = spectrum_components_dict["wimps"]["recoil_energy_kev_list"],
                abspath_spectra_files           = spectrum__resources,
                exposure_t_y                    = detector__runtime_y*detector__fiducial_mass_t,
                drift_field_v_cm                = detector__drift_field_v_cm,
                xyz_pos_mm                      = "-1",
                spectrum_dict_default_values    = spectrum__default_spectrum_profiles,
                differential_rate_parameters    = {
                    "mw"                       : wimp_mass_gev, # GeV
                    "sigma_nucleon"            : 1e-45, # cm^2
                },
                number_of_simulated_signatures  = simulation__number_of_samples_for_expectation_value_computation,
                selection_window_recoil_energy  = limit__er_eroi_kev,
                selection_window_s1             = [],
                selection_window_s2             = [],
                flag_verbose                    = flag_verbose_low_level,)
            # updating the 'spectrum_components_dict'
            if flag_verbose : print(f"\t\tupdating the 'spectrum_components_dict'")
            spectrum_components_dict["wimps"]["number_of_expected_events_within_eroi"] = number_of_expected_wimp_events_within_wimp_eroi
            # finishing
            ct = time.time()-ct
            ctd = timedelta(seconds=ct)
            if flag_verbose : print(f"\t\tfinished within {ctd} h'")


        # simulating 'simulation__number_of_upper_limit_simulations_per_wimp_mass'-many background-only datasets
        if flag_verbose : print(f"\tsimulating {simulation__number_of_upper_limit_simulations_per_wimp_mass} background-only datasets")
        if flag_verbose : print(f"\t\tgenerating the 'spectrum_dict's")
        er_background_spectrum_dict_list = give_spectrum_dict(
            spectrum_name                           = spectrum__er_background_model,
            recoil_energy_kev_list                  = spectrum_components_dict["er_background"]["recoil_energy_kev_list"],
            abspath_spectra_files                   = spectrum__resources,
            exposure_t_y                            = detector__runtime_y*detector__fiducial_mass_t,
            num_events                              = "exposure_poisson",
            seed                                    = randrange(10000001),
            drift_field_v_cm                        = detector__drift_field_v_cm,
            xyz_pos_mm                              = "-1",
            flag_spectrum_type                      = "integral",
            flag_verbose                            = flag_verbose_low_level,
            flag_return_non_integer_events          = flag_verbose_low_level,
            flag_inhibit_scaling                    = flag_verbose_low_level,
            flag_number_of_output_spectrum_dicts    = simulation__number_of_upper_limit_simulations_per_wimp_mass,
            spectrum_dict_default_values            = spectrum__default_spectrum_profiles,
            differential_rate_parameters            = {},)
        nr_background_spectrum_dict_list = give_spectrum_dict(
            spectrum_name                           = spectrum__nr_background_model,
            recoil_energy_kev_list                  = spectrum_components_dict["nr_background"]["recoil_energy_kev_list"],
            abspath_spectra_files                   = spectrum__resources,
            exposure_t_y                            = detector__runtime_y*detector__fiducial_mass_t,
            num_events                              = "exposure_poisson",
            seed                                    = randrange(10000001),
            drift_field_v_cm                        = detector__drift_field_v_cm,
            xyz_pos_mm                              = "-1",
            flag_spectrum_type                      = "integral",
            flag_verbose                            = flag_verbose_low_level,
            flag_return_non_integer_events          = flag_verbose_low_level,
            flag_inhibit_scaling                    = flag_verbose_low_level,
            flag_number_of_output_spectrum_dicts    = simulation__number_of_upper_limit_simulations_per_wimp_mass,
            spectrum_dict_default_values            = spectrum__default_spectrum_profiles,
            differential_rate_parameters            = {},)
        if flag_verbose : print(f"\t\tgenerating the ER and NR background 'signature's")
        er_background_signature_list = []
        nr_background_signature_list = []
        for l in range(simulation__number_of_upper_limit_simulations_per_wimp_mass):
            er_background_signature = execNEST(
                spectrum_dict                           = er_background_spectrum_dict_list[l],
                baseline_detector_dict                  = detector__nest_parameter_dict,
                baseline_drift_field_v_cm               = detector__drift_field_v_cm,
                detector_dict                           = {},
                detector_name                           = detector__detector_name,
                abspath_list_detector_dict_json_output  = [],
                flag_verbose                            = flag_verbose_low_level,
                flag_print_stdout_and_stderr            = False,)
            er_background_signature = reduce_nest_signature_to_eroi(
                sim_ndarray = er_background_signature,
                eroi = limit__er_eroi_kev,
                detector_dict = detector__nest_parameter_dict)
            er_background_signature_list.append(er_background_signature)
            nr_background_signature = execNEST(
                spectrum_dict                           = nr_background_spectrum_dict_list[l],
                baseline_detector_dict                  = detector__nest_parameter_dict,
                baseline_drift_field_v_cm               = detector__drift_field_v_cm,
                detector_dict                           = {},
                detector_name                           = detector__detector_name,
                abspath_list_detector_dict_json_output  = [],
                flag_verbose                            = flag_verbose_low_level,
                flag_print_stdout_and_stderr            = False,)
            nr_background_signature = reduce_nest_signature_to_eroi(
                sim_ndarray = nr_background_signature,
                eroi = limit__er_eroi_kev,
                detector_dict = detector__nest_parameter_dict)
            nr_background_signature_list.append(nr_background_signature)


        # calculating the maximum likelihood parameter estimators
        if flag_verbose : print(f"\tcalculating the maximum likelihood parameter estimators")
        mle_sigma_list = []
        mle_thetavec_list = []
        for l in range(simulation__number_of_upper_limit_simulations_per_wimp_mass):
            # a priori definitions and calculations
            if flag_verbose : print(f"\t\ta priori calculations")
            er_data = er_background_signature_list[l]
            nr_data = nr_background_signature_list[l]
            bin_edges_s2 = spectrum_components_dict["cs2_bin_edges"]
            bin_edges_s1 = spectrum_components_dict["cs1_bin_edges"]
            lambda_er = spectrum_components_dict["er_background"]["number_of_expected_events_within_eroi"]
            lambda_nr = spectrum_components_dict["nr_background"]["number_of_expected_events_within_eroi"]
            lambda_wimps = spectrum_components_dict["wimps"]["number_of_expected_events_within_eroi"]
            pdf_er = spectrum_components_dict["er_background"]["spectral_pdf"]
            pdf_nr = spectrum_components_dict["nr_background"]["spectral_pdf"]
            pdf_wimps = spectrum_components_dict["wimps"]["spectral_pdf"]
            theta_er_sigma = 0.3
            theta_nr_sigma = 0.2
            n_obs_er = []
            n_obs_nr = []
            for b_row, be_row in enumerate(bin_edges_s2[:-1]):
                n_obs_er_row_list = []
                n_obs_nr_row_list = []
                for b_column, be_column in enumerate(bin_edges_s1[:-1]):
                    n_obs_b_er = len(er_data[
                        ( er_data["S2_3Dcorr [phd]"] <= bin_edges_s2[b_row] ) &
                        ( er_data["S2_3Dcorr [phd]"] > bin_edges_s2[b_row+1] ) &
                        ( er_data["S1_3Dcor [phd]"] <= bin_edges_s1[b_column+1] ) &
                        ( er_data["S1_3Dcor [phd]"] > bin_edges_s1[b_column] ) ] )
                    n_obs_b_nr = len(nr_data[
                        ( nr_data["S2_3Dcorr [phd]"] <= bin_edges_s2[b_row] ) &
                        ( nr_data["S2_3Dcorr [phd]"] > bin_edges_s2[b_row+1] ) &
                        ( nr_data["S1_3Dcor [phd]"] <= bin_edges_s1[b_column+1] ) &
                        ( nr_data["S1_3Dcor [phd]"] > bin_edges_s1[b_column] ) ] )
                    n_obs_er_row_list.append(n_obs_b_er)
                    n_obs_nr_row_list.append(n_obs_b_nr)
                n_obs_er.append(n_obs_er_row_list)
                n_obs_nr.append(n_obs_nr_row_list)
#            print(f"#######################################")
#            print(pdf_er)
#            print(pdf_nr)
#            print(lambda_er)
#            print(lambda_nr)
#            print(lambda_wimps)

#            # defining the likelihood function
            def neg_likelihood_function(
                i_sigma, # SI WIMP-nucleon cross-section, Note: due to computational reasons 'lambda_wimps' was defined to correspond to a sigma of 1e-45 --> one needs to correcto for that factor later
                i_theta_er,
                i_theta_nr,
            ):
                # initial definitions
                lf_val = np.array(1)
                sigma = np.array(i_sigma)
                theta_er = np.array(i_theta_er)
                theta_nr = np.array(i_theta_nr)
                # Poisson factor product: looping over all bins of the cS1-cS2 observable space
                for b_row, be_row in enumerate(bin_edges_s2[:-1]):
                    for b_column, be_column in enumerate(bin_edges_s1[:-1]):
                        n_obs_b = n_obs_er[b_row][b_column] +n_obs_nr[b_row][b_column]
                        lambda_b = pdf_er[b_row][b_column]*lambda_er*theta_er +pdf_nr[b_row][b_column]*lambda_nr*theta_nr +pdf_wimps[b_row][b_column]*lambda_wimps*sigma
                        lf_val = lf_val *(lambda_b**n_obs_b *np.exp(-lambda_b)) # neglecting the expression 'n_obs_b!' since those don't depend on the parameters but are extremely expensive to compute
                # Gaussian factor product: looping over the nuissance parameter PDFS
                lf_val = lf_val *(1/(theta_er_sigma*np.sqrt(2*math.pi)) *np.exp(-0.5*((theta_er-1)/theta_er_sigma)**2))
                lf_val = lf_val *(1/(theta_nr_sigma*np.sqrt(2*math.pi)) *np.exp(-0.5*((theta_nr-1)/theta_nr_sigma)**2))
                return np.float64(-1)*lf_val

            # defining the likelihood function
            if flag_verbose : print(f"\t\tdefining the 'log_likelihood_function'")
            def neg_log_likelihood_function(
                i_sigma, # SI WIMP-nucleon cross-section, Note: due to computational reasons 'lambda_wimps' was defined to correspond to a sigma of 1e-45 --> one needs to correcto for that factor later
                i_theta_er,
                i_theta_nr,
            ):
                # initial definitions
                llf_val = np.array(0)
                sigma = np.array(i_sigma)
                theta_er = np.array(i_theta_er)
                theta_nr = np.array(i_theta_nr)
#                print("types:###########")
#                print(f"type('sigma') = '{type(sigma)}'")
#                print(f"type('llf_val') = '{type(llf_val)}'")
#                print(f"type('n_obs_b') = '{type(n_obs_er[2][3])}'")
#                print(f"type('lambda_er') = '{type(lambda_er)}'")
#                print(f"type('pdf_er_b') = '{type(pdf_er[2][3])}'")
                
                # Poisson factor product: looping over all bins of the cS1-cS2 observable space
                for b_row, be_row in enumerate(bin_edges_s2[:-1]):
                    for b_column, be_column in enumerate(bin_edges_s1[:-1]):
                        n_obs_b = 2*n_obs_er[b_row][b_column] +n_obs_nr[b_row][b_column]
#                        print(type(pdf_er[b_row][b_column]*lambda_er))
#                        print(type(theta_er))
#                        print(type(pdf_nr[b_row][b_column]*lambda_nr))
#                        print(type(theta_nr))
#                        print(type(pdf_wimps[b_row][b_column]*lambda_wimps))
#                        print(type(sigma))
                        lambda_b = pdf_er[b_row][b_column]*lambda_er*theta_er +pdf_nr[b_row][b_column]*lambda_nr*theta_nr +pdf_wimps[b_row][b_column]*lambda_wimps*sigma
#                        print(f"sanity check: observed={n_obs_b}, expected={pdf_er[b_row][b_column]*lambda_er+pdf_nr[b_row][b_column]*lambda_nr}")
                        llf_val = llf_val + n_obs_b*np.log(lambda_b) -lambda_b
                # Gaussian factor product: looping over the nuissance parameter PDFS
                llf_val = llf_val -np.log(theta_er_sigma) -0.5*np.log(2*math.pi) -0.5*((theta_er-1)/theta_er_sigma)**2
                llf_val = llf_val -np.log(theta_nr_sigma) -0.5*np.log(2*math.pi) -0.5*((theta_nr-1)/theta_nr_sigma)**2
                return np.float64(-1)*llf_val

            # plotting the neg_log_likelihood function
            x_data = np.geomspace(start=0.000001, stop=100, num=150, endpoint=True)
            y_data = [neg_likelihood_function(x,1,1) for x in x_data]
            print(x_data)
            print(y_data)
            plt.plot(x_data,y_data)
            plt.xscale("log")
            plt.show()

            # determining the maximum likelihood estimators
            if flag_verbose : print(f"\t\tminimizing -1*'log_likelihood_function'")
            mle = minimize(
                fun = lambda x : neg_likelihood_function(x[0],x[1],x[2]),
                x0   = [0.01,0.9,0.91],
                bounds = [[0,10000], [0.00001,5], [0.00001,5]],
                method = None,
            )

            mle_sigma = mle.x[0]
            mle_thetavec = [mle.x[1],mle.x[2]]
            mle_sigma_list.append(mle_sigma)
            mle_thetavec_list.append(mle_thetavec)
            if flag_verbose : print(f"\t\tmaximum likelihood estimators")
            if flag_verbose : print(f"\t\tsigmas: {mle_sigma_list}")
            if flag_verbose : print(f"\t\tthetas: {mle_thetavec_list}")

#        # calculating the upper limit for each of the 'simulation__number_of_upper_limit_simulations_per_wimp_mass'-many background-only datasets
#        upper_limit_list = []
#        for l in range(simulation__number_of_upper_limit_simulations_per_wimp_mass):
#            if flag_verbose : print(f"\tstarting upper limit loop with l={l}/{simulation__number_of_upper_limit_simulations_per_wimp_mass}")


            # determining the upper limit

        # determining the median upper limit along with the 1-sigma and 2-sigma bands


#infer_two_dimensional_pdf_value_for_observation(
#    observation,
#    bin_edges_x,
#    bin_edges_y,
#    pdf,)


    # compiling the output dictionary
    if flag_verbose : print(f"{fn}: filling the output dictionary")
    output_dict = {
#        "input" : {
#            "detector__drift_field_v_cm"   : detector__drift_field_v_cm,
#            "spectrum__er_eroi_kev" : spectrum__er_eroi_kev,
#            "spectrum__nr_eroi_kev" : spectrum__nr_eroi_kev,
#            "spectrum__er_simulation_window_kev" : spectrum__er_simulation_window_kev,
#            "spectrum__nr_simulation_window_kev" : spectrum__nr_simulation_window_kev,
#            "limit_wimp_mass_gev_list" : limit_wimp_mass_gev_list,
#            "flag_verbose" : flag_verbose,
#        },
#        "output" : {},
        "spectrum_components" : spectrum_components_dict,
    }


    # finishing
    ctd = timedelta(seconds=time.time()-start_time)
    if flag_verbose : print(f"{fn}: finished within {ctd} h")
    return output_dict



































