



############################################
### imports
############################################


import subprocess
import numpy as np
import json
import scipy.integrate as integrate
import scipy.constants as constants
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
import wimprates
import os





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
    SE = elYield *detector_dict["g1_gas"]

    # computing 'g2'
    g2 = ExtEff*SE
    return g2





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
            "x,y,z-position[mm]" : "-1 -1 -1",
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
            "x,y,z-position[mm]" : "-1 -1 -1",
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
    "nr_pp"		                                : {
        "latex_label"                           : r"pp",
        "color"                                 : color_pp_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_pep"		                            : {
        "latex_label"                           : r"pep",
        "color"                                 : color_pep_default,
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
    "nr_be7_384"	                            : {
        "latex_label"                           : r"$^{7}\mathrm{Be}\,(384\,\mathrm{keV})$",
        "color"                                 : color_be7_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_be7_861"	                            : {
        "latex_label"                           : r"$^{7}\mathrm{Be}\,(861\,\mathrm{keV})$",
        "color"                                 : color_be7_default,
        "linestyle"                             : "--",
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
    "nr_o15"		                            : {
        "latex_label"                           : r"$^{15}\mathrm{O}$",
        "color"                                 : color_cno_default,
        "linestyle"                             : "-",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_f17"		                            : {
        "latex_label"                           : r"$^{17}\mathrm{F}$",
        "color"                                 : color_cno_default,
        "linestyle"                             : "-.",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
    },
    "nr_n13"		                            : {
        "latex_label"                           : r"$^{13}\mathrm{N}$",
        "color"                                 : color_cno_default,
        "linestyle"                             : "--",
        "linewidth"                             : 1,
        "zorder"                                : 1,
        "differential_rate_computation"         : "interpolation_from_file",
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
        "constituent_spectra_list"              : ["nr_atm", "nr_hep", "nr_atm", "nr_b8", "nr_dsnb"],
    },
    # This is the combined ER background model used for the SFS study
    "combined_er_background"                    : {
        "latex_label"                           : r"combined ER background",
        "color"                                 : color_ers_default,
        "linestyle"                             : "-",
        "linewidth"                             : 2,
        "zorder"                                : 2,
        "differential_rate_computation"         : "spectrum_sum",
        "constituent_spectra_list"              : ["er_be7_384", "er_be7_861", "er_o15", "er_n13", "er_nunubetabeta", "er_pp"],
    },
})


def give_spectrum_dict(
    spectrum_name,
    recoil_energy_kev_list,
    # 
    abspath_spectra_files,
    exposure_t_y = 40*5,
    num_events = -1,
    # nest parameters
    seed = 0,
    drift_field_v_cm = 200,
    xyz_pos_mm = "-1 -1 -1",
    # flags
    flag_spectrum_type = ["differential", "integral"][0],
    flag_verbose = False,
    # keywords
    spectrum_dict_default_values = spectrum_dict_default_dict, # default 'spectrum_dict' values
    **kwargs, # additional keyword argument values overwriting those from 'spectrum_dict_default_values'
):

    """
    This function is used to generate a 'spectrum_dict' based on one of the templates in 'spectrum_dict_default_dict'.
    If 'flag_spectrum_type' == "differential" the 'spectrum_dict' resembles the differential rate (in events per tonne x year x keV, computed for every 'recoil_energy_kev' value).
    If 'flag_spectrum_type' == "integral" the 'spectrum_dict' resembles the integrated number of events per aequidistant recoil energy bin (bin centers are given by the 'recoil_energy_kev' values).
    """

    # initializing
    fn = "give_spectrum_dict"
    if flag_verbose: print(f"{fn}: initializing 'spectrum_dict'")
    if flag_verbose: print(f"\tcopying entry from 'spectrum_dict_default_values'")
    spectrum_dict = spectrum_dict_default_values[spectrum_name].copy()
    if flag_verbose: print(f"\tupdating 'spectrum_dict' with specified keyword arguments")
    spectrum_dict.update({
        "recoil_energy_kev_list"    : list(recoil_energy_kev_list),
        "exposure_t_y"              : exposure_t_y,
        "num_events"                : num_events,
        "seed"                      : seed,
        "field_drift[V/cm]"         : drift_field_v_cm,
        "x,y,z-position[mm]"        : xyz_pos_mm,
        "flag_verbose"              : flag_verbose,
        "flag_spectrum_type"        : flag_spectrum_type,
    })
    for k in [*kwargs]:
        if k == "differential_rate_parameters":
            for kk in [*kwargs["differential_rate_parameters"]]:
                spectrum_dict["differential_rate_parameters"].update({kk:kwargs["differential_rate_parameters"][kk]})
        else:
            spectrum_dict.update({k:kwargs[k]})

    # case: specified spectrum is sum of many constituent profiles ---> recursively compute the 'spectrum_dict' for every constituent dictionary
    if spectrum_dict["differential_rate_computation"] == "spectrum_sum":
        constituent_spectrum_dict_list = []
        # recursively computing the 'spectrum_dict's for all constituent spectra
        for constituent_spectrum_name in spectrum_dict["constituent_spectra_list"]:
            constituent_spectrum_dict_list.append(give_spectrum_dict(
                spectrum_name = constituent_spectrum_name,
                recoil_energy_kev_list = recoil_energy_kev_list,
                abspath_spectra_files = abspath_spectra_files,
                exposure_t_y = exposure_t_y,
                num_events = num_events,
                seed = seed,
                drift_field_v_cm = drift_field_v_cm,
                xyz_pos_mm = xyz_pos_mm,
                flag_spectrum_type = flag_spectrum_type,
                flag_verbose = flag_verbose,
                spectrum_dict_default_values = spectrum_dict_default_values,
                **kwargs,
            ))
        # summing the number of entries of the individual constituent dicts
        #spectrum_dict = constituent_spectrum_dict_list[0].copy()
        if spectrum_dict["flag_spectrum_type"] == "differential":
            y_data_summed = compute_array_sum([csd["differential_recoil_rate_events_t_y_kev"] for csd in constituent_spectrum_dict_list])
            spectrum_dict.update({
                "differential_recoil_rate_events_t_y_kev" : y_data_summed})
        elif spectrum_dict["flag_spectrum_type"] == "integral":
            y_data_summed = compute_array_sum([csd["numEvts"] for csd in constituent_spectrum_dict_list])
            spectrum_dict.update({
                "numEvts"               : list(y_data_summed),
                "type_interaction"      : str(constituent_spectrum_dict_list[0]["type_interaction"]),
                "E_min[keV]"            : list(recoil_energy_kev_list),
                "E_max[keV]"            : list(recoil_energy_kev_list),
            })

        # returning the 'spectrum_dict' with summed entries
        if callable(spectrum_dict["differential_rate_computation"]):
            spectrum_dict.pop("differential_rate_computation")
        return spectrum_dict

    # case: specified spectrum is a single profile ---> infer differential rate and - if specified - integrated rate
    else:

        # inferring the differential rate computation method
        if flag_verbose: print(f"{fn}: assessing differential rate computation method")
        if spectrum_dict["differential_rate_computation"] == "interpolation_from_file":
            digitized_spectrum_ndarray = convert_grabbed_csv_to_ndarray(abspath_spectra_files +spectrum_name +".csv")
            differential_rate_function = np.interp
            differential_rate_param_dict = {"xp" : digitized_spectrum_ndarray["x_data"], "fp" : digitized_spectrum_ndarray["y_data"], "left" : 0, "right" : 0}
        elif callable(spectrum_dict["differential_rate_computation"]):
            differential_rate_function = spectrum_dict["differential_rate_computation"]
            differential_rate_param_dict = spectrum_dict["differential_rate_parameters"]

        # case: computing the differential rate
        if spectrum_dict["flag_spectrum_type"] == "differential":
            if flag_verbose: print(f"{fn}: computing the differential rate")
            differential_recoil_rate_events_t_y_kev = [differential_rate_function(e, **differential_rate_param_dict) for e in recoil_energy_kev_list]
            spectrum_dict.update({
                "recoil_energy_kev_list"                        : list(recoil_energy_kev_list),
                "differential_recoil_rate_events_t_y_kev"       : list(differential_recoil_rate_events_t_y_kev),
            })

        # case: computing the integrated rate
        # code adapted from C. Hock's 'give_spectrum' function
        elif spectrum_dict["flag_spectrum_type"] == "integral":
            # computing the number of events per energy bin via integration
            binwidth_kev = recoil_energy_kev_list[1] -recoil_energy_kev_list[0]
            recoil_energy_kev_bin_edges_list = [bc-0.5*binwidth_kev for bc in recoil_energy_kev_list] +[recoil_energy_kev_list[-1]+0.5*binwidth_kev]
            args_tuple = (differential_rate_param_dict[key] for key in [*differential_rate_param_dict])
            number_of_events_per_energy_bin = [
                integrate.quad(
                    differential_rate_function,
                    bc-0.5*binwidth_kev,
                    bc+0.5*binwidth_kev,
                    args = tuple([differential_rate_param_dict[key] for key in [*differential_rate_param_dict]])
                )[0] for bc in recoil_energy_kev_list]
            # scaling the integrated number of events either according to 'exposure_t_y' or to 'num_events'
            if num_events <= 0:
                number_of_events_per_energy_bin = [noe*exposure_t_y for noe in number_of_events_per_energy_bin]
            else:
                total = np.sum(number_of_events_per_energy_bin)
                num_scale_factor = num_events/total
                number_of_events_per_energy_bin = [num_scale_factor*noe for noe in number_of_events_per_energy_bin]
            # rounding the entries of 'number_of_events_per_energy_bin' to integer values:
            number_of_events_per_energy_bin = [int(noe) for noe in number_of_events_per_energy_bin]
            # updating the 'spectrum_dict'
            if flag_verbose: print(f"{fn}: computing the integrated rate")
            spectrum_dict.update({
                "numEvts"               : list(number_of_events_per_energy_bin),
                "type_interaction"      : list(spectrum_name.split("_"))[0].upper(),
                "E_min[keV]"            : list(recoil_energy_kev_list),
                "E_max[keV]"            : list(recoil_energy_kev_list),
                "field_drift[V/cm]"     : str(drift_field_v_cm),
                "x,y,z-position[mm]"    : str(xyz_pos_mm),
                "seed"                  : str(seed),})

    # returning the 'spectrum_dict'
    if flag_verbose: print(f"{fn}: finished compiling the 'spectrum_dict'")
    if callable(spectrum_dict["differential_rate_computation"]):
        spectrum_dict.pop("differential_rate_computation")
    return spectrum_dict


def gen_spectrum_plot(
    spectra_list, # list of 'spectra_dict' keys, e.g., ["nr_wimps", "nr_atm", "nr_dsnb"]
    abspath_spectra_files,
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
            x_data_size = 301
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
                abspath_spectra_files = abspath_spectra_files,)
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
    baseline_detector_dict, # dict, 'detector_dict' of the DARWIN baseline detector
    detector_dict = {}, # dict or abspath-string, dictionary or .json-file resembling the detector the spectrum is supposed to be simulated in
    detector_name = "", # string, name of the detector, only required when not referring to an existing file
    abspath_list_detector_dict_json_output = [], # list of strings, list of abspaths into which the detector_dict.json files are saved
    abspathfile_execNEST_binary = abspathfile_nest_installation_execNEST_bin, # string, abspathfile of the 'execNEST' executiable generated in the 'install' NEST folder
    flag_verbose = False, # bool, flag indicating whether the print-statements are being printed
    flag_print_stdout_and_stderr = False, # bool, flag indicating whether the 'stdout' and 'stderr' values returned by executing the 'execNEST' C++ executable are being printed
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
    if detector_dict == {}:
        if flag_verbose: print(f"{fn}: no detector specified --> running with the pre-installed detector")
        pass
    else:
        if type(detector_dict)==str:
            if detector_dict.endswith(".hh"):
                detector_name = list(detector_dict[:-3].split("/"))[-1]
                if flag_verbose: print(f"{fn}: specified detector '{detector_name}' as .hh-file: {detector_dict}")
                new_detector_hh_abspathfile = detector_dict
            elif detector_dict.endswith(".json"):
                detector_name = list(detector_dict[:-5].split("/"))[-1]
                if flag_verbose: print(f"{fn}: specified detector '{detector_name}' as .json-file: {detector_dict}")
                if flag_verbose: print(f"{fn}: updating baseline detector: {abspathfile_baseline_detector_json}")
                new_detector_dict = baseline_detector_dict.copy()
                new_detector_dict.update(get_dict_from_json(detector_dict))
                convert_detector_dict_into_detector_header()
        elif type(detector_dict)==dict:
            if detector_name=="" : raise Exception(f"ERROR: You did not specify a detector name!")
            if flag_verbose: print(f"{fn}: specified detector '{detector_name}'as dictionary: {detector_dict}")
            if flag_verbose: print(f"{fn}: updating baseline detector")
            new_detector_dict = baseline_detector_dict.copy()
            new_detector_dict.update(detector_dict)
            print(new_detector_dict)
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
        default_xyz = "-1 -1 -1"
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
        if flag_verbose: print(f"{fn}: writing the NEST output into an ndarray")
        this_nest_run_tuple_list = []
        for line in stdout.split("\n"):
            line_list = list(line.split("\t"))
            if len(line_list) == 12 and "Nph" not in line:
                execNEST_output_tuple = tuple(line_list)
                this_nest_run_tuple_list.append(execNEST_output_tuple)
            else:
                continue
        execNEST_output_tuple_list += this_nest_run_tuple_list
        #num = int(spectrum_dict["numEvts"][k]) if hasattr(spectrum_dict["numEvts"], "__len__") else int(spectrum_dict["numEvts"])
        #num = cmd_string
        #e_min = str(spectrum_dict["E_min[keV]"][k]) if hasattr(spectrum_dict["E_min[keV]"], "__len__") else int(spectrum_dict["E_min[keV]"])
        #e_max = str(spectrum_dict["E_max[keV]"][k]) if hasattr(spectrum_dict["E_max[keV]"], "__len__") else int(spectrum_dict["E_max[keV]"])
        #if len(this_nest_run_tuple_list) != num: raise Exception(f"This NEST run yielded {len(this_nest_run_tuple_list)} events instead of the specified {num} at E_min={e_min} and E_max={max}.")

    ### casting the 'execNEST_output_tuple_list' into a ndarray
    if flag_verbose: print(f"{fn}: casting 'execNEST_output_tuple_list' into numpy ndarray")
    execNEST_output_ndarray = np.array(execNEST_output_tuple_list, execNEST_dtype)

    return execNEST_output_ndarray


#def gen_signature_plot(
#    signature_list, # list of signature ndarrays to be plotted onto the canvas
#    abspath_spectra_files,
#    # plot parameters
#    plot_fontsize_axis_label = 11,
#    plot_figure_size_x_inch = 5.670,
#    plot_aspect_ratio = 9/16,
#    plot_log_y_axis = False,
#    plot_log_x_axis = False,
#    plot_xlim = [],
#    plot_ylim = [],
#    plot_axes_units = ["cs2_over_cs1_vs_cs1_over_g1"][0],
#    plot_legend = True,
#    plot_legend_bbox_to_anchor = [0.45, 0.63, 0.25, 0.25],
#    plot_legend_labelspacing = 0.5,
#    plot_legend_fontsize = 9,
#    # flags
#    flag_output_abspath_list = [],
#    flag_output_filename = "signature_plot.png",
#    flag_profile = ["default"][0],
#    flag_verbose = False,
#):

#    """
#    This function is used to generate a plot displaying all signatures specified in 'signature_list'.
#    """

#    # initialization
#    fn = "gen_signature_plot"
#    if flag_verbose: print(f"{fn}: initializing")

#    # setting up the canvas
#    if flag_verbose: print(f"{fn}: setting up canvas and axes")
#    fig = plt.figure(
#        figsize = [plot_figure_size_x_inch, plot_figure_size_x_inch*plot_aspect_ratio],
#        dpi = 150,
#        constrained_layout = True) 

#    # axes
#    ax1 = fig.add_subplot()
##    if plot_log_y_axis: ax1.set_yscale('log')
##    if plot_log_x_axis: ax1.set_xscale('log')

#    # looping over all specified spectra
#    for signature in signature_list:


#        # plotting the current signature
##        ax1.plot(
##            plot_x_data,
##            plot_y_data,
##            label = spectrum_dict["latex_label"],
##            linestyle = spectrum_dict["linestyle"],
##            linewidth = spectrum_dict["linewidth"],
##            zorder = spectrum_dict["zorder"],
##            color = spectrum_dict["color"],)

#    # shading the WIMP EROI
##    if flag_shade_wimp_eroi != []:
##        ax1.axvspan(
##            flag_shade_wimp_eroi[0],
##            flag_shade_wimp_eroi[1],
##            alpha = 0.2,
##            linewidth = 0,
##            color = "grey",
##            zorder = -1)

#    # legend
#    if plot_legend : ax1.legend(
#        loc = "center",
#        labelspacing = plot_legend_labelspacing,
#        fontsize = plot_legend_fontsize,
#        bbox_to_anchor = plot_legend_bbox_to_anchor,
#        bbox_transform = ax1.transAxes,)

#    # saving
#    plt.show()
#    for abspath in flag_output_abspath_list:
#        fig.savefig(abspath +flag_output_filename)







############################################
### ER/NR discrimination
############################################


def calc_er_nr_discrimination_line(
    er_spectrum,
    nr_spectrum,

    g1, #photoelectrons per photon
    g2, # photons per electron
    w, #eV
    min_energy, #in keV
    max_energy, # in keV
    bin_size, #in keV, not exceeding max_energy
    nr_acceptance = 0.30, # portion of the nr-spectrum below the discrimination line
    approx_depth = 10,# only integers allowed
    # upper bound for errors in the discrimination line is
    # ~ 2**(-approx_depth-1) * (max(nr_spectrum[S2]/nr_spectrum[S1]) - min(nr_spectrum[S2]/nr_spectrum[S1]))
    verbose =True,
    ):
    #generate energy bins of size energy_bin_size and an array of the bin edges (energy_bin_edges).
    #Bins are centerd around the entries in the energy_bins array.

    if verbose: print("calc_er_nr_discrimination_line running.")
    energy_bins = [min_energy + bin_size/2]
    bin_edges = [min_energy]
    while energy_bins[-1]+bin_size*3/2<=max_energy:
        energy_bins.append(energy_bins[-1]+bin_size)
        bin_edges.append(bin_edges[-1]+bin_size)
    bin_edges.append(bin_edges[-1]+bin_size)


    dtype = np.dtype([("S2/S1", np.float64)]) #dtype for nr_energies and er_energies numpy arrays

    total_er_remaining = 0
    total_er = 0
    total_nr_below_discr_line = 0
    total_nr = 0

    #x-data and y-data for the discrimination line
    dl_x_data=[] # cS1/g1 [ph]
    dl_y_data=[] # cS1/cS2

    #array contain reconstructed energies for ER- and NR-events (but only accurate for ER)
    energy_er=w*(er_spectrum_corr["S1_3Dcor [phd]"]/g1 +er_spectrum_corr["S2_3Dcorr [phd]"]/g2)/1000

    energy_nr=w*(nr_spectrum_corr["S1_3Dcor [phd]"]/g1 +nr_spectrum_corr["S2_3Dcorr [phd]"]/g2)/1000

    #plt.hist(energy_er)
    #plt.hist(energy_nr)

    nr_below_dl=np.array([])
    nr_below_dl.dtype = nr_spectrum.dtype
    for bin_index in range(len(energy_bins)):

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

        if len(nr_y)==0 or len(er_y)==0:
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

        dl_y_data.append(guess)
        dl_y_data.append(guess)

    er_rejection = total_er_remaining/total_er
    if verbose:
        print("Total number of ER-events:",total_er)
        print("Total number of NR-events:",total_nr)
        print("Input NR-acceptance:", nr_acceptance)
        print("Actual NR-acceptance:", total_nr_below_discr_line/total_nr)
        print("Ratio of ER-Events left below the discrimination line:",er_discrimination)

    output_dict = {
        "dl_x_data_s1_over_g1": dl_x_data,
        "dl_y_data_s2_over_s1": dl_y_data,
        "nr_acceptance":nr_acceptance,
        "er_rejection": er_rejection,
        "nr_below_dl": nr_below_dl,
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


############################################
### likelihood stuff
############################################











