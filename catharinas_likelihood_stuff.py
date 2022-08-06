############ IMPORTS ################
import sys
sys.path.insert(0, '/home/catharina/Desktop/praktikum_freiburg_2022/sfs/sfsv2')

import sfs
import scipy.integrate as integrate
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.optimize import minimize
import random
import copy
import wimprates
from scipy.stats import poisson #looks *fishy* (Parce que c'est un poisson, hahaha)
import time
# import more_time
# import happiness
# from ALDI import snickers


################## DARWIN BASELINE DETECTOR DEFINITION #######################
darwin_baseline_detector_dict = {
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
#    "inGas"             : "false",                  # (duh),                                                                               JN: "false", LUX_Run03: commented out, XENON10: "false"
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

############## PATHS ################
# should be changed
abspath_sfs_repo = "/home/catharina/Desktop/praktikum_freiburg_2022/sfs/sfsv2/"
abspath_study = "/home/catharina/Desktop/praktikum_freiburg_2022/study/"
abspath_detectors = abspath_study +"detectors/"
abspath_spectra = abspath_study +"spectra/"
abspath_resources = abspath_sfs_repo +"resources/"
abspath_list_plots = [abspath_study +"plots/"]


############ Parameters #############
darwin_default_drift_field_v_cm = 200
exposure_t_y = 40*5
wimp_eroi_kev_ee = [1.4, 10.6]
wimp_eroi_kev_nr = [4.9, 40.9]
recoil_energy_simulation_window_er = [0, wimp_eroi_kev_ee[1]*1.5]
recoil_energy_simulation_window_nr = [0, wimp_eroi_kev_nr[1]*1.5]
simulation_energy_bins = 50

g1 = darwin_baseline_detector_dict["g1"]
g2 = sfs.compute_g2_from_detector_configuration(darwin_baseline_detector_dict)
w=13.6 #eV

S1_bounds = [1, 150] #Somewhat arbitrary, change if you have a better idea
S2_bounds = [1, 65000] #Somewhat arbitrary
num_S1_bins = num_S2_bins = 50

######################################
def reduce_NEST_data_to_eroi(
    sim_ndarray, #ndarry, outpur from execNEST
    eroi, # array containing start and end of the energy region of interest in keV
    detector_dict = darwin_baseline_detector_dict
    ):

    g1 = detector_dict["g1"]
    g2 = sfs.compute_g2_from_detector_configuration(detector_dict)
    w=13.6 #eV

    sim_ndarray= sim_ndarray[(wimp_eroi_kev_ee[0]<w/1000*(sim_ndarray["S1_3Dcor [phd]"]/g1 + sim_ndarray["S2_3Dcorr [phd]"]/g2))&
                                         (w/1000*(sim_ndarray["S1_3Dcor [phd]"]/g1 + sim_ndarray["S2_3Dcorr [phd]"]/g2)<wimp_eroi_kev_ee[1])&
                            (sim_ndarray["S1_3Dcor [phd]"]>0) & (sim_ndarray["S2_3Dcorr [phd]"]>0)]

    return sim_ndarray

def give_mu(
    energy_simulation_window,
    simulation_energy_bins=200,
    itype = ["er", "nr", "wimp"][0], #interaction type / particle
    mw = 25, #WIMP mass, only required for WIMPs in GeV
    sigma = 1e-47, #WIMP cross-section, only required for WIMPs
    exposure_t_y = 5e1, # just an arbitrary value. Chosen to be big to estimate the expected number of events.
    verbose =True,
    detector_dict =darwin_baseline_detector_dict,
    ):

    if verbose: print("give mu running")
    g1 = detector_dict["g1"]
    g2 = sfs.compute_g2_from_detector_configuration(detector_dict)
    w=13.6 #eV
    mu_list =[]
    default_dict = sfs.spectrum_dict_default_dict
    exposure_t_y_saved = copy.deepcopy(exposure_t_y)

    if itype =="er":
        spectrum_name ="combined_er_background"
    if itype =="nr":
        spectrum_name ="combined_nr_background"

    if isinstance(sigma, float) or isinstance(sigma, int):
        if verbose: print("Only one sigma entered")
        sigma = [sigma]
    for s in sigma:
        if verbose: print(f"simulating sigma {s}")
        if itype=="wimp":
            nr_wimps_wimprates_custom= {
            "latex_label"                           : r"WIMPs",
            "color"                                 : sfs.color_wimps_default,
            "linestyle"                             : "-",
            "linewidth"                             : 2,
            "zorder"                                : 2,
            "differential_rate_computation"         : wimprates.rate_wimp_std,
            "differential_rate_parameters"          : {
                "mw"                                : mw,
                "sigma_nucleon"                     : s,
            }}

            spectrum_name ="nr_wimps_wimprates_custom"
            default_dict[spectrum_name]=nr_wimps_wimprates_custom
            # scale exposure according to sigma for better accuracy by also avoiding too much computation.
            exposure_t_y = np.power(10, -np.log10(s)-46)*exposure_t_y_saved

        # generate energy bins based on interaction type and spectrum
        spectrum_dict = sfs.give_spectrum_dict(
            spectrum_name = spectrum_name,
            recoil_energy_kev_list = sfs.bin_centers_from_interval(energy_simulation_window, simulation_energy_bins),
            #
            abspath_spectra_files = abspath_resources,
            exposure_t_y = exposure_t_y,
            num_events = -1,
            # nest parameters
            seed = 0,
            drift_field_v_cm = darwin_default_drift_field_v_cm,
            xyz_pos_mm = "-1 -1 -1",
            # flags
            flag_spectrum_type = ["differential", "integral"][1],
            flag_verbose = False,
            # keywords
            spectrum_dict_default_values = default_dict, # default 'spectrum_dict' values
        )

        if verbose: print("Number of events:",sum(spectrum_dict["numEvts"]))
        if sum(spectrum_dict["numEvts"]) == 0:
            print("WARNING: 0 events simulated.")
            mu_list.append(0)
            continue
        if verbose: print("simulating events with NEST")
        #simulate events using spectrum_dict with NEST
        sim_ndarray = sfs.execNEST(
            spectrum_dict = spectrum_dict,
            baseline_detector_dict = detector_dict,
            detector_dict = {}, # simply use baseline detector
            detector_name = "random_test_detector",
            flag_verbose = False,
            flag_print_stdout_and_stderr = False,
        )


        if verbose: print("number of events simulated: ",sum(np.array(spectrum_dict["numEvts"])))
        if verbose: print("removing events outside of energy region of interest")

        sim_ndarray = reduce_NEST_data_to_eroi(sim_ndarray, wimp_eroi_kev_ee, detector_dict)
        mu = len(sim_ndarray)/exposure_t_y
        mu_list.append(mu)
        if verbose: print("mu: ",mu)

    if verbose: print("give_mu finshed.")
    output_dict = {
        "expected number of events/t/y": mu_list,
        "interaction type/ particle": itype,
    }
    return output_dict

def generate_pdf_from_spectrum(
    energy_simulation_window,
    simulation_energy_bins,
    itype = ["er", "nr", "wimp"][0], #interaction type / particle
    mw = 25, #WIMP mass, only required for WIMPs
    sigma = 1e-47, #WIMP cross-section, only required for WIMPs
    num_S1_bins = num_S1_bins, #[phd] default is arbitrary.
    num_S2_bins = num_S2_bins, #[phd] default is arbitrary
    S1_bounds =S1_bounds,
    S2_bounds = S2_bounds,
    exposure_t_y = 20, # just an arbitrary value. Chosen to be big to estimate the expected number of events.
    num_events = 1e5, # real number of simulated events is off by a factor of 5-6. Reason unknown. Daniel will propably fix it. (To Do)
    bin_flag = ["bins", "events"][1], # iterate over bin_flag for binning. "events" is faster for nr-events.
    # For a test run under similiar conditions:
    # bin_flag:bins, time required: 206s for er, 90s for nr, 26s for wimps
    # bin_flag:events 88s for er, 23s for nr, 7s for wimps
    verbose =False,
    detector_dict =darwin_baseline_detector_dict,
):
    if verbose: print("generate pdf from spectrum running")
    if verbose: print("interaction type/particle: ", itype)

    g1 = detector_dict["g1"]
    g2 = sfs.compute_g2_from_detector_configuration(detector_dict)
    w=13.6 #eV

    default_dict = sfs.spectrum_dict_default_dict
    if itype =="er":
        spectrum_name ="combined_er_background"
    elif itype =="nr":
        spectrum_name ="combined_nr_background"
    else:
        nr_wimps_wimprates_custom= {
        "latex_label"                           : r"WIMPs",
        "color"                                 : sfs.color_wimps_default,
        "linestyle"                             : "-",
        "linewidth"                             : 2,
        "zorder"                                : 2,
        "differential_rate_computation"         : wimprates.rate_wimp_std,
        "differential_rate_parameters"          : {
            "mw"                                : mw,
            "sigma_nucleon"                     : sigma,
        }}

        spectrum_name ="nr_wimps_wimprates_custom"
        default_dict[spectrum_name]=nr_wimps_wimprates_custom

    # generate energy bins based on interaction type and spectrum
    spectrum_dict = sfs.give_spectrum_dict(
        spectrum_name = spectrum_name,
        recoil_energy_kev_list = sfs.bin_centers_from_interval(energy_simulation_window, simulation_energy_bins),
        #
        abspath_spectra_files = abspath_resources,
        exposure_t_y = 1,
        num_events = num_events,
        # nest parameters
        seed = 0,
        drift_field_v_cm = darwin_default_drift_field_v_cm,
        xyz_pos_mm = "-1 -1 -1",
        # flags
        flag_spectrum_type = ["differential", "integral"][1],
        flag_verbose = False,
        # keywords
        spectrum_dict_default_values = default_dict, # default 'spectrum_dict' values
        #**kwargs, # additional keyword argument values overwriting those from 'spectrum_dict_default_values'
    )

    if verbose: print("simulating",sum(spectrum_dict["numEvts"]), "events with NEST")
    #simulate events using spectrum_dict with NEST
    sim_ndarray = sfs.execNEST(
        spectrum_dict = spectrum_dict,
        baseline_detector_dict = detector_dict,
        detector_dict = {}, # simply use baseline detector
        detector_name = "random_test_detector",
        flag_verbose = False,
        flag_print_stdout_and_stderr = False,
    )

    minS1, maxS1 = S1_bounds
    minS2, maxS2 = S2_bounds

    if verbose: print("number of events simulated: ",sum(np.array(spectrum_dict["numEvts"])))

    S1_bins = np.linspace(minS1, maxS1, num_S1_bins+1)
    S2_bins = np.linspace(minS2, maxS2, num_S2_bins+1)
    pdf_grid=[]


    sim_ndarray = reduce_NEST_data_to_eroi(sim_ndarray, wimp_eroi_kev_ee, detector_dict)
    if verbose: print("calculating pdf by binning simulated data")
    if bin_flag =="bins":
        print("You are using the bin_flag 'bins'. This can take considerably longer than using the bin_flag 'events'.Proceed with caution.")

        for s1_i in range(num_S1_bins):
            pdf_grid.append([])
            for s2_i in range(num_S2_bins):

                bin_data = sim_ndarray[(S1_bins[s1_i]<sim_ndarray["S1_3Dcor [phd]"])&
                                       (sim_ndarray["S1_3Dcor [phd]"]<S1_bins[s1_i+1])&
                                        (S2_bins[s2_i]<sim_ndarray["S2_3Dcorr [phd]"])&
                                       (sim_ndarray["S2_3Dcorr [phd]"]<S2_bins[s2_i+1])]
                                        #E = w(S1/g1 +S2/g2) in eV, not in keV
                pdf_grid[-1].append(len(bin_data["S1_3Dcor [phd]"]))
    else:
        for s1_i in range(num_S1_bins):
            pdf_grid.append(np.zeros(num_S2_bins))

        for event in sim_ndarray:
            s1_i = int(np.interp(event["S1_3Dcor [phd]"], S1_bins, range(len(S1_bins))))
            s2_i = int(np.interp(event["S2_3Dcorr [phd]"], S2_bins, range(len(S2_bins))))
            if s1_i <0 or s1_i >=len(S1_bins):
                continue
            if s2_i<0 or s2_i >= len(S2_bins):
                continue
            pdf_grid[s1_i][s2_i]+=1

    #if verbose: print("calculating mu")
    #mu = give_mu(energy_simulation_window,simulation_energy_bins,itype=itype,
    #mw=mw,sigma=sigma,exposure_t_y = exposure_t_y,verbose =verbose,detector_dict =detector_dict)
    #normalization, but still needs checking. (To Do)
    pdf_grid = np.array(pdf_grid)/len(sim_ndarray["S1_3Dcor [phd]"])
    output_dict = {
        "S1_bins": S1_bins,
        "S2_bins": S2_bins,
        "pdf": pdf_grid,
        "NEST output":sim_ndarray,
        "interaction type/particle": itype,
        #"expected number of events/t/y":mu["expected number of events/t/y"],

    }
    if itype =="wimp":
        output_dict["mw"] = mw
        output_dict["sigma"]= sigma

    if verbose: print("generate pdf from spectrum finshed.")
    return output_dict

def log_Likelihood(
    # returns the log of the Likelihood function.
    data_ndarray, # data simulated by NEST
    er_pdf_dict, #output of generate_pdf_from_spectrum
    nr_pdf_dict,
    wimp_pdf_dict,
    mu_er,
    mu_nr,
    mu_wimp,
    exposure_t_y = exposure_t_y,
    mw = 25, #WIMP mass, only required for WIMPs
    sigma = 1e-47, #WIMP cross-section, only required for WIMPs
    eroi = wimp_eroi_kev_ee, # Energy region of interest in keVee
    flag = ["individual events", "bins"][1],
    detector_dict = darwin_baseline_detector_dict,
    verbose=True,
    ):
    num_events = len(data_ndarray["S1_3Dcor [phd]"])

    mu_wimp = mu_wimp * exposure_t_y
    mu_er = mu_er*exposure_t_y
    mu_nr = mu_nr*exposure_t_y
    mu_tot = mu_wimp+mu_er+mu_nr

    if verbose: print(f"Events expected: {mu_tot}, Events observed: {num_events}")
    L=np.log(poisson.pmf(num_events, mu_tot))
    #L=np.log(poisson.pmf(num_events, mu_tot))
    if verbose: print("Calculating log_Likelihood for sigma: ", sigma)
    if verbose: print("Poisson: ",poisson.pmf(num_events, mu_tot))

    #assert not binning or wimp_pdf_dict["S1_bins"] == er_pdf_dict["S1_bins"] == nr_pdf_dict["S1_bins"]
    #assert not binning or wimp_pdf_dict["S2_bins"] == er_pdf_dict["S2_bins"] == nr_pdf_dict["S2_bins"]
    if abs(wimp_pdf_dict["mw"]-mw)>0.05:
        if verbose: print("entered wimp_pdf_dict has different mw compared to enetered parameters.")
        if verbose: print(f"Calculating new wimp_pdf_dict based on input parameters (sigma:{sigma}, mw:{mw}) . This can take a moment.")
        wimp_pdf_dict = generate_pdf_from_spectrum( eroi, simulation_energy_bins,  itype="wimp")

    if flag=="individual events":
        assert False, "flag 'individual events' is still under construction. Sorry."
        return None

    else:
        S1_bins = wimp_pdf_dict["S1_bins"]
        S2_bins = wimp_pdf_dict["S2_bins"]
        data_ndarray = reduce_NEST_data_to_eroi(data_ndarray, eroi, detector_dict = detector_dict)

        for event in data_ndarray:
            s1_i = int(np.interp(event["S1_3Dcor [phd]"], S1_bins, range(len(S1_bins))))kj
            s2_i = int(np.interp(event["S2_3Dcorr [phd]"], S2_bins, range(len(S2_bins))))
            if s1_i <0 or s1_i >=len(S1_bins):
                continue
            if s2_i<0 or s2_i >= len(S2_bins):
                continue

            # mus dependent on spacing in s1-s2-space?
            sum_over_interaction_types=mu_er*er_pdf_dict["pdf"][s1_i][s2_i]
            sum_over_interaction_types+=mu_nr*nr_pdf_dict["pdf"][s1_i][s2_i]
            sum_over_interaction_types+=mu_wimp*wimp_pdf_dict["pdf"][s1_i][s2_i]
            sum_over_interaction_types /= mu_tot
            if sum_over_interaction_types == 0:
                print("sum_over_interaction_types =0!")
                print(event)
            L+= np.log(sum_over_interaction_types)
    return L

def t( # aka q(sigma)
    sigma, # cross-section
    mw, # WIMP mass
    data_ndarray, # data simulated by NEST
    er_pdf_dict, #output of generate_pdf_from_spectrum
    nr_pdf_dict,
    wimp_pdf_dict,
    mu_er,
    mu_nr,
    mu_wimp_dict,
    binning = True,
    eroi = wimp_eroi_kev_ee, # Energy region of interest in keVee
    # binning =True -> L will be calculated based on bins. Events in the same bin are treated together
    # binning= False -> every event is treated seperately,
    detector_dict = darwin_baseline_detector_dict,
     ):

    def concise_log_Likelihood(sigma):
        mu_wimp = np.interp(sigma,mu_wimp_dict["sigma_list"], mu_wimp_dict["mu_wimp_list"])
        return -log_Likelihood(data_ndarray, er_pdf_dict, nr_pdf_dict,wimp_pdf_dict, mu_er = mu_er, mu_nr = mu_nr,
                              mu_wimp = mu_wimp,
                mw= mw,sigma = sigma,binning = binning,eroi = eroi, detector_dict = detector_dict) # - because scipy minimizes

    bounds = [(1e-51,1e-45)] # arbitrary bounds. (To Do: check)
    initial_guess = [sigma] # also an arbitraty choice.
    global_optimum_dict = minimize(concise_log_Likelihood, initial_guess, bounds=bounds)
    L_sigma = concise_log_Likelihood(sigma)

    return global_optimum_dict


############# STUFF TO TEST log_Likelihood ####################
"""
##### Calculating mu_wimp, mu_er, mu_nr, pdf_wimp_dict, pdf_nr_dict, pdf_er_dict for log_Likelihood
# Calculate with constant simulation_energy_bins!!

sigmas = np.logspace(-51, -45, num=5, base = 10)
print(sigmas)
simulation_energy_bins = 200
mw = 25 # GeV

mu_wimp_dict = {
    "sigma_list":sigmas,
    "mu_wimp_list": give_mu(wimp_eroi_kev_nr,
        simulation_energy_bins,
        itype = "wimp",
        mw = mw,
        sigma = sigmas,
        exposure_t_y = 1e3,
        verbose =True,
        detector_dict =darwin_baseline_detector_dict)["expected number of events/t/y"]
}

mu_er = give_mu(recoil_energy_simulation_window_er,
        simulation_energy_bins,
        itype = "er",
        exposure_t_y = 1e3,
        verbose =True,
        detector_dict =darwin_baseline_detector_dict)['expected number of events/t/y'][0]

mu_nr = give_mu(recoil_energy_simulation_window_nr,
        simulation_energy_bins,
        itype = "nr",
        exposure_t_y = 1e3,
        verbose =True,
        detector_dict =darwin_baseline_detector_dict)['expected number of events/t/y'][0]


wimp_pdf_dict = generate_pdf_from_spectrum( wimp_eroi_kev_nr, simulation_energy_bins,  itype="wimp", verbose=True)
er_pdf_dict = generate_pdf_from_spectrum( recoil_energy_simulation_window_er, simulation_energy_bins, itype="er", verbose=True)
nr_pdf_dict = generate_pdf_from_spectrum( recoil_energy_simulation_window_nr, simulation_energy_bins, itype="nr", verbose=True)


####### PLOTTTING THE PDFs
x = [0,10,20,30,40]
x_values = [wimp_pdf_dict["S2_bins"][xx] for xx in x]

y = [0,10,20,30,40]
y_values = [wimp_pdf_dict["S1_bins"][yy] for yy in y]

plt.figure(figsize=(22, 7))
plt.subplot(1,3,1)
plt.xticks(x,x_values)
plt.yticks(y, y_values)
plt.title("WIMP")
plt.xlabel("cS2 [phd]")
plt.ylabel("cS1 [phd]")
plt.imshow(wimp_pdf_dict["pdf"], origin="lower", cmap="gray")
plt.colorbar()

plt.subplot(1,3,2)
plt.xticks(x,x_values)
plt.yticks(y, y_values)
plt.title("ER")
plt.xlabel("cS2 [phd]")
plt.ylabel("cS1 [phd]")
plt.imshow(er_pdf_dict["pdf"], origin="lower",cmap="gray")
plt.colorbar()

plt.subplot(1,3,3)
plt.xticks(x,x_values)
plt.yticks(y, y_values)
plt.title("NR")
plt.xlabel("cS2 [phd]")
plt.ylabel("cS1 [phd]")
plt.imshow(nr_pdf_dict["pdf"], origin="lower", cmap="gray")
plt.colorbar()

###### Creating Test-Data to plug into log_Likelihood() #######
spectra_names_array= ["combined_er_background", "combined_nr_background","nr_wimps_wimprates_custom"]
eroi_array = [recoil_energy_simulation_window_er,recoil_energy_simulation_window_nr,wimp_eroi_kev_ee]
detector_dict = darwin_baseline_detector_dict
exposure_t_y = (40*5)*10
sigma = 1e-48
mw = 25 #GeV

default_dict = sfs.spectrum_dict_default_dict
simulation_energy_bins=200
assert len(spectra_names_array) == len(eroi_array), "number of erois not equal to number of spectra names."
if "nr_wimps_wimprates_custom" in spectra_names_array:
    nr_wimps_wimprates_custom= {
    "latex_label"                           : r"WIMPs",
    "color"                                 : sfs.color_wimps_default,
    "linestyle"                             : "-",
    "linewidth"                             : 2,
    "zorder"                                : 2,
    "differential_rate_computation"         : wimprates.rate_wimp_std,
    "differential_rate_parameters"          : {
        "mw"                                : mw,
        "sigma_nucleon"                     : sigma,
    }}
    if sigma>1e-52:
        # only simulate wimp events if there is at least one event. Otherwise, NEST is angry
        default_dict["nr_wimps_wimprates_custom"]=nr_wimps_wimprates_custom
    print(sigma)

pre_output = []
for i in range(len(spectra_names_array)):
    #generating test data
    spectrum_dict = sfs.give_spectrum_dict(
        spectrum_name = spectra_names_array[i],
        recoil_energy_kev_list = sfs.bin_centers_from_interval(eroi_array[i], simulation_energy_bins),
        #
        abspath_spectra_files = abspath_resources,
        exposure_t_y = exposure_t_y,
        num_events = -1,
        # nest parameters
        seed = 0,
        drift_field_v_cm = darwin_default_drift_field_v_cm,
        xyz_pos_mm = "-1 -1 -1",
        # flags
        flag_spectrum_type = ["differential", "integral"][1],
        flag_verbose = False,
        # keywords
        spectrum_dict_default_values = default_dict,
    )
    if sum(spectrum_dict["numEvts"])==0:
        print("0 events")
        continue

    total_num_events = sum(spectrum_dict["numEvts"])
    print(f"Simulating {total_num_events} {spectra_names_array[i]}-events")
    sim_ndarray = sfs.execNEST(
        spectrum_dict = spectrum_dict,
        baseline_detector_dict = detector_dict,
        detector_dict = {}, # simply use baseline detector
        detector_name = "random_test_detector",
        flag_verbose = False,
        flag_print_stdout_and_stderr = False,
    )
    sim_ndarray = reduce_NEST_data_to_eroi(sim_ndarray, wimp_eroi_kev_ee,detector_dict = detector_dict)
    pre_output.append(sim_ndarray)


test_data = []
for pre in pre_output:
    for p in pre:
        test_data.append(p)


test_data = np.array(test_data)
test_data.dtype = pre.dtype


########## PLOTTING test_data ###################

plt.figure(figsize=(10,10))
plt.xlim([0,800])
plt.yscale("log")
plt.xlabel("cS1/g1 [phd]")
plt.ylabel("cS2/cS1")
plt.scatter(test_data["S1_3Dcor [phd]"]/g1, test_data["S2_3Dcorr [phd]"]/test_data["S1_3Dcor [phd]"],
            alpha=0.1, s=1)
print(exposure_t_y)
print(default_dict["nr_wimps_wimprates_custom"])


####### TESTING log_Likelihood #######
sigmas = np.logspace(-51, -45, num=20, base = 10)
log_likelihood=[]
for s in sigmas:
    mu_wimp = np.interp(s,mu_wimp_dict["sigma_list"], mu_wimp_dict["mu_wimp_list"])
    print(f"mu_wimp: {mu_wimp}")
    L = log_Likelihood(test_data,  er_pdf_dict, nr_pdf_dict, wimp_pdf_dict,
        mu_er=mu_er, mu_nr=mu_nr, mu_wimp = mu_wimp, exposure_t_y=exposure_t_y,mw = mw, sigma = s,
        binning = True,eroi = wimp_eroi_kev_ee,
        detector_dict = detector_dict,verbose=True,
        )
    print(f"Likelihood calculated as {L}")
    log_likelihood.append(L)

plt.xscale("log")
plt.xlabel("cross section")
plt.ylabel("log(L(sigma))")
plt.scatter(sigmas, log_likelihood)
print(log_likelihood)

"""

