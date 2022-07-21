



############################################
### imports
############################################


import subprocess
import numpy as np
import json
import scipy.integrate as integrate





############################################
### general definitions
############################################


# paths and files: NEST installation
abspath_nest_installation = "/home/daniel/Desktop/arbeitsstuff/sfs/nest_v_2_3_9/"
abspath_nest_installation_install = abspath_nest_installation +"install/"
abspath_nest_installation_build = abspath_nest_installation +"build/"
abspath_nest_installation_nest = abspath_nest_installation +"nest/"
abspath_nest_installation_nest_include_detectors = abspath_nest_installation_nest +"include/Detectors/"
abspathfile_nest_installation_execNEST_cpp = abspath_nest_installation_nest +"src/execNEST.cpp"
abspathfile_nest_installation_execNEST_bin = abspath_nest_installation_install +"bin/execNEST"


# paths and files: sfs repo
abspath_sfs_repo = "/home/daniel/Desktop/arbeitsstuff/sfs/github_repo_v2/"
abspath_sfs_repo_detectors = abspath_sfs_repo +"detectors/"
abspath_sfs_repo_spectra = abspath_sfs_repo +"spectra/"


# darwin baseline detector design
darwin_baseline_detector_dict = {
    # primary scintillation (S1) parameters
    "g1"                : 0.1170,                   # phd per S1 phot at dtCntr (not phe), divide out 2-PE effect,                          JN: 0.119, LUX_Run03: 0.1170 (0.117+/-0.003 WS,0.115+/-0.005 D-D,0.115+/-0.005 CH3T,0.119+/-0.001 LUXSim), XENON10: 0.073
    "sPEres"            : 0.37,                     # single phe (=PE=photoelectrons) resolution (Gaussian assumed),                        JN: 0.38, LUX_Run03: 0.37 (arXiv:1910.04211.), XENON10: 0.58
    "sPEthr"            : (0.3 * 1.173) / 0.915,    # POD threshold in phe, usually used IN PLACE of sPEeff,                                JN: 0.35, LUX_Run03: (0.3 * 1.173) / 0.915 (arXiv:1910.04211.), XENON10: 0.35
    "sPEeff"            : 1.00,                     # actual efficiency, can be used in lieu of POD threshold, units: fractional,           JN: 0.90, LUX_Run03: 1.00 (arXiv:1910.04211), XENON10: 1.00
    "noiseBaseline[0]"  : 0.00,                     # baseline noise mean in PE (Gaussian),                                                 JN: 0.0, LUX_Run03: 0.00 (arXiv:1910.04211 says -0.01), XENON10: 0.0
    "noiseBaseline[1]"  : 0.08,                     # baseline noise width in PE (Gaussian),                                                JN: 0.0, LUX_Run03: 0.08 (arXiv:1910.04211), XENON10: 0.0
    "noiseBaseline[2]"  : 0.0,                      # baseline noise mean in e- (for grid wires),                                           JN: none, LUX_Run03: 0.0, XENON10: 0.0
    "noiseBaseline[3]"  : 0.0,                      # baseline noise width in e- (for grid wires),                                          JN: none, LUX_Run03: 0.0, XENON10: 0.0
    "P_dphe"            : 0.173,                    # chance 1 photon makes 2 phe instead of 1 in Hamamatsu PMT, units: fractional,         JN: 0.22, LUX_Run03: 0.173 (arXiv:1910.04211), XENON10: 0.2
    "coinWind"          : 100,                      # S1 coincidence window in ns,                                                          JN: 100, LUX_Run03: 100 (1310.8214), XENON10: 100
    "coinLevel"         : 2,                        # how many PMTs have to fire for an S1 to count,                                        JN: 3, LUX_Run03: 2 (1512.03506), XENON10: 2
    "numPMTs"           : 119,                      # for coincidence calculation,                                                          JN: 494, LUX_Run03: 119 (122 minus 3 off), XENON10: 89
    "OldW13eV"          : "true",                   # default true, which means use "classic" W instead of Baudis / EXO's,                  JN: none, LUX_Run03: "true", XENON10: "true"
    "noiseLinear[0]"    : 0.0e-2,                   # S1->S1 Gaussian-smeared with noiseL[0]*S1, units: fraction NOT %!                     JN: none, LUX_Run03: 0.0e-2 (1910.04211 p.12, to match 1610.02076 Fig. 8.), XENON10: 3e-2
    "noiseLinear[1]"    : 0.0e-2,                   # S2->S2 Gaussian-smeared with noiseL[1]*S2, units: fraction NOT %!                     JN: none, LUX_Run03: 0.0e-2 (1910.04211 p.12, to match 1610.02076 Fig. 8.), XENON10: 3e-2
    # ionization and secondary scintillation (S2) parameters
    "g1_gas"            : 0.1,                      # phd per S2 photon in gas, used to get SE size, units: phd per e-,                     JN: 0.102, LUX_Run03: 0.1 (0.1 in 1910.04211), XENON10: 0.0655
    "s2Fano"            : 3.6,                      # Fano-like fudge factor for SE width, dimensionless,                                   JN: 3.61, LUX_Run03: 3.6 (3.7 in 1910.04211; this matches 1608.05381 better), XENON10: 3.61
    "s2_thr"            : (150.0 * 1.173) / 0.915,  # the S2 threshold in phe or PE, *not* phd. Affects NR most,                            JN: 100.0, LUX_Run03: (150.0 * 1.173) / 0.915 (65-194 pe in 1608.05381), XENON10: 300.0
    "E_gas"             : 6.25,                     # field in kV/cm between liquid/gas border and anode,                                   JN: 10.85, LUX_Run03: 6.25 (6.55 in 1910.04211), XENON10: 12.0
    "eLife_us"          : 800.0,                    # the drift electron mean lifetime in micro-seconds,                                    JN: 1600.0, LUX_Run03: 800.0 (p.44 of James Verbus PhD thesis Brown), XENON10: 2200.0
    # thermodynamic properties
#    "inGas"             : "false",                  # (duh),                                                                               JN: "false", LUX_Run03: commented out, XENON10: "false"
    "T_Kelvin"          : 173.0,                    # for liquid drift speed calculation, temperature in Kelvin,                            JN: 175.0, LUX_Run03: 173.0 (1910.04211), XENON10: 177.0
    "p_bar"             : 1.57,                     # gas pressure in units of bars, it controls S2 size,                                   JN: 2.0, LUX_Run03: 1.57 (1910.04211), XENON10: 2.14
    # data analysis parameters and geometry
    "dtCntr"            : 160.0,                    # center of detector for S1 corrections, in usec.,                                      JN: 822.0, LUX_Run03: 160.0 (p.61 Dobi thesis UMD, 159 in 1708.02566), XENON10: 40.0
    "dt_min"            : 38.0,                     # minimum. Top of detector fiducial volume, units: microseconds,                        JN: 75.8, LUX_Run03: 38.0 (1608.05381), XENON10: 20.0
    "dt_max"            : 305.0,                    # maximum. Bottom of detector fiducial volume, units: microseconds,                     JN: 1536.5, LUX_Run03: 305.0 (1608.05381), XENON10: 60.0
    "radius"            : 200.0,                    # millimeters (fiducial rad), units: millimeters,                                       JN: 1300., LUX_Run03: 200.0 (1512.03506), XENON10: 50.0
    "radmax"            : 235.0,                    # actual physical geo. limit, units: millimeters,                                       JN: 1350., LUX_Run03: 235.0 (1910.04211), XENON10: 50.0
    "TopDrift"          : 544.95,                   # top of drif volume in mm not cm or us, i.e., this *is* where dt=0, z=0mm is cathode,  JN: 3005.0, LUX_Run03: 544.95 (544.95 in 1910.04211), XENON10: 150.0
    "anode"             : 549.2,                    # the level of the anode grid-wire plane in mm,                                         JN: 3012.5, LUX_Run03: 549.2 (1910.04211 and 549 in 1708.02566), XENON10: 152.5
    "gate"              : 539.2,                    # mm. this is where the E-field changes (higher),                                       JN: 3000.0, LUX_Run03: 539.2 (1910.04211 and 539 in 1708.02566), XENON10: 147.5
    "cathode"           : 55.90,                    # mm. defines point below which events are gamma-X                                      JN: 250, LUX_Run03: 55.90 (55.9-56 in 1910.04211,1708.02566), XENON10: 1.00
    # 2D (xy) position reconstruction
    "PosResExp"         : 0.015,                    # exp increase in pos recon res at hi r, units: 1/mm,                                   JN: 0.015, LUX_Run03: 0.015 (arXiv:1710.02752 indirectly), XENON10: 0.015
    "PosResBase"        : 70.8364,                  # baseline unc in mm, see NEST.cpp for usage,                                           JN: 30.0, LUX_Run03: 70.8364 ((1710.02752 indirectly), XEONON10: 70.8364
}



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





############################################
### spectra computation
############################################


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





############################################
### NEST interfacing
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
    detector_dict = {}, # dict or abspath-string, dictionary or .json-file resembling the detector the spectrum is supposed to be simulated in
    detector_name = "", # string, name of the detector, only required when not referring to an existing file
    abspathfile_execNEST_binary = abspathfile_nest_installation_execNEST_bin, # string, abspathfile of the 'execNEST' executiable generated in the 'install' NEST folder
    baseline_detector_dict = darwin_baseline_detector_dict, # string, abspathfile of the DARWIN baseline detector
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
                abspath_output_list = [abspath_sfs_repo_detectors],
                detector_name = detector_name,
                flag_verbose = flag_verbose,
            )
        text_add="\n" if detector_dict != {} else ""
        if flag_verbose: print(text_add +f"{fn}: installing new detector header file")
        install_detector_header_file(
            abspathfile_new_detector_hh = abspath_sfs_repo_detectors +detector_name +".hh",
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
        num = int(spectrum_dict["numEvts"][k]) if hasattr(spectrum_dict["numEvts"], "__len__") else int(spectrum_dict["numEvts"])
        e_min = str(spectrum_dict["E_min[keV]"][k]) if hasattr(spectrum_dict["E_min[keV]"], "__len__") else int(spectrum_dict["E_min[keV]"])
        e_max = str(spectrum_dict["E_max[keV]"][k]) if hasattr(spectrum_dict["E_max[keV]"], "__len__") else int(spectrum_dict["E_max[keV]"])
        if len(this_nest_run_tuple_list) != num: raise Exception(f"This NEST run yielded {len(this_nest_run_tuple_list)} events instead of the specified {num} at E_min={e_min} and E_max={max}.")

    ### casting the 'execNEST_output_tuple_list' into a ndarray
    if flag_verbose: print(f"{fn}: casting 'execNEST_output_tuple_list' into numpy ndarray")
    execNEST_output_ndarray = np.array(execNEST_output_tuple_list, execNEST_dtype)

    return execNEST_output_ndarray





############################################
### ER/NR discrimination
############################################







############################################
### likelihood stuff
############################################





