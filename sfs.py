



############################################
### imports
############################################


import subprocess
import numpy as np





############################################
### helper functions
############################################









############################################
### NEST interfacing
############################################


def execNEST(
    abspath_execNEST_binary, # string, abspath to the 'execNEST' executiable generated in the 'install' NEST folder
    spectrum_dict = {}, # dict, dictionary resembling the input spectrum to be simulated by NEST
    detector_dict = {}, # dict, dictionary resembling the detector the spectrum is supposed to be simulated in
    flag_verbose = True, # bool, flag indicating whether the print-statements are being printed
):

    """
    This function is used to execute the 'execNEST' C++ executable from the NEST installation.
    Returns the NEST output in the form of a numpy structured array.
    """

    ### initializing
    fn = "execNEST" # name of this function, required for the print statements
    if flag_verbose == True: print(f"{fn}: initializing")
    execNEST_output_tuple_list = []
    cmd_list = []
    debug_list = []

    ### executing the 'execNEST' executable
    if flag_verbose == True: print(f"{fn}: compiling the 'execNEST' command strings")
    if spectrum_dict["type_interaction"]=="ER":
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
                abspath_execNEST_binary,
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
                if flag_verbose == True: print(f"\tinput 'spectrum_dict' appears valid")
            else:
                raise Exception(f"ERROR: len(spectrum_dict['numEvts'])==len(spectrum_dict['E_min[keV]'])==len(spectrum_dict['E_max[keV]'])")
            # looping over all resulting 'cmd_strings'
            for k, num in enumerate(spectrum_dict["numEvts"]):
                cmd_string = " ".join([
                    abspath_execNEST_binary,
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
        if flag_verbose == True: print(f"{fn}: executing '$ {cmd_string}'")
        cmd_return = subprocess.run(cmd_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout = cmd_return.stdout.decode('utf-8')
        stderr = cmd_return.stderr.decode('utf-8')
        debug_dict = {
            "stderr" : stderr,
            "cmd_string" : cmd_string,
        }

        ### writing the 'execNEST' output into 'execNEST_output_tuple_list'
        if flag_verbose == True: print(f"{fn}: writing the 'execNEST' data into 'execNEST_output_tuple_list'")
        this_nest_run_tuple_list = []
        state = "searching_for_header_line"
        # looping over all 'execNEST' output lines
        if flag_verbose == True: print(f"\tlooping over 'execNEST' output lines")
        for line in stdout.split("\n"):
            line_list = list(line.split("\t"))
            # skipping all lines that can neither be headers nor simulation output
            if len(line_list) in [1,4,6]:
                continue
            # extracting header columns and simulation data
            else:
                if state == "searching_for_header_line":
                    if "Nph" in line_list:
                        state = "reading_output"
                        dtype_list = []
                        for header in line_list:
                            if header in ["X,Y,Z [mm]"]:
                                dtype_list.append((header, np.unicode_, 16))
                            else:
                                dtype_list.append((header, np.float64))

                        execNEST_dtype = np.dtype(dtype_list)
                        if flag_verbose == True: print(f"\textracted header: {line_list}")
                    else:
                        continue
                elif state == "reading_output":
                    execNEST_output_tuple = tuple(line_list)
                    this_nest_run_tuple_list.append(execNEST_output_tuple)
        execNEST_output_tuple_list += this_nest_run_tuple_list
        num = int(spectrum_dict["numEvts"][k]) if hasattr(spectrum_dict["numEvts"], "__len__") else int(spectrum_dict["numEvts"])
        if len(this_nest_run_tuple_list) != num: raise Exception(f"")

    ### casting the 'execNEST_output_tuple_list' into a ndarray
    if flag_verbose == True: print(f"{fn}: casting 'execNEST_output_tuple_list' into numpy ndarray")
    execNEST_output_ndarray = np.array(execNEST_output_tuple_list, execNEST_dtype)

    return execNEST_output_ndarray





############################################
### ER/NR discrimination
############################################







############################################
### likelihood stuff
############################################





