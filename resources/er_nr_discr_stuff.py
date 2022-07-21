


def calc_er_nr_discrimination_line(
    er_spectrum,
    nr_spectrum,
    nr_acceptance = 0.30,
    g1,
    g2,
    W,
):

    output_dict = {
        "dl_x_data_s1_over_g1",
        "dl_y_data",
        "nr_acceptance",
        "er_discrimination",
    }













# This function is used to extract discrimination data from simulated input ER and NR signatures.
# I.e. the input signatures are sliced into bins (according to 'flag_slicing') and every subset of data is then analyzed in terms of leakage.
def get_discrdata_from_simdata(
    input_er_data,
    input_nr_data,
    bin_edges,
    threshold_events_per_bin = 20,
    nr_acceptances = [50, 85], # former: leakage_fraction_percentile
    savestring = "",
    flag_slicing = ["s1_g1", "er_ee"][1],
    flag_returnsubdatasets = True,
    **kwargs
):

    # definitions
    # bins
    bin_width = bin_edges[1] -bin_edges[0]
    bin_centers = [bin_edges[i] +0.5*(bin_edges[i+1] -bin_edges[i]) for i in range(len(bin_edges)-1)]
    # output data
    popdata_dtype = [
        ("bin_center", np.float64)]
    for i in range(len(nr_acceptances)):
        nracc_add_string = "nracc_" +f"{nr_acceptances[i]:.1f}".replace(".","_") +"__"
        popdata_dtype = popdata_dtype +[
            (nracc_add_string +"threshold_value", np.float64),
            (nracc_add_string +"discriminationline_x_left", np.float64),
            (nracc_add_string +"discriminationline_x_right", np.float64),
            (nracc_add_string +"n_nr_events_in_bin", np.uint64),
            (nracc_add_string +"n_er_events_in_bin", np.uint64),
            (nracc_add_string +"n_er_events_below_threshold", np.uint64),
            (nracc_add_string +"leakage_fraction_in_bin", np.float64),
            (nracc_add_string +"leakage_fraction_in_bin_error", np.float64),
            (nracc_add_string +"er_rejection_in_bin", np.float64),
            (nracc_add_string +"er_rejection_in_bin_error", np.float64)]
    popdata_tuple_list = []
    sliced_data_er = []
    sliced_data_nr = []

    # looping over the bins/slices to generate 'popdata' data and add it to the 'popdata_tuple_list'
    for j in range(len(bin_edges)-1):

        # selecting the data corresponding to the current bin/slice
        if flag_slicing == "er_ee":
            er_bin_data = input_er_data[
                (input_er_data["s2_phe"] >= ((kwargs["g2"]/kwargs["w"])*bin_edges[j]*1000) -((kwargs["g2"]/kwargs["g1"])*input_er_data["s1_phe"]) ) &
                (input_er_data["s2_phe"] <= ((kwargs["g2"]/kwargs["w"])*bin_edges[j+1]*1000) -((kwargs["g2"]/kwargs["g1"])*input_er_data["s1_phe"]))]
            nr_bin_data = input_nr_data[
                (input_nr_data["s2_phe"] >= ((kwargs["g2"]/kwargs["w"])*bin_edges[j]*1000) -((kwargs["g2"]/kwargs["g1"])*input_nr_data["s1_phe"]) ) &
                (input_nr_data["s2_phe"] <= ((kwargs["g2"]/kwargs["w"])*bin_edges[j+1]*1000) -((kwargs["g2"]/kwargs["g1"])*input_nr_data["s1_phe"]))]
        elif flag_slicing == "s1_g1": # this I still need to implement
            er_bin_data = input_er_data
            nr_bin_data = input_nr_data
        else:
            raise Exception("undefined 'flag_slicing'")
            
        # extracting data from to the current bin/slice
        n_nr = len(nr_bin_data) # <--- popdata: "n_nr_events_in_bin"
        n_er = len(er_bin_data) # <--- popdata: "n_er_events_in_bin"

        # checking whether there are sufficient events within the current bin/slice
        if (n_er > threshold_events_per_bin) and (n_nr > threshold_events_per_bin):

            # looping over all NR acceptances
            popdata_tuple = (bin_edges[j]+0.5*bin_width, )
            for k in range(len(nr_acceptances)):

                # calculating the leakage beneath the threshold
                percentile_index = int(len(nr_bin_data)*(nr_acceptances[k]/100)) # number of events for an NR acceptance of 'nr_acceptances[k]'
                percentile_threshold_value = sorted(list(nr_bin_data["log_s2_s1"]))[percentile_index] # 'log_s2_s1' value corresponding to the NR acceptance defined above
                n_er_below_threshold = len(er_bin_data[(er_bin_data["log_s2_s1"] <= percentile_threshold_value)]) # <--- popdata: "n_er_events_below_threshold"
                leakage_fraction_within_current_bin = n_er_below_threshold/n_er # <--- popdata: "leakage_fraction_in_bin"
                leakage_fraction_within_current_bin_error = np.sqrt(n_er_below_threshold)/n_er # <--- popdata: "leakage_fraction_in_bin_error"
                er_rejection_within_current_bin = (n_er-n_er_below_threshold)/n_er # <--- popdata: "er_rejection_in_bin"
                er_rejection_within_current_bin_error = np.sqrt(n_er-n_er_below_threshold)/n_er # <--- popdata: "er_rejection_in_bin_error"

                # calculating the x values of the discrimination line for the log_s2_s1 over s1_g1 observable space
                if flag_slicing == "er_ee":
                    c_star = 10**(percentile_threshold_value)
                    first_factor_low = (1/c_star) *(kwargs["g2"]/(kwargs["w"]*kwargs["g1"])) *bin_edges[j]*1000
                    first_factor_high = (1/c_star) *(kwargs["g2"]/(kwargs["w"]*kwargs["g1"])) *bin_edges[j+1]*1000
                    second_factor = 1/(1 +(1/c_star)*(kwargs["g2"]/kwargs["g1"]))
                    discrline_x_left = first_factor_low *second_factor
                    discrline_x_right = first_factor_high *second_factor
                elif flag_slicing == "s1_g1":
                    a = 3
                else:
                    raise Exception("invalid 'flag_slicing'")

                # adding data to the 'popdata_tuple'
                popdata_tuple = popdata_tuple +(
                    percentile_threshold_value,
                    discrline_x_left,
                    discrline_x_right,
                    n_nr,
                    n_er,
                    n_er_below_threshold,
                    leakage_fraction_within_current_bin,
                    leakage_fraction_within_current_bin_error,
                    er_rejection_within_current_bin,
                    er_rejection_within_current_bin_error)

            # adding data to the 'popdata_tuple_list'
            popdata_tuple_list.append(popdata_tuple)

        # saving the subdatasets
        if flag_returnsubdatasets == True:
            sliced_data_nr.append(nr_bin_data)
            sliced_data_er.append(er_bin_data)

    # generating the output 'popdata' ndarray
    popdata_ndarray = np.array(popdata_tuple_list, popdata_dtype)
    if savestring != "":
        np.save(savestring, popdata_ndarray)

    # end of program: returning stuff
    if flag_returnsubdatasets == True:
        return popdata_ndarray, sliced_data_nr, sliced_data_er
    else:
        return popdata_ndarray

