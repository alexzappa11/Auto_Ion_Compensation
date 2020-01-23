# def gradient_calc(starting_vals):
#         ###### Initial gradient calculation ##########
#  # first step to calculate gradient (step_in_x_weight, step_in_y_weight)
#  #    n0_x = np.array([0.1, 0])
#  #    n0_y = np.array([0, 0.1])
#  #    x0_1 = starting_vals
#  #    x0_2 = starting_vals + n0_x
#  #
#  #    y0_1 = starting_vals
#  #    y0_2 = starting_vals + n0_y
#
#
#     initial_photon_count = take_photon(photon_channel)# read photon count
#     photon_count1 = np.array([initial_photon_count, initial_photon_count])# store photon before step (photon count for change in x, photon count for change in y)
#
#     # DO: apply weights to voltage x
#     # DO: send voltage
#     weight_Params_x = np.array([weight_Params[0]+0.1,weight_Params[1], weight_Params[2], weight_Params[3]])
#     weight_Params_y = np.array([weight_Params[0], weight_Params[1]+0.1, weight_Params[2], weight_Params[3]])
#     final_voltage = get_Voltage(weight_Params, position)
#     print(final_voltage)
#     write_voltage(final_voltage)
#
#     ### Pause ###
#     # DO: read photon count for x channel 4
#     photon_countx = take_photon(photon_channel)
#
#     # DO: apply weights to voltage y
#     # DO: send voltage
#     ### Pause ###
#
#     photon_county = take_photon(photon_channel)  # DO: read photon count for y
#
#     # read photon after voltage with new weight applied
#     photon_count2 = np.array([photon_countx, photon_county])
#
#     print(photon_count2)
#
#     ft_dash = np.divide((photon_count2 - photon_count1),
#                         (x0_y0_2 - x0_y0_1))  # calculate gradient
#     print("initial grad", ft_dash)
#
#     return ft_dash
