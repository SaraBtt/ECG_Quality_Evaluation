




def compute_avg_segments(start_ms = 13, end_ms = 21, given_path = False, given_model=False):
    '''
    Input ECG dict, 
    
    start_ms = Sample index of ECG beat where the st segment STARTS
    end_ms = Sample index of ECG beat where the st segment ENDS
    
    Note:
    QRS complex is between 70 to 100 ms
    and ST segment lasts generally 80 ms
    '''

    imi_real, asmi_real, norm_real, asmi_gen_norm, imi_gen_norm, norm_gen_asmi, norm_gen_imi = load_data(given_path, given_model)
    ecgs = [imi_real, asmi_real, norm_real, asmi_gen_norm, imi_gen_norm, norm_gen_asmi, norm_gen_imi]
    ecgs_labels = ['imi_real', 'asmi_real', 'norm_real', 'asmi_gen_norm', 'imi_gen_norm', 'norm_gen_asmi', 'norm_gen_imi']

    #compute st_segments for each class
    #And the average for each ST segment
    ecg_segments = []
    ecg_segm_avg = []
    for ecg_class in ecgs:
        segments = ecg_class[:, start_ms: end_ms, :]
        avg_segments = np.average(segments, axis=1)
        ecg_segments.append(segments)
        ecg_segm_avg.append(avg_segments)

    #build dictionary with ecg label, ecg_full, ecg_segments, ecg_segments_average
    ecg_dict={}
    for ind, label in enumerate(ecgs_labels):
        ecg_dict[label]={ 'full_ecg' : ecgs[ind],
                          'st_segments' : ecg_segments[ind],
                          'st_average' : ecg_segm_avg[ind]
                        }
        

    return ecg_dict