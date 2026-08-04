from dataclasses import dataclass
from typing import Any


@dataclass
class ECGSample:
    label: str
    data: Any


@dataclass
class ECGDataPair:
    real: ECGSample
    fake: ECGSample



def make_data_class(real_labels_list, real_data_list, gen_labels_list, gen_data_list):
    
    '''
    Create paired real and generated ECG samples.

    
    Parameters
    ----------
    real_labels_list : sequence
        Labels associated with the real ECG samples.

    real_data_list : sequence
        ECG data associated with the real labels.

    gen_labels_list : sequence
        Labels associated with the generated ECG samples.

    gen_data_list : sequence
        ECG data associated with the generated labels.
    
    Example
    -------
    real_labels = ["NORM", "NORM"]
    real_data = [real_ecg_1, real_ecg_2]
    generated_labels = ["ASMI_FROM_NORM", "ASMI_FROM_NORM"]
    generated_data = [generated_ecg_1, generated_ecg_2]

    pairs = make_data_class( real_labels,  real_data, generated_labels, generated_data)
    pairs[0].real.data -> real_ecg_1
    pairs[0].fake.data -> generated_ecg_1
    pairs[0].real.label -> 'NORM'
    pairs[0].fake.label -> 'ASMI_FROM_NORM'
    """

    '''
    
    pairs = []
    
    assert len(real_labels_list) == len(real_data_list) == len(gen_labels_list) == len(gen_data_list), "All input lists must have the same length."
    
    for i in range(len(real_labels_list)):
        data_pair = ECGDataPair(real=ECGSample(label=real_labels_list[i], data=real_data_list[i]),
                             fake=ECGSample(label=gen_labels_list[i], data=gen_data_list[i]))
        pairs.append(data_pair)

    
    return pairs
