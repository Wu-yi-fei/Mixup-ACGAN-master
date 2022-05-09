import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from load import load_solar_data
import numpy as np


def interpret_for_pattern():
    high_peak_power_bound = 0.8
    normal_peak_power_bound = 0.6

    persistent_duration_bound = 288
    long_duration_bound = 240
    normal_duration_bound = 210

    X = load_solar_data()
    pattern_interpret_label = []

    for s in range(X.shape[0]):
        x = X[s]
        if np.max(x) >= high_peak_power_bound:
            if np.count_nonzero(x) >= persistent_duration_bound:
                pattern_interpret_label.append(0)  # Persistent high irradiation day

            elif long_duration_bound <= np.count_nonzero(x) <= persistent_duration_bound:
                    pattern_interpret_label.append(1)  # High irradiation day of long duration

            elif normal_duration_bound <= np.count_nonzero(x) <= long_duration_bound:
                pattern_interpret_label.append(2)   # High irradiation day of normal duration

            elif np.count_nonzero(x) <= normal_duration_bound:
                pattern_interpret_label.append(3)   # High irradiation day of short-lived duration

        elif normal_peak_power_bound <= np.max(x) <= high_peak_power_bound:
            if np.count_nonzero(x) >= persistent_duration_bound:
                pattern_interpret_label.append(4)  # Persistent normal irradiation day

            elif long_duration_bound <= np.count_nonzero(x) <= persistent_duration_bound:
                pattern_interpret_label.append(5)  # Normal irradiation day of long duration

            elif normal_duration_bound <= np.count_nonzero(x) <= long_duration_bound:
                pattern_interpret_label.append(6)   # Normal irradiation day of normal duration

            elif np.count_nonzero(x) <= normal_duration_bound:
                pattern_interpret_label.append(7)  # Normal irradiation day of short-lived duration

        elif np.max(x) <= normal_peak_power_bound:
            if np.count_nonzero(x) >= persistent_duration_bound:
                pattern_interpret_label.append(8)  # Persistent low irradiation day

            elif long_duration_bound <= np.count_nonzero(x) <= persistent_duration_bound:  # Low irradiation day of long duration
                pattern_interpret_label.append(9)

            elif normal_duration_bound <= np.count_nonzero(x) <= long_duration_bound:
                pattern_interpret_label.append(10)   # Low irradiation day of normal duration

            elif np.count_nonzero(x) <= normal_duration_bound:
                pattern_interpret_label.append(11)  # Low irradiation days of short-lived duration

    pattern_interpret_label = np.array(pattern_interpret_label)
    return pattern_interpret_label



if __name__ == '__main__':
    pattern_interpret_label = interpret_for_pattern()