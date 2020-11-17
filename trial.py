import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import biosppy.signals.ecg as ecg


signal = np.load('one_hb.npy')


print('shape of signal: ', signal.shape)

ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, sampling_rate=300, show=False)
templates, rpeaks_ = ecg.extract_heartbeats(filtered, rpeaks = rpeaks, sampling_rate=300)

print('shape templates: ', templates.shape)

np.save('one_templates', templates)
