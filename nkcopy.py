import numpy as np
import pandas as pd
import sklearn
import nolds
import mne
import biosppy
import scipy.signal

from scipy import stats

def ecg_hrv(rpeaks=None, rri=None, sampling_rate=1000, hrv_features=["time", "frequency", "nonlinear"]):
    """
    Computes the Heart-Rate Variability (HRV). Shamelessly stolen from the `hrv <https://github.com/rhenanbartels/hrv/blob/develop/hrv>`_ package by Rhenan Bartels. All credits go to him.

    Parameters
    ----------
    rpeaks : list or ndarray
        R-peak location indices.
    rri: list or ndarray
        RR intervals in the signal. If this argument is passed, rpeaks should not be passed.
    sampling_rate : int
        Sampling rate (samples/second).
    hrv_features : list
        What HRV indices to compute. Any or all of 'time', 'frequency' or 'nonlinear'.

    Returns
    ----------
    hrv : dict
        Contains hrv features and percentage of detected artifacts.

    Example
    ----------
    >>> import neurokit as nk
    >>> sampling_rate = 1000
    >>> hrv = nk.bio_ecg.ecg_hrv(rpeaks=rpeaks, sampling_rate=sampling_rate)

    Notes
    ----------
    *Details*

    - **HRV**: Heart-Rate Variability (HRV) is a finely tuned measure of heart-brain communication, as well as a strong predictor of morbidity and death (Zohar et al., 2013). It describes the complex variation of beat-to-beat intervals mainly controlled by the autonomic nervous system (ANS) through the interplay of sympathetic and parasympathetic neural activity at the sinus node. In healthy subjects, the dynamic cardiovascular control system is characterized by its ability to adapt to physiologic perturbations and changing conditions maintaining the cardiovascular homeostasis (Voss, 2015). In general, the HRV is influenced by many several factors like chemical, hormonal and neural modulations, circadian changes, exercise, emotions, posture and preload. There are several procedures to perform HRV analysis, usually classified into three categories: time domain methods, frequency domain methods and non-linear methods.

       - **sdNN**: The standard deviation of the time interval between successive normal heart beats (*i.e.*, the RR intervals). Reflects all influences on HRV including slow influences across the day, circadian variations, the effect of hormonal influences such as cortisol and epinephrine. It should be noted that total variance of HRV increases with the length of the analyzed recording.
       - **meanNN**: The the mean RR interval.
       - **CVSD**: The coefficient of variation of successive differences (van Dellen et al., 1985), the RMSSD divided by meanNN.
       - **cvNN**: The Coefficient of Variation, *i.e.* the ratio of sdNN divided by meanNN.
       - **RMSSD** is the root mean square of the RR intervals (*i.e.*, square root of the mean of the squared differences in time between successive normal heart beats). Reflects high frequency (fast or parasympathetic) influences on HRV (*i.e.*, those influencing larger changes from one beat to the next).
       - **medianNN**: Median of the Absolute values of the successive Differences between the RR intervals.
       - **madNN**: Median Absolute Deviation (MAD) of the RR intervals.
       - **mcvNN**: Median-based Coefficient of Variation, *i.e.* the ratio of madNN divided by medianNN.
       - **pNN50**: The proportion derived by dividing NN50 (The number of interval differences of successive RR intervals greater than 50 ms) by the total number of RR intervals.
       - **pNN20**: The proportion derived by dividing NN20 (The number of interval differences of successive RR intervals greater than 20 ms) by the total number of RR intervals.
       - **Triang**: The HRV triangular index measurement is the integral of the density distribution (that is, the number of all RR intervals) divided by the maximum of the density distribution (class width of 8ms).
       - **Shannon_h**: Shannon Entropy calculated on the basis of the class probabilities pi (i = 1,...,n with n—number of classes) of the NN interval density distribution (class width of 8 ms resulting in a smoothed histogram suitable for HRV analysis).
       - **VLF** is the variance (*i.e.*, power) in HRV in the Very Low Frequency (.003 to .04 Hz). Reflect an intrinsic rhythm produced by the heart which is modulated by primarily by sympathetic activity.
       - **LF**  is the variance (*i.e.*, power) in HRV in the Low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and parasympathetic activity, but in long-term recordings like ours, it reflects sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol (McCraty & Atkinson, 1996).
       - **HF**  is the variance (*i.e.*, power) in HRV in the High Frequency (.15 to .40 Hz). Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per minute) (Kawachi et al., 1995) and decreased by anticholinergic drugs or vagal blockade (Hainsworth, 1995).
       - **Total_Power**: Total power of the density spectra.
       - **LFHF**: The LF/HF ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal balance.
       - **LFn**: normalized LF power LFn = LF/(LF+HF).
       - **HFn**: normalized HF power HFn = HF/(LF+HF).
       - **LFp**: ratio between LF and Total_Power.
       - **HFp**: ratio between H and Total_Power.
       - **DFA**: Detrended fluctuation analysis (DFA) introduced by Peng et al. (1995) quantifies the fractal scaling properties of time series. DFA_1 is the short-term fractal scaling exponent calculated over n = 4–16 beats, and DFA_2 is the long-term fractal scaling exponent calculated over n = 16–64 beats.
       - **Shannon**: Shannon Entropy over the RR intervals array.
       - **Sample_Entropy**: Sample Entropy (SampEn) over the RR intervals array with emb_dim=2.
       - **Correlation_Dimension**: Correlation Dimension over the RR intervals array with emb_dim=2.
       - **Entropy_Multiscale**: Multiscale Entropy over the RR intervals array  with emb_dim=2.
       - **Entropy_SVD**: SVD Entropy over the RR intervals array with emb_dim=2.
       - **Entropy_Spectral_VLF**: Spectral Entropy over the RR intervals array in the very low frequency (0.003-0.04).
       - **Entropy_Spectral_LF**: Spectral Entropy over the RR intervals array in the low frequency (0.4-0.15).
       - **Entropy_Spectral_HF**: Spectral Entropy over the RR intervals array in the very high frequency (0.15-0.40).
       - **Fisher_Info**: Fisher information over the RR intervals array with tau=1 and emb_dim=2.
       - **Lyapunov**: Lyapunov Exponent over the RR intervals array with emb_dim=58 and matrix_dim=4.
       - **FD_Petrosian**: Petrosian's Fractal Dimension over the RR intervals.
       - **FD_Higushi**: Higushi's Fractal Dimension over the RR intervals array with k_max=16.

    """
    # Check arguments: exactly one of rpeaks or rri has to be given as input
    if rpeaks is None and rri is None:
        raise ValueError("Either rpeaks or RRIs needs to be given.")

    if rpeaks is not None and rri is not None:
        raise ValueError("Either rpeaks or RRIs should be given but not both.")

    # Initialize empty dict
    hrv = {}

    # Preprocessing
    # ==================
    # Extract RR intervals (RRis)
    if rpeaks is not None:
        # Rpeaks is given, RRis need to be computed
        RRis = np.diff(rpeaks)
    else:
        # Case where RRis are already given:
        RRis = rri


    # Basic resampling to 1Hz to standardize the scale
    RRis = RRis/sampling_rate
    RRis = RRis.astype(float)



     # Sanity check
    if len(RRis) <= 1:
        print("NeuroKit Warning: ecg_hrv(): Not enough R peaks to compute HRV :/")
        return(hrv)

    # Artifacts treatment
    hrv["n_Artifacts"] = pd.isnull(RRis).sum()/len(RRis)
    # artifacts_indices = RRis.index[RRis.isnull()]  # get the artifacts indices
    # RRis = RRis.drop(artifacts_indices)  # remove the artifacts


    # Rescale to 1000Hz
    RRis = RRis*1000
    hrv["RR_Intervals"] = RRis  # Values of RRis

    # Sanity check after artifact removal
    if len(RRis) <= 1:
        print("NeuroKit Warning: ecg_hrv(): Not enough normal R peaks to compute HRV :/")
        return(hrv)

    # Time Domain
    # ==================
    if "time" in hrv_features:
        hrv["RMSSD"] = np.sqrt(np.mean(np.diff(RRis) ** 2))
        hrv["meanNN"] = np.mean(RRis)
        hrv["sdNN"] = np.std(RRis, ddof=1)  # make it calculate N-1
        hrv["cvNN"] = hrv["sdNN"] / hrv["meanNN"]
        hrv["CVSD"] = hrv["RMSSD"] / hrv["meanNN"]
        hrv["medianNN"] = np.median(abs(RRis))
        hrv['madNN'] = stats.median_absolute_deviation(RRis, axis=None)
        #hrv["madNN"] = mad(RRis, constant=1)
        hrv["mcvNN"] = hrv["madNN"] / hrv["medianNN"]
        nn50 = sum(abs(np.diff(RRis)) > 50)
        nn20 = sum(abs(np.diff(RRis)) > 20)
        hrv["pNN50"] = nn50 / len(RRis) * 100
        hrv["pNN20"] = nn20 / len(RRis) * 100


    return(hrv)




