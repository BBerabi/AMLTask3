import numpy as np 
import pandas as pd 
import sklearn 
from scipy import signal 
import biosppy 

hb = np.load('one_templates.npy')
#print(hb.shape)

freq_band = {"ULF": [0, 0.0033],
			  "VLF": [0.0033, 0.04],
			  "LF": [0.04, 0.15],
			  "HF": [0.15, 0.4],
			  "VHF": [0.4, 0.5]}

#freqs, psd = signal.welch(hb)
#print(freqs)
#freqs, psd = signal.welch(hb[:200])
#print(freqs)

def get_frequency_features(heartbeats):

	features = {}

	sig = heartbeats.mean(axis=0)
	
	# Energy properties
	freqs, psd = signal.welch(sig)

	lower = np.where(freqs>=freq_band["ULF"][0])
	upper = np.where(freqs<freq_band["ULF"][1])
	indexes = np.intersect1d(lower,upper)
	features["ULF"] = np.trapz(y=psd[indexes], x=freqs[indexes])
	lower = np.where(freqs>=freq_band["VLF"][0])
	upper = np.where(freqs<freq_band["VLF"][1])
	indexes = np.intersect1d(lower,upper)
	features["VLF"] = np.trapz(y=psd[indexes], x=freqs[indexes])
	lower = np.where(freqs>=freq_band["LF"][0])
	upper = np.where(freqs<freq_band["LF"][1])
	indexes = np.intersect1d(lower,upper)
	features["LF"] = np.trapz(y=psd[indexes], x=freqs[indexes])
	lower = np.where(freqs>=freq_band["HF"][0])
	upper = np.where(freqs<freq_band["HF"][1])
	indexes = np.intersect1d(lower,upper)
	features["HF"] = np.trapz(y=psd[indexes], x=freqs[indexes])
	lower = np.where(freqs>=freq_band["VHF"][0])
	upper = np.where(freqs<freq_band["VHF"][1])
	indexes = np.intersect1d(lower,upper)
	features["VHF"] = np.trapz(y=psd[indexes], x=freqs[indexes])
	features["Total"] = np.trapz(y=psd,x=freqs)

	#Some additional features to test
	features["extra1"]=features["LF"]/(features["LF"]+features["HF"])
	features["extra2"]=features["HF"]/(features["LF"]+features["HF"])
	features["extra3"]=features["LF"]/features["HF"]
	features["extra4"]=features["HF"]/features["Total"]
	features["extra5"]=features["HF"]/features["Total"]

	# Autocorrelation properties

	autocorr = np.correlate(sig,sig,"full")
	features["corr_max"]=np.max(autocorr)
	features["corr_min"]=np.min(autocorr)

	sig_bar = np.mean(sig)

	N = len(freqs)
	normalization = np.sum(np.square(sig-sig_bar))
	len1 = N - int(N/4)
	r1 = np.sum((sig[:len1]-sig_bar)*(sig[len1:2*len1]-sig_bar))/np.sum(np.square(sig-sig_bar))
	len2 = N - int(N/2)
	r2 = np.sum((sig[:len2]-sig_bar)*(sig[len2:2*len2]-sig_bar))/np.sum(np.square(sig-sig_bar))
	len3 = N - int(3*N/4)
	r3 = np.sum((sig[:len3]-sig_bar)*(sig[len3:2*len3]-sig_bar))/np.sum(np.square(sig-sig_bar))
	#r1 = np.sum((sig[:int(N-3*N/4)]-sig_bar)*(sig[int(3*N/4):int(N-3*N/4)]-sig_bar))/np.sum(np.square(sig-sig_bar))
	#r2 = np.sum((sig[:int(N-2*N/4)]-sig_bar)*(sig[int(2*N/4):int(N-2*N/4)]-sig_bar))/np.sum(np.square(sig-sig_bar))
	#r3 = np.sum((sig[:int(N-1*N/4)]-sig_bar)*(sig[int(N/4):int(N-N/4)]-sig_bar))/np.sum(np.square(sig-sig_bar)) 
	
	features["corr1"]=r1
	features["corr2"]=r2
	features["corr3"]=r3

	return features


features = get_frequency_features(hb)
print(features)












