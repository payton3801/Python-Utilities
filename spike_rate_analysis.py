# %%
import numpy as np
from brpylib import NsxFile
# %%
nsx = NsxFile("/snel/braingate/data/T16/2024-02-26/RawData/NSP_Data/t16_240226_000/20240226-134233/Hub1-20240226-134233-001.ns5")
# %%
data = np.vstack(nsx.getdata()['data'])
print(data)
# %%
##filter 
from scipy.signal import butter, sosfiltfilt

# Define the cutoff frequency and order of the filter 
cutoff_freq = 250.0 
sampling_rate = 1000.0
order = 6  

normalized_cutoff_freq = cutoff_freq / (0.5 * sampling_rate)
sos = butter(order, normalized_cutoff_freq, btype='high', analog=False, output='sos')

filtered_data = sosfiltfilt(sos, data)


# %%
##compute and apply linear regression reference parameters
def calc_lrr_params_parallel(channel, group, decimate=1):

    grp = np.setdiff1d(group, channel)

    X = all_data[grp, ::decimate].T
    y = all_data[channel, ::decimate].reshape(1, -1)
    params = np.linalg.solve(X.T @ X, X.T @ y.T).T

    return channel, grp, params


reref_params = np.zeros((256, 256), dtype=np.float64)
if reref == 'car':
    ch_count = 0
    for g, s in zip(reref_groups, reref_sizes):
        reref_params[ch_count:ch_count+s, g] = 1./len(g)
        ch_count += s

elif reref == 'lrr':
    # use single-precision for faster compute
    all_data = data.astype(np.float32)
    with Parallel(n_jobs=-1, require='sharedmem') as parallel:
        # loop through the groups and compute LRR for each one
        ch_count = 0
        for idx, (g, s) in enumerate(zip(reref_groups, reref_sizes)):
            # compute the LRR parameters for each channel in this group
            tasks = [
                delayed(calc_lrr_params_parallel)(channel=ch, group=g)
                for ch in range(ch_count, ch_count + s)
            ]
            lrr_params = parallel(tasks)
            # unpack the parallel execution results - assign the LRR parameters
            # to the reref_params array
            for item in lrr_params:
                ch, grp, output = item
                reref_params[ch, grp] = output

            ch_count += s
    data = data.astype(np.float64)
# Re-reference the data
data = rereference_data(data, reref_params)
logging.debug('Finished rereferencing')

#################################

# %%
#compute -4.5 RMS threshold
if np.isnan(data).any():
    print("Warning: NaN values found in the data. They will be ignored.")

thresholds = -4.5 * np.sqrt(np.mean(np.square(data), axis=1)).reshape(-1, 1)

#%%
##extract spikes
spikes = np.where(data <= thresholds)[0]