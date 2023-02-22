import numpy as np
import os
import pickle
import dill

def foldering(folder_dir):
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)

case_root = "D:\\SpatiotemporalModelingCMAMERevision\\data\\case2\\"

datasets = ['training\\', 'testing1\\', 'testing2\\', 'testing3\\']


''' SEA NORMALIZATION '''

###### obtain the upper and lower bounds from the training set for normalization

global sea_maxi, sea_mini
sea_maxi = 0.
sea_mini = 0.
for file in os.listdir(case_root + 'training\\energy\\sea_npy\\'):
    
    sea = np.load(case_root + 'training\\energy\\sea_npy\\' + file)
    sea_maxi = max(sea_maxi, np.max(sea))
    sea_mini = min(sea_mini, np.min(sea))

print("sea max min:", sea_maxi, sea_mini)


def normalize_sea(x):
    normed_x = x/float(sea_maxi - sea_mini)
    normed_x = normed_x - float(sea_mini)
    return normed_x

def denormalize_sea(normed_x):
    x = normed_x + float(sea_mini)
    x = x * float(sea_maxi - sea_mini)
    return x

with open(case_root + "nomalization_sea.dill",  '+wb') as f:
    dill.dump(normalize_sea, f, recurse=True)

with open(case_root + "denomalization_sea.dill",  '+wb') as f:
    dill.dump(denormalize_sea, f, recurse=True)



with open(case_root + "nomalization_sea.dill",  '+rb') as f:
    normalize_sea = dill.load(f)

with open(case_root + "denomalization_sea.dill",  '+rb') as f:
    denormalize_sea = dill.load(f)

for subset in datasets:
    
    normed_sea_folder = case_root + subset + 'energy\\normed_sea\\'
    foldering(normed_sea_folder)

    for file in os.listdir(case_root + subset + 'energy\\sea_npy\\'):
        
        sea = np.load(case_root + subset + 'energy\\sea_npy\\' + file)
        normed_sea = normalize_sea(sea)
        np.save(normed_sea_folder + file, normed_sea)
        print(f" normed sea file {file} : {float(np.max(normed_sea))} ~ {float(np.min(normed_sea))}")
        
        
    
''' CF NORMALIZATION '''
    
###### obtain the upper and lower bounds from the training set for normalization
cf0_maxi = 0.
cf0_mini = 0.
cf1_maxi = 0.
cf1_mini = 0.
cf2_maxi = 0.
cf2_mini = 0.
for file in os.listdir(case_root + 'training\\force\\filtered_cf\\'):
    
    cf = np.load(case_root + 'training\\force\\filtered_cf\\' + file)
    cf0_maxi = max(cf0_maxi, np.max(cf[:,:,0]))
    cf0_mini = min(cf0_mini, np.min(cf[:,:,0]))
    cf1_maxi = max(cf1_maxi, np.max(cf[:,:,1]))
    cf1_mini = min(cf1_mini, np.min(cf[:,:,1]))
    cf2_maxi = max(cf2_maxi, np.max(cf[:,:,2]))
    cf2_mini = min(cf2_mini, np.min(cf[:,:,2]))

print("cf0 max min:", cf0_maxi, cf0_mini)
print("cf1 max min:", cf1_maxi, cf1_mini)
print("cf2 max min:", cf2_maxi, cf2_mini)

def normalize_cf0(x):
    normed_x = x/float(cf0_maxi - cf0_mini)
    normed_x = normed_x - float(cf0_mini)
    return normed_x
def denormalize_cf0(normed_x):
    x = normed_x + float(cf0_mini)
    x = x * float(cf0_maxi - cf0_mini)
    return x

def normalize_cf1(x):
    normed_x = x/float(cf1_maxi - cf1_mini)
    normed_x = normed_x - float(cf1_mini)
    return normed_x
def denormalize_cf1(normed_x):
    x = normed_x + float(cf1_mini)
    x = x * float(cf1_maxi - cf1_mini)
    return x

def normalize_cf2(x):
    normed_x = x/float(cf2_maxi - cf2_mini)
    normed_x = normed_x - float(cf2_mini)
    return normed_x
def denormalize_cf2(normed_x):
    x = normed_x + float(cf2_mini)
    x = x * float(cf2_maxi - cf2_mini)
    return x

with open(case_root + "nomalization_cf0.dill",  '+wb') as f:
    dill.dump(normalize_cf0, f, recurse=True)
with open(case_root + "denomalization_cf0.dill",  '+wb') as f:
    dill.dump(denormalize_cf0, f, recurse=True)

with open(case_root + "nomalization_cf1.dill",  '+wb') as f:
    dill.dump(normalize_cf1, f, recurse=True)
with open(case_root + "denomalization_cf1.dill",  '+wb') as f:
    dill.dump(denormalize_cf1, f, recurse=True)

with open(case_root + "nomalization_cf2.dill",  '+wb') as f:
    dill.dump(normalize_cf2, f, recurse=True)
with open(case_root + "denomalization_cf2.dill",  '+wb') as f:
    dill.dump(denormalize_cf2, f, recurse=True)
    


with open(case_root + "nomalization_cf0.dill",  '+rb') as f:
    normalize_cf0 = dill.load(f)
with open(case_root + "denomalization_cf0.dill",  '+rb') as f:
    denormalize_cf0 = dill.load(f)
    
with open(case_root + "nomalization_cf1.dill",  '+rb') as f:
    normalize_cf1 = dill.load(f)
with open(case_root + "denomalization_cf1.dill",  '+rb') as f:
    denormalize_cf1 = dill.load(f)

with open(case_root + "nomalization_cf2.dill",  '+rb') as f:
    normalize_cf2 = dill.load(f)
with open(case_root + "denomalization_cf2.dill",  '+rb') as f:
    denormalize_cf2 = dill.load(f)

cf_normalizers = [normalize_cf0, normalize_cf1, normalize_cf2]
for subset in datasets:
    
    filtered_normed_cf_folder = case_root + subset + 'force\\filtered_normed_cf\\'
    foldering(filtered_normed_cf_folder)

    for file in os.listdir(case_root + subset + 'force\\filtered_cf\\'):
        
        cf = np.load(case_root + subset + 'force\\filtered_cf\\' + file)
        filtered_normed_cf = []
        for nc, channel in enumerate(np.split(cf, 3, -1)):
            filtered_normed_cf.append(cf_normalizers[nc](channel))
        filtered_normed_cf = np.concatenate(filtered_normed_cf, axis=-1)
        np.save(filtered_normed_cf_folder + file, filtered_normed_cf)
        print(f" normed cf file {file} {filtered_normed_cf.shape} : range {float(np.max(filtered_normed_cf))} ~ {float(np.min(filtered_normed_cf))}")
    
    
    