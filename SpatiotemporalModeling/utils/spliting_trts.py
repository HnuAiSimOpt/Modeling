import numpy as np
import random
import shutil, os

root_path = "D:\\SpatiotemporalModelingCMAMERevision\\data\\case2\\"

testing_cf_path = root_path + "testing1\\force\\filtered_cf\\"
testing_sea_path = root_path + "testing1\\energy\\sea_npy\\"

total_cf_path = root_path + "training\\force\\filtered_cf\\"
total_sea_path = root_path + "training\\energy\\sea_npy\\"

total_dvs_path = root_path + "training\\valid_design_variables.npy"

testing_n_samples = 100




if not os.path.isdir(testing_cf_path):
    os.makedirs(testing_cf_path)

if not os.path.isdir(testing_sea_path):
    os.makedirs(testing_sea_path)

total_cf_files = os.listdir(total_cf_path)
total_n_samples = len(total_cf_files)

total_sea_files = os.listdir(total_sea_path)
total_dvs = np.load(total_dvs_path)
total_dvs = np.split(total_dvs, total_dvs.shape[0], 0)

testing_files = random.sample(total_cf_files, testing_n_samples)


testing_indexes = []

for file_name in testing_files:
    
    testing_indexes.append(total_cf_files.index(file_name))
    
    shutil.move(total_cf_path + file_name, testing_cf_path + file_name)
    shutil.move(total_sea_path + file_name, testing_sea_path + file_name)
    print(f" sea and cf file :: {file_name} have been moved to the testing folders")

print(len(os.listdir(testing_cf_path)))
print(len(os.listdir(testing_sea_path)))

testing_dvs = []
training_dvs = []
for idx in range(total_n_samples):
    if idx in testing_indexes:
        testing_dvs.append(total_dvs[idx])
    else:
        training_dvs.append(total_dvs[idx])

training_dvs = np.concatenate(training_dvs, axis=0)
testing_dvs = np.concatenate(testing_dvs, axis=0)

np.save(root_path + "training\\training_dvs.npy", training_dvs)
np.save(root_path + "testing1\\testing1_dvs.npy", testing_dvs)
print(f" training dvs :: {training_dvs.shape}")
print(f" testing dvs :: {testing_dvs.shape}")


for sea_file in os.listdir(root_path + "testing2\\energy\\sea_npy\\"):
    if sea_file not in os.listdir(root_path + "testing2\\force\\filtered_cf\\"):
        os.remove(root_path + "testing2\\energy\\sea_npy\\" + sea_file)
        
for sea_file in os.listdir(root_path + "testing3\\energy\\sea_npy\\"):
    if sea_file not in os.listdir(root_path + "testing3\\force\\filtered_cf\\"):
        os.remove(root_path + "testing3\\energy\\sea_npy\\" + sea_file)        

        
