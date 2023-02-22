import numpy as np



plot_data = np.load(r"D:\SpatiotemporalModelingCMAMERevision\data\case2\training\force\filtered_normed_cf\0008.npy")

import matplotlib.pyplot as plt
from filtering import BWfilter, BWfilter1D

D0 = 10

xx, yy = np.meshgrid(np.arange(180), np.arange(310), indexing='ij')

# filtered = BWfilter1D(np.squeeze(plot_data), wn=15)
filtered = []
for channel in np.split(plot_data, 3, -1):
    # filtered.append(BWfilter(np.squeeze(channel), D0=5, type='lp', N=4))
    filtered.append(BWfilter1D(np.squeeze(channel), wn=20))
filtered = np.stack(filtered, axis=-1)

plt.contourf(xx, yy, plot_data[:,:,0], levels=310, cmap='jet')
plt.show()
plt.close()
plt.contourf(xx, yy, plot_data[:,:,1], levels=310, cmap='jet')
plt.show()
plt.close()
plt.contourf(xx, yy, plot_data[:,:,2], levels=310, cmap='jet')
plt.show()
plt.close()

plt.style.use('classic')
for i in range(40):
    
    plt.plot(plot_data[:,i*6,0], c='r')
    plt.plot(plot_data[:,i*6,1], c='g')
    plt.plot(plot_data[:,i*6,2], c='b')
    
    plt.show()
    plt.close()


# from filtering import filter_dataset


# filter_dataset(
#     root_path = "E:\\SpatiotemporalModelingCMAMERevision\\data\\case2\\training",
#     valid_shape = cf.shape,
#     D0 = 10
# )


