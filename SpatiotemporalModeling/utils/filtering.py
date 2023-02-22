import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import cv2 as cv
import os
import cv2 as cv
from scipy import signal


def BWfilter(img, D0, W=None, N=2, type='lp', filter='butterworth'):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        W: 带宽
        N: butterworth和指数滤波器的阶数
        type: lp, hp, bp, bs即低通、高通、带通、带阻
        filter:butterworth、ideal、exponential即巴特沃斯、理想、指数滤波器

    Returns:
        imgback：滤波后的图像

    '''

    #离散傅里叶变换
    float_img = np.float32(img)
    dft=cv.dft(float_img,flags=cv.DFT_COMPLEX_OUTPUT)
    #中心化
    dtf_shift=np.fft.fftshift(dft)

    rows,cols=img.shape
    crow,ccol=int(rows/2),int(cols/2) #计算频谱中心
    mask=np.ones((rows,cols,2)) #生成rows行cols列的2纬矩阵
    for i in range(rows):
        for j in range(cols):
            D = np.sqrt((i-crow)**2+(j-ccol)**2)
            if(filter.lower() == 'butterworth'):
                if(type == 'lp'):
                    mask[i, j] = 1/(1+(D/D0)**(2*N))
                elif(type == 'hp'):
                    mask[i, j] = 1/(1+(D0/D)**(2*N))
                elif(type == 'bs'):
                    mask[i, j] = 1/(1+(D*W/(D**2-D0**2))**(2*N))
                elif(type == 'bp'):
                    mask[i, j] = 1/(1+((D**2-D0**2)/D*W)**(2*N))
                else:
                    assert('type error')
            elif(filter.lower() == 'ideal'): #理想滤波器
                if(type == 'lp'):
                    if(D > D0):
                        mask[i, j] = 0
                elif(type == 'hp'):
                    if(D < D0):
                        mask[i, j] = 0
                elif(type == 'bs'):
                    if(D > D0 and D < D0+W):
                        mask[i, j] = 0
                elif(type == 'bp'):
                    if(D < D0 and D > D0+W):
                        mask[i, j] = 0
                else:
                    assert('type error')
            elif(filter.lower() == 'exponential'): #指数滤波器
                if(type == 'lp'):
                    mask[i, j] = np.exp(-(D/D0)**(2*N))
                elif(type == 'hp'):
                    mask[i, j] = np.exp(-(D0/D)**(2*N))
                elif(type == 'bs'):
                    mask[i, j] = np.exp(-(D*W/(D**2 - D0**2))**(2*N))
                elif(type == 'bp'):
                    mask[i, j] = np.exp(-((D**2 - D0**2)/D*W)**(2*N))
                else:
                    assert('type error')
           
    fshift = dtf_shift*mask

    f_ishift=np.fft.ifftshift(fshift)
    img_back=cv.idft(f_ishift)
    img_back=cv.magnitude(img_back[:,:,0],img_back[:,:,1]) #计算像素梯度的绝对值
    img_back=np.abs(img_back)
    img_back=(img_back-np.amin(img_back))/(np.amax(img_back)-np.amin(img_back))

    return img_back

def BWfilter1D(x, wn):
    sos_t = signal.butter(4, wn, 'lp', fs=1000, output='sos')
    sos_s = signal.butter(4, wn, 'lp', fs=1000, output='sos')
    
    spatial_filtered = []
    for frame in np.split(x, x.shape[0], 0):
        spatial_filtered.append(
            signal.sosfilt(sos_s, np.squeeze(frame))
        )
    spatial_filtered = np.stack(spatial_filtered, axis=0)

    filtered = []
    for frame in np.split(spatial_filtered, spatial_filtered.shape[1], 1):
        filtered.append(
            signal.sosfilt(sos_t, np.squeeze(frame))
        )
    filtered = np.stack(filtered, axis=1)
    return filtered
    

def filter_dataset(root_path, valid_shape, filter_param, method='1d'):
    dvs = np.load(root_path + "\\design_variables.npy")

    cf_folder = root_path + "\\force\\cf_npy\\"
    filtered_folder = root_path + "\\force\\filtered_cf\\"
    
    if not os.path.isdir(filtered_folder):
        os.makedirs(filtered_folder)

    valid_dvs = []
    for cf_file, dv in zip(os.listdir(cf_folder), np.split(dvs, dvs.shape[0], axis=0)):

        cf = np.load(cf_folder + cf_file)
        
        if cf.shape == valid_shape:
            valid_dvs.append(dv)
            filtered = []
            for channel in np.split(cf, 3, -1):
                if method == '2d':
                    single_filtered = BWfilter(np.squeeze(channel), D0=filter_param, type='lp', N=4)
                elif method == '1d':
                    single_filtered = BWfilter1D(np.squeeze(channel), wn=filter_param)
                filtered.append(single_filtered)
            filtered = np.stack(filtered, axis=-1)
            
            np.save(filtered_folder + cf_file, filtered)
            print(f" {cf_file}::{filtered.shape} has been filtered and saved")
        
        else:
            print(f" {cf_file} is an invalid file, skip")

    valid_dvs = np.concatenate(valid_dvs, axis=0)
    np.save(root_path + "\\valid_design_variables.npy", valid_dvs)
    print(f" valid dvs::{valid_dvs.shape} has been saved")


if __name__ == "__main__":

    for dataset in ['training', 'testing2', 'testing3']:
        filter_dataset(
            root_path=f"D:\\SpatiotemporalModelingCMAMERevision\\data\\case2\\{dataset}",
            valid_shape=(180,310,3),
            filter_param=20
        )
        
