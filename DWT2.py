import traceback
import time

import numpy as np 
import matplotlib.pyplot as plt

from PIL import Image

import pywt
import pywt.data

#bior1.3
const_dwt_method = 'bior1.3'

def dwt_act(img):
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    coeffs = pywt.dwt2(img, const_dwt_method)
    LL, (LH, HL, HH) = coeffs
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.tight_layout()
    plt.show()

    return LL, (LH, HL, HH)

def dwt(img):
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    coeffs = pywt.dwt2(img, const_dwt_method)
    LL, (LH, HL, HH) = coeffs
    
    return LL, (LH, HL, HH)

def np_info(A, title: str):
    print(f'Title : {title}')
    print(np.shape(A))
    print(f'Width: {len(A[0])} / Height: {len(A)}' )

#Read Image
im = Image.open("C:/Users/shinj/Desktop/test.JPEG")
#Convert to Grey Scale
im_grey = im.convert('L')

#im.show()
#im_grey.show()



try:
    img_data = np.array(im_grey)
    np_info(img_data, str('original'))
    LL, cD = dwt(img_data) #Lev 1
    print('type-LL ',type(LL))
    print('type-image_data ',type(img_data))
    print('shape-LL ',np.shape(LL))
    print('shape-image_data ',np.shape(img_data))

    LL, cD = dwt(LL) # Lev 2
    LL, cD = dwt(LL) # Lev 3

    print(f'original H-SUM --> {np.sum(cD)}')
    #LL with Original H-F
    A = pywt.idwt2((LL, cD), const_dwt_method)
    np_info(A, str('idwt2'))

    temp = cD[0]
    art = np.zeros((len(temp), len(temp[0])))
    cD = (art,art,art)

    #LL with zeros
    B = pywt.idwt2((LL, cD), const_dwt_method)
    np_info(B, str('idwt2'))

    print(np.array_equal(A,B))

    LL2, cD2 = dwt_act(B)
    print(f'second H-SUM --> {np.sum(cD2)}')

    '''
    print(img_data[0][0:10])
    print(A[0].astype('int32')[0:10])
    print(B[0].astype('int32')[0:10])

    print(img_data[1][0:10])
    print(A[1].astype('int32')[0:10])
    print(B[1].astype('int32')[0:10])
    '''
    
    
    
    #img_ORI = Image.fromarray(img_data)
    #img_B = Image.fromarray(B)
    
    #img_ORI.show()
    #img_A.show()
    #img_B.show()
except:
    print(traceback.format_exc())
    while True:
        pass
while True:
    pass