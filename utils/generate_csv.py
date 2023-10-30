import numpy as np

# 这个函数是对mask进行rle编码,所以输入的值非0即1
def mask2rle(img):
    '''
    Convert mask to rle.
    img: numpy array,
    1 - mask,
    0 - background

    Returns run length as string formated
    '''
    # print("看下输入的img", img)
    pixels = img.T.flatten()  # 转置后看图像
    # print("pixels进行flatten以后=", pixels)
    # pixels进行flatten以后= [1 1 0 0 0 0 0 0 0 0 0 0 1 1]#14位
    pixels = np.concatenate([[0], pixels, [0]])
    # print("pixels=", pixels)
    #                 pixels = [0 1 1 0 0 0 0 0 0 0 0 0 0 1 1 0]#16位
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # print("runs=", runs)  # 这个记录的是bit值开始变化的位置,这里+1是为了位置的调整
    runs[1::2] -= runs[::2]
    # 这句代码写得很抽象,其实是在进行编码.
    # 运行前的结果是：
    # runs= [ 1  3 13  15]   #runs中的每个数值都代表像素值发生变化的位置
    # 运行后的结果是:
    # runs= [ 1  2 13  2]
    # 意思是第1个位置算起，共有2个bit是相同的，所以用3-1得到
    # 意思是第13个位置算起，共有2个bit是相同的，所以用15-13得到。
    # 对应上面头部和末尾的两个11

    # print("runs=", runs)
    return ' '.join(str(x) for x in runs)


# 这个是用来解码train.csv中的Encoded Pixels的
def rle_decode(mask_rle: str = '', shape: tuple = (1400, 2100)):
    '''
    Decode rle encoded mask.

    :param mask_rle: run-length as string formatted (start length)
    :param shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()  # 这个运算前后没啥区别
    # print("-----------------------------------------------------------")
    # print("s[0:][::2]=", s[0:][::2])  # 这个获取的是变化的像素的位置序号的列表
    # ['1', '13']
    # print("s[1:][::2]=", s[1:][::2])  # 这个获取的是相同像素的长度列表（分别记录每个变化的像素后面连续的同等像素值的连续长度）
    # ['2', '2']

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # print("看下最初的starts=", starts)  # 变化的像素的位置序号的列表
    # print("lengths=", lengths)
    starts -= 1
    ends = starts + lengths
    # print("ends=", ends)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):  # 进行恢复
        img[lo:hi] = 1
    return img.reshape(shape, order='F')



