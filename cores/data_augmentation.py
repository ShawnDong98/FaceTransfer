import numpy as np
import cv2

from umeyama import umeyama

random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w//2, h//2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(
        image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


# get pair of random warped images from aligened face image
def random_warp(image, res=64):
    assert image.shape == (256,256,3)
    res_scale = res//64
    assert res_scale >= 1, f"Resolution should be >= 64. Recieved {res}."
    
    interp_param = 80 * res_scale
     
    interp_slice = slice(interp_param//10,9*interp_param//10)
    dst_pnts_slice = slice(0,65*res_scale,16*res_scale)

    
    # rand_coverage = np.random.randint(20) + 78 # random warping coverage
    rand_coverage = 100
    # 标准差
    rand_scale = np.random.uniform(5., 6.2) # random warping scale
    
    # 这里产生的是坐标位置 
    range_ = np.linspace(128-rand_coverage, 128+rand_coverage, 5)
    # 产生25个点， 值在128-rand_coverage到128+rand_coverage之间，步长为5
    # 按行顺序从小到大
    mapx = np.broadcast_to(range_, (5,5))
    # 转置，按列顺序从小到大
    mapy = mapx.T
    
    # 在原来行的基础上按元素加上随机产生的均值为0标准差为rand_scale的25个点
    # np.random.normal: loc: mean, scale: std
    mapx = mapx + np.random.normal(size=(5,5), scale=rand_scale)
    mapy = mapy + np.random.normal(size=(5,5), scale=rand_scale)
    
    # 将生成的25个点resize成interp_param， 再截取中间interp_slice
    interp_mapx = cv2.resize(mapx, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
    interp_mapy = cv2.resize(mapy, (interp_param,interp_param))[interp_slice,interp_slice].astype('float32')
    
    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)
    
    # 在最后一维加维度: (5x5, 2)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    # shape: (2, 5, 5) -> (5, 5, 2) -> (25, 2)
    # 左闭右开区间
    # 相当于还是分出来了5个点， 对应之前的五个点
    dst_points = np.mgrid[dst_pnts_slice,dst_pnts_slice].T.reshape(-1,2)
    # 这个函数返回了一个加了噪声的点云和没加噪声的点云之间的仿射变换矩阵
    # 类似于getAffineTransform
    mat = umeyama(src_points, dst_points, True)[0:2]
    
    # 将原图又还原回来
    target_image = cv2.warpAffine(image, mat, (res,res))

    return warped_image, target_image

    

if __name__ == "__main__":
    img = cv2.imread(r"D:\GoogleDrive1\FaceTransfer\00000267.jpg")
    # print(img)
    warped_image, target_image = random_warp(img)
    for i in warped_image.shape:
        print(i)

    
    target_image = cv2.copyMakeBorder(target_image,8,8,8,8,cv2.BORDER_CONSTANT,value=0)
    
    cv2.imshow("test", target_image)
    cv2.waitKey(0)

    # mask = np.zeros([256, 256], dtype=np.float)
    # for i in range(256):
    #     for j in range(256):
    #         dist = np.sqrt((i-128)**2 + (j-128)**2)/128
    #         dist = np.minimum(dist, 1)
    #         mask[i, j] = 1-dist
    # mask = cv2.dilate(mask, None, iterations=20)
    # mask = np.expand_dims(mask, 2)

    # dst = img * mask/255.
    # for i in mask.shape:
    #     print(i)

    # cv2.imshow("mask", dst)
    # cv2.waitKey(0)