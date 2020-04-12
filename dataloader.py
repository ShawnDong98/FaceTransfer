import torch
from torchvision import transforms, datasets
from torch.utils import data

from PIL import Image
import glob
import numpy as np
import cv2


def umeyama( src, dst, estimate_scale ):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}

def random_transform( image, rotation_range, zoom_range, shift_range, random_flip ):
    h,w = image.shape[0:2]
    rotation = np.random.uniform( -rotation_range, rotation_range )
    scale = np.random.uniform( 1 - zoom_range, 1 + zoom_range )
    tx = np.random.uniform( -shift_range, shift_range ) * w
    ty = np.random.uniform( -shift_range, shift_range ) * h
    mat = cv2.getRotationMatrix2D( (w//2,h//2), rotation, scale )
    mat[:,2] += (tx,ty)
    result = cv2.warpAffine( image, mat, (w,h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:,::-1]
    return result



# get pair of random warped images from aligened face image
def random_warp(image):
    assert image.shape == (256,256,3)
    range_ = np.linspace( 128-80, 128+80, 5 )
    mapx = np.broadcast_to(range_, (5,5))
    mapy = mapx.T

    # np.random.normal: loc: mean, scale: std
    mapx = mapx + np.random.normal(size=(5,5), scale=5)
    mapy = mapy + np.random.normal(size=(5,5), scale=5)

    interp_mapx = cv2.resize(mapx, (160,160))[16:144,16:144].astype('float32')
    interp_mapy = cv2.resize(mapy, (160,160))[16:144,16:144].astype('float32')

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    # 在最后一维加维度: (5x5, 2)
    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    # shape: (2, 5, 5) -> (5, 5, 2) -> (25, 2)
    # 左闭右开区间
    dst_points = np.mgrid[0:129:32,0:129:32].T.reshape(-1,2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    # 将原图又还原回来
    target_image = cv2.warpAffine(image, mat, (128,128))

    return warped_image, target_image

class Dataset(data.Dataset):
    def __init__(self, root, transforms=None):
        self.imgs = glob.glob(root+'/*')
        self.trans = transforms
        

    def __getitem__(self, index):
        path = self.imgs[index]
        img = Image.open(path)
        img = random_transform(np.array(img), **random_transform_args)
        warped_image, target_image = random_warp(img)
        warped_image = self.trans(Image.fromarray(warped_image))
        target_image = self.trans(Image.fromarray(target_image))

        return warped_image, target_image

    def __len__(self):
        return len(self.imgs)

class Data_Loader():
    def __init__(self, img_size, img_path, batch_size):
        super(Data_Loader, self).__init__()
        self.img_size = img_size
        self.img_path = img_path
        self.batch_size = batch_size
        self.trans = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)), 
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        )
        self.dataset = Dataset(img_path, self.trans)
        

    def loader(self):
        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        return loader


# t = np.mgrid[0:129:32,0:129:32].T.reshape(-1, 2)
# print(t)
# for i in t.shape:
#     print(i)

# path = r"D:\Deepfake\data\data_src\sqys\lyf\00181_0.jpg"
# img = cv2.imread(path)
# warped_image, target_image = random_warp(img)
# for i in warped_image.shape:
#      print(i)
# cv2.imshow("warped_image", warped_image)
# cv2.imshow("target_image", target_image)
# cv2.waitKey(0)





