import torch
from torchvision import transforms, datasets
from torch.utils import data

from PIL import Image
import glob

from data_augmentation import *

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
