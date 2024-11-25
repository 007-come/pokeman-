import csv
import os
import random

import torch
import glob
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms #一些常用的图片变换器
from PIL import Image #图片处理工具


"""
第一步：先将准备好的数据集分类建好文件夹
第二步：在加载数据类的初始化方法中根据分好类的文件夹建立对应的 类别：标签 imageToLabel字典中
第三步：用glob.glob()方法将每张图片的路径信息（根目录/类别/*.jpg）保存到一个image_route数组中
第四步：利用路径信息中的类别和imageToLabel，将图片路径信息和对应的标签保存到csv文件中
第五步：一行一行的读取csv文件，将所有的图片的路径信息和对应的标签信息分别保存到images和labels中返回

"""


"""
加载自定义的数据集
数据集的初始化
"""
class Pokemon(Dataset):
    """
    root:要加载数据的根目录
    resize：要将图片resize成多大尺寸的图片进行输入
    mode：当前要进行数据加载的模式是：train（训练集） validate（验证集） test（测试集）
    """
    def __init__(self,root,resize,mode):
        super(Pokemon,self).__init__()
        self.root=root
        self.resize=resize
        self.mode=mode

        # 创建标签信息
        self.nameToLabel={}#数据集中的各个类别对应的标签 皮卡丘：3
        # 这里要做的是将文件中的各种类别的角色对应一个标签
        # bulbasaur:0， charmander:1 mewtwo:2 pikachu:3 squirtle:4
        # sorted()先将根目录下的文件夹先进行排序（字母的顺序），再遍历二级目录名字
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root,name)):
                continue
            # 文件夹中第一个加载进来的角色类别 bulbasaur:0， charmander:1
            self.nameToLabel[name]=len(self.nameToLabel.keys())

            # {'bulbasaur': 0, 'charmander': 1, 'mewtwo': 2, 'pikachu': 3, 'squirtle': 4}
        print(self.nameToLabel)

        #得到图片路径值和对应的标签值
        self.images,self.labels=self.load_csv(filename='images.csv')

        # 对图片进行裁剪操作，不同模式下的图片数量不同
        if mode=='train':#60%
            self.images=self.images[:int(0.6*len(self.images))]
            self.labels=self.labels[:int(0.6*len(self.labels))]
        elif mode=='val': #20%
            self.images=self.images[int(0.6*len(self.images)):int(0.8*len(self.images))]
            self.labels=self.labels[int(0.6*len(self.labels)):int(0.8*len(self.labels))]
        else: #20%
            self.images=self.images[int(0.8*len(self.images)):]
            self.labels=self.labels[int(0.8*len(self.labels)):]




    # 将我们数据集中的图片的路径信息保存在filename文件家中,返回图片路径信息和对应的标签信息的数组
    def load_csv(self,filename):
        if not os.path.exists(os.path.join(self.root,filename)):
            images=[] #保存图片的路径信息
            for name in self.nameToLabel:
                # glob作用文件匹配：根据给定的模式匹配文件名，返回符合条件的文件路径列表。
                #        支持通配符：可以使用通配符（如 *、? 和 []）来匹配文件名。
                images+=glob.glob(os.path.join(self.root,name,'*.jpg'))
                images+=glob.glob(os.path.join(self.root,name,'*.png'))
                images+=glob.glob(os.path.join(self.root,name,'*.jpeg'))


            # 1167 ['pokeman\\bulbasaur\\00000006.jpg'
            print(len(images),images)

            # 这个方法不返回新的列表，而是直接修改传入的可变序列（如列表）。
            # 将保存的图片地址信息打乱，使其变得没有规律，方便训练
            random.shuffle(images)
            #将图片路径信息和对应的标签保存到root/filename路径下
            with open(os.path.join(self.root,filename),mode='w',newline='') as f:
                #这里采用csv的方法写进文件  路径,标签 的方式
                write=csv.writer(f)
                for img in images: #img:pokeman\\bulbasaur\\00000006.jpg
                    name=img.split(os.sep)[-2]  #os.seq 代表路径分隔符，防止操作系统不同，路径分隔符不同而报错
                    label=self.nameToLabel[name]
                    write.writerow([img,label])

                    print('write into csv file',filename)

        #将写入csv文件中的图片路径信息和标签信息分别读出，存储到images 和 labels中去
        images, labels = [], []
        with open(os.path.join(self.root,filename),'r') as f:
            reader=csv.reader(f)
            for row in reader:
                # row pokeman\\bulbasaur\\00000006.jpg,0
                #  每一行都会被打印为一个列表
                # print('每一行对应元素：',row) #每一行对应元素： ['pokeman\\mewtwo\\00000032.jpg', '2']
                img,label=row
                label=int(label)
                images.append(img)
                labels.append(label)
        #用于进行调试和验证程序的假设。它的主要作用是检查一个条件是否为真，
        # 如果条件为假，则会引发 AssertionError 异常。
        assert len(images)==len(labels)

        return images,labels







    def __len__(self):

        return len(self.images)
    def denormalize(self,x_hat):
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
        mean=torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std=torch.tensor(std).unsqueeze(1).unsqueeze(1)
        x=x_hat*std+mean
        return x

    def __getitem__(self, idx):
        """
        第一步：先将对应索引的图片和对应标签取出
        第二步：再利用torchvision的transforms对图片进行处理【得到图片数据，resize】
        """
        # img:pokeman\\bulbasaur\\00000006.jpg   lab:0
        # idx~[0,len(self.images)]
        img,lab=self.images[idx],self.labels[idx]
        #tranforms图片变换器   根据图片路径信息加载图片
        tf=transforms.Compose([
            lambda x:Image.open(x).convert('RGB'),#根据图片路径-》image data
            transforms.Resize((int(self.resize*1.25),int(self.resize*1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),#先将图片放大1.25倍，然后进行旋转15度，最后进行中心裁剪。这样不会使旋转后的图片出现黑点
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])  #这里标准化的均值和方差是统计imageNet得出来的


        ])
        img=tf(img)
        label=torch.tensor(lab)
        return img,label



def main():
    import visdom
    import time
    import torchvision


    #方案一：如果我们的建立的图片文件符合二级目录，我们可以使用
    # torchvision.datasets.ImageFolder(root='pokeman',transform=tf)
    # 直接利用根目录将图片加载进来，而不需要一步一步的将各目录下的图片和标签分类
    # viz = visdom.Visdom()
    # tf=transforms.Compose([
    #     transforms.Resize((64,64)),
    #     transforms.ToTensor()
    # ])
    # db=torchvision.datasets.ImageFolder(root='pokeman',transform=tf)
    # print(db.class_to_idx)
    # loader=DataLoader(db,batch_size=32,shuffle=True) #DataLoader主要的作用是批量记载图片数据
    # for x,y in loader:
    #     viz.images(x,nrow=8,win='batch',opts=dict(title='batch'))
    #     viz.text(str(y.numpy()),win='label',opts=dict(title='batch-y'))
    #     time.sleep(10)

    viz=visdom.Visdom()
    pokeman = Pokemon('pokeman', 64, 'train')
    x,y=next(iter(pokeman)) #将输出的数据变成可迭代对象，然后取出第一个
    # for x,y in iter(pokeman):
    #     viz.images(pokeman.denormalize(x), win='sample_x', opts=dict(title='sample_x'))
    #     time.sleep(2)

    print('sample:',x.shape,y.shape,y)
    viz.images(pokeman.denormalize(x),win='sample_x',opts=dict(title='sample_x'))

    #一个一个batch的数据加载
    # loader=DataLoader(pokeman,batch_size=32,shuffle=True,num_workers=8)#num_workers=8以8个线程去取数据，而不是一个一个去取

    for x,y in pokeman:
        viz.images(pokeman.denormalize(x),nrow=8,win='batch',opts=dict(title='batch'))
        viz.text(str(y.numpy()),win='label',opts=dict(title='batch-y'))
        time.sleep(10)

if __name__ == '__main__':
    main()