import torch
import visdom
import torchvision
from torch import nn,optim
from torch.utils.data import DataLoader
from pokemon import Pokemon
from resnet import ResNet18

batchsz=32
lr=1e-3
epochs=10
device=torch.device('cpu')
torch.manual_seed(1234) #设置随机数种子，方便代码复现



train_db = Pokemon('pokeman', 224, 'train')
print(len(train_db))
validation_db = Pokemon('pokeman', 224, 'val')
test_db = Pokemon('pokeman', 224, 'test')
train_db = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
validation_db = DataLoader(validation_db, batch_size=batchsz, num_workers=2)
test_db = DataLoader(test_db, batch_size=batchsz, shuffle=True, num_workers=2)

viz=visdom.Visdom()

#验证集：找到训练过程中表现最好的参数，并将其记下来
def validation(model,db):
    """
    model:网络模型
    db:验证集的数据集
    """
    correct=0  #正确率
    total_len=len(db.dataset) #数据集的总长度

    for x,y in validation_db:
        logist=model(x)
        with torch.no_grad():
            pre=logist.argmax(dim=1)
        correct+=torch.eq(pre,y).sum().float().item()

    return correct/total_len




def main():
    res = ResNet18(5)
    criteon = nn.CrossEntropyLoss()  # 损失函数
    optimizer = optim.Adam(res.parameters(), lr=lr)  # 优化器
    best_epoch,best_accurate=0,0
    glo_step=0
    # 初始状态
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))
    for epoch in range(epochs):
        for step,(x,y) in enumerate(train_db):
            logist=res(x)
            loss=criteon(logist,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            viz.line([loss.item()], [glo_step], win='loss', update='append')
            glo_step+=1


        if epoch % 2==0:
            val_correct=validation(res,validation_db)
            viz.line([val_correct], [glo_step], win='val_acc', update='append')
            if val_correct>best_accurate:
                best_accurate=val_correct
                best_epoch=epoch
                torch.save(res.state_dict(),'best.model')

    print('best_epoch:',best_epoch,'best_accurate:',best_accurate)

    res.load_state_dict(torch.load('best.model'))
    print('loaded from validation!')
    test_acc=validation(res,test_db) #在验证集得到的最好模型下运行test数据集
    print('test_acc',test_acc)





if __name__ == '__main__':
    main()