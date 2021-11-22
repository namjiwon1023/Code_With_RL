import torch

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cuda'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)


        # loss = loss_fn(outputs, labels)
        # l1_lambda = 0.001
        # l1_norm = sum(p.abs().sum() for p in model.parameters())

        # loss = loss + l1_lambda * l1_norm


        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())

        # l2_lambda = 5e-4
        # l2_norm = torch.sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm



        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss=weight_decay*reg_loss
        return reg_loss

    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")



# # 检查GPU是否可用
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print("-----device:{}".format(device))
# print("-----Pytorch version:{}".format(torch.__version__))

# weight_decay=100.0 # 正则化参数

# model = my_net().to(device)
# # 初始化正则化
# if weight_decay>0:
#    reg_loss=Regularization(model, weight_decay, p=2).to(device)
# else:
#    print("no regularization")


# criterion= nn.CrossEntropyLoss().to(device) # CrossEntropyLoss=softmax+cross entropy
# optimizer = optim.Adam(model.parameters(),lr=learning_rate)#不需要指定参数weight_decay

# # train
# batch_train_data=...
# batch_train_label=...

# out = model(batch_train_data)

# # loss and regularization
# loss = criterion(input=out, target=batch_train_label)
# if weight_decay > 0:
#    loss = loss + reg_loss(model)
# total_loss = loss.item()

# # backprop
# optimizer.zero_grad()#清除当前所有的累积梯度
# total_loss.backward()
# optimizer.step()
# optimizer weight_decay置为0
# 此外更改参数p，如当p=0表示L2正则化，p=1表示L1正则化
# 策略网络正则化比较好， （Dropout和BatchNormal 存有异议）