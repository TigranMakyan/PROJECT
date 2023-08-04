import torch.nn as nn

class Head_Bottleneck(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel, padding, lin_input, lin_hidden, lin_next) -> None:
        
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dr1 = nn.Dropout(0.1)
        self.pool1 = nn.AvgPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dr2 = nn.Dropout(0.1)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        
        
        self.lin1 = nn.Linear(lin_input, lin_hidden)
        self.lin_bn1 = nn.BatchNorm1d(lin_hidden)
        self.lin_relu1 = nn.ReLU()
        self.lin2 = nn.Linear(lin_hidden, lin_next)
        self.lin_bn2 = nn.BatchNorm1d(lin_next)
        self.lin_relu2 = nn.ReLU()
        self.lin3 = nn.Linear(lin_next, 1)
        

    def forward(self, memory):
        x = self.conv1(memory)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dr1(x)
        x = self.pool1(x)
       # print('arajin convnet: ', x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dr2(x)
        x = self.pool2(x)
       # print('Aper convnety anca')
        x = x.flatten(1)
       # print(x.shape)
        x = self.lin1(x)
        x = self.lin_bn1(x)
        x = self.lin_relu1(x)
        x = self.lin2(x)
        x = self.lin_bn2(x)
        x = self.lin_relu2(x)

        result = self.lin3(x)
        return result

def build_head(cfg):
    print("Building head block...")
    head = Head_Bottleneck(input_dim=cfg.MODEL.HEAD.CONVNET.INPUT_DIM, 
        hidden_dim=cfg.MODEL.HEAD.CONVNET.HIDDEN_DIM, 
        output_dim=cfg.MODEL.HEAD.CONVNET.OUTPUT_DIM,
        kernel=cfg.MODEL.HEAD.CONVNET.KERNEL,
        padding=cfg.MODEL.HEAD.CONVNET.PADDING,
        lin_input=cfg.MODEL.HEAD.LINEARNET.INPUT_DIM,
        lin_hidden=cfg.MODEL.HEAD.LINEARNET.HIDDEN_DIM,
        lin_next=cfg.MODEL.HEAD.LINEARNET.NEXT_DIM)

    return head


