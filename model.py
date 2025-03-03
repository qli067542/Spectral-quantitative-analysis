class MBML_Net(nn.Module):
    def __init__(self):
        super(Con1, self).__init__()
        self.conv_in = nn.Conv1d(1, 16, 3, 1)
        self.conv1x1 = nn.Conv1d(16, 16, 3, 1, padding=0)
        self.conv1x1_1 = nn.Conv1d(16, 16, 3, 1, padding=2)
        self.conv1x1_2 = nn.Conv1d(16, 16, 3, 1, padding=2)
        self.conv1x1_3 = nn.Conv1d(16, 16, 3, 1, padding=2)
        self.conv1x1_4 = nn.Conv1d(16, 16, 3, 1, padding=2)
        self.conv1x1_5 = nn.Conv1d(16, 16, 3, 1, padding=2)
        self.conv1x1_6 = nn.Conv1d(16, 16, 3, 1, padding=2)
        self.conv1x1_7 = nn.Conv1d(16, 16, 3, 1, padding=1)
        self.conv1x1_8 = nn.Conv1d(16, 16, 3, 1, padding=1)
        self.BN = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1808, 1)
        self.flat = nn.Flatten()
        self.Dropout = nn.Dropout(0.2)
    def forward(self, x):
        x = self.conv_in(x)
        x = self.BN(x)
        xc = self.conv1x1_1(x)
        xc = self.leaky_relu(xc)
        x11 = xc
        xc = self.leaky_relu(self.conv1x1_2(x11))
        x22 = xc
        xc = self.leaky_relu(self.conv1x1_3(x22))
        x33 = xc
        xc = self.leaky_relu(self.conv1x1_4(x33))
        x44 = xc
        xc = self.leaky_relu(self.conv1x1_5(x44))
        x55 = xc
        xc = self.leaky_relu(self.conv1x1_6(x55))
        x66 = xc
        x1 = self.conv1x1(x11)
        x2 = self.conv1x1(x22)
        x3 = self.conv1x1(x33)
        x4 = self.conv1x1(x44)
        x5 = self.conv1x1(x55)
        x6 = self.conv1x1(x66)
        out = torch.cat((x1, x2, x3, x4, x5, x6, x6), dim=1)
        out = self.Dropout(out)
        out = self.flat(out)
        out = self.fc(out)

        return out