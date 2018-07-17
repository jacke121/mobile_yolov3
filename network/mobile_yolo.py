import torch
import torch.nn as nn
import network.mobilenet as mobilenet


class Mobile_YOLO(nn.Module):
    def __init__(self, config, is_training):
        super(Mobile_YOLO, self).__init__()
        self.training = is_training
        #  backbone
        self.backbone = mobilenet.mobilenetv2(config.backbone_pretrained)
        _out_filters = self.backbone.layers_out_filters
        #  embedding0
        self.embedding0_adj = nn.Sequential(
            nn.Conv2d(_out_filters[-1], 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
        )
        final_out_filter0 = len(config.anchors[0]) * (5 + config.classes_num)
        self.embedding0 = self._make_embedding(1024, 256, final_out_filter0)  # 1024 256 255
        #  embedding1
        self.embedding1_adj = nn.Sequential(
            nn.Conv2d(_out_filters[-2], 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.embedding1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        final_out_filter1 = len(config.anchors[1]) * (5 + config.classes_num)
        self.embedding1 = self._make_embedding(512, 128, final_out_filter1)
        #  embedding2
        self.embedding2_adj = nn.Sequential(
            nn.Conv2d(_out_filters[-3], 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )
        self.embedding2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        final_out_filter2 = len(config.anchors[2]) * (5 + config.classes_num)
        self.embedding2 = self._make_embedding(256, 64, final_out_filter2)

    def _make_embedding(self, in_filter, middle_filter, out_filter):
        module = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_filter, middle_filter, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_filter),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(middle_filter, middle_filter, kernel_size=3, padding=1, groups=middle_filter, bias=False),
                nn.BatchNorm2d(middle_filter),
                nn.ReLU6(),
            ),
            nn.Sequential(
                nn.Conv2d(middle_filter, in_filter, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_filter),
            ),
            nn.Conv2d(in_filter, out_filter, kernel_size=1))
        return module

    def forward(self, x):
        def _branch(_embedding, _in):
            for i, e in enumerate(_embedding):
                _in = e(_in)
                if i == 0:
                    out_branch = _in
            return _in, out_branch

        #  backbone
        x52, x26, x13 = self.backbone(x)
        #  yolo branch 0
        x13 = self.embedding0_adj(x13)
        out0, out0_branch = _branch(self.embedding0, x13)
        #  yolo branch 1
        x26 = self.embedding1_adj(x26)
        x1_in = self.embedding1_upsample(out0_branch)
        x1_in = torch.cat([x1_in, x26], 1)
        out1, out1_branch = _branch(self.embedding1, x1_in)
        #  yolo branch 2
        x52 = self.embedding2_adj(x52)
        x2_in = self.embedding2_upsample(out1_branch)
        x2_in = torch.cat([x2_in, x52], 1)
        out2, out2_branch = _branch(self.embedding2, x2_in)
        return out0, out1, out2


if __name__ == "__main__":
    import training.config_train as config

    m = Mobile_YOLO(config)
    x = torch.randn(1, 3, 416, 416)
    y0, y1, y2 = m(x)
    print(y0.size())
    print(y1.size())
    print(y2.size())
