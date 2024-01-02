
import torch
import torch.nn as nn
from tdcnpp import TDCNpp, MaskNetworkPlus


class EmbeddingTDCNpp(nn.Module):
    def __init__(self, emb_dim, enc_dim, enc_ker, conv_in_channels, conv_out_channels, conv_ker, n_blocks, n_repeats, input_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.enc_stride = enc_ker//2
        self.input_size = input_size

        self.encoder = nn.Conv1d(1, enc_dim, enc_ker, stride=self.enc_stride, bias=False)
        self.en_time = (input_size - enc_ker) // self.enc_stride + 1

        self.embed_net = MaskNetworkPlus(enc_dim, 
                                        enc_ker, 
                                        conv_in_channels, 
                                        conv_out_channels, 
                                        conv_ker, 
                                        n_blocks, 
                                        n_repeats, 
                                        1, 
                                        False)
        # self.extractor = nn.Linear(enc_dim*self.en_time, enc_dim)
        self.extractor = nn.Conv1d(self.en_time, 1, 1, bias=True)
        # f_extractor = 64
        # self.extractor = nn.Sequential( nn.Conv2d(1, f_extractor, 3, stride=2),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(f_extractor, f_extractor*2, 3, stride=2),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(f_extractor*2, f_extractor*4, 3, stride=2),
        #                                 nn.ReLU(),
        #                                 nn.Conv2d(f_extractor*4, enc_dim, 3, stride=2),
        #                                 nn.ReLU(),
        #                                 )
        # self.fc = nn.Sequential(nn.BatchNorm1d(enc_dim), 
        #                         nn.Linear(enc_dim, enc_dim),
        #                         nn.Dropout(0.5),
        #                         nn.ReLU()
        #                         )
        
        self.embedding = nn.Linear(self.enc_dim, self.emb_dim) 


    def _align_num_frames_with_strides(self, input: torch.Tensor):
            batch_size, num_channels, num_frames = input.shape
            is_odd = self.enc_ker % 2
            num_strides = (num_frames - is_odd) // self.enc_stride
            num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
            if num_remainings == 0:
                return input, 0

            num_paddings = self.enc_stride - num_remainings
            pad = torch.zeros(
                batch_size,
                num_channels,
                num_paddings,
                dtype=input.dtype,
                device=input.device,
            )
            return torch.cat([input, pad], 2), num_paddings

    def forward(self, x):
        padded, num_pads = self._align_num_frames_with_strides(x)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        transformed = self.encoder(padded) # B, F, M

        mix_emb, _ = self.embed_net(transformed)  # B, 1, F, M

        # for conv extractor
        # # mix_emb = mix_emb.squeeze(1)  # B, F, M
        # mix_emb = self.extractor(mix_emb) # B, 1, F, M
        # # mix_emb = mix_emb.mean(-1) # global pooling over time B, F
        # mix_emb = mix_emb.mean((-1, 2)) # global pooling over H and W -> B, F
        # mix_emb = self.embedding(mix_emb) # B, E
        
        mix_emb = mix_emb.squeeze(1)  # B, F, M
        mix_emb = mix_emb.permute(0, 2, 1) # B, M, F in order to convolve along M
        mix_emb = self.extractor(mix_emb) # B, 1, F
        mix_emb = nn.ReLU()(mix_emb)
        mix_emb = mix_emb.squeeze(1) # B, F
        mix_emb = self.embedding(mix_emb) # B, E

        mix_emb = nn.functional.normalize(mix_emb, p=2, dim=1, eps=1e-5)

        return mix_emb


class EmbeddingTDCNpp2(nn.Module):
    def __init__(self, emb_dim,
                 enc_dim, enc_ker, 
                 conv_in_channels,
                 conv_out_channels, 
                 conv_ker, 
                 n_blocks, 
                 n_repeats, 
                 input_size):
        
        super().__init__()
        self.emb_dim = emb_dim

        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.enc_stride = enc_ker//2
        self.input_size = input_size

        self.encoder = nn.Conv1d(1, enc_dim, enc_ker, stride=self.enc_stride, bias=False)
        self.en_time = (input_size - enc_ker) // self.enc_stride + 1

        self.embed_net = MaskNetworkPlus(enc_dim, 
                                        enc_ker, 
                                        conv_in_channels, 
                                        conv_out_channels, 
                                        conv_ker, 
                                        n_blocks, 
                                        n_repeats, 
                                        1, 
                                        False)
        self.extractor = nn.Sequential(
             nn.BatchNorm1d(self.en_time),
             nn.Conv1d(self.en_time, 1, 1, bias=True),
             nn.ReLU()
        )
        self.hidden = nn.Sequential(
             nn.Dropout(0.3),
             nn.Linear(enc_dim, enc_dim),
             nn.ReLU(),
        )
        self.embedding_species = nn.Linear(self.enc_dim, self.emb_dim) 
        # self.embedding_family = nn.Linear(self.enc_dim, self.emb_dim)
        # self.embedding_order = nn.Linear(self.enc_dim, self.emb_dim)

    def _align_num_frames_with_strides(self, input: torch.Tensor):
            batch_size, num_channels, num_frames = input.shape
            is_odd = self.enc_ker % 2
            num_strides = (num_frames - is_odd) // self.enc_stride
            num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
            if num_remainings == 0:
                return input, 0

            num_paddings = self.enc_stride - num_remainings
            pad = torch.zeros(
                batch_size,
                num_channels,
                num_paddings,
                dtype=input.dtype,
                device=input.device,
            )
            return torch.cat([input, pad], 2), num_paddings

    def _apply_extractor(self, x):
        x = x.squeeze(1)
        x = x.permute(0, 2, 1) # B, M, F in order to convolve along M
        x = self.extractor(x) # B, 1, F
        x = x.squeeze(1) # B, F
        return x

    def forward(self, x):
        padded, num_pads = self._align_num_frames_with_strides(x)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        transformed = self.encoder(padded) # B, F, M

        features, _ = self.embed_net(transformed)  # B, 1, F, M

        features = self._apply_extractor(features)
        features = self.hidden(features) # B, F
        emb = self.embedding_species(features) # B, E

        # emb_f = self.embedding_family(features) # B, E_f
        # emb_o = self.embedding_order(features) # B, E_o

        emb = nn.functional.normalize(emb, p=2, dim=1, eps=1e-5)
        # emb_f = nn.functional.normalize(emb_f, p=2, dim=1, eps=1e-5)
        # emb_o = nn.functional.normalize(emb_o, p=2, dim=1, eps=1e-5)

        # return emb, emb_f, emb_o
        return emb




class EmbeddingTDCNpp3(nn.Module):
    def __init__(self, emb_dim,
                 enc_dim, enc_ker, 
                 conv_in_channels,
                 conv_out_channels, 
                 conv_ker, 
                 n_blocks, 
                 n_repeats, 
                 input_size):
        
        super().__init__()
        self.emb_dim = emb_dim

        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.enc_stride = enc_ker//2
        self.input_size = input_size

        self.encoder = nn.Conv1d(1, enc_dim, enc_ker, stride=self.enc_stride, bias=False)
        self.en_time = (input_size - enc_ker) // self.enc_stride + 1

        self.embed_net = MaskNetworkPlus(enc_dim, 
                                        enc_ker, 
                                        conv_in_channels, 
                                        conv_out_channels, 
                                        conv_ker, 
                                        n_blocks, 
                                        n_repeats, 
                                        1, 
                                        False)
        # self.extractor = nn.Sequential(
        #      nn.BatchNorm1d(self.enc_dim),
        #      nn.Conv1d(self.enc_dim, enc_dim//2, enc_ker, bias=True),
        #      nn.ReLU()
        # )

        self.extractor = nn.Sequential(
             nn.BatchNorm1d(self.enc_dim),
             nn.Conv1d(self.enc_dim, enc_dim//2, 8, stride=4, bias=True),
             nn.ReLU()
        ) 

        # f_extractor = 64
        # self.extractor = nn.Sequential( nn.Conv2d(1, f_extractor, 3, stride=1),
        #                                 nn.ReLU(),
        #                                 nn.MaxPool2d(2),
        #                                 nn.Conv2d(f_extractor, f_extractor*2, 3, stride=1),
        #                                 nn.ReLU(),
        #                                 nn.MaxPool2d(2),
        #                                 nn.Conv2d(f_extractor*2, enc_dim//2, 3, stride=1),
        #                                 nn.ReLU(),
        #                                 )
        
        self.hidden = nn.Sequential(
             nn.Dropout(0.3),
             nn.Linear(enc_dim//2, enc_dim//2),
             nn.ReLU(),
        )
        self.embedding_species = nn.Linear(self.enc_dim//2, self.emb_dim) 
        # self.embedding_family = nn.Linear(self.enc_dim, self.emb_dim)
        # self.embedding_order = nn.Linear(self.enc_dim, self.emb_dim)

    def _align_num_frames_with_strides(self, input: torch.Tensor):
            batch_size, num_channels, num_frames = input.shape
            is_odd = self.enc_ker % 2
            num_strides = (num_frames - is_odd) // self.enc_stride
            num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
            if num_remainings == 0:
                return input, 0

            num_paddings = self.enc_stride - num_remainings
            pad = torch.zeros(
                batch_size,
                num_channels,
                num_paddings,
                dtype=input.dtype,
                device=input.device,
            )
            return torch.cat([input, pad], 2), num_paddings

    def _apply_extractor(self, x):
        # x  B, 1, F,M
        x = x.squeeze(1) # B, F, M
        x = self.extractor(x) # B, 2F, M'
        x = x.mean(-1) # B, 2F
        # print(x.shape)
        return x

    def forward(self, x):
        padded, num_pads = self._align_num_frames_with_strides(x)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        transformed = self.encoder(padded) # B, F, M

        features, _ = self.embed_net(transformed)  # B, 1, F, M

        features = self._apply_extractor(features)
        features = self.hidden(features) # B, F
        emb = self.embedding_species(features) # B, E

        # emb_f = self.embedding_family(features) # B, E_f
        # emb_o = self.embedding_order(features) # B, E_o

        emb = nn.functional.normalize(emb, p=2, dim=1, eps=1e-5)
        # emb_f = nn.functional.normalize(emb_f, p=2, dim=1, eps=1e-5)
        # emb_o = nn.functional.normalize(emb_o, p=2, dim=1, eps=1e-5)

        # return emb, emb_f, emb_o
        return emb


class EmbeddingTDCNpp4(nn.Module):
    def __init__(self, emb_dim,
                 enc_dim, enc_ker, 
                 conv_in_channels,
                 conv_out_channels, 
                 conv_ker, 
                 n_blocks, 
                 n_repeats, 
                 input_size,
                 norm=True):
        
        super().__init__()
        self.emb_dim = emb_dim

        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.enc_stride = enc_ker//2
        self.input_size = input_size

        self.norm = norm

        self.encoder = nn.Conv1d(1, enc_dim, enc_ker, stride=self.enc_stride, bias=False)
        self.en_time = (input_size - enc_ker) // self.enc_stride + 1

        self.embed_net = MaskNetworkPlus(enc_dim, 
                                        enc_ker, 
                                        conv_in_channels, 
                                        conv_out_channels, 
                                        conv_ker, 
                                        n_blocks, 
                                        n_repeats, 
                                        1, 
                                        False)

        self.extractor = nn.Sequential(
             nn.BatchNorm1d(self.enc_dim),
             nn.Conv1d(self.enc_dim, self.enc_dim//2, 8, stride=4, bias=True),
             nn.Hardswish(),
             nn.Conv1d(self.enc_dim//2, self.emb_dim, 16, stride=8, bias=True),
             nn.Hardswish()
        ) 

        
        self.hidden = nn.Sequential(
             nn.Linear(self.emb_dim, self.emb_dim),
             nn.Dropout(0.3),
             nn.ReLU(),
        )
        self.embedding_species = nn.Linear(self.emb_dim, self.emb_dim) 
        # self.embedding_family = nn.Linear(self.enc_dim, self.emb_dim)
        # self.embedding_order = nn.Linear(self.enc_dim, self.emb_dim)

    def _align_num_frames_with_strides(self, input: torch.Tensor):
            batch_size, num_channels, num_frames = input.shape
            is_odd = self.enc_ker % 2
            num_strides = (num_frames - is_odd) // self.enc_stride
            num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
            if num_remainings == 0:
                return input, 0

            num_paddings = self.enc_stride - num_remainings
            pad = torch.zeros(
                batch_size,
                num_channels,
                num_paddings,
                dtype=input.dtype,
                device=input.device,
            )
            return torch.cat([input, pad], 2), num_paddings

    def _apply_extractor(self, x):
        # x  B, 1, F,M
        x = x.squeeze(1) # B, F, M
        x = self.extractor(x) # B, 2F, M'
        x = x.mean(-1) # B, 2F
        # print(x.shape)
        return x

    def forward(self, x):
        padded, num_pads = self._align_num_frames_with_strides(x)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        transformed = self.encoder(padded) # B, F, M

        features, _ = self.embed_net(transformed)  # B, 1, F, M

        emb = self._apply_extractor(features)
        features = self.hidden(emb) # B, F
        features = self.embedding_species(features) # B, E

        # emb_f = self.embedding_family(features) # B, E_f
        # emb_o = self.embedding_order(features) # B, E_o

        if self.norm:
            emb = nn.functional.normalize(emb, p=2, dim=1, eps=1e-5)

            features = nn.functional.normalize(features, p=2, dim=1, eps=1e-5)
            # emb_f = nn.functional.normalize(emb_f, p=2, dim=1, eps=1e-5)
            # emb_o = nn.functional.normalize(emb_o, p=2, dim=1, eps=1e-5)

        # return emb, emb_f, emb_o
        # return (emb, features)
        return features


class EmbeddingTDCNpp5(nn.Module):
    def __init__(self, emb_dim,
                 enc_dim, enc_ker, 
                 conv_in_channels,
                 conv_out_channels, 
                 conv_ker, 
                 n_blocks, 
                 n_repeats, 
                 input_size,
                 norm_emb = True):
        
        super().__init__()
        self.emb_dim = emb_dim
        self.norm_emb = norm_emb

        self.enc_dim = enc_dim
        self.enc_ker = enc_ker
        self.enc_stride = enc_ker//2
        self.input_size = input_size

        self.encoder = nn.Conv1d(1, enc_dim, enc_ker, stride=self.enc_stride, bias=False)
        self.en_time = (input_size - enc_ker) // self.enc_stride + 1

        self.embed_net = MaskNetworkPlus(enc_dim, 
                                        enc_ker, 
                                        conv_in_channels, 
                                        conv_out_channels, 
                                        conv_ker, 
                                        n_blocks, 
                                        n_repeats, 
                                        1, 
                                        False)

        self.extractor = nn.Sequential(
             nn.BatchNorm1d(self.enc_dim),
             nn.Conv1d(self.enc_dim, self.enc_dim, 8, stride=4, bias=True),
             nn.Hardswish(),
             nn.Conv1d(self.enc_dim, self.emb_dim, 16, stride=8, bias=True),
             nn.Hardswish()
        ) 

        
        self.projection = nn.Sequential(
             nn.Linear(self.emb_dim, self.emb_dim//8),
             nn.Hardswish(),
        )
        self.embedding_species = nn.Linear(self.emb_dim, self.emb_dim)

    def _align_num_frames_with_strides(self, input: torch.Tensor):
            batch_size, num_channels, num_frames = input.shape
            is_odd = self.enc_ker % 2
            num_strides = (num_frames - is_odd) // self.enc_stride
            num_remainings = num_frames - (is_odd + num_strides * self.enc_stride)
            if num_remainings == 0:
                return input, 0

            num_paddings = self.enc_stride - num_remainings
            pad = torch.zeros(
                batch_size,
                num_channels,
                num_paddings,
                dtype=input.dtype,
                device=input.device,
            )
            return torch.cat([input, pad], 2), num_paddings

    def _apply_extractor(self, x):
        # x  B, 1, F,M
        x = x.squeeze(1) # B, F, M
        x = self.extractor(x) # B, 2F, M'
        x = x.mean(-1) # B, 2F
        # print(x.shape)
        return x

    def forward(self, x):
        padded, num_pads = self._align_num_frames_with_strides(x)  # B, 1, L'
        batch_size, num_padded_frames = padded.shape[0], padded.shape[2]

        transformed = self.encoder(padded) # B, F, M

        features, _ = self.embed_net(transformed)  # B, 1, F, M
        features = self._apply_extractor(features)

        emb = self.embedding_species(features) # B, E

        features = self.projection(emb) # B, F
        features = nn.functional.normalize(features, p=2, dim=1, eps=1e-5) # for loss calculation
        if self.norm_emb:
            emb = nn.functional.normalize(emb, p=2, dim=1, eps=1e-5) # for distance calculation
        return (emb, features)


class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes, num_of_hidden, size_of_layers, activation = nn.ReLU(), num_of_f = None, num_of_o = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.num_of_hidden = num_of_hidden
        self.size_of_layers = size_of_layers
        self.activation = activation

        self.input = nn.Linear(emb_dim, size_of_layers)
        self.hidden = nn.ModuleList([nn.Linear(size_of_layers, size_of_layers) for i in range(num_of_hidden)])
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(size_of_layers) for i in range(num_of_hidden)])
        self.head_spe = nn.Linear(size_of_layers, num_classes)

        self.dropout = nn.Dropout(0.3)
        
        self.num_of_f = num_of_f
        self.num_of_o = num_of_o
        if num_of_f is not None and num_of_o is not None:
            self.head_f = nn.Linear(size_of_layers, num_of_f)
            self.head_o = nn.Linear(size_of_layers, num_of_o)
        
    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)
        for i in range(self.num_of_hidden):
            x = self.hidden[i](x)
            x = self.batch_norm[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        new_emb = x
        logit_spe = self.head_spe(x)
        # logit_spe = torch.softmax(logit_spe, dim=1)
        out = logit_spe
        if self.num_of_f is not None and self.num_of_o is not None and self.training:
            logit_f = self.head_f(x)
            logit_o = self.head_o(x)
            # logit_f = torch.softmax(logit_f, dim=1)
            # logit_o = torch.softmax(logit_o, dim=1)
            out = {'species': logit_spe, 'family': logit_f, 'order': logit_o}
            
        return out, nn.functional.normalize(new_emb, p=2, dim=1, eps=1e-5)

class EmbeddingClassifierLayer(nn.Module):
    def __init__(self, emb_dim, num_classes, activation = nn.Hardswish(), num_of_f = None, num_of_o = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        
        self.activation = activation
        self.input = nn.Linear(emb_dim, emb_dim//2)

        self.head_spe = nn.Linear(emb_dim//2, num_classes)
        
        self.num_of_f = num_of_f
        self.num_of_o = num_of_o
        if num_of_f is not None and num_of_o is not None:
            self.head_f = nn.Linear(emb_dim//2, num_of_f)
            self.head_o = nn.Linear(emb_dim//2, num_of_o)
        
    def forward(self, x):
        x = self.input(x)
        x = self.activation(x)

        logit_spe = self.head_spe(x)

        out = logit_spe
        if self.num_of_f is not None and self.num_of_o is not None and self.training:
            logit_f = self.head_f(x)
            logit_o = self.head_o(x)
            out = {'species': logit_spe, 'family': logit_f, 'order': logit_o}
            
        return out

class DownstreamClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(DownstreamClassifier, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
        
    def forward(self, x):
        x = self.fc(x)
        return x
