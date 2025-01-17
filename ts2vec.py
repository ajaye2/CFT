import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss, expclr_loss, quadratic_contrastive_loss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math


class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None, 
        use_projection_head=True
    ):
        ''' Initialize a TS2Vec model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, None Type]): The maximum allowed sequence length for training. For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length>.
            temporal_unit (int): The minimum unit to perform temporal contrast. When training on a very long sequence, this param helps to reduce the cost of time and memory.
            after_iter_callback (Union[Callable, None Type]): A callback function that would be called after each iteration.
            after_epoch_callback (Union[Callable, None Type]): A callback function that would be called after each epoch.
        '''
        
        super().__init__()
        # self.device = device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Device is ", self.device)
        
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.use_projection_head = use_projection_head
        
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
        self.net  = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self._projection_head = nn.Linear(output_dims, output_dims).to(self.device) ## TODO: Add parameters for more flexibilty 
        self.projection_head  = torch.optim.swa_utils.AveragedModel(self._projection_head)
        self.projection_head.update_parameters(self._projection_head)

        self._rnn = nn.LSTM(output_dims, output_dims, batch_first=True).to(self.device) ## TODO: Add parameters for more flexibilty 
        self.rnn  = torch.optim.swa_utils.AveragedModel(self._rnn)      ## TODO: Figure out what this does
        self.rnn.update_parameters(self._rnn)
        
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
    
    def fit(self, train_data, expert_features=None, n_epochs=None, n_iters=None, verbose=False, temperature=1, delta=1, loss_weight_scale=0.35, use_expclr_loss=False):
        ''' Training the TS2Vec model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            expert_features (numpy.ndarray): The expert_features data. It should have a shape of (n_instance, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, None Type]): The number of epochs. When this reaches, the training stops.
            n_iters (Union[int, None Type]): The number of iterations. When this reaches, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3

        if expert_features is not None:
            assert expert_features.ndim == 2

        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
        
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)
                ## TODO: Add code for expert_features
            

        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)

        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
            ## TODO: Add code for expert_features
        
        idx        = ~np.isnan(train_data).all(axis=2).all(axis=1)
        train_data = train_data[idx]
        
        if expert_features is not None:
            expert_features = expert_features[idx]
            idx             = ~np.isnan(expert_features).all(axis=1)

            train_data      = train_data[idx]
            expert_features = expert_features[idx]

        
        if expert_features is not None:
            train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float), torch.from_numpy(expert_features).to(torch.float))
        else:
            train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))

        train_loader  = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        # optimizer = torch.optim.AdamW([self._net.parameters(), self._projection_head.parameters()], lr=self.lr)
        optimizer = torch.optim.AdamW([{"params": self._net.parameters()},{"params":  self._projection_head.parameters()}, {"params":  self._rnn.parameters()}], lr=self.lr)
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]

                if expert_features is not None:
                    expf = batch[1]

                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                    ## TODO: Add code for expert_features

                x = x.to(self.device)
                if expert_features is not None:
                    expf = expf.to(self.device)
                
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                optimizer.zero_grad()
                
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                out1 = self._projection_head(out1)

                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                out2 = self._projection_head(out2) ### Added projection layer
                
                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )

                loss *= loss_weight_scale
 
                if expert_features is not None:
                    _, (out1, _)  = self._rnn(out1) 
                    _, (out2, _)  = self._rnn(out2) 

                    out1 = out1.squeeze(0) #out1.view(out1.shape[1:])
                    out2 = out2.squeeze(0) #out2.view(out2.shape[1:])

                    if use_expclr_loss:
                        l1 = expclr_loss(out1, expf, temp=temperature, delta=delta) 
                        l2 = expclr_loss(out2, expf, temp=temperature, delta=delta)
                    else:
                        l1 = quadratic_contrastive_loss(out1, expf, delta=delta)
                        l2 = quadratic_contrastive_loss(out2, expf, delta=delta) 
                    loss += l1 + l2
                    
                loss.backward()
                optimizer.step()

                self.net.update_parameters(self._net)
                self.projection_head.update_parameters(self._projection_head)
                self.rnn.update_parameters(self._rnn)
                    
                cum_loss += loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
            
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        out = self.net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)

        elif encoding_window == 'net_compression':
            if slicing is not None:
                out = out[:, slicing]
            _, (out, _)  = self.rnn(out) 
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
            
        return out.cpu()
    
    def encode(self, data, mask=None, encoding_window=None, casual=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations using the model.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be specified with this parameter. This can be set to 'binomial', 'continuous', 'all_true', 'all_false' or 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            casual (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, None Type]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, None Type]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        assert self.net is not None, 'please train or load a net first'
        assert data.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        n_samples, ts_l, _ = data.shape

        org_training = self.net.training
        org_training_projection_head = self.projection_head.training
        org_training_rnn = self.rnn.training
        self.net.eval()
        self.projection_head.eval()
        self.rnn.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer_l = 0
                    for i in range(0, ts_l, sliding_length):
                        l = i - sliding_padding
                        r = i + sliding_length + (sliding_padding if not casual else 0)
                        x_sliding = torch_pad_nan(
                            x[:, max(l, 0) : min(r, ts_l)],
                            left=-l if l<0 else 0,
                            right=r-ts_l if r>ts_l else 0,
                            dim=1
                        )
                        if n_samples < batch_size:
                            if calc_buffer_l + n_samples > batch_size:
                                out = self._eval_with_pooling(
                                    torch.cat(calc_buffer, dim=0),
                                    mask,
                                    slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                    encoding_window=encoding_window
                                )
                                reprs += torch.split(out, n_samples)
                                calc_buffer = []
                                calc_buffer_l = 0
                            calc_buffer.append(x_sliding)
                            calc_buffer_l += n_samples
                        else:
                            out = self._eval_with_pooling(
                                x_sliding,
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs.append(out)

                    if n_samples < batch_size:
                        if calc_buffer_l > 0:
                            out = self._eval_with_pooling(
                                torch.cat(calc_buffer, dim=0),
                                mask,
                                slicing=slice(sliding_padding, sliding_padding+sliding_length),
                                encoding_window=encoding_window
                            )
                            reprs += torch.split(out, n_samples)
                            calc_buffer = []
                            calc_buffer_l = 0
                    
                    out = torch.cat(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                    
                    if encoding_window == 'net_compression':
                        # self.rnn.flatten_parameters()
                        _, (out, _)  = self.rnn(out) 
                        out = out.squeeze(0) 
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                    elif encoding_window == 'net_compression':
                        # self.rnn.flatten_parameters()
                        out = out.squeeze(0) 
                        
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        self.projection_head.train(org_training_projection_head)
        self.rnn.train(org_training_rnn)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        if fn[-4:] == '.pkl':
            fn = fn[:-4]

        torch.save(self.net.state_dict(), fn + "_net.pkl")
        torch.save(self.projection_head.state_dict(), fn + "_projection_head.pkl")
        torch.save(self.rnn.state_dict(), fn + "_rnn.pkl")
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        if fn[-4:] == '.pkl':
            fn = fn[:-4]

        net_state_dict = torch.load(fn + "_net.pkl", map_location=self.device)
        self.net.load_state_dict(net_state_dict)

        projection_head_state_dict = torch.load(fn + "_projection_head.pkl", map_location=self.device)
        self.projection_head.load_state_dict(projection_head_state_dict)

        rnn_state_dict = torch.load(fn + "_rnn.pkl", map_location=self.device)
        self.rnn.load_state_dict(rnn_state_dict)
    