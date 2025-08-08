import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianSmoothingLayer(nn.Module):
    """
    Apply Gaussian smoothing to a tensor.
    """
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self._make_kernel()
        
    def _make_kernel(self):
        # Create a Gaussian kernel
        kernel_1d = torch.exp(-torch.arange(-(self.kernel_size//2), self.kernel_size//2 + 1)**2 / (2*self.sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize
        self.register_buffer('kernel_1d', kernel_1d.view(1, 1, -1))
        
    def forward(self, x):
        # x shape: [batch, time_steps, nodes]
        batch_size, time_steps, num_nodes = x.shape
        
        # Apply smoothing over the time dimension (dim=1)
        padding = self.kernel_size // 2
        x_padded = F.pad(x, (0, 0, padding, padding), mode='replicate')
        
        # Reshape for 1D convolution
        x_reshaped = x_padded.permute(0, 2, 1).contiguous()  # [batch, nodes, time_steps+padding*2]
        
        # Apply 1D convolution with Gaussian kernel
        x_smoothed = F.conv1d(
            x_reshaped,
            self.kernel_1d.expand(num_nodes, 1, -1),
            groups=num_nodes
        )  # [batch, nodes, time_steps]
        
        # Reshape back
        return x_smoothed.permute(0, 2, 1).contiguous()  # [batch, time_steps, nodes]

class CustomFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
    def compl_mul2d(self, input_fft, weights):
        # (B, C_in, H, W) × (C_in, C_out, H, W) → (B, C_out, H, W)
        return torch.einsum("bcih,wcoh->bcoh", input_fft, weights)
        
    def forward(self, x):  # x: [B, C_in, H, W]
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        m1 = min(self.modes1, x_ft.shape[-2])  # H (time dim)
        m2 = min(self.modes2, x_ft.shape[-1])  # W (node dim / rfft size)
        out_ft = torch.zeros(B, self.out_channels, H, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2]
        )
        x = torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")
        return x.real

class LatentCorrelationGCN(nn.Module):
    """
    A GCN layer that learns the adjacency matrix (connections between nodes)
    instead of using predefined edge features.
    """
    def __init__(self, in_features, out_features, num_nodes, 
                 k_neighbors=None, temperature=1.0, add_self_loops=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.k_neighbors = k_neighbors  # If None, use dense attention. If int, use top-k sparsification
        self.temperature = temperature  # Temperature for softmax (lower = more discrete)
        self.add_self_loops = add_self_loops
        
        # Linear transformations
        self.query_proj = nn.Linear(in_features, out_features)
        self.key_proj = nn.Linear(in_features, out_features)
        self.value_proj = nn.Linear(in_features, out_features)
        
        # Optional: learnable edge weights matrix
        self.learn_static_correlations = True
        if self.learn_static_correlations:
            # Initialize with identity-like pattern (stronger self-connections)
            init_adj = torch.ones(num_nodes, num_nodes) * 0.1
            if self.add_self_loops:
                init_adj = init_adj + torch.eye(num_nodes) * 0.9
            self.static_edge_weights = nn.Parameter(init_adj)
        
        # Output MLP
        self.out_mlp = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
    def forward(self, x, batch_size=None):
        """
        Args:
            x: Node features [B*N, F] or [B, N, F]
            batch_size: Batch size (needed if x is [B*N, F])
        """
        # Reshape if necessary
        if x.dim() == 2:
            if batch_size is None:
                batch_size = 1
            # Reshape from [B*N, F] to [B, N, F]
            x = x.view(batch_size, self.num_nodes, self.in_features)
        
        # Now x is [B, N, F]
        B, N, F = x.shape
        
        # Compute query, key, value projections
        q = self.query_proj(x)  # [B, N, out_features]
        k = self.key_proj(x)    # [B, N, out_features]
        v = self.value_proj(x)  # [B, N, out_features]
        
        # Compute dynamic graph topology through attention
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / (self.out_features ** 0.5)  # [B, N, N]
        
        # Add static correlations if enabled
        if self.learn_static_correlations:
            # Broadcast static weights to all batches
            static_weights = self.static_edge_weights.unsqueeze(0).expand(B, -1, -1)
            attn_scores = attn_scores + static_weights
        
        # Optional: Use top-k sparsification
        if self.k_neighbors is not None and self.k_neighbors < N:
            # Find top-k values per node
            topk_values, topk_indices = torch.topk(attn_scores, k=self.k_neighbors, dim=-1)
            
            # Create mask for topk
            mask = torch.zeros_like(attn_scores)
            mask.scatter_(2, topk_indices, 1)
            
            # Apply mask (set non-top-k values to -inf)
            attn_scores = torch.where(mask > 0, attn_scores, torch.tensor(-1e9, device=attn_scores.device))
        
        # Apply softmax to get normalized attention weights
        # Using full namespace to avoid any potential name conflicts
        attn_weights = torch.nn.functional.softmax(attn_scores / self.temperature, dim=-1)  # [B, N, N]
        
        # Compute messages via weighted sum
        messages = torch.bmm(attn_weights, v)  # [B, N, out_features]
        
        # Apply output MLP
        output = self.out_mlp(messages)  # [B, N, out_features]
        
        # Option: use residual connection
        output = output + v
        
        # Reshape back if input was flattened
        if x.dim() == 2:
            output = output.view(B * N, self.out_features)
        
        return output, attn_weights

class Seq2SeqLatentGNN(nn.Module):
    """
    Sequence-to-sequence model with learnable graph structure that predicts future values 
    for all nodes in a graph.
    """
    def __init__(self, node_input_dim, hidden_dim, num_nodes, out_seq_len, 
                 k_neighbors=4, teacher_forcing_ratio=0.5):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_seq_len = out_seq_len
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        # Gaussian smoothing for output
        self.gaussian_smoother = GaussianSmoothingLayer(kernel_size=5, sigma=1.0)
        
        # Encoding layers
        self.fourier_encoder = CustomFourierLayer(
            in_channels=node_input_dim, 
            out_channels=node_input_dim, 
            modes1=16, 
            modes2=16
        )
        
        # Latent graph layers (learns connections dynamically)
        self.latent_gcn1 = LatentCorrelationGCN(
            in_features=node_input_dim,
            out_features=hidden_dim,
            num_nodes=num_nodes,
            k_neighbors=k_neighbors
        )
        
        self.latent_gcn2 = LatentCorrelationGCN(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_nodes=num_nodes,
            k_neighbors=k_neighbors
        )
        
        # Temporal modeling
        self.encoder_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.2
        )
        
        self.decoder_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.2
        )
        
        # Output layers
        self.fc_out = nn.Linear(hidden_dim * 2, hidden_dim)  # Extra hidden for attention context
        self.final_out = nn.Linear(hidden_dim, 1)
        self.value_to_hidden = nn.Linear(1, hidden_dim)
        
        # Smoothing layers
        self.output_smoother = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=5, padding=2)
        )
        
        # Initialize LSTMs with orthogonal initialization for better gradient flow
        for layer_p in self.encoder_lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.orthogonal_(self.encoder_lstm.__getattr__(p))
        
        for layer_p in self.decoder_lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    nn.init.orthogonal_(self.decoder_lstm.__getattr__(p))
    
    def forward(self, node_features, targets=None, teacher_forcing=False):
        """
        Perform sequence-to-sequence prediction with latent graph structure
        
        Args:
            node_features: Input node features [batch_size, in_seq_len, num_nodes, feat_dim]
            targets: Target values for teacher forcing [batch_size, out_seq_len, num_nodes]
            teacher_forcing: Whether to use teacher forcing
            
        Returns:
            Predicted sequence [batch_size, out_seq_len, num_nodes]
        """
        batch_size, in_seq_len, num_nodes, feat_dim = node_features.shape
        
        # Save input for skip connection
        skip_input = node_features  # [B, T, N, F]
        
        # Fourier Encoding for global patterns
        fno_input = node_features.permute(0, 3, 1, 2)         # [B, F, T, N]
        fno_output = self.fourier_encoder(fno_input)          # [B, F, T, N]
        node_features = fno_output.permute(0, 2, 3, 1)        # [B, T, N, F]
        node_features = node_features + skip_input            # [B, T, N, F]
        
        # Process through latent graph layers for each timestep
        encoder_outputs = []
        graph_attentions = []  # Store attention matrices for visualization/analysis
        
        for t in range(in_seq_len):
            x_t = node_features[:, t]  # [B, N, F]
            x_t_flat = x_t.reshape(batch_size * num_nodes, -1)  # [B*N, F]
            
            # First latent GCN layer
            h1, attn1 = self.latent_gcn1(x_t_flat, batch_size)  # [B*N, H], [B, N, N]
            
            # Second latent GCN layer
            h2, attn2 = self.latent_gcn2(h1, batch_size)  # [B*N, H], [B, N, N]
            
            # Reshape to [B, N, H]
            h_t = h2.view(batch_size, num_nodes, -1)
            
            encoder_outputs.append(h_t)
            graph_attentions.append(attn2)  # Save the attention weights from the last GCN layer
        
        # Stack timesteps
        encoder_outputs = torch.stack(encoder_outputs, dim=1)  # [B, T, N, H]
        
        # Get initial decoder input from last timestep
        decoder_input = encoder_outputs[:, -1]  # [B, N, H]
        
        # Prepare encoder LSTM input - flatten nodes into batch dimension
        lstm_input = encoder_outputs.permute(0, 2, 1, 3)  # [B, N, T, H]
        lstm_input = lstm_input.reshape(batch_size * num_nodes, in_seq_len, self.hidden_dim)  # [B*N, T, H]
        
        # Run encoder LSTM
        _, (h, c) = self.encoder_lstm(lstm_input)
        
        # Tracking predictions for smoothing
        last_predictions = []
        outputs = []
        
        # Decoder loop
        for t in range(self.out_seq_len):
            # Reshape decoder input for LSTM
            decoder_input_reshaped = decoder_input.reshape(batch_size * num_nodes, 1, self.hidden_dim)  # [B*N, 1, H]
            
            # Run decoder timestep
            decoder_output, (h, c) = self.decoder_lstm(decoder_input_reshaped, (h, c))  # [B*N, 1, H]
            
            # === Attention mechanism ===
            query = decoder_output  # [B*N, 1, H]
            key = value = lstm_input  # [B*N, T_enc, H]
            
            # Compute attention scores
            attn_scores = torch.bmm(query, key.transpose(1, 2)) / (self.hidden_dim ** 0.5)  # [B*N, 1, T]
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B*N, 1, T]
            context = torch.bmm(attn_weights, value)  # [B*N, 1, H]
            
            # Combine attention context with decoder state
            combined = torch.cat([decoder_output.squeeze(1), context.squeeze(1)], dim=-1)  # [B*N, 2H]
            
            # Generate output
            projection = self.fc_out(combined)  # [B*N, H]
            prediction = self.final_out(projection).view(batch_size, num_nodes)  # [B, N]
            
            # Maintain history for smoothing
            last_predictions.append(prediction.unsqueeze(1))  # [B, 1, N]
            if len(last_predictions) > 5:
                last_predictions.pop(0)  # Keep only most recent predictions
            
            # Apply temporal smoothing if we have enough predictions
            if len(last_predictions) >= 3:
                # Concatenate recent predictions
                recent_preds = torch.cat(last_predictions[-3:], dim=1)  # [B, 3, N]
                
                # Apply smoothing for each node
                for node_idx in range(num_nodes):
                    node_preds = recent_preds[:, :, node_idx:node_idx+1].transpose(1, 2)  # [B, 1, 3]
                    smoothed = self.output_smoother(node_preds).transpose(1, 2)  # [B, 3, 1]
                    # Update the latest prediction
                    # inside Seq2SeqLatentGNN.forward
                    last_predictions[-1] = last_predictions[-1].clone()  # <- add this line
                    last_predictions[-1][:, 0, node_idx] = smoothed[:, -1, 0]  # safe now
            
            # Add to output sequence
            outputs.append(last_predictions[-1])  # [B, 1, N]
            
            # Teacher forcing or use own predictions
            if teacher_forcing and targets is not None and torch.rand(1).item() < self.teacher_forcing_ratio:
                # Use ground truth as next input - FIXED SHAPE HANDLING
                next_input = targets[:, t].unsqueeze(-1)  # [B, N, 1]
                decoder_input = self.value_to_hidden(next_input)  # [B, N, H]
            else:
                # Use own prediction as next input (autoregressive) - FIXED SHAPE HANDLING
                next_input = last_predictions[-1].squeeze(1).unsqueeze(-1)  # [B, N, 1]
                decoder_input = self.value_to_hidden(next_input)  # [B, N, H]
        
        # Concatenate all outputs
        output_seq = torch.cat(outputs, dim=1)  # [B, T_out, N]
        
        # Apply final Gaussian smoothing
        smoothed_output = self.gaussian_smoother(output_seq)
        
        return smoothed_output, graph_attentions[-1]  # Return predictions and last graph attention matrix
