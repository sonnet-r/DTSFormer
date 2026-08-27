import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    """使用端点填充的移动平均提取趋势项。"""

    def __init__(self, kernel_size: int):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("moving_avg must be a positive odd number")
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = (self.kernel_size - 1) // 2
        front = x[:, :1].expand(-1, padding, -1)
        end = x[:, -1:].expand(-1, padding, -1)
        padded = torch.cat((front, x, end), dim=1)
        return self.pool(padded.transpose(1, 2)).transpose(1, 2)


class SeriesDecomposition(nn.Module):
    """将输入分解为季节项和趋势项。"""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, output_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        return projected + self.block(projected)


class TemporalInformationFusion(nn.Module):
    """论文中的局部季节-趋势信息融合注意力（TIF）。"""

    def __init__(
        self,
        d_model: int,
        window_size: int,
        dropout: float,
        correction_scale: float,
        correction_floor: float,
    ):
        super().__init__()
        if window_size < 0:
            raise ValueError("tif_window must be non-negative")
        self.window_size = window_size
        self.memory_projection = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.value = nn.Linear(d_model, d_model)
        self.output_projection = nn.Linear(d_model * 2, d_model)
        self.correction_floor = correction_floor
        initial_scale = max(correction_scale - correction_floor, 1e-6)
        raw_scale = torch.log(torch.expm1(torch.tensor(initial_scale)))
        self.correction_scale = nn.Parameter(raw_scale)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def _local_windows(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_size == 0:
            return x.unsqueeze(2)
        padded = F.pad(
            x.transpose(1, 2),
            (self.window_size, self.window_size),
            mode="replicate",
        ).transpose(1, 2)
        windows = padded.unfold(1, 2 * self.window_size + 1, 1)
        return windows.permute(0, 1, 3, 2).contiguous()

    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        trend_windows = self._local_windows(trend)
        window_width = trend_windows.size(2)
        seasonal_query = seasonal.unsqueeze(2).expand(-1, -1, window_width, -1)
        memory_input = torch.cat((trend_windows, seasonal_query), dim=-1)
        scores = self.memory_projection(memory_input).squeeze(-1)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        values = self.value(trend_windows)
        context = (weights.unsqueeze(-1) * values).sum(dim=2)
        correction = self.output_projection(torch.cat((seasonal, context), dim=-1))
        correction_scale = self.correction_floor + F.softplus(self.correction_scale)
        fused = self.norm1(seasonal + trend + correction_scale * correction)
        return self.norm2(fused + self.dropout(self.feed_forward(fused)))


class AdaptiveGraph(nn.Module):
    """为趋势项或季节项学习独立的稀疏有向图。"""

    def __init__(
        self,
        num_nodes: int,
        node_dim: int,
        top_k: int,
        alpha: float,
        temperature: float,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        max_neighbors = max(1, num_nodes - 1)
        self.top_k = min(max(1, top_k), max_neighbors)
        self.alpha = alpha
        self.temperature = temperature
        self.source_embedding = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.target_embedding = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.source_projection = nn.Linear(node_dim, node_dim)
        self.target_projection = nn.Linear(node_dim, node_dim)
        nn.init.xavier_uniform_(self.source_embedding)
        nn.init.xavier_uniform_(self.target_embedding)

    def forward(self) -> torch.Tensor:
        source = torch.tanh(self.alpha * self.source_projection(self.source_embedding))
        target = torch.tanh(self.alpha * self.target_projection(self.target_embedding))
        score = source @ target.transpose(0, 1) - target @ source.transpose(0, 1)
        logits = self.alpha * score
        adjacency = F.relu(torch.tanh(logits))
        diagonal = torch.eye(
            self.num_nodes,
            device=adjacency.device,
            dtype=torch.bool,
        )
        candidates = adjacency.masked_fill(diagonal, -torch.inf)
        indices = candidates.topk(self.top_k, dim=1).indices
        mask = torch.zeros_like(adjacency, dtype=torch.bool).scatter_(1, indices, True)
        scaled_logits = logits / max(self.temperature, 1e-6)
        return torch.softmax(
            scaled_logits.masked_fill(~mask, -torch.inf), dim=1
        )


class MixHopDiffusion(nn.Module):
    """带特征注意力和保真项的 mix-hop diffusion。"""

    def __init__(
        self,
        d_model: int,
        depth: int,
        residual_alpha: float,
        epsilon: float,
        dropout: float,
        self_loop_weight: float,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("gcn_depth must be at least 1")
        self.depth = depth
        self.residual_alpha = residual_alpha
        self.epsilon = epsilon
        self.self_loop_weight = self_loop_weight
        self.transforms = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(depth))
        self.output_projection = nn.Linear((depth + 1) * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _row_normalize(self, adjacency: torch.Tensor) -> torch.Tensor:
        """将有向邻接矩阵转换为行随机游走矩阵。"""
        identity = torch.eye(adjacency.size(0), device=adjacency.device, dtype=adjacency.dtype)
        neighbors = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return self.self_loop_weight * identity + (
            1.0 - self.self_loop_weight
        ) * neighbors

    def _normalized_step(
        self,
        transformed: torch.Tensor,
        graph: torch.Tensor,
    ) -> torch.Tensor:
        """实现论文中的特征权重扩散、保真项和归一化。"""
        coefficients = torch.softmax(transformed, dim=-1)
        payoff = torch.einsum("ij,bjd->bid", graph, coefficients)
        payoff = payoff + self.epsilon * coefficients
        normalized = payoff / payoff.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # 概率权重的均值为 1 / d_model；尺度补偿使门控初始幅值接近恒等映射。
        return normalized * transformed.size(-1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph = self._row_normalize(adjacency)
        initial = x
        hidden = x
        states = [x]
        for transform in self.transforms:
            transformed = transform(hidden)
            gate = self._normalized_step(transformed, graph)
            update = F.relu(gate * transformed)
            hidden = self.residual_alpha * initial + (1.0 - self.residual_alpha) * update
            hidden = self.dropout(hidden)
            states.append(hidden)
        output = self.output_projection(torch.cat(states, dim=-1))
        return self.norm(initial + output)


class CrossDiffusionAttention(nn.Module):
    """基于 E-step/M-step 迭代的季节-趋势图信息融合。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        iterations: int,
        epsilon: float,
        dropout: float,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if iterations < 1:
            raise ValueError("cross_diffusion_iters must be at least 1")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.iterations = iterations
        self.epsilon = epsilon
        self.trend_qkv = nn.Linear(d_model, d_model * 3)
        self.seasonal_qkv = nn.Linear(d_model, d_model * 3)
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, _ = x.shape
        return x.view(batch, nodes, self.num_heads, self.head_dim).transpose(1, 2)

    def _qkv(self, projection: nn.Linear, x: torch.Tensor):
        query, key, value = projection(x).chunk(3, dim=-1)
        return tuple(self._split_heads(item) for item in (query, key, value))

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        q_trend, k_trend, v_trend = self._qkv(self.trend_qkv, trend)
        q_seasonal, k_seasonal, v_seasonal = self._qkv(self.seasonal_qkv, seasonal)

        for _ in range(self.iterations):
            trend_to_season = torch.softmax(
                (q_trend @ k_seasonal.transpose(-2, -1)) * self.scale,
                dim=-1,
            )
            season_to_trend = torch.softmax(
                (q_seasonal @ k_trend.transpose(-2, -1)) * self.scale,
                dim=-1,
            )
            trend_context = trend_to_season @ v_seasonal
            seasonal_context = season_to_trend @ v_trend
            k_trend = self.epsilon * k_trend + (1.0 - self.epsilon) * trend_context
            k_seasonal = self.epsilon * k_seasonal + (1.0 - self.epsilon) * seasonal_context

        fused = (k_trend + k_seasonal).transpose(1, 2).contiguous()
        fused = fused.view(trend.size(0), trend.size(1), -1)
        residual = self.norm1(trend + seasonal + self.output_projection(fused))
        return self.norm2(residual + self.dropout(self.feed_forward(residual)))


class Model(nn.Module):
    """DTSFormer：解耦时序融合与空间图扩散的长期预测模型。"""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.temporal_d_model = getattr(configs, "temporal_d_model", None) or self.d_model
        self.spatial_d_model = getattr(configs, "spatial_d_model", None) or self.d_model

        dropout = configs.dropout
        d_ff = configs.d_ff
        temporal_d_ff = getattr(configs, "temporal_d_ff", None) or d_ff
        spatial_d_ff = getattr(configs, "spatial_d_ff", None) or d_ff
        moving_avg = getattr(configs, "moving_avg", 25)
        tif_window = getattr(configs, "tif_window", 4)
        tif_correction_scale = getattr(configs, "tif_correction_scale", 0.1)
        tif_scale_floor = getattr(configs, "tif_scale_floor", 0.0)
        graph_top_k = getattr(configs, "graph_top_k", getattr(configs, "subgraph_size", 3))
        graph_alpha = getattr(configs, "graph_alpha", getattr(configs, "tanhalpha", 3.0))
        graph_temperature = getattr(configs, "graph_temperature", 1.0)
        graph_self_loop_weight = getattr(configs, "graph_self_loop_weight", 0.5)
        node_dim = getattr(configs, "node_dim", 16)
        diffusion_depth = getattr(configs, "gcn_depth", 3)
        diffusion_alpha = getattr(configs, "propalpha", 0.3)
        diffusion_epsilon = getattr(configs, "diffusion_epsilon", 0.1)
        cross_iterations = getattr(configs, "cross_diffusion_iters", 3)
        cross_epsilon = getattr(configs, "cross_diffusion_epsilon", 0.3)

        self.decomposition = SeriesDecomposition(moving_avg)

        self.temporal_seasonal_mlp = ResidualMLP(
            self.enc_in, temporal_d_ff, self.temporal_d_model, dropout
        )
        self.temporal_trend_mlp = ResidualMLP(
            self.enc_in, temporal_d_ff, self.temporal_d_model, dropout
        )
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.seq_len, self.temporal_d_model)
        )
        nn.init.trunc_normal_(self.temporal_position, std=0.02)
        self.temporal_fusion = TemporalInformationFusion(
            self.temporal_d_model,
            tif_window,
            dropout,
            tif_correction_scale,
            tif_scale_floor,
        )
        self.temporal_channel_projection = nn.Linear(
            self.temporal_d_model, self.enc_in
        )
        self.temporal_horizon_projection = nn.Linear(self.seq_len, self.pred_len)

        self.spatial_seasonal_mlp = ResidualMLP(
            self.seq_len, spatial_d_ff, self.spatial_d_model, dropout
        )
        self.spatial_trend_mlp = ResidualMLP(
            self.seq_len, spatial_d_ff, self.spatial_d_model, dropout
        )
        self.trend_graph = AdaptiveGraph(
            self.enc_in,
            node_dim,
            graph_top_k,
            graph_alpha,
            graph_temperature,
        )
        self.seasonal_graph = AdaptiveGraph(
            self.enc_in,
            node_dim,
            graph_top_k,
            graph_alpha,
            graph_temperature,
        )
        self.trend_diffusion = MixHopDiffusion(
            self.spatial_d_model,
            diffusion_depth,
            diffusion_alpha,
            diffusion_epsilon,
            dropout,
            graph_self_loop_weight,
        )
        self.seasonal_diffusion = MixHopDiffusion(
            self.spatial_d_model,
            diffusion_depth,
            diffusion_alpha,
            diffusion_epsilon,
            dropout,
            graph_self_loop_weight,
        )
        self.cross_diffusion = CrossDiffusionAttention(
            self.spatial_d_model,
            configs.n_heads,
            cross_iterations,
            cross_epsilon,
            dropout,
        )
        self.spatial_horizon_projection = nn.Linear(
            self.spatial_d_model, self.pred_len
        )
        self.output_projection = nn.Linear(self.enc_in * 2, self.enc_in)

    def _normalize(self, x: torch.Tensor):
        """按样本和变量执行可逆实例归一化。"""
        mean = x.mean(dim=1, keepdim=True).detach()
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(variance + 1e-5).detach()
        return (x - mean) / std, mean, std

    def _temporal_path(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        seasonal_feature = self.temporal_seasonal_mlp(seasonal) + self.temporal_position
        trend_feature = self.temporal_trend_mlp(trend) + self.temporal_position
        temporal = self.temporal_fusion(seasonal_feature, trend_feature)
        temporal = self.temporal_channel_projection(temporal).transpose(1, 2)
        return self.temporal_horizon_projection(temporal).transpose(1, 2)

    def _spatial_path(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        seasonal_feature = self.spatial_seasonal_mlp(seasonal.transpose(1, 2))
        trend_feature = self.spatial_trend_mlp(trend.transpose(1, 2))

        trend_feature = self.trend_diffusion(trend_feature, self.trend_graph())
        seasonal_feature = self.seasonal_diffusion(
            seasonal_feature, self.seasonal_graph()
        )
        spatial = self.cross_diffusion(trend_feature, seasonal_feature)
        return self.spatial_horizon_projection(spatial).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.ndim != 3:
            raise ValueError("x_enc must have shape [batch, sequence, channel]")
        if x_enc.size(1) != self.seq_len or x_enc.size(2) != self.enc_in:
            raise ValueError(
                f"expected [B, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )

        normalized, mean, std = self._normalize(x_enc)
        seasonal, trend = self.decomposition(normalized)
        temporal_prediction = self._temporal_path(seasonal, trend)
        spatial_prediction = self._spatial_path(seasonal, trend)
        prediction = self.output_projection(
            torch.cat((temporal_prediction, spatial_prediction), dim=-1)
        )
        return prediction * std + mean
