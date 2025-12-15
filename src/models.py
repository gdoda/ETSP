import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import config


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        channels = config.cnn_channels

        # CNN architecture
        self.conv1 = nn.Conv2d(
            config.in_channels, channels[0], kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(
            channels[0], channels[1], kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(channels[1])
        self.conv3 = nn.Conv2d(
            channels[1], channels[2], kernel_size=3, stride=1, padding=1
        )
        self.bn3 = nn.BatchNorm2d(channels[2])
        self.conv4 = nn.Conv2d(
            channels[2], channels[3], kernel_size=3, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(channels[3])

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[3], channels[3], kernel_size=1),
            nn.Sigmoid(),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.classifier = nn.Linear(channels[3], config.num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        attention_weights = self.attention(x)
        x = x * attention_weights

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        patch_size = config.vit_patch_size
        embed_dim = config.vit_embed_dim

        self.patch_size = patch_size
        self.n_patches = (config.img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            config.in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))

        # Transformer encoder layers
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=config.vit_num_heads,
                    dim_feedforward=int(embed_dim * config.vit_mlp_ratio),
                    dropout=config.vit_dropout,
                    activation="gelu",
                    batch_first=True,
                )
                for _ in range(config.vit_depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, config.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head
        x = self.norm(x)
        x = x[:, 0]  # Class token
        x = self.head(x)

        return x


class AudioRNN(nn.Module):
    """
    RNN model for audio classification using MFCC features.

    Input: MFCC features of shape (batch, n_frames, n_features)
           where n_features = n_mfcc * 3 (static + delta + delta-delta) if use_mfcc_deltas
    """

    def __init__(self):
        super(AudioRNN, self).__init__()

        input_size = config.mfcc_feature_dim  # 39 if use_mfcc_deltas else 13
        hidden_size = config.rnn_hidden_size

        self.hidden_size = hidden_size
        self.num_layers = config.rnn_num_layers

        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            config.rnn_num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.rnn_num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(hidden_size, config.num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: MFCC features of shape (batch, n_frames, n_features)

        Returns:
            logits: shape (batch, num_classes)
        """
        # Project input features
        x = self.input_proj(x)  # (batch, n_frames, hidden_size)

        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)  # (batch, n_frames, hidden_size * 2)

        # Attention mechanism
        attention_scores = self.attention(lstm_out).squeeze(-1)  # (batch, n_frames)
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, n_frames)
        context_vector = torch.sum(
            lstm_out * attention_weights.unsqueeze(-1), dim=1
        )  # (batch, hidden_size * 2)

        # Classification
        logits = self.classifier(context_vector)  # (batch, num_classes)

        return logits
