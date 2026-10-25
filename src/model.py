"""Архитектура нейросети для классификации ЭКГ.

Модель: лёгкий 1D ResNet. Мультилейбл-классификация:
на выходе 5 независимых сигмоид (по одной на суперкласс),
потому что одна ЭКГ может иметь несколько диагнозов сразу.

Контракт данных:
    вход  — torch.Tensor формы (batch, 12, 1000), float32
    выход — torch.Tensor формы (batch, 5), логиты (до сигмоиды)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.preprocessing import load_config


class ResidualBlock1D(nn.Module):
    """Два свёрточных слоя + skip-связь (классический residual)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7,
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5,
                               padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # skip-связь: приводим размерность входа к выходу, если они различаются
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.shortcut(x))
        return out


class EcgResNet1D(nn.Module):
    """Лёгкий 1D ResNet для ЭКГ.

    Слои сжимают временную ось в 8 раз (1000 → 125 отсчётов),
    затем GlobalAveragePooling убирает время совсем —
    остаётся компактное представление сигнала для классификации.
    """

    def __init__(
        self,
        num_leads: int = 12,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_leads, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = ResidualBlock1D(32, 32)
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        self.layer3 = ResidualBlock1D(64, 128, stride=2)
        self.layer4 = ResidualBlock1D(128, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)  # логиты, без сигмоиды

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)      # (batch, 32, 1000)
        x = self.layer1(x)    # (batch, 32, 1000)
        x = self.layer2(x)    # (batch, 64, 500)
        x = self.layer3(x)    # (batch, 128, 250)
        x = self.layer4(x)    # (batch, 128, 125)
        x = self.pool(x)      # (batch, 128, 1)
        x = x.squeeze(-1)     # (batch, 128)
        x = self.dropout(x)
        x = self.fc(x)        # (batch, 5)
        return x


def build_model(config: dict | None = None) -> EcgResNet1D:
    """Создаёт модель с настройками из config.yaml."""
    if config is None:
        config = load_config()
    model_cfg = config["model"]
    sig_cfg = config["signal"]
    return EcgResNet1D(
        num_leads=sig_cfg["num_leads"],
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
    )