import inspect

import torch

import typeguard
import typing


@typeguard.typechecked
class VanillaAutoencoder(torch.nn.Module):
    def __init__(
        self, num_feat: int, num_hidden: int, num_encoding: int, dropout_rate: float
    ) -> None:
        super().__init__()

        args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.num_feat, track_running_stats=True),
            torch.nn.Linear(
                in_features=self.num_feat, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_encoding, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(
                num_features=self.num_encoding, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_encoding, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_feat, bias=True
            ),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(input=x)
        return self.decoder(input=x)


@typeguard.typechecked
class ConvolutionalAutoencoder(torch.nn.Module):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int,
        pool_size: int,
        num_feat: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.num_feat, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.num_feat,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.MaxPool1d(
                kernel_size=self.pool_size,
                stride=self.pool_size,
                padding=0,
                dilation=1,
                return_indices=False,
                ceil_mode=False,
            ),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Upsample(scale_factor=self.pool_size),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.BatchNorm1d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv1d(
                in_channels=self.filters,
                out_channels=self.num_feat,
                kernel_size=self.kernel_size,
                stride=self.strides,
                dilation=1,
                bias=True,
                padding=int((self.kernel_size - 1) / 2),
            ),
            torch.nn.ConstantPad1d(padding=(1, 0), value=0),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(input=x)
        return self.decoder(input=x)


@typeguard.typechecked
class Seq2SeqAutoencoder(torch.nn.Module):
    def __init__(
        self,
        num_feat: int,
        num_hidden: int,
        num_layers: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.encoder = torch.nn.LSTM(
            input_size=self.num_feat,
            hidden_size=self.num_hidden,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            dropout=self.dropout_rate,
            bidirectional=False,
        )

        self.intermediate = torch.nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            num_layers=self.num_layers - 1,
            batch_first=True,
            bias=True,
            dropout=self.dropout_rate,
            bidirectional=False,
        )

        self.decoder = torch.nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_feat,
            num_layers=1,
            batch_first=True,
            bias=True,
            dropout=0.0,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding, (_, _) = self.encoder(input=x)
        encoding = (
            encoding[:, -1, :].unsqueeze(dim=1).repeat(1, encoding.size(dim=1), 1)
        )

        intermediate, (_, _) = self.intermediate(input=encoding)
        decoding, (_, _) = self.decoder(input=intermediate)

        return decoding


@typeguard.typechecked
class VariationalAutoencoder(torch.nn.Module):
    def __init__(
        self, num_feat: int, num_hidden: int, num_encoding: int, dropout_rate: float
    ) -> None:
        super().__init__()

        args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=self.num_feat, track_running_stats=True),
            torch.nn.Linear(
                in_features=self.num_feat, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_encoding, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
        )

        self.mu = torch.nn.Linear(
            in_features=self.num_encoding, out_features=self.num_encoding, bias=True
        )
        self.log_sigma = torch.nn.Linear(
            in_features=self.num_encoding, out_features=self.num_encoding, bias=True
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(
                num_features=self.num_encoding, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_encoding, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_hidden, bias=True
            ),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Dropout(p=self.dropout_rate, inplace=False),
            torch.nn.BatchNorm1d(
                num_features=self.num_hidden, track_running_stats=True
            ),
            torch.nn.Linear(
                in_features=self.num_hidden, out_features=self.num_feat, bias=True
            ),
            torch.nn.Sigmoid(),
        )

    def forward(
        self, x: torch.Tensor, prior: torch.Tensor
    ) -> typing.Dict[str, torch.Tensor]:
        out = {}
        encoding = self.encoder(input=x)
        out["mu"] = self.mu(input=encoding)
        out["log_sigma"] = self.log_sigma(input=encoding)

        posterior = out["mu"] + (torch.exp(input=out["log_sigma"] / 2) * prior)

        out["decoding"] = self.decoder(input=posterior)

        return out

    def vae_loss(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mu: torch.Tensor,
        log_sigma: torch.Tensor,
        beta: float,
    ) -> torch.Tensor:
        recon = torch.nn.MSELoss(reduction="mean")(input=y_pred, target=y_true)
        kl = -0.5 * torch.sum(
            input=1 + log_sigma - torch.exp(input=log_sigma) - mu.pow(exponent=2),
            axis=-1,
        )
        return recon + (beta * torch.mean(input=kl))

    def feature_extractor(self, x: torch.Tensor) -> typing.Dict[str, torch.Tensor]:
        out = {}
        encoding = self.encoder(input=x)
        out["mu"] = self.mu(input=encoding)
        out["log_sigma"] = self.log_sigma(input=encoding)
        return out


@typeguard.typechecked
class Convolutional2DAutoencoder(torch.nn.Module):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int,
        pool_size: int,
        num_channels: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        args, _, _, values = inspect.getargvalues(frame=inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

        self.encoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(
                num_features=self.num_channels, track_running_stats=True
            ),
            torch.nn.Conv2d(
                in_channels=self.num_channels,
                out_channels=self.filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.strides, self.strides),
                dilation=(1, 1),
                bias=True,
                padding=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            ),
            torch.nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.BatchNorm2d(num_features=self.filters, track_running_stats=True),
            torch.nn.Dropout2d(p=self.dropout_rate, inplace=False),
            torch.nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.strides, self.strides),
                dilation=(1, 1),
                bias=True,
                padding=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            ),
            torch.nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.BatchNorm2d(num_features=self.filters, track_running_stats=True),
            torch.nn.Dropout2d(p=self.dropout_rate, inplace=False),
            torch.nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.strides, self.strides),
                dilation=(1, 1),
                bias=True,
                padding=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            ),
            torch.nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.MaxPool2d(
                kernel_size=(self.pool_size, self.pool_size),
                stride=(self.pool_size, self.pool_size),
                padding=(0, 0),
                dilation=(1, 1),
                return_indices=False,
                ceil_mode=False,
            ),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm2d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.strides, self.strides),
                dilation=(1, 1),
                bias=True,
                padding=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            ),
            torch.nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.Upsample(scale_factor=self.pool_size),
            torch.nn.BatchNorm2d(num_features=self.filters, track_running_stats=True),
            torch.nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.filters,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.strides, self.strides),
                dilation=(1, 1),
                bias=True,
                padding=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            ),
            torch.nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0),
            torch.nn.ELU(alpha=1.0, inplace=False),
            torch.nn.BatchNorm2d(num_features=self.filters, track_running_stats=True),
            torch.nn.Dropout2d(p=self.dropout_rate, inplace=False),
            torch.nn.Conv2d(
                in_channels=self.filters,
                out_channels=self.num_channels,
                kernel_size=(self.kernel_size, self.kernel_size),
                stride=(self.strides, self.strides),
                dilation=(1, 1),
                bias=True,
                padding=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            ),
            torch.nn.ConstantPad2d(padding=(1, 0, 1, 0), value=0),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoding = self.encoder(input=x)
        return self.decoder(input=encoding)
