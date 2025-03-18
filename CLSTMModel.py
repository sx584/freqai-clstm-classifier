import torch
import torch.nn as nn


class CLSTMModel(nn.Module):
    """
    CLSTM (Convolutional LSTM) Modell basierend auf der Studie:
    Khattak et al. Journal of Big Data (2024) 11:58

    Das Modell kombiniert CNN-Schichten zur Merkmalsextraktion mit LSTM für
    zeitliche Abhängigkeiten und Dense-Schichten für die endgültige Klassifikation.

    Erweitert um mehrschichtige LSTM und Attention-Mechanismus.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cnn_blocks: int = 2,
        lstm_units: int = 64,
        lstm_layers: int = 1,
        dense_layers: int = 2,
        dense_neurons: int = 14,
        dropout_percent: float = 0.2,
        use_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention

        # CNN-Schichten
        self.cnn_layers = nn.Sequential()
        for i in range(cnn_blocks):
            # Wichtig: Für die erste Schicht ist in_channels=1,
            # für nachfolgende Schichten in_channels=32
            in_channels = 1 if i == 0 else 32
            out_channels = 32

            self.cnn_layers.add_module(
                f"conv1d_{i}",
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            self.cnn_layers.add_module(f"batchnorm_{i}", nn.BatchNorm1d(out_channels))
            self.cnn_layers.add_module(f"relu_{i}", nn.ReLU())
            if i < cnn_blocks - 1:  # Kein Dropout nach dem letzten CNN-Block
                self.cnn_layers.add_module(f"dropout_{i}", nn.Dropout(dropout_percent))

        # Mehrschichtige LSTM
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_percent if lstm_layers > 1 else 0
        )

        # Attention-Mechanismus (optional)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_units, lstm_units // 2),
                nn.Tanh(),
                nn.Linear(lstm_units // 2, 1)
            )

        # Dense-Schichten
        self.dense_layers = nn.Sequential()
        input_size = lstm_units
        for i in range(dense_layers):
            output_size = dense_neurons if i < dense_layers - 1 else dense_neurons * 2
            self.dense_layers.add_module(
                f"dense_{i}",
                nn.Linear(input_size, output_size)
            )
            self.dense_layers.add_module(f"relu_dense_{i}", nn.ReLU())
            if i < dense_layers - 1:  # Kein Dropout nach der letzten Dense-Schicht
                self.dense_layers.add_module(f"dropout_dense_{i}", nn.Dropout(dropout_percent))
            input_size = output_size

        # Ausgabeschicht
        self.output_layer = nn.Linear(input_size, output_dim)

    def attention_net(self, lstm_output):
        """
        Attention-Mechanismus für LSTM-Ausgaben
        """
        # (batch_size, seq_len, lstm_units)
        attn_weights = self.attention(lstm_output).squeeze(-1)
        soft_attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(soft_attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-Pass durch das Netzwerk

        :param x: Eingabedaten mit Form (batch_size, features)
        :return: Vorhersagen mit Form (batch_size, output_dim)
        """
        # Umformen für CNN: (batch_size, channels=1, sequence_length=features)
        x = x.unsqueeze(1)

        # CNN-Schichten anwenden
        x = self.cnn_layers(x)

        # Umformen für LSTM: (batch_size, sequence_length, input_size)
        x = x.permute(0, 2, 1)

        # LSTM anwenden
        lstm_out, _ = self.lstm(x)

        # Attention oder letzter LSTM-Output
        if self.use_attention:
            x = self.attention_net(lstm_out)
        else:
            x = lstm_out[:, -1, :]

        # Dense-Schichten anwenden
        x = self.dense_layers(x)

        # Ausgabeschicht anwenden
        x = self.output_layer(x)

        return x
