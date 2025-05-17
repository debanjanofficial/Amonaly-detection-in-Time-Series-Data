# src/models/deep_learning_models.py

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout_prob, device):
        super(LSTMAutoencoder, self).__init__()

        self.input_dim = input_dim          # Number of features in input
        self.embedding_dim = embedding_dim  # Dimension of the latent space (bottleneck)
        self.hidden_dim = hidden_dim        # Dimension of LSTM hidden states
        self.n_layers = n_layers            # Number of LSTM layers
        self.dropout_prob = dropout_prob
        self.device = device

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True, # Input: (batch_size, seq_len, input_dim)
            dropout=dropout_prob if n_layers > 1 else 0 # Dropout only between LSTM layers
        )
        # Linear layer to further reduce to embedding dimension
        self.encoder_fc = nn.Linear(hidden_dim, embedding_dim)

        # Decoder
        # Linear layer to expand from embedding dimension
        self.decoder_fc = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, # Input to decoder LSTM will be the expanded embedding
            hidden_size=hidden_dim, # Outputting hidden_dim to match encoder's LSTM output before final FC
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_prob if n_layers > 1 else 0
        )
        # Final fully connected layer to reconstruct the original input_dim
        self.output_fc = nn.Linear(hidden_dim, input_dim)

        self.to(self.device)


    def forward(self, x_batch):
        # x_batch shape: (batch_size, sequence_length, input_dim)
        batch_size = x_batch.shape[0]
        seq_len = x_batch.shape[1]

        # ---- Encoder ----
        # Initialize hidden and cell states for encoder
        # h_n, c_n shape: (n_layers * num_directions, batch_size, hidden_dim)
        h0_encoder = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c0_encoder = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)

        # Pass input through encoder LSTM
        # lstm_out_encoder shape: (batch_size, seq_len, hidden_dim)
        # hidden_encoder is a tuple (h_n, c_n)
        lstm_out_encoder, (h_n_encoder, c_n_encoder) = self.encoder_lstm(x_batch, (h0_encoder, c0_encoder))

        # We are interested in the output of the last LSTM layer at the last time step
        # Or, use the output of all time steps from the last layer
        # For embedding, often the last hidden state h_n_encoder[-1] is used or output of last time step
        # Let's use the output of the LSTM at the last time step of the sequence from the final layer
        encoded_output = lstm_out_encoder[:, -1, :] # (batch_size, hidden_dim)
        embedding = self.encoder_fc(encoded_output) # (batch_size, embedding_dim)


        # ---- Decoder ----
        # The input to the decoder LSTM needs to be a sequence.
        # Repeat the embedding vector 'seq_len' times.
        # decoder_input_sequence shape: (batch_size, seq_len, embedding_dim) but LSTM expects hidden_dim
        decoder_hidden_inflated = self.decoder_fc(embedding) # (batch_size, hidden_dim)
        # Repeat this vector seq_len times to form the input sequence for the decoder LSTM
        decoder_input_sequence = decoder_hidden_inflated.unsqueeze(1).repeat(1, seq_len, 1)
        # Shape: (batch_size, seq_len, hidden_dim)

        # Initialize hidden and cell states for decoder
        # Use the final hidden/cell states of the encoder as initial states for the decoder,
        # but here feeding the embedding through an FC layer first.
        # Let's initialize decoder's h0, c0 like encoder's for simplicity or pass encoder's last state.
        # Using the encoder's last hidden state (after passing through decoder_fc to match dimensions if needed)
        # Let's initialize with zeros, as the full embedding is repeated as input.
        h0_decoder = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        c0_decoder = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        # Alternatively, could use h_n_encoder, c_n_encoder if they align with decoder_lstm input.

        lstm_out_decoder, _ = self.decoder_lstm(decoder_input_sequence, (h0_decoder, c0_decoder))
        # lstm_out_decoder shape: (batch_size, seq_len, hidden_dim)

        # Pass the decoder LSTM's output through the final FC layer
        reconstructed_x = self.output_fc(lstm_out_decoder)
        # reconstructed_x shape: (batch_size, seq_len, input_dim)

        return reconstructed_x