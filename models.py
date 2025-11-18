from typing import Optional
import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 50,
        hidden_dim: int = 50,  # change to 128 to match paper if needed
        use_time_deltas: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.use_time_deltas = use_time_deltas

        # map multi-hot visit vector -> embedding. bias=False so zero input -> zero embedding
        self.code_embed = nn.Linear(vocab_size, emb_dim, bias=False)

        # If using time deltas, project them to emb_dim then concat
        if self.use_time_deltas:
            self.delta_proj = nn.Linear(1, emb_dim, bias=True)
            gru_input_size = emb_dim * 2
        else:
            self.delta_proj = None
            gru_input_size = emb_dim

        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        codes: torch.Tensor,
        deltas: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        codes: [B, T, V] (float multi-hot per visit)
        deltas: [B, T] or [B, T, 1] (optional)
        lengths: [B] lengths in time steps (int)
        """
        # codes -> [B, T, emb_dim]
        x = self.code_embed(codes)  # linear with no bias: zeros -> zeros
        x = torch.relu(x)

        if self.use_time_deltas:
            if deltas is None:
                raise ValueError("Model configured to use_time_deltas but deltas is None")
            # ensure deltas shape [B, T, 1]
            if deltas.dim() == 2:
                d = deltas.unsqueeze(-1)
            else:
                d = deltas
            d_emb = torch.relu(self.delta_proj(d))
            x = torch.cat([x, d_emb], dim=-1)  # [B, T, emb_dim*2]

        if lengths is not None:
            # pack padded sequence so GRU doesn't process padded timesteps
            # lengths must be on CPU for pack_padded_sequence
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, h_n = self.gru(packed)
            # h_n: (num_layers * num_directions, batch, hidden_size)
            # for a single-layer unidirectional GRU, h_n[-1] is the last hidden for each sequence
            last_hidden = h_n[-1]  # [B, hidden_dim]
        else:
            out, _ = self.gru(x)  # out [B, T, H]
            last_hidden = out[:, -1, :]  # last step

        logits = self.fc(last_hidden)  # [B, V]
        return logits


MODEL_REGISTRY = {
    "GRUBaseline": GRUBaseline,
}