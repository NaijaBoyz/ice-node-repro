"""
ICE-NODE and baseline models implementation.

ICE-NODE architecture from the paper:
1. Clinical embeddings module (matrix or GRAM)
2. Neural ODE dynamics function  
3. Update function (GRU-based memory update)
4. Decoder function (MLP)

Fixed with paper-specific configurations:
- ODE solver: dopri5 with rtol=1e-3, atol=1e-4
- Embedding dim: 300, Memory dim: 30
- Dynamics function: 3-layer MLP (mlp3)
- Decoder: 2-layer MLP
"""

from typing import Optional, Tuple, Dict, Type, List
import torch
import torch.nn as nn
from torchdiffeq import odeint


class ClinicalEmbedding(nn.Module):
    """Simple matrix embedding: multi-hot vector -> dense embedding"""

    def __init__(self, vocab_size: int, emb_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        # Paper eq (4): g = W_M * v + b_M
        self.linear = nn.Linear(vocab_size, emb_dim, bias=True)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            codes: [B, V] or [B, T, V] multi-hot vectors
        Returns:
            embeddings: [B, emb_dim] or [B, T, emb_dim]
        """
        return self.linear(codes)


class ODEFunc(nn.Module):
    """
    Dynamics function f_d for the ODE: dh/dt = f_d(h; theta_d)

    From paper Section B.2: optimal architecture is mlp3
    (3-layer MLP with tanh activation, no bias)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # MLP with 3 layers, each of dimension hidden_dim (no bias as in paper)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

    def forward(self, t, h):
        """
        Args:
            t: scalar time (not used in autonomous ODE)
            h: [B, hidden_dim] hidden state
        Returns:
            dh_dt: [B, hidden_dim] time derivative
        """
        return self.net(h)


class MemoryUpdate(nn.Module):
    """
    Update function f_U that adjusts memory state at each timestamp.

    From paper eq (12):
    h_m(t_k+) = GRU(W_U [h_m(t_k-); g(t_k)] + b_U, h_m(t_k-))
    """

    def __init__(self, memory_dim: int, embedding_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        # Linear layer to project [h_m; g] to embedding_dim
        self.linear = nn.Linear(memory_dim + embedding_dim, embedding_dim, bias=True)
        
        # GRU cell for update (using standard GRU as in paper)
        self.gru_cell = nn.GRUCell(embedding_dim, memory_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        h_memory: torch.Tensor,
        g_obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_memory: [B, memory_dim] memory state before update
            g_obs: [B, embedding_dim] embedding of observed codes at t_k
        Returns:
            h_memory_new: [B, memory_dim] updated memory state
        """
        # Concatenate memory and observed embedding
        concat = torch.cat([h_memory, g_obs], dim=-1)  # [B, memory_dim + embedding_dim]

        # Project to embedding_dim
        x = self.linear(concat)  # [B, embedding_dim]

        # Update memory with GRU
        h_memory_new = self.gru_cell(x, h_memory)  # [B, memory_dim]

        return h_memory_new


class Decoder(nn.Module):
    """
    Decoder function f_D: embedding state -> predicted codes

    From paper Section B.3: MLP with 2 layers
    """

    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        # 2-layer MLP as in paper
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, vocab_size),
            # Note: sigmoid is applied in loss (BCEWithLogitsLoss), not here
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, h_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_embedding: [B, embedding_dim]
        Returns:
            logits: [B, vocab_size] (raw logits, not probabilities)
        """
        return self.net(h_embedding)


class ICENode(nn.Module):
    """
    ICE-NODE: Integration of Clinical Embeddings with Neural ODEs

    Architecture:
    - h(t) = [h_m(t); h_e(t)] where h_m is memory state, h_e is embedding state
    - d_h = d_m + d_e (paper uses d_m=30, d_e=300 as optimal)

    Training procedure (from paper Section 3.3.1):
    1. Initialize h(t_0) = [0; g(t_0)] where g(t_0) = embedding of codes at t_0
    2. For each timestamp t_k:
       a. Integrate: h(t_k-) = ODESolve(f_d, h(t_{k-1}+), [t_{k-1}, t_k])
       b. Predict: v_hat(t_k) = f_D(h_e(t_k-))
       c. Update: h_m(t_k+) = f_U(h_m(t_k-), g(t_k))
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,      # Paper optimal: 300
        memory_dim: int = 30,          # Paper optimal: 30  
        ode_method: str = "dopri5",    # Runge-Kutta 4(5) as in paper
        rtol: float = 1e-3,            # Paper values
        atol: float = 1e-4,            # Paper values
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.hidden_dim = memory_dim + embedding_dim
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol

        # Modules with paper-optimal configurations
        self.embedding = ClinicalEmbedding(vocab_size, embedding_dim)
        self.ode_func = ODEFunc(self.hidden_dim)
        self.memory_update = MemoryUpdate(memory_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, vocab_size)

        print(f"ICE-NODE initialized: hidden_dim={self.hidden_dim}, "
              f"embedding_dim={embedding_dim}, memory_dim={memory_dim}")

    def forward(
        self,
        times: torch.Tensor,   # [B, T]
        codes: torch.Tensor,   # [B, T, V]
        lengths: torch.Tensor, # [B]
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass implementing the integrate-predict-update cycle.

        Args:
            times: [B, max_T] timestamps in days since first discharge
            codes: [B, max_T, V] multi-hot code vectors
            lengths: [B] actual sequence lengths
            return_trajectory: if True, return list of hidden trajectories per patient

        Returns:
            predictions: [B, max_T-1, V] logits for codes at t_1, ..., t_{T-1}
            trajectories: optional list of tensors with hidden states per patient
        """
        batch_size = codes.size(0)
        max_len = codes.size(1)
        device = codes.device

        all_predictions = []
        all_hidden_states = [] if return_trajectory else None

        for b in range(batch_size):
            seq_len = lengths[b].item()
            if seq_len < 2:
                # No valid transitions for this patient
                all_predictions.append(
                    torch.zeros(0, self.vocab_size, device=device)
                )
                if return_trajectory:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))
                continue

            # Extract this patient's sequence
            patient_times = times[b, :seq_len]    # [T]
            patient_codes = codes[b, :seq_len]    # [T, V]

            # 1. Initialize at t_0: h(t_0) = [0; g(t_0)]
            g_0 = self.embedding(patient_codes[0])          # [emb_dim]
            h_memory = torch.zeros(self.memory_dim, device=device)  # [mem_dim]
            h_embedding = g_0                               # [emb_dim]
            h = torch.cat([h_memory, h_embedding], dim=0)   # [hidden_dim]

            patient_predictions = []
            patient_states = [] if return_trajectory else None

            # 2. Integrate-Predict-Update cycle for t_1, ..., t_{seq_len-1}
            for k in range(1, seq_len):
                # Time interval for integration
                t_span = torch.stack(
                    [patient_times[k - 1], patient_times[k]]
                )  # [2], same dtype/device as times

                # a. Integrate ODE: h(t_k-) = ODESolve(h(t_{k-1}+), [t_{k-1}, t_k])
                h0 = h.unsqueeze(0)  # [1, hidden_dim]
                
                try:
                    h_integrated = odeint(
                        self.ode_func,
                        h0,
                        t_span,
                        method=self.ode_method,
                        rtol=self.rtol,
                        atol=self.atol,
                    )  # [2, 1, hidden_dim]
                    h = h_integrated[-1, 0, :]  # [hidden_dim] at t_k-
                except Exception as e:
                    print(f"ODE integration failed: {e}")
                    # Fallback: use previous state
                    h = h0[0]

                # Split into memory and embedding states
                h_memory = h[:self.memory_dim]
                h_embedding = h[self.memory_dim:]

                # b. Predict: v_hat(t_k) = f_D(h_e(t_k-))
                logits_k = self.decoder(h_embedding.unsqueeze(0))  # [1, V]
                patient_predictions.append(logits_k)

                if return_trajectory and patient_states is not None:
                    patient_states.append(h.clone())

                # c. Update memory: h_m(t_k+) = f_U(h_m(t_k-), g(t_k))
                g_k = self.embedding(patient_codes[k])  # [emb_dim]
                h_memory_updated = self.memory_update(
                    h_memory.unsqueeze(0),
                    g_k.unsqueeze(0),
                ).squeeze(0)  # [mem_dim]

                # Reconstruct h for next iteration
                h = torch.cat([h_memory_updated, h_embedding], dim=0)

            # Stack predictions for this patient
            if len(patient_predictions) > 0:
                patient_preds_tensor = torch.cat(patient_predictions, dim=0)  # [seq_len-1, V]
            else:
                patient_preds_tensor = torch.zeros(0, self.vocab_size, device=device)

            all_predictions.append(patient_preds_tensor)

            if return_trajectory and patient_states is not None:
                if len(patient_states) > 0:
                    all_hidden_states.append(torch.stack(patient_states))  # [seq_len-1, H]
                else:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))

        # Pad predictions to max_len-1 (since we predict starting from t_1)
        max_pred_len = max_len - 1
        predictions = torch.zeros(batch_size, max_pred_len, self.vocab_size, device=device)

        for b in range(batch_size):
            pred_len = max(0, lengths[b].item() - 1)
            if pred_len > 0 and len(all_predictions[b]) > 0:
                predictions[b, :pred_len, :] = all_predictions[b][:pred_len]

        if return_trajectory:
            return predictions, all_hidden_states

        return predictions, None

    def predict_future(
        self,
        times: torch.Tensor,  # [T]
        codes: torch.Tensor,  # [T, V]
        future_time: float,
    ) -> torch.Tensor:
        """
        Predict codes at a future time given patient history.

        Args:
            times: [T] timestamps of patient history
            codes: [T, V] code vectors of patient history
            future_time: scalar, time to predict at
        Returns:
            logits: [V] predicted code logits at future_time
        """
        device = codes.device
        seq_len = len(times)

        # Initialize
        g_0 = self.embedding(codes[0])
        h_memory = torch.zeros(self.memory_dim, device=device)
        h_embedding = g_0
        h = torch.cat([h_memory, h_embedding], dim=0)

        # Process all history with integrate-update cycle (no predictions stored)
        for k in range(1, seq_len):
            t_span = torch.stack([times[k - 1], times[k]])

            # Integrate
            h0 = h.unsqueeze(0)
            h_integrated = odeint(
                self.ode_func, h0, t_span,
                method=self.ode_method, rtol=self.rtol, atol=self.atol
            )
            h = h_integrated[-1, 0, :]

            # Update with observed codes
            h_memory = h[:self.memory_dim]
            h_embedding = h[self.memory_dim:]
            g_k = self.embedding(codes[k])
            h_memory = self.memory_update(
                h_memory.unsqueeze(0),
                g_k.unsqueeze(0),
            ).squeeze(0)

            h = torch.cat([h_memory, h_embedding], dim=0)

        # Integrate to future time
        t_span = torch.stack([times[-1], torch.tensor(future_time, device=device)])
        h0 = h.unsqueeze(0)
        h_future = odeint(
            self.ode_func, h0, t_span,
            method=self.ode_method, rtol=self.rtol, atol=self.atol
        )[-1, 0, :]

        # Predict at future time
        h_embedding_future = h_future[self.memory_dim:]
        logits = self.decoder(h_embedding_future.unsqueeze(0)).squeeze(0)

        return logits


# [Keep the rest of your baseline models unchanged - GRUBaseline, RETAIN, etc.]
class GRUBaseline(nn.Module):
    """GRU baseline from paper (Choi et al. 2017)."""
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 50,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.code_embed = nn.Linear(vocab_size, emb_dim, bias=False)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        batch_size = codes.size(0)
        max_len = codes.size(1)

        # Embed codes
        x = self.code_embed(codes)  # [B, T, emb_dim]
        x = torch.relu(x)

        # Pack and run through GRU
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_len
        )  # [B, T, hidden_dim]

        # Predict at each timestep (except first)
        predictions = self.fc(out[:, :-1, :])  # [B, T-1, V]

        return predictions, None


class RETAIN(nn.Module):
    """RETAIN: Reverse Time Attention Model (Choi et al. 2016)"""
    
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Embedding
        self.code_embed = nn.Linear(vocab_size, emb_dim, bias=False)

        # Two GRUs for alpha (visit-level) and beta (variable-level) attention
        self.alpha_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.beta_gru = nn.GRU(emb_dim, hidden_dim, batch_first=True)

        # Attention generation
        self.alpha_fc = nn.Linear(hidden_dim, 1)
        self.beta_fc = nn.Linear(hidden_dim, emb_dim)

        # Output
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        batch_size = codes.size(0)
        max_len = codes.size(1)

        # Embed codes
        emb = self.code_embed(codes)  # [B, T, emb_dim]
        emb = torch.relu(emb)

        # Reverse time order for attention
        predictions_list = []

        for k in range(1, max_len):
            # For each patient, get history up to k-1
            history_emb = emb[:, :k, :]  # [B, k, emb_dim]

            # Reverse time order
            rev_emb = torch.flip(history_emb, dims=[1])  # [B, k, emb_dim]

            # Generate attention weights
            alpha_out, _ = self.alpha_gru(rev_emb)  # [B, k, hidden_dim]
            beta_out, _ = self.beta_gru(rev_emb)    # [B, k, hidden_dim]

            # Visit-level attention (alpha)
            alpha = torch.softmax(self.alpha_fc(alpha_out), dim=1)  # [B, k, 1]

            # Variable-level attention (beta)
            beta = torch.tanh(self.beta_fc(beta_out))  # [B, k, emb_dim]

            # Weighted sum
            context = torch.sum(alpha * beta * rev_emb, dim=1)  # [B, emb_dim]

            # Predict
            pred_k = self.fc(context)  # [B, V]
            predictions_list.append(pred_k.unsqueeze(1))

        predictions = torch.cat(predictions_list, dim=1)  # [B, T-1, V]

        # Mask predictions beyond sequence length
        for b in range(batch_size):
            seq_len = lengths[b].item()
            valid_steps = max(0, seq_len - 1)
            if valid_steps < (max_len - 1):
                predictions[b, valid_steps:, :] = 0.0

        return predictions, None


class ICENodeUniform(nn.Module):
    """ICE-NODE with uniform time intervals (1 week)."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        memory_dim: int = 30,
        ode_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.hidden_dim = memory_dim + embedding_dim
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol

        # Same modules as ICENode
        self.embedding = ClinicalEmbedding(vocab_size, embedding_dim)
        self.ode_func = ODEFunc(self.hidden_dim)
        self.memory_update = MemoryUpdate(memory_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, vocab_size)

    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[object]]:
        batch_size = codes.size(0)
        max_len = codes.size(1)
        device = codes.device

        all_predictions = []
        all_hidden_states = [] if return_trajectory else None

        for b in range(batch_size):
            seq_len = lengths[b].item()
            if seq_len < 2:
                all_predictions.append(
                    torch.zeros(0, self.vocab_size, device=device)
                )
                if return_trajectory:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))
                continue

            patient_codes = codes[b, :seq_len]

            # Initialize
            g_0 = self.embedding(patient_codes[0])
            h_memory = torch.zeros(self.memory_dim, device=device)
            h_embedding = g_0
            h = torch.cat([h_memory, h_embedding], dim=0)

            patient_predictions = []
            patient_states = []

            # Use fixed 7-day intervals instead of actual times
            for k in range(1, seq_len):
                # Fixed interval of 7 days
                t_span = torch.tensor([0.0, 7.0], device=device)

                # Integrate
                h0 = h.unsqueeze(0)
                h_integrated = odeint(
                    self.ode_func, h0, t_span,
                    method=self.ode_method, rtol=self.rtol, atol=self.atol
                )
                h = h_integrated[-1, 0, :]

                h_memory = h[:self.memory_dim]
                h_embedding = h[self.memory_dim:]

                # Predict
                logits_k = self.decoder(h_embedding.unsqueeze(0))
                patient_predictions.append(logits_k)

                if return_trajectory:
                    patient_states.append(h)

                # Update
                g_k = self.embedding(patient_codes[k])
                h_memory = self.memory_update(
                    h_memory.unsqueeze(0),
                    g_k.unsqueeze(0),
                ).squeeze(0)

                h = torch.cat([h_memory, h_embedding], dim=0)

            if len(patient_predictions) > 0:
                patient_preds_tensor = torch.cat(patient_predictions, dim=0)
            else:
                patient_preds_tensor = torch.zeros(0, self.vocab_size, device=device)

            all_predictions.append(patient_preds_tensor)

            if return_trajectory:
                if len(patient_states) > 0:
                    all_hidden_states.append(torch.stack(patient_states))
                else:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))

        max_pred_len = max_len - 1
        predictions = torch.zeros(batch_size, max_pred_len, self.vocab_size, device=device)

        for b in range(batch_size):
            pred_len = max(0, lengths[b].item() - 1)
            if pred_len > 0:
                predictions[b, :pred_len, :] = all_predictions[b]

        if return_trajectory:
            return predictions, all_hidden_states

        return predictions, None


class LogisticRegressionBaseline(nn.Module):
    """Simple logistic regression that ignores temporal information."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.fc = nn.Linear(vocab_size, vocab_size)

    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        batch_size = codes.size(0)
        max_len = codes.size(1)

        predictions_list = []

        for k in range(1, max_len):
            # Bag of codes from history (max over time)
            history = codes[:, :k, :]  # [B, k, V]
            bag = torch.max(history, dim=1)[0]  # [B, V]

            # Predict
            pred_k = self.fc(bag)  # [B, V]
            predictions_list.append(pred_k.unsqueeze(1))

        predictions = torch.cat(predictions_list, dim=1)  # [B, T-1, V]

        return predictions, None


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ICENode": ICENode,
    "ICENodeUniform": ICENodeUniform,
    "GRUBaseline": GRUBaseline,
    "RETAIN": RETAIN,
    "LogReg": LogisticRegressionBaseline,
}