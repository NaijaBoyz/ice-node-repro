from typing import Optional, Tuple, Dict, Type, List
import warnings

import torch
import torch.nn as nn
from torch.autograd import grad  # kept in case you want to extend
from torchdiffeq import odeint

SECOND = 1 / 3600.0
HOUR = 1.0
DAY = 24.0 * HOUR
WEEK = 7.0 * DAY
MONTH = 30.0 * DAY
YEAR = 365.0 * DAY


class ClinicalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 300):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.linear = nn.Linear(vocab_size, emb_dim, bias=True)

        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.linear(codes)


class DemographicEmbedding(nn.Module):
    def __init__(self, demographic_dim: int, emb_dim: int):
        super().__init__()
        self.demographic_dim = demographic_dim
        self.emb_dim = emb_dim
        self.embedding_network = nn.Sequential(
            nn.Linear(demographic_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        for m in self.embedding_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, demographic_features: torch.Tensor) -> torch.Tensor:
        return self.embedding_network(demographic_features)


class AugmentedODEFunc(nn.Module):
    def __init__(self, base_ode_func: nn.Module, reg_order: int = 1):
        super().__init__()
        self.base_func = base_ode_func
        self.reg_order = reg_order
        self.nfe = 0

    def forward(self, t, augmented_state):
        self.nfe += 1

        *batch_dims, state_dim = augmented_state.shape
        h = augmented_state[..., :-1]
        with torch.set_grad_enabled(True):
            if not h.requires_grad:
                h = h.requires_grad_(True)
            dh_dt = self.base_func(t, h)
            if self.reg_order == 1:
                derivative = dh_dt
            elif self.reg_order == 2:
                d2h_dt2 = self._compute_second_derivative(h, dh_dt)
                derivative = d2h_dt2
            elif self.reg_order == 3:
                derivative = dh_dt
            else:
                derivative = dh_dt
            dreg_dt = (derivative ** 2).sum(dim=-1, keepdim=True)

        augmented_derivative = torch.cat([dh_dt, dreg_dt], dim=-1)
        return augmented_derivative

    def _compute_second_derivative(self, h, dh_dt):
        dh_dt = torch.clamp(dh_dt, min=-10.0, max=10.0)

        try:
            jvp = torch.autograd.functional.jvp(
                lambda x: self.base_func(0, x),
                h,
                dh_dt,
                create_graph=True,
            )[1]

            return torch.clamp(jvp, min=-10.0, max=10.0)

        except RuntimeError:
            return dh_dt


class ODEFunc(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)

    def forward(self, t, h):
        h = torch.clamp(h, min=-10.0, max=10.0)
        dh_dt = self.net(h)
        dh_dt = torch.clamp(dh_dt, min=-1.0, max=1.0)
        return dh_dt


class MemoryUpdate(nn.Module):
    def __init__(self, memory_dim: int, embedding_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        self.embedding_dim = embedding_dim

        self.linear = nn.Linear(memory_dim + embedding_dim, embedding_dim, bias=True)
        self.gru_cell = nn.GRUCell(embedding_dim, memory_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, h_memory: torch.Tensor, g_obs: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([h_memory, g_obs], dim=-1)
        x = self.linear(concat)
        h_memory_new = self.gru_cell(x, h_memory)
        return h_memory_new


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, vocab_size),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, h_embedding: torch.Tensor) -> torch.Tensor:
        return self.net(h_embedding)


class ICENodeAugmented(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        memory_dim: int = 30,
        ode_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_dt: float = 365.0,
        reg_order: int = 1,
        demographic_dim: int = 3,
        timescale: float = 7.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.memory_dim = memory_dim
        self.hidden_dim = memory_dim + embedding_dim
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol
        self.max_dt = max_dt
        self.reg_order = reg_order
        self.demographic_dim = demographic_dim
        self.timescale = timescale

        self.SECOND = SECOND
        self.embedding = ClinicalEmbedding(vocab_size, embedding_dim)
        self.demographic_embedding = DemographicEmbedding(
            demographic_dim=demographic_dim,
            emb_dim=embedding_dim,
        )
        self.ode_func = ODEFunc(self.hidden_dim)
        self.augmented_ode_func = AugmentedODEFunc(self.ode_func, reg_order=reg_order)
        self.memory_update = MemoryUpdate(memory_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, vocab_size)

        print(
            f"ICE-NODE (AUGMENTED) initialized: hidden_dim={self.hidden_dim}, "
            f"embedding_dim={embedding_dim}, memory_dim={memory_dim}, "
            f"reg_order={reg_order}, max_dt={max_dt}, "
            f"demographic_dim={demographic_dim}, SECOND={SECOND}"
        )

    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        compute_regularization: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        batch_size = codes.size(0)
        max_len = codes.size(1)
        device = codes.device

        all_predictions: List[torch.Tensor] = []
        all_hidden_states: Optional[List[torch.Tensor]] = [] if return_trajectory else None
        total_reg = torch.tensor(0.0, device=device)
        total_ode_time = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            seq_len = lengths[b].item()
            if seq_len < 2:
                all_predictions.append(torch.zeros(0, self.vocab_size, device=device))
                if return_trajectory and all_hidden_states is not None:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))
                continue

            patient_times = times[b, :seq_len]
            patient_codes = codes[b, :seq_len]

            g_0 = self.embedding(patient_codes[0])
            h_memory = torch.zeros(self.memory_dim, device=device)

            if demographic_features is not None:
                patient_demographics = demographic_features[b]
                demographic_embedding = self.demographic_embedding(
                    patient_demographics.unsqueeze(0)
                ).squeeze(0)
                h_embedding = g_0 + demographic_embedding
            else:
                h_embedding = g_0

            h = torch.cat([h_memory, h_embedding], dim=0)

            patient_predictions: List[torch.Tensor] = []
            patient_states: Optional[List[torch.Tensor]] = [] if return_trajectory else None

            for k in range(1, seq_len):
                t_prev = patient_times[k - 1]
                t_curr = patient_times[k]
                dt_days = float((t_curr - t_prev).item())
                dt_days = min(dt_days, self.max_dt)
                dt = dt_days / self.timescale

                if dt >= 1e-6 and compute_regularization:
                    h_aug = torch.cat([h, torch.zeros(1, device=device)])
                    h_aug = h_aug.unsqueeze(0)

                    t_span = torch.tensor([0.0, dt], device=device, dtype=torch.float32)

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            aug_trajectory = odeint(
                                self.augmented_ode_func,
                                h_aug,
                                t_span,
                                method=self.ode_method,
                                rtol=self.rtol,
                                atol=self.atol,
                                options={"max_num_steps": 1000},
                            )

                            final_aug_state = aug_trajectory[-1, 0, :]

                            h_new = final_aug_state[:-1]
                            reg_increment = final_aug_state[-1]

                            if torch.isnan(h_new).any() or torch.isinf(h_new).any():
                                h_new = h
                                reg_increment = torch.tensor(0.0, device=device)

                            h = h_new
                            total_reg = total_reg + reg_increment
                            total_ode_time = total_ode_time + dt

                    except Exception:
                        pass

                elif dt >= 1e-6:
                    t_span = torch.tensor([0.0, dt], device=device, dtype=torch.float32)
                    h0 = h.unsqueeze(0)

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            h_integrated = odeint(
                                self.ode_func,
                                h0,
                                t_span,
                                method=self.ode_method,
                                rtol=self.rtol,
                                atol=self.atol,
                                options={"max_num_steps": 1000},
                            )

                            h_new = h_integrated[-1, 0, :]

                            if not (torch.isnan(h_new).any() or torch.isinf(h_new).any()):
                                h = h_new
                    except Exception:
                        pass

                h_memory = h[: self.memory_dim]
                h_embedding = h[self.memory_dim :]

                logits_k = self.decoder(h_embedding.unsqueeze(0)).squeeze(0)
                patient_predictions.append(logits_k)

                if return_trajectory and patient_states is not None:
                    patient_states.append(h.clone())

                g_k = self.embedding(patient_codes[k])
                h_memory_updated = self.memory_update(
                    h_memory.unsqueeze(0),
                    g_k.unsqueeze(0),
                ).squeeze(0)

                h = torch.cat([h_memory_updated, h_embedding], dim=0)

            if len(patient_predictions) > 0:
                patient_preds_tensor = torch.stack(patient_predictions)
            else:
                patient_preds_tensor = torch.zeros(0, self.vocab_size, device=device)

            all_predictions.append(patient_preds_tensor)

            if return_trajectory and all_hidden_states is not None and patient_states is not None:
                if len(patient_states) > 0:
                    all_hidden_states.append(torch.stack(patient_states))
                else:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))

        max_pred_len = max_len - 1
        predictions = torch.zeros(batch_size, max_pred_len, self.vocab_size, device=device)
        for b in range(batch_size):
            pred_len = max(0, lengths[b].item() - 1)
            if pred_len > 0 and all_predictions[b].size(0) > 0:
                actual_len = min(pred_len, all_predictions[b].size(0))
                predictions[b, :actual_len, :] = all_predictions[b][:actual_len]

        if compute_regularization and total_ode_time > 0:
            reg_value = total_reg / total_ode_time
        else:
            reg_value = None

        return predictions, all_hidden_states, reg_value


class GRUBaseline(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 300, hidden_dim: int = 128):
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
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.code_embed.weight)
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        times,
        codes,
        lengths,
        demographic_features=None,
        return_trajectory: bool = False,
        compute_regularization: bool = False,
    ):
        batch_size = codes.size(0)
        max_len = codes.size(1)

        x = self.code_embed(codes)
        x = torch.relu(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=max_len
        )

        logits = self.decoder(out[:, :-1, :])

        return logits, None, None


class RETAINBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 300,
        state_a_size: int = 128,
        state_b_size: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.state_a_size = state_a_size
        self.state_b_size = state_b_size

        self.code_embed = nn.Linear(vocab_size, emb_dim, bias=False)

        self.gru_a = nn.GRU(
            input_size=emb_dim,
            hidden_size=state_a_size,
            num_layers=1,
            batch_first=True,
        )
        self.gru_b = nn.GRU(
            input_size=emb_dim,
            hidden_size=state_b_size,
            num_layers=1,
            batch_first=True,
        )

        self.att_a = nn.Linear(state_a_size, 1)
        self.att_b = nn.Linear(state_b_size, emb_dim)

        self.decoder = nn.Linear(emb_dim, vocab_size)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.code_embed.weight)
        for layer in [self.att_a, self.att_b, self.decoder]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        times,
        codes,
        lengths,
        demographic_features=None,
        return_trajectory: bool = False,
        compute_regularization: bool = False,
    ):
        device = codes.device
        batch_size, max_len, vocab_size = codes.shape

        v = self.code_embed(codes)
        out_a, _ = self.gru_a(v)
        out_b, _ = self.gru_b(v)
        e = self.att_a(out_a).squeeze(-1)
        b = torch.tanh(self.att_b(out_b))
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        e_masked = e.masked_fill(~mask, float('-inf'))
        alpha = torch.softmax(e_masked, dim=1)
        alpha = alpha.unsqueeze(-1)
        w = alpha * b * v
        cumsum_w = torch.cumsum(w, dim=1)
        context = cumsum_w[:, :-1, :]
        logits = self.decoder(context)

        return logits, None, None


class ICENode(ICENodeAugmented):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        memory_dim: int = 30,
        ode_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_dt: float = 365.0,
        reg_order: int = 1,
        timescale: float = 7.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            memory_dim=memory_dim,
            ode_method=ode_method,
            rtol=rtol,
            atol=atol,
            max_dt=max_dt,
            reg_order=reg_order,
            demographic_dim=0,
            timescale=timescale,
        )
        print(f"ICE-NODE (non-augmented) initialized: no demographics")


class ICENodeUniform(ICENodeAugmented):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        memory_dim: int = 30,
        ode_method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-4,
        max_dt: float = 365.0,
        reg_order: int = 1,
        demographic_dim: int = 3,
        timescale: float = 7.0,
        uniform_dt: float = 7.0,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            memory_dim=memory_dim,
            ode_method=ode_method,
            rtol=rtol,
            atol=atol,
            max_dt=max_dt,
            reg_order=reg_order,
            demographic_dim=demographic_dim,
            timescale=timescale,
        )
        self.uniform_dt = uniform_dt
        print(f"ICE-NODE UNIFORM initialized: fixed dt={uniform_dt} days")
    
    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        compute_regularization: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        batch_size = codes.size(0)
        max_len = codes.size(1)
        device = codes.device

        all_predictions: List[torch.Tensor] = []
        all_hidden_states: Optional[List[torch.Tensor]] = [] if return_trajectory else None
        total_reg = torch.tensor(0.0, device=device)
        total_ode_time = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            seq_len = lengths[b].item()
            if seq_len < 2:
                all_predictions.append(torch.zeros(0, self.vocab_size, device=device))
                if return_trajectory and all_hidden_states is not None:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))
                continue

            patient_codes = codes[b, :seq_len]

            g_0 = self.embedding(patient_codes[0])
            h_memory = torch.zeros(self.memory_dim, device=device)

            if demographic_features is not None:
                patient_demographics = demographic_features[b]
                demographic_embedding = self.demographic_embedding(
                    patient_demographics.unsqueeze(0)
                ).squeeze(0)
                h_embedding = g_0 + demographic_embedding
            else:
                h_embedding = g_0

            h = torch.cat([h_memory, h_embedding], dim=0)

            patient_predictions: List[torch.Tensor] = []
            patient_states: Optional[List[torch.Tensor]] = [] if return_trajectory else None

            for k in range(1, seq_len):
                dt = self.uniform_dt / self.timescale

                if dt >= 1e-6 and compute_regularization:
                    h_aug = torch.cat([h, torch.zeros(1, device=device)])
                    h_aug = h_aug.unsqueeze(0)

                    t_span = torch.tensor([0.0, dt], device=device, dtype=torch.float32)

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            aug_trajectory = odeint(
                                self.augmented_ode_func,
                                h_aug,
                                t_span,
                                method=self.ode_method,
                                rtol=self.rtol,
                                atol=self.atol,
                                options={"max_num_steps": 1000},
                            )

                            final_aug_state = aug_trajectory[-1, 0, :]
                            h_new = final_aug_state[:-1]
                            reg_increment = final_aug_state[-1]

                            if torch.isnan(h_new).any() or torch.isinf(h_new).any():
                                h_new = h
                                reg_increment = torch.tensor(0.0, device=device)

                            h = h_new
                            total_reg = total_reg + reg_increment
                            total_ode_time = total_ode_time + dt

                    except Exception:
                        pass

                elif dt >= 1e-6:
                    t_span = torch.tensor([0.0, dt], device=device, dtype=torch.float32)
                    h0 = h.unsqueeze(0)

                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")

                            h_integrated = odeint(
                                self.ode_func,
                                h0,
                                t_span,
                                method=self.ode_method,
                                rtol=self.rtol,
                                atol=self.atol,
                                options={"max_num_steps": 1000},
                            )

                            h_new = h_integrated[-1, 0, :]

                            if not (torch.isnan(h_new).any() or torch.isinf(h_new).any()):
                                h = h_new
                    except Exception:
                        pass

                h_memory_part = h[: self.memory_dim]
                h_embedding_part = h[self.memory_dim :]

                pred_logits = self.decoder(h_embedding_part)
                patient_predictions.append(pred_logits)

                if return_trajectory and patient_states is not None:
                    patient_states.append(h.detach().cpu())

                g_k = self.embedding(patient_codes[k])
                h_memory_part = self.memory_update(
                    h_memory_part.unsqueeze(0),
                    g_k.unsqueeze(0),
                ).squeeze(0)
                h = torch.cat([h_memory_part, h_embedding_part], dim=0)

            if patient_predictions:
                stacked_preds = torch.stack(patient_predictions, dim=0)
                all_predictions.append(stacked_preds)
            else:
                all_predictions.append(torch.zeros(0, self.vocab_size, device=device))

            if return_trajectory and all_hidden_states is not None:
                if patient_states:
                    all_hidden_states.append(torch.stack(patient_states, dim=0))
                else:
                    all_hidden_states.append(torch.zeros(0, self.hidden_dim, device=device))

        max_pred_len = max((p.size(0) for p in all_predictions), default=0)
        if max_pred_len == 0:
            logits_out = torch.zeros(batch_size, 0, self.vocab_size, device=device)
        else:
            logits_out = torch.zeros(batch_size, max_pred_len, self.vocab_size, device=device)
            for i, preds in enumerate(all_predictions):
                if preds.size(0) > 0:
                    logits_out[i, : preds.size(0), :] = preds

        avg_reg = total_reg / max(total_ode_time, 1e-6) if compute_regularization else None

        return logits_out, all_hidden_states, avg_reg


class LogRegBaseline(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(vocab_size, vocab_size, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        print(f"LogReg baseline initialized: simple bag-of-codes")
    
    def forward(
        self,
        times: torch.Tensor,
        codes: torch.Tensor,
        lengths: torch.Tensor,
        demographic_features: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        compute_regularization: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        batch_size = codes.size(0)
        max_len = codes.size(1)
        device = codes.device
        
        all_predictions: List[torch.Tensor] = []
        
        for b in range(batch_size):
            seq_len = lengths[b].item()
            if seq_len < 2:
                all_predictions.append(torch.zeros(0, self.vocab_size, device=device))
                continue
            
            patient_codes = codes[b, :seq_len]
            patient_predictions: List[torch.Tensor] = []
            
            for k in range(1, seq_len):
                cumulative_codes = patient_codes[:k].sum(dim=0)
                cumulative_codes = torch.clamp(cumulative_codes, 0, 1)
                logits = self.linear(cumulative_codes)
                patient_predictions.append(logits)
            
            if patient_predictions:
                stacked_preds = torch.stack(patient_predictions, dim=0)
                all_predictions.append(stacked_preds)
            else:
                all_predictions.append(torch.zeros(0, self.vocab_size, device=device))
        
        max_pred_len = max((p.size(0) for p in all_predictions), default=0)
        if max_pred_len == 0:
            logits_out = torch.zeros(batch_size, 0, self.vocab_size, device=device)
        else:
            logits_out = torch.zeros(batch_size, max_pred_len, self.vocab_size, device=device)
            for i, preds in enumerate(all_predictions):
                if preds.size(0) > 0:
                    logits_out[i, : preds.size(0), :] = preds

        l1_penalty = torch.tensor(0.0, device=device)
        if compute_regularization:
            num_params = self.linear.weight.numel()
            l1_penalty = self.linear.weight.abs().sum() / num_params

        return logits_out, None, l1_penalty


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "ICENode": ICENode,
    "ICENodeAugmented": ICENodeAugmented,
    "ICENodeUniform": ICENodeUniform,
    "GRUBaseline": GRUBaseline,
    "RETAINBaseline": RETAINBaseline,
    "LogRegBaseline": LogRegBaseline,
}