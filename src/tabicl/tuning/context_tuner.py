from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

from tabicl import TabICL


class ContextTuner(nn.Module):
    """Learnable context tokens (prompt tuning) for a frozen TabICL model.

    Prepends ``n_context_tokens`` learnable vectors at the TF_icl input,
    analogous to TuneTables-style prompt tuning. The frozen model's
    column embedder and row interactor produce representations as usual;
    the context tokens are inserted before the ICL transformer so that
    test queries attend to them alongside (or instead of) real labeled data.

    Parameters
    ----------
    model : TabICL
        A pre-trained TabICL model. All its parameters will be frozen.

    n_context_tokens : int, default=128
        Number of learnable context tokens (*p*).

    init_strategy : str, default="normal"
        Initialization for context tokens:

        - ``"normal"``: Truncated normal with std 0.02.
        - ``"zeros"``: All zeros.
        - ``"uniform"``: Uniform on [-0.02, 0.02].
    """

    def __init__(
        self,
        model: TabICL,
        n_context_tokens: int = 128,
        init_strategy: str = "normal",
    ):
        super().__init__()
        self.model = model
        self.n_context_tokens = n_context_tokens

        # Freeze the entire pretrained model
        for param in model.parameters():
            param.requires_grad_(False)
        model.eval()

        # d_model for TF_icl = embed_dim * row_num_cls
        d_model = model.embed_dim * model.row_num_cls
        self.d_model = d_model

        # Learnable context tokens
        self.context_tokens = nn.Parameter(torch.empty(n_context_tokens, d_model))
        self._init_context_tokens(init_strategy)

    def _init_context_tokens(self, strategy: str) -> None:
        if strategy == "normal":
            trunc_normal_(self.context_tokens, std=0.02)
        elif strategy == "zeros":
            nn.init.zeros_(self.context_tokens)
        elif strategy == "uniform":
            nn.init.uniform_(self.context_tokens, -0.02, 0.02)
        else:
            raise ValueError(f"Unknown init_strategy '{strategy}'")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        X: Tensor,
        y_train: Tensor,
        use_context_only: bool = False,
        return_frozen: bool = False,
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        """Forward pass with learnable context tokens prepended at TF_icl.

        Parameters
        ----------
        X : Tensor
            Input of shape ``(B, T, H)`` where ``T = train_size + test_size``.

        y_train : Tensor
            Training labels of shape ``(B, train_size)``.

        use_context_only : bool, default=False
            If True (NC mode), only the context tokens serve as ICL context;
            real labeled data is *not* included in the key/value set.

        return_frozen : bool, default=False
            If True, also return the frozen model's predictions (standard ICL
            without context tokens). Used for KL divergence regularization.

        Returns
        -------
        Tensor or tuple[Tensor, Tensor]
            Predictions for the test portion, shape ``(B, test_size, out_dim)``.
            If ``return_frozen=True``, returns ``(tuned_out, frozen_out)``.
        """
        B = X.shape[0]
        train_size = y_train.shape[1]
        p = self.n_context_tokens
        icl = self.model.icl_predictor

        # Stages 1-2: frozen col_embedder + row_interactor
        # Use training codepath to bypass InferenceManager device handling;
        # dropout=0 so train/eval mode produces identical results.
        with torch.no_grad():
            self.model.col_embedder.train()
            self.model.row_interactor.train()
            embeddings = self.model.col_embedder(X, y_train=y_train)
            R = self.model.row_interactor(embeddings)  # (B, T, d_model)
            self.model.col_embedder.eval()
            self.model.row_interactor.eval()
        R = R.detach()

        # Compute target embeddings (shared between frozen and tuned paths)
        with torch.no_grad():
            if icl.max_classes > 0:
                Ry = icl.y_encoder(y_train.float())
            else:
                Ry = icl.y_encoder(y_train.unsqueeze(-1))

        # Frozen predictions: standard ICL without context tokens
        frozen_out = None
        if return_frozen:
            with torch.no_grad():
                R_frozen = R.clone()
                R_frozen[:, :train_size] = R_frozen[:, :train_size] + Ry
                src_f = icl.tf_icl(R_frozen, train_size=train_size)
                if icl.norm_first:
                    src_f = icl.ln(src_f)
                frozen_out = icl.decoder(src_f)[:, train_size:]

        # Add target embeddings to training rows (tuned path)
        if not use_context_only:
            R[:, :train_size] = R[:, :train_size] + Ry

        # Prepend context tokens
        C = self.context_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, p, d_model)
        R_aug = torch.cat([C, R], dim=1)  # (B, p + T, d_model)

        if use_context_only:
            effective_train_size = p
        else:
            effective_train_size = p + train_size

        # Stage 3: ICL transformer (gradient flows through C)
        src = icl.tf_icl(R_aug, train_size=effective_train_size)
        if icl.norm_first:
            src = icl.ln(src)
        out = icl.decoder(src)

        # Slice: skip context tokens + train rows
        out = out[:, p + train_size:]

        if return_frozen:
            return out, frozen_out
        return out

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_steps: int = 100,
        lr: float = 0.01,
        context_fraction: float = 0.8,
        use_context_only: bool = False,
        kl_weight: float = 0.0,
        seed: int = 42,
        verbose: bool = True,
    ) -> "ContextTuner":
        """Optimize context tokens on a training dataset.

        Each step randomly splits the data into a context subset (with labels,
        used as ICL context) and a query subset (predictions scored by the loss).

        Parameters
        ----------
        X : ndarray of shape (N, H)
            Training features.

        y : ndarray of shape (N,)
            Training targets.

        n_steps : int, default=100
            Number of optimization steps.

        lr : float, default=0.01
            Learning rate for AdamW.

        context_fraction : float, default=0.8
            Fraction of samples used as labeled context each step.

        use_context_only : bool, default=False
            If True, train in NC mode (only context tokens as ICL context).

        kl_weight : float, default=0.0
            Weight for KL divergence regularization between tuned and frozen
            (no context tokens) predictions.  When > 0 the loss becomes
            ``(1 - kl_weight) * task_loss + kl_weight * kl_loss``.
            Useful for small datasets to prevent context tokens from drifting
            too far from the pretrained model's behavior.

        seed : int, default=42
            Random seed for reproducibility.

        verbose : bool, default=True
            Print loss every 10 steps.

        Returns
        -------
        self
        """
        device = self.context_tokens.device
        rng = torch.Generator(device="cpu").manual_seed(seed)

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
        N = X_t.shape[0]

        ctx_size = max(1, int(context_fraction * N))
        qry_size = N - ctx_size

        optimizer = torch.optim.AdamW([self.context_tokens], lr=lr)
        is_classification = self.model.max_classes > 0

        # Only context_tokens requires grad; keep model in eval mode
        self.model.eval()

        for step in range(n_steps):
            perm = torch.randperm(N, generator=rng)
            ctx_idx = perm[:ctx_size]
            qry_idx = perm[ctx_size : ctx_size + qry_size]

            X_batch = torch.cat([X_t[ctx_idx], X_t[qry_idx]], dim=0).unsqueeze(0)
            y_ctx = y_t[ctx_idx].unsqueeze(0)
            y_qry = y_t[qry_idx]

            if kl_weight > 0:
                pred, frozen_pred = self(
                    X_batch, y_train=y_ctx,
                    use_context_only=use_context_only, return_frozen=True,
                )
                pred = pred.squeeze(0)
                frozen_pred = frozen_pred.squeeze(0)
            else:
                pred = self(X_batch, y_train=y_ctx, use_context_only=use_context_only)
                pred = pred.squeeze(0)

            if is_classification:
                task_loss = F.cross_entropy(pred, y_qry.long())
            else:
                task_loss = self._pinball_loss(pred, y_qry)

            if kl_weight > 0:
                kl = self._kl_loss(pred, frozen_pred, is_classification)
                loss = (1 - kl_weight) * task_loss + kl_weight * kl
            else:
                loss = task_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and (step + 1) % 10 == 0:
                msg = f"Step {step + 1}/{n_steps}  loss={loss.item():.4f}"
                if kl_weight > 0:
                    msg += f"  task={task_loss.item():.4f}  kl={kl.item():.4f}"
                print(msg)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        use_context_only: bool = False,
        softmax_temperature: float = 0.9,
        n_estimators: int = 1,
        feat_shuffle_method: str = "none",
        random_state: Optional[int] = 42,
    ) -> np.ndarray:
        """Predict using tuned context tokens.

        Parameters
        ----------
        X_train : ndarray of shape (N_train, H)
            Training features (used as ICL context in C mode).

        y_train : ndarray of shape (N_train,)
            Training labels.

        X_test : ndarray of shape (N_test, H)
            Test features.

        use_context_only : bool, default=False
            If True, only the context tokens provide context (NC mode).

        softmax_temperature : float, default=0.9
            Temperature for softmax in classification.

        n_estimators : int, default=1
            Number of ensemble members. When > 1 with a shuffle method,
            predictions are averaged over latin-hypercube feature permutations,
            matching ``TabICLRegressor`` ensemble behavior.

        feat_shuffle_method : str, default="none"
            Feature shuffling method passed to ``Shuffler``.
            One of ``"none"``, ``"latin"``, ``"random"``, ``"shift"``.

        random_state : int or None, default=42
            Random seed for reproducible shuffling.

        Returns
        -------
        ndarray
            For classification: probabilities of shape (N_test, n_classes).
            For regression: point predictions of shape (N_test,).
        """
        from tabicl.sklearn.preprocessing import Shuffler

        device = self.context_tokens.device

        X_tr = torch.from_numpy(np.asarray(X_train, dtype=np.float32)).to(device)
        y_tr = torch.from_numpy(np.asarray(y_train, dtype=np.float32)).to(device)
        X_te = torch.from_numpy(np.asarray(X_test, dtype=np.float32)).to(device)
        y = y_tr.unsqueeze(0)  # (1, N_train)

        self.eval()
        is_classification = self.model.max_classes > 0

        shuffler = Shuffler(
            n_elements=X_tr.shape[1],
            method=feat_shuffle_method,
            random_state=random_state,
        )
        patterns = shuffler.shuffle(n_estimators)

        all_outputs = []
        for perm in patterns:
            perm_t = torch.tensor(perm, device=device)
            X = torch.cat([X_tr[:, perm_t], X_te[:, perm_t]], dim=0).unsqueeze(0)
            out = self(X, y_train=y, use_context_only=use_context_only)
            all_outputs.append(out.squeeze(0))  # (N_test, out_dim)

        if is_classification:
            n_classes = int(y_tr.unique().numel())
            probs_list = [
                torch.softmax(o[..., :n_classes] / softmax_temperature, dim=-1)
                for o in all_outputs
            ]
            return torch.stack(probs_list).mean(dim=0).cpu().numpy()
        else:
            means = [o.mean(dim=-1) for o in all_outputs]
            return torch.stack(means).mean(dim=0).cpu().numpy()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _pinball_loss(pred: Tensor, target: Tensor) -> Tensor:
        """Pinball (quantile) loss matching TabICL regression training."""
        Q = pred.shape[-1]
        taus = torch.linspace(0.5 / Q, 1 - 0.5 / Q, Q, device=pred.device)
        errors = target.unsqueeze(-1) - pred  # (N, Q)
        loss = torch.where(errors >= 0, taus * errors, (taus - 1) * errors)
        return loss.mean()

    @staticmethod
    def _kl_loss(tuned: Tensor, frozen: Tensor, is_classification: bool) -> Tensor:
        """KL divergence between frozen and tuned predictions (regularizer).

        For classification, computes forward KL ``KL(frozen || tuned)`` on
        softmax outputs.  For regression, uses MSE between quantile predictions
        since quantile outputs are not probability distributions.
        """
        if is_classification:
            return F.kl_div(
                F.log_softmax(tuned, dim=-1),
                F.log_softmax(frozen, dim=-1),
                log_target=True,
                reduction="batchmean",
            )
        return F.mse_loss(tuned, frozen)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        task: str = "classification",
        n_context_tokens: int = 128,
        checkpoint_version: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> "ContextTuner":
        """Create a ContextTuner from a pretrained TabICL checkpoint.

        Parameters
        ----------
        task : str, default="classification"
            ``"classification"`` or ``"regression"``.

        n_context_tokens : int, default=128
            Number of learnable context tokens.

        checkpoint_version : str, optional
            Checkpoint filename on ``jingang/TabICL``. Defaults to the latest
            version for the specified task.

        model_path : str or Path, optional
            Local checkpoint path. If None, downloads from HuggingFace Hub.

        device : str or torch.device, optional
            Target device.

        **kwargs
            Extra keyword arguments forwarded to ``ContextTuner.__init__``.

        Returns
        -------
        ContextTuner
        """
        from huggingface_hub import hf_hub_download

        default_versions = {
            "classification": "tabicl-classifier-v2-20260212.ckpt",
            "regression": "tabicl-regressor-v2-20260212.ckpt",
        }
        if checkpoint_version is None:
            checkpoint_version = default_versions[task]

        if model_path is None:
            model_path = Path(hf_hub_download(repo_id="jingang/TabICL", filename=checkpoint_version))
        else:
            model_path = Path(model_path)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model = TabICL(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        tuner = cls(model=model, n_context_tokens=n_context_tokens, **kwargs)
        if device is not None:
            tuner = tuner.to(device)
        return tuner
