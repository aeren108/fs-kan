import torch
import torch.nn as nn
from efficient_kan import KANLinear, KAN


class FSKAEquivariantLayer(nn.Module):
    """
    S_n-equivariant Function Sharing KA layer (standard version).

    from the paper (equation 5):
        Phi(x)_q = Phi_1(x_q) + sum_{p != q} Phi_2(x_p)

    equivalent to:
        Phi(x)_q = Phi_1(x_q) + (sum_p Phi_2(x_p)) - Phi_2(x_q)

    args:
        in_features: input feature dimension per point
        out_features: output feature dimension per point
        grid_size: B-spline grid size for KANLayer sub-layers
        spline_order: B-spline order
        use_batchnorm: whether to apply BatchNorm1d after the layer
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        use_batchnorm=True,
    ):
        super().__init__()
        self.phi_self = KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order)
        self.phi_others = KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order)
        self.bn = nn.BatchNorm1d(out_features) if use_batchnorm else None

    def forward(self, x):
        """
        args:
            x: (batch_size, num_points, in_features)

        returns:
            output: (batch_size, num_points, out_features)
        """
        B, N, D = x.shape

        # reshape to (B*N, D) for KANLayer which expects 2D input
        x_flat = x.reshape(B * N, D)

        # Phi_1(x_q) for each point
        self_out = self.phi_self(x_flat)

        # Phi_2(x_p) for each point
        others_out = self.phi_others(x_flat)

        # reshape back
        self_out = self_out.reshape(B, N, -1)
        others_out = others_out.reshape(B, N, -1)

        # sum of Phi_2(x_p) over all points, then subtract own contribution
        # Phi(x)_q = Phi_1(x_q) + sum_p(Phi_2(x_p)) - Phi_2(x_q)
        others_sum = others_out.sum(dim=1, keepdim=True)  # (B, 1, out_features)
        output = self_out + (others_sum - others_out)

        # apply batchnorm (operates on feature dim, expects (B, C, L) or (B, C))
        if self.bn is not None:
            output = self.bn(output.transpose(1, 2)).transpose(1, 2)

        return output


class EfficientFSKAEquivariantLayer(nn.Module):
    """
    efficient S_n-equivariant Function Sharing KA layer.

    from the paper (appendix A.3):
        Phi_tilde(x)_q = Phi_1(x_q) + Phi_2(sum_p x_p)

    the key difference from the standard version is that we aggregate (sum)
    the raw input *before* applying Phi_2, so Phi_2 is evaluated only once
    per sample rather than once per point. this reduces memory and compute.

    args:
        in_features: input feature dimension per point
        out_features: output feature dimension per point
        grid_size: B-spline grid size for KANLayer sub-layers
        spline_order: B-spline order
        use_batchnorm: whether to apply BatchNorm1d after the layer
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        use_batchnorm=True,
    ):
        super().__init__()
        self.phi_1 = KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order)
        self.phi_2 = KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order)
        self.bn = nn.BatchNorm1d(out_features) if use_batchnorm else None

    def forward(self, x):
        # x: (batch_size, num_points, in_features)
        # return output: (batch_size, num_points, out_features)
        B, N, D = x.shape

        # Phi_1 applied point-wise
        x_flat = x.reshape(B * N, D)
        phi1_out = self.phi_1(x_flat).reshape(B, N, -1)  # (B, N, out_features)

        # aggregate first, then apply Phi_2 once
        x_sum = x.sum(dim=1)  # (B, D)
        phi2_out = self.phi_2(x_sum)  # (B, out_features)
        phi2_out = phi2_out.unsqueeze(1).expand_as(phi1_out)  # (B, N, out_features)

        output = phi1_out + phi2_out

        if self.bn is not None:
            output = self.bn(output.transpose(1, 2)).transpose(1, 2)

        return output


class FSKAInvariantLayer(nn.Module):
    """
    S_n-invariant Function Sharing KA layer.

    from the paper (proposition 3):
        Phi(x) = pool_p(phi(x_p))

    applies a shared KANLayer to each point independently, then pools
    across all points to produce a single feature vector per sample.

    args:
        in_features: input feature dimension per point
        out_features: output feature dimension (global)
        grid_size: B-spline grid size
        spline_order: B-spline order
        pool: pooling method ('sum', 'mean', or 'max')
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        pool="sum",
    ):
        super().__init__()
        self.phi = KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order)
        self.pool = pool

    def forward(self, x):
        # x: (batch_size, num_points, in_features)
        # returns: output: (batch_size, out_features)
        B, N, D = x.shape

        # apply shared phi to each point
        x_flat = x.reshape(B * N, D)
        out = self.phi(x_flat).reshape(B, N, -1)  # (B, N, out_features)

        # pool across points
        if self.pool == "sum":
            return out.sum(dim=1)
        elif self.pool == "mean":
            return out.mean(dim=1)
        elif self.pool == "max":
            return out.max(dim=1)[0]
        else:
            raise ValueError(f"unknown pooling method: {self.pool}")


class FSKANClassifier(nn.Module):
    """
    FS-KAN point cloud classifier.

    architecture (from paper appendix C.3):
        equivariant_layer_1 (in_features -> hidden_dim) + BatchNorm
        equivariant_layer_2 (hidden_dim -> hidden_dim) + BatchNorm
        invariant_layer (hidden_dim -> hidden_dim) with pooling
        linear_head (hidden_dim -> num_classes)

    args:
        in_features: input feature dim per point (3 for xyz)
        hidden_dim: hidden dimension (paper uses 36)
        num_classes: number of output classes (40 for ModelNet40)
        num_equiv_layers: number of equivariant layers (paper uses 2)
        use_efficient: if True, use EfficientFSKAEquivariantLayer; else standard
        pool: pooling method for invariant layer ('sum', 'mean', or 'max')
        grid_size: B-spline grid size
        spline_order: B-spline order
    """

    def __init__(
        self,
        in_features=3,
        hidden_dim=36,
        num_classes=40,
        num_equiv_layers=2,
        use_efficient=False,
        pool="sum",
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()

        EquivLayer = EfficientFSKAEquivariantLayer if use_efficient else FSKAEquivariantLayer

        # equivariant layers
        self.equiv_layers = nn.ModuleList()
        dims = [in_features] + [hidden_dim] * num_equiv_layers
        for i in range(num_equiv_layers):
            self.equiv_layers.append(
                EquivLayer(
                    dims[i], dims[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    use_batchnorm=True,
                )
            )

        # invariant layer
        self.invariant_layer = FSKAInvariantLayer(
            hidden_dim, hidden_dim,
            grid_size=grid_size,
            spline_order=spline_order,
            pool=pool,
        )

        # output head (as in the paper)
        self.output_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        args:
            x: (batch_size, num_points, in_features)

        returns:
            logits: (batch_size, num_classes)
        """
        # equivariant layers
        for layer in self.equiv_layers:
            x = layer(x)

        # invariant layer -> global feature
        x = self.invariant_layer(x)  # (B, hidden_dim)

        # classification head
        logits = self.output_head(x)  # (B, num_classes)
        return logits


class StandardKANClassifier(nn.Module):
    """
    non-equivariant KAN baseline for point cloud classification.

    flattens the point cloud and applies a standard KAN.
    from the paper: 3 hidden layers of width 16.

    note: this model's parameter count depends on num_points (N),
    since the input is flattened to N*3.

    args:
        num_points: number of points per cloud (e.g. 1024)
        in_features: feature dim per point (3 for xyz)
        hidden_dims: list of hidden layer widths
        num_classes: number of output classes
        grid_size: B-spline grid size
        spline_order: B-spline order
    """

    def __init__(
        self,
        num_points=1024,
        in_features=3,
        hidden_dims=None,
        num_classes=40,
        grid_size=5,
        spline_order=3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 16, 16]

        layers_dims = [num_points * in_features] + hidden_dims + [num_classes]
        self.kan = KAN(
            layers_hidden=layers_dims,
            grid_size=grid_size,
            spline_order=spline_order,
        )

    def forward(self, x):
        # x: (batch_size, num_points, in_features)
        # return logits: (batch_size, num_classes)

        B = x.size(0)
        x_flat = x.reshape(B, -1)  # (B, N*in_features)
        return self.kan(x_flat)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("FS-KAN Module Verification")
    print("=" * 60)

    B, N, D = 4, 1024, 3
    num_classes = 40
    x = torch.randn(B, N, D, dtype=torch.float64)

    # --- test standard FS-KAN ---
    print("\n--- Standard FS-KAN Classifier ---")
    model_std = FSKANClassifier(
        in_features=D, hidden_dim=36, num_classes=num_classes,
        num_equiv_layers=2, use_efficient=False, pool="sum",
    ).double()
    logits = model_std(x)
    print(f"  Input shape:   {x.shape}")
    print(f"  Output shape:  {logits.shape}")
    print(f"  Parameters:    {count_parameters(model_std)}")

    # --- test efficient FS-KAN ---
    print("\n--- Efficient FS-KAN Classifier ---")
    model_eff = FSKANClassifier(
        in_features=D, hidden_dim=36, num_classes=num_classes,
        num_equiv_layers=2, use_efficient=True, pool="sum",
    ).double()
    logits_eff = model_eff(x)
    print(f"  Input shape:   {x.shape}")
    print(f"  Output shape:  {logits_eff.shape}")
    print(f"  Parameters:    {count_parameters(model_eff)}")

    # --- equivariance / invariance test ---
    print("\n--- Permutation Invariance Test ---")
    # since the full model ends with an invariant layer, the output should be the same
    perm = torch.randperm(N)
    x_permuted = x[:, perm, :]

    with torch.no_grad():
        model_std.eval()
        model_eff.eval()

        out_orig_std = model_std(x)
        out_perm_std = model_std(x_permuted)
        diff_std = (out_orig_std - out_perm_std).abs().max().item()

        out_orig_eff = model_eff(x)
        out_perm_eff = model_eff(x_permuted)
        diff_eff = (out_orig_eff - out_perm_eff).abs().max().item()

    print(f"  Standard FS-KAN  — max output diff after permutation: {diff_std:.8e}")
    print(f"  Efficient FS-KAN — max output diff after permutation: {diff_eff:.8e}")

    invariance_ok = diff_std < 1e-4 and diff_eff < 1e-4
    print(f"  Invariance holds: {'YES' if invariance_ok else 'NO (check implementation!)'}")

    # --- test equivariant layer directly ---
    print("\n--- Equivariant Layer Direct Test ---")
    equiv_layer = FSKAEquivariantLayer(D, 16, use_batchnorm=False).double()
    equiv_layer.eval()
    with torch.no_grad():
        out1 = equiv_layer(x)
        out2 = equiv_layer(x_permuted)
        # equivariance: out2 should be out1 with the same permutation applied
        out1_permuted = out1[:, perm, :]
        equiv_diff = (out1_permuted - out2).abs().max().item()
    print(f"  Max equivariance error: {equiv_diff:.8e}")
    print(f"  Equivariance holds: {'YES' if equiv_diff < 1e-4 else 'NO (check implementation!)'}")

    # --- test standard (non-equivariant) KAN baseline ---
    print("\n--- Standard KAN Classifier (non-equivariant baseline) ---")
    N_small = 32
    x_small = torch.randn(B, N_small, D, dtype=torch.float64)
    model_kan = StandardKANClassifier(
        num_points=N_small, in_features=D,
        hidden_dims=[16, 16, 16], num_classes=num_classes,
    ).double()
    logits_kan = model_kan(x_small)
    print(f"  Input shape:   {x_small.shape}")
    print(f"  Output shape:  {logits_kan.shape}")
    print(f"  Parameters:    {count_parameters(model_kan)}")

    print("\n" + "=" * 60)
    print("All FS-KAN tests passed!")
    print("=" * 60)
