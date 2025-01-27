import torch
import torch.nn as nn
import numpy as np

class poseHead(nn.Module):
    def __init__(self):
        super(poseHead, self).__init__()
        
    def _homo(self,x):
        """Convert 2D points to homogeneous coordinates."""
        ones = torch.ones(*x.shape[:-1], 1, device=x.device)  # Shape: (B, C, K, 1)
        return torch.cat([x, ones], dim=-1)  # Shape: (B, C, K, 3)
    def _de_homo(self,x_homo):
        """Convert homogeneous coordinates back to 2D."""
        return x_homo[..., :2] / (x_homo[..., 2:] + 1e-10)  # Shape: (B, C, K, 2)
    def normalize(self, X):
        X = self._homo(X)  # Shape: (B, C, K, 3)
        
        mean_X = torch.mean(X[..., :2], dim=2, keepdim=True)  # Shape: (B, C, 1, 2)
        S_X = np.sqrt(2) / torch.mean(torch.norm(X[..., :2] - mean_X, dim=-1), dim=2, keepdim=True)  # Shape: (B, C, 1, 1)
        
        # Construct transformation matrix T1 for X
        # print(mean_X.shape)
        
        # print(mean_X[..., 0, 0].shape)
        # print(S_X.squeeze(-1).shape)
        
        T1 = torch.zeros(X.size(0), X.size(1), 3, 3, device=X.device)  # Shape: (B, C, 3, 3)
        T1[..., 0, 0] = S_X.squeeze(-1)
        T1[..., 1, 1] = S_X.squeeze(-1)
        T1[..., 0, 2] = -S_X.squeeze(-1) * mean_X[..., 0, 0]
        T1[..., 1, 2] = -S_X.squeeze(-1) * mean_X[..., 0, 1]
        T1[..., 2, 2] = 1

        # Normalize X
        X_normalized = torch.einsum('bcij,bckj->bcki', T1, X)  # Shape: (B, C, K, 3)
        X_normalized = self._de_homo(X_normalized)  # Shape: (B, C, K, 2)
        
        return X_normalized, T1
        
    def _normalize_XY(self,X, Y):
        """Normalize batched 2D points using Hartley normalization."""
        if X.size() != Y.size():
            raise ValueError("Input tensors X and Y must have the same shape.")
        
        X_normalized, T1 = self.normalize(X)
        Y_normalized, T2 = self.normalize(Y)
        return X_normalized, Y_normalized, T1, T2
    def _E_from_XY(self, X, Y, K, W=None, if_normzliedK=False, normalize=True, show_debug=False):
        """Compute Essential Matrix for batched inputs."""
        if if_normzliedK:
            X_normalizedK = X
            Y_normalizedK = Y
        else:
            # Apply inverse of K to normalize points
            K_inv = torch.inverse(K)  # Shape: (B, 3, 3)
            X_normalizedK = torch.einsum('bij,bckj->bcki', K_inv, self._homo(X))  # Shape: (B, C, K, 3)
            X_normalizedK = self._de_homo(X_normalizedK)  # Shape: (B, C, K, 2)
            Y_normalizedK = torch.einsum('bij,bckj->bcki', K_inv, self._homo(Y))  # Shape: (B, C, K, 3)
            Y_normalizedK = self._de_homo(Y_normalizedK)  # Shape: (B, C, K, 2)

        if normalize:
            X, Y, T1, T2 = self._normalize_XY(X_normalizedK, Y_normalizedK)
        else:
            X, Y = X_normalizedK, Y_normalizedK

        # Construct the linear system for the Essential Matrix
        xx = torch.cat([X, Y], dim=-1)  # Shape: (B, C, K, 4)
        XX = torch.stack([
            xx[..., 2] * xx[..., 0], xx[..., 2] * xx[..., 1], xx[..., 2],
            xx[..., 3] * xx[..., 0], xx[..., 3] * xx[..., 1], xx[..., 3],
            xx[..., 0], xx[..., 1], torch.ones_like(xx[..., 0])
        ], dim=-1)  # Shape: (B, C, K, 9)
        if W is not None:
            XX = torch.einsum('bck,bckj->bckj', W, XX)  # Apply weights if provided

        # Reshape XX to (B*C, K, 9) to compute SVD independently for each batch and channel
        XX_reshaped = XX.reshape(-1, XX.size(2), 9)  # Shape: (B*C, K, 9)

        # Initialize a tensor to store the recovered Essential Matrices
        F_recover = torch.zeros(XX_reshaped.size(0), 3, 3, device=XX.device)  # Shape: (B*C, 3, 3)

        # Compute SVD for each batch and channel
        for i in range(XX_reshaped.size(0)):
            U, D, V = torch.svd(XX_reshaped[i])  # Compute SVD for the i-th batch and channel
            F_recover[i] = V[:, -1].reshape(3, 3)  # Recover the Essential Matrix

        # Reshape F_recover back to (B, C, 3, 3)
        F_recover = F_recover.reshape(X.size(0), X.size(1), 3, 3)  # Shape: (B, C, 3, 3)

        # Enforce rank-2 constraint
        U_F, D_F, V_F = torch.svd(F_recover)
        S_110 = torch.diag_embed(torch.tensor([1., 1., 0.], device=X.device)).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 3, 3)
        E_recover_110 = torch.einsum('bcij,bcjk,bckl->bcil', U_F, S_110, V_F.transpose(-1, -2))

        if normalize:
            E_recover_110 = torch.einsum('bcij,bcjk,bckl->bcil', T2.transpose(-1, -2), E_recover_110, T1)

        return E_recover_110

    def decompose_E_batched(self, E):
        """Decompose batched Essential Matrix into rotation and translation."""
        # Compute batched SVD of E
        U, D, V = torch.linalg.svd(E)  # Shapes: (B, C, 3, 3), (B, C, 3), (B, C, 3, 3)

        # Ensure the determinant of U and V is +1 (to ensure proper rotation matrices)
        det_U = torch.det(U)
        det_V = torch.det(V)
        U = torch.where((det_U < 0).unsqueeze(-1).unsqueeze(-1), -U, U)
        V = torch.where((det_V < 0).unsqueeze(-1).unsqueeze(-1), -V, V)

        # Define the W matrix
        W = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=E.dtype, device=E.device)  # Shape: (3, 3)

        # Compute the two possible rotations
        R1 = U @ W.unsqueeze(0).unsqueeze(0) @ V.transpose(-1, -2)  # Shape: (B, C, 3, 3)
        R2 = U @ W.T.unsqueeze(0).unsqueeze(0) @ V.transpose(-1, -2)  # Shape: (B, C, 3, 3)

        # Compute the two possible translations
        t1 = U[..., 2]  # Shape: (B, C, 3)
        t2 = -t1  # Shape: (B, C, 3)

        return R1, R2, t1, t2
    def check_cheirality_fast(self, X, Y, R, t):
        """Check the cheirality condition without explicit triangulation."""
        # Convert 2D points to homogeneous coordinates
        X_homo = self._homo(X)  # Shape: (B, C, K, 3)
        Y_homo = self._homo(Y)  # Shape: (B, C, K, 3)

        # Compute the cross product of X and Y in the second camera's coordinate system
        Y_cam2 = (R @ Y_homo.transpose(-1, -2)).transpose(-1, -2)  # Shape: (B, C, K, 3)
        cross = torch.cross(X_homo, Y_cam2, dim=-1)  # Shape: (B, C, K, 3)

        # Compute the dot product between t and the cross product
        dot = (t.unsqueeze(-2) * cross).sum(dim=-1)  # Shape: (B, C, K)

        # Check if the dot product is positive (cheirality condition)
        mask = dot > 0  # Shape: (B, C, K)

        # Return the number of points satisfying the cheirality condition
        return mask.sum(dim=-1)  # Shape: (B, C)
    def select_best_Rt_batched(self, E, X, Y):
        """Select the best R and t from the four possible combinations (batched)."""
        # Decompose E into R and t
        R1, R2, t1, t2 = self.decompose_E_batched(E)

        # List all four combinations
        combinations = [
            (R1, t1),
            (R1, t2),
            (R2, t1),
            (R2, t2)
        ]

        # Initialize variables to track the best combination
        best_score = torch.full((E.size(0), E.size(1)), -1, dtype=torch.int32, device=E.device)  # Shape: (B, C)
        best_R = torch.zeros_like(R1)  # Shape: (B, C, 3, 3)
        best_t = torch.zeros_like(t1)  # Shape: (B, C, 3)

        # Iterate over all combinations
        for R, t in combinations:
            # Check the cheirality condition
            score = self.check_cheirality_fast(X, Y, R, t)  # Shape: (B, C)

            # Update the best combination
            update_mask = score > best_score
            best_R = torch.where(update_mask.unsqueeze(-1).unsqueeze(-1), R, best_R)
            best_t = torch.where(update_mask.unsqueeze(-1), t, best_t)
            best_score = torch.where(update_mask, score, best_score)

        return best_R, best_t
    def triangulate_points_fast(self, X, Y, K, R, t):
        """Triangulate 3D points from corresponding 2D points and generate a depth map."""
        # Convert 2D points to homogeneous coordinates
        X_homo = self._homo(X)  # Shape: (B, C, K, 3)
        Y_homo = self._homo(Y)  # Shape: (B, C, K, 3)

        # Compute the projection matrices for the two cameras
        P1 = K @ torch.eye(3, 4, dtype=K.dtype, device=K.device)  # First camera: [I | 0], Shape: (B, 3, 4)
        P2 = K @ torch.cat([R, t.unsqueeze(-1)], dim=-1)  # Second camera: [R | t], Shape: (B, 3, 4)

        # Construct the matrix A for the linear system
        A = torch.stack([
            X_homo[..., 0:1] * P1[..., 2:3, :] - P1[..., 0:1, :],
            X_homo[..., 1:2] * P1[..., 2:3, :] - P1[..., 1:2, :],
            Y_homo[..., 0:1] * P2[..., 2:3, :] - P2[..., 0:1, :],
            Y_homo[..., 1:2] * P2[..., 2:3, :] - P2[..., 1:2, :]
        ], dim=-2)  # Shape: (B, C, K, 4, 4)

        # Solve the linear system using SVD (batched)
        _, _, V = torch.svd(A.reshape(-1, 4, 4))  # Flatten batch and channel dimensions, Shape: (B*C*K, 4, 4)

        # The solution is the last column of V (homogeneous coordinates)
        points_3d_homo = V[..., -1]  # Shape: (B*C*K, 4)

        # Convert to non-homogeneous coordinates
        points_3d = points_3d_homo[..., :3] / points_3d_homo[..., 3:]  # Shape: (B*C*K, 3)

        # Reshape back to the original batch and channel dimensions
        points_3d = points_3d.reshape(X.size(0), X.size(1), X.size(2), 3)  # Shape: (B, C, K, 3)

        # Extract Z-values (depth) in camera coordinates
        depths = points_3d[..., 2]  # Shape: (B, C, K)

        return depths
    def filterFlow(self, flow_map, motion_map, depth_map, n_min=8):
        # Number of top elements to select
        top_k = n_min

        # Unpack the shape
        B, C, H, W = motion_map.shape

        # Flatten the spatial dimensions (H, W) while keeping batch and channel dimensions
        flat_confidence = motion_map.view(B, C, -1)  # Shape: (B, C, H*W)

        # Find the top_k indices and values for each (B, C) independently
        top_values, top_indices = torch.topk(flat_confidence, min(top_k, H * W), dim=-1, sorted=False)

        # Compute the coordinates from the flattened indices
        row_indices = top_indices // W  # Compute row indices (H dimension)
        col_indices = top_indices % W   # Compute column indices (W dimension)

        # Stack coordinates for each top-k point
        coordinates = torch.stack([row_indices, col_indices], dim=3)  # Shape: (B, C, k, 2)
        # Unpack dimensions
        # B, C, _, _ = coordinates.shape

        # Gather the flow vectors at the specified coordinates
        # Split the coordinates into x and y for indexing
        x_coords = coordinates[..., 1]  # Shape: (B, C, K)TODO
        y_coords = coordinates[..., 0]  # Shape: (B, C, K)
        # print(y_coords.shape)

        # Gather flow values for x and y
        # Permute flow_map to shape (B, H, W, 2) for easier indexing
        flow_map_permuted = flow_map.permute(0, 2, 3, 1)  # Shape: (B, H, W, 2)
        depth_map_permuted = depth_map.permute(0, 2, 3, 1)

        # Use advanced indexing to gather flow vectors
        flow_at_points = flow_map_permuted[torch.arange(B).unsqueeze(-1).unsqueeze(-1),
                                        y_coords, x_coords]  # Shape: (B, C, K, 2)

        # Add the flow to the original coordinates
        new_coordinates = coordinates + flow_at_points

        selected_depth = depth_map_permuted[torch.arange(B).unsqueeze(-1).unsqueeze(-1),
                                        y_coords, x_coords]  # Shape: (B, C, K, 2)
        
        return coordinates, new_coordinates, selected_depth
    
    def forward(self, K, flow_map, motion_map, depth_map, threshold=None):
        B, C, H, W = motion_map.shape
        X, Y, selected_depths = self.filterFlow(flow_map, motion_map, depth_map, 8)
        K1 = K.expand(B, -1, -1)  # Intrinsic matrix
        # Compute Essential Matrix
        E = self._E_from_XY(X, Y, K1)

        # Select the best R and t
        R, t = self.select_best_Rt_batched(E, X, Y)
        K1 = K.expand(B, C, -1, -1)  # Intrinsic matrix
        # print(K.shape)
        # Triangulate 3D points using the optimized function
        depth = self.triangulate_points_fast(X, Y, K, R, t)
        scale = (depth / selected_depths.squeeze(-1)).mean(-1)
        t = t * scale.unsqueeze(-1)
        # Reshape translation to (B, C, 3, 1)
        translation = t.unsqueeze(-1)

        # Create the bottom row [0, 0, 0, 1]
        bottom_row = torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(B, C, 1, 4)

        # Concatenate rotation and translation to form the top 3 rows
        top_rows = torch.cat([R, translation], dim=-1)  # Shape: (B, C, 3, 4)

        # Concatenate the bottom row to form the full transformation matrix
        T = torch.cat([top_rows, bottom_row], dim=-2)  # Shape: (B, C, 4, 4)

        
        if threshold != None:
            # Create masks by comparing each channel to the maximum value
            masks = (motion_map > threshold)  # Shape: (B, C, H, W)
        else:
            # Find the maximum value along the channel dimension
            max_values = torch.max(motion_map, dim=1, keepdim=True)[0]  # Shape: (B, 1, H, W)

            # Create masks by comparing each channel to the maximum value
            masks = (motion_map == max_values)  # Shape: (B, C, H, W)
        
        return T, masks

if __name__ == "__main__":
    model = poseHead()
    device = 'cpu'
    model.to(device)
    model.eval()
    K = torch.tensor([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]], dtype=torch.float32).to(device)
    
    B, C, H, W = 2, 5, 640, 960
    flow = torch.randn(B, 2, H, W).to(device)
    motion = torch.rand(B, C, H, W).to(device)
    depth = torch.rand(B, 1, H, W).to(device)
    import time
    s = time.time()
    # coordinates, new_coordinates, selected_depth, masks = model.filterFlow(flow, motion, depth)
    for i in range(100):
        T, masks = model(K, flow, motion, depth)
    print(100/(time.time()-s))
    
    print(T.shape)
    