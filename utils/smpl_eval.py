import torch

try:
    from smplx import SMPL
except ImportError:
    SMPL = None


class SMPLForward:
    """
    Lightweight SMPL forward wrapper for evaluation.

    Notes:
    - Requires `smplx` to be installed.
    - Requires a valid SMPL model directory.
    - Assumes input pose is [B, T, 24, 3] in axis-angle format.
    """

    def __init__(self, model_path: str, gender: str = "neutral", device: str = "cuda"):
        if SMPL is None:
            raise ImportError(
                "smplx is not installed. Please install smplx before using SMPLForward."
            )

        self.device = device
        self.model = SMPL(
            model_path=model_path,
            gender=gender,
            batch_size=1,
        ).to(device)

    def pose_to_joints(self, pose_axis_angle: torch.Tensor, betas: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pose_axis_angle: [B, T, 24, 3]
            betas:
                - None, or
                - [B, 10], or
                - [B, T, 10]

        Returns:
            joints: [B, T, J, 3] in meters
        """
        batch_size, seq_len, joint_num, dim = pose_axis_angle.shape
        if joint_num != 24 or dim != 3:
            raise ValueError(
                f"Expected pose shape [B, T, 24, 3], got {pose_axis_angle.shape}"
            )

        flat_pose = pose_axis_angle.reshape(batch_size * seq_len, 24, 3)
        global_orient = flat_pose[:, 0, :]           # [B*T, 3]
        body_pose = flat_pose[:, 1:, :].reshape(batch_size * seq_len, 23 * 3)

        if betas is None:
            betas = torch.zeros(
                batch_size * seq_len,
                10,
                dtype=pose_axis_angle.dtype,
                device=pose_axis_angle.device,
            )
        elif betas.dim() == 2:
            betas = betas[:, None, :].expand(batch_size, seq_len, betas.shape[-1])
            betas = betas.reshape(batch_size * seq_len, -1)
        elif betas.dim() == 3:
            betas = betas.reshape(batch_size * seq_len, -1)
        else:
            raise ValueError(f"Unsupported betas shape: {betas.shape}")

        output = self.model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas,
            return_verts=False,
        )

        joints = output.joints.reshape(batch_size, seq_len, -1, 3)
        return joints