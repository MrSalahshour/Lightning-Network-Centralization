# from typing import Callable, Dict, List, Optional, Tuple, Type, Union

# from gymnasium import spaces
# import torch as th
# from torch import nn

# from stable_baselines3 import PPO
# from stable_baselines3.common.policies import ActorCriticPolicy

# class TransformerNetwork(nn.Module):
#     """
#     Custom network for policy and value function using a Transformer.
#     It receives as input the features extracted by the features extractor.

#     :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
#     :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
#     :param last_layer_dim_vf: (int) number of units for the last layer of the value network
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         last_layer_dim_pi: int = 64,
#         last_layer_dim_vf: int = 64,
#         nhead: int = 4,
#         num_encoder_layers: int = 4,
#         dim_feedforward: int = 128,
#     ):
#         super().__init__()

#         # Save output dimensions, used to create the distributions
#         self.latent_dim_pi = last_layer_dim_pi
#         self.latent_dim_vf = last_layer_dim_vf

#         # Transformer encoder layer for policy network
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
#         )
#         self.policy_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         # Linear layer for the policy network
#         self.policy_net = nn.Linear(feature_dim, last_layer_dim_pi)

#         # Transformer encoder layer for value network
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
#         )
#         self.value_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
#         # Linear layer for the value network
#         self.value_net = nn.Linear(feature_dim, last_layer_dim_vf)

#     def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
#         return self.forward_actor(features), self.forward_critic(features)

#     def forward_actor(self, features: th.Tensor) -> th.Tensor:
#         # Add batch and sequence dimensions
#         features = features.unsqueeze(1)
#         transformer_output = self.policy_transformer(features)
#         transformer_output = transformer_output.squeeze(1)
#         return self.policy_net(transformer_output)

#     def forward_critic(self, features: th.Tensor) -> th.Tensor:
#         # Add batch and sequence dimensions
#         features = features.unsqueeze(1)
#         transformer_output = self.value_transformer(features)
#         transformer_output = transformer_output.squeeze(1)
#         return self.value_net(transformer_output)


# class TransformerActorCriticPolicy(ActorCriticPolicy):
#     def __init__(
#         self,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         lr_schedule: Callable[[float], float],
#         *args,
#         **kwargs,
#     ):
#         # Disable orthogonal initialization
#         kwargs["ortho_init"] = False
#         super().__init__(
#             observation_space,
#             action_space,
#             lr_schedule,
#             # Pass remaining arguments to base class
#             *args,
#             **kwargs,
#         )

#     def _build_mlp_extractor(self) -> None:
#         self.mlp_extractor = TransformerNetwork(self.features_dim)