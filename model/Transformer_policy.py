from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from transformers import BertModel, BertConfig
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import gymnasium as gym



import torch as th
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Union, List, Dict, Type, Tuple
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule

from torch.distributions import Bernoulli, Categorical, Normal

from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.distributions import CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution
from functools import partial
import numpy as np

SelfTransformerMultiCategoricalDistribution = TypeVar("SelfTransformerMultiCategoricalDistribution", bound="TransformerMultiCategoricalDistribution")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as thg


from model.GNNFeatureExtractor import GraphFeaturesExtractor2

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.

    Used by the ``FlattenExtractor`` to compute the input shape.

    :param observation_space:
    :return:
    """
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        # Use Gym internal method
        return spaces.utils.flatdim(observation_space)
         
'''
The following three classes are a custom transformer implemented specifically for this task
'''    
class MaskedAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MaskedAttention, self).__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first = True)

    def forward(self, x, mask):
        # x: [batch_size, n_nodes, hidden_dim]
        # mask: [batch_size, n_nodes]

        # Perform attention with the modified mask
        attn_output, _ = self.multihead_attn(x, x, x, attn_mask=mask)
        
        # Transpose the attention output back to the original input shape
        # attn_output = attn_output.permute(1, 0, 2)  # [batch_size, n_nodes, hidden_dim]

        return attn_output

class TransformerLayerWithMaskedAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(TransformerLayerWithMaskedAttention, self).__init__()
        self.masked_attn = MaskedAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * num_heads),
            nn.ReLU(),
            nn.Linear(hidden_dim * num_heads, hidden_dim)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        # x: [batch_size, n_nodes, hidden_dim]
        # mask: [batch_size, n_nodes]

        # Apply masked attention
        attn_output = self.masked_attn(x, mask)
        
        # Add & norm
        x = self.layer_norm1(x + attn_output)
        
        # Feedforward network
        ff_output = self.feedforward(x)
        
        # Add & norm
        x = self.layer_norm2(x + self.dropout(ff_output))

        return x

class TransformerWithMaskedAttention(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, num_layers=4):
        super(TransformerWithMaskedAttention, self).__init__()
        
        self.num_heads = num_heads
        
        # Stacking multiple Transformer layers with masked attention
        self.layers = nn.ModuleList([
            TransformerLayerWithMaskedAttention(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])
        

    def forward(self, x, mask):
        # x: [batch_size, n_nodes, hidden_dim]
        # mask: [batch_size, n_nodes]
        
        # Expand mask to match the dimensions of the attention matrix
        mask = mask.unsqueeze(1).expand(-1, mask.size(1), -1)  # [batch_size, n_nodes, n_nodes]

        # Modify the attention mechanism by applying the mask
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 0, float(0.0))

        # Check each row and set to zero if all elements in the row are -inf
        # Here, -inf is represented by the most negative number, so we use this to identify such rows
        # row_all_inf = torch.isinf(mask).all(dim=-1)
        # mask[row_all_inf] = 0.0  # Set those rows to all 0
        
        # Set all diagonal elements [:, i, i] to 0
        # diag_indices = torch.arange(mask.size(1))
        # mask[:, diag_indices, diag_indices] = 0.0

        # Repeat the mask for multi-head attention
        mask = mask.repeat_interleave(self.num_heads, dim=0)  # [batch_size * num_heads, n_nodes, n_nodes]

        # MultiheadAttention expects the mask to be of shape [n_nodes, n_nodes] or [batch_size*num_heads, n_nodes, n_nodes]
        # Here, we'll use [batch_size, n_nodes, n_nodes] which works with MultiheadAttention

        for layer in self.layers:
            x = layer(x, mask)


        return x

'''
end
'''    
          
class CustomTransformerExtractor(nn.Module):
    """
    Constructs a transformer-based model that receives the output from a previous features extractor
    (i.e., a CNN) or directly the observations (if no features extractor is applied) as input and outputs
    a latent representation for the policy and a value network.

    :param feature_dim: Dimension of the feature vector (can be the output of a CNN)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: The activation function to use for the networks.
    :param device: PyTorch device.
    :param hidden_dim: The hidden dimension size of the transformer.
    :param num_layers: Number of transformer layers.
    :param num_heads: Number of attention heads in the transformer.
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
        num_heads = 4,
        num_layers = 4,
        max_position_embedding = 50,
        ) -> None:
        super().__init__()
        
        device = get_device(device)
        
        # NOTE: only for MLP: we need an embedder to bring all node features to an embedding dim
        self.embedder_pi = nn.Linear(4, feature_dim).to(device)
        self.transformer_pi = TransformerWithMaskedAttention(feature_dim, num_heads=num_heads, num_layers=num_layers)

        # self.embedder_vf = nn.Linear(4, feature_dim).to(device)
        # self.transformer_vf = TransformerWithMaskedAttention(feature_dim, num_heads=num_heads, num_layers=num_layers)

        #to ensure that the ending sums to 1 for each batch
        self.score_activation = nn.Softmax(dim=-1)        
        
        #general flatten function
        self.flatten = nn.Flatten(start_dim=1).to(device)
        
        value_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
        
        # Networks for allocation
        # std_net: List[nn.Module] = []
        # mean_net: List[nn.Module] = []
        
        # Network for node scoring
        scoring_net: List[nn.Module] = []
                
        
        last_layer_dim_vf = feature_dim * max_position_embedding
        # last_layer_dim_pi = feature_dim 

        # for transformerOnly, this shall be the latent dim:
        last_layer_dim_pi = feature_dim * max_position_embedding
        
        # save dimensions of layers in policy and value nets
        if isinstance(net_arch, dict):
            # Note: if key is not specified, assume linear network
            pi_layers_dims = net_arch.get("pi", [])  # Layer sizes of the policy network
            vf_layers_dims = net_arch.get("vf", [])  # Layer sizes of the value network
        else:
            pi_layers_dims = vf_layers_dims = net_arch
            
        # Iterate through the policy layers and build the policy net
        # for curr_layer_dim in pi_layers_dims:
        #     policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
        #     policy_net.append(activation_fn())
        #     last_layer_dim_pi = curr_layer_dim
        
        # Bringing all hidden dims to a score and then summing all to 1
        scoring_net.append(nn.Linear(feature_dim, 1))
        scoring_net.append(self.flatten)
        # scoring_net.append(self.score_activation)        
        
        # Network for learning the standard deviation of allocation vectors
        # std_net.append(nn.Linear(feature_dim, 1))
        # std_net.append(activation_fn())
        # std_net.append(self.flatten)
        # std_net.append(nn.Linear(max_position_embedding, 1))
        # std_net.append(activation_fn())

        
        # Network for leanring the mean value of the allocation network
        # mean_net.append(nn.Linear(feature_dim, 9))
        # mean_net.append(activation_fn())
        # mean_net.append(self.flatten)
        # mean_net.append(nn.Linear(max_position_embedding * 9, 9))
        # mean_net.append(nn.Softmax(dim=-1))
        
        
        # Iterate through the value layers and build the value net
        value_net.append(self.flatten)
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim
        

        # Save dim, used to create the distributions
        #NOT IMPORTANT
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Create networks
        # If the list of layers is empty, the network will just act as an Identity module
        self.value_net = nn.Sequential(*value_net).to(device)
        # self.policy_net = nn.Sequential(*policy_net).to(device)
        
        # self.std_net  = nn.Sequential(*std_net).to(device)
        # self.mean_net = nn.Sequential(*mean_net).to(device)
        self.scoring_net = nn.Sequential(*scoring_net).to(device)
 
    def normal_distribution_tensor(self,center_value, size=9, std_dev=1.0):
        """
        Generate a tensor representing a normal distribution centered around the given value for each batch element.
        
        Args:
        - center_value (tensor): A tensor of shape [batch_size, 1] containing values between 0 and 10.
        - size (int): The size of the output tensor (default is 10, representing values from 0 to 9).
        - std_dev (float): The standard deviation of the normal distribution (default is 1.0).
        
        Returns:
        - Tensor: A tensor of shape [batch_size, size] where each row is a normalized distribution.
        """
        batch_size = center_value.shape[0]
        x = torch.arange(size, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)  # Shape: [batch_size, size]
        distribution = torch.exp(-0.5 * ((x - center_value) / std_dev) ** 2)
        dist_sum = distribution.sum(dim=1, keepdim=True)  # Normalize to make the sum of each distribution equal to 1
        distribution_final = distribution / dist_sum
        return distribution_final
        
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:

        x, mask = features
        # embedded = self.transformer(self.embedder(x), mask)
        # embedded = self.embedder_pi(x)
        # transformer_output = self.transformer_pi(x, mask)
        # Acts as the body of the policy network, either transformer or mlp
        # node_embeddings = self.policy_net(embedded)
        
        # Node scores
        node_scores = self.scoring_net(x)
        
        # STD and Mean
        # std_dev = self.std_net(transformer_output) + 1.1
        # dist_mean = self.mean_net(transformer_output) # Gives a tensor of size [batch_size, 10]
        
        # # Find the maximum value along the `num_nodes` dimension for each batch
        # _, cap_alloc_init = torch.max(dist_mean, dim=-1, keepdim=True)
        
        # # The output is the distibution by which the node capacity allocation is done
        # allocation_distribution = self.normal_distribution_tensor(cap_alloc_init, size = 9, std_dev= std_dev) 
        return node_scores, self.flatten(x)
        
    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        #intuition: value net should not rely on a transformer:: Proven wrong
        input, mask = features
        # embedded = self.transformer(self.embedder(x), mask)
        # embedded = self.embedder_pi(input)
        # transformer_output = self.transformer_pi(input, mask)
        return self.value_net(input)
    
class NullFeatureExtractor(BaseFeaturesExtractor):
    """
    acts nothing, gives the tensor of the exact shape to the transformer.

    :param observation_space:
    """

    def __init__(self, observation_space: gym.Space, features_dim: int) -> None:
        super().__init__(observation_space, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # return (observations["node_features"],  (observations["node_features"][:, :, 3] != 0).long())
        return observations, (observations[:,:,3] != 0).long()

class TransformerMultiCategoricalDistribution(Distribution):
    """
    MultiCategorical distribution for multi discrete actions.

    :param action_dims: List of sizes of discrete action spaces
    """

    def __init__(self, action_dims: List[int]):
        super().__init__()
        self.action_dims = action_dims

        
    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits (flattened) of the MultiCategorical distribution.
        You can then get probabilities using a softmax on each sub-space.

        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        self.action_dims
        action_logits = nn.Linear(latent_dim, self.action_dims[-1])
        return action_logits

    def proba_distribution(
        self: SelfTransformerMultiCategoricalDistribution, action_logits: th.Tensor
    ) -> SelfTransformerMultiCategoricalDistribution:

        self.distribution = [Categorical(logits=split) for split in th.split(action_logits, list(self.action_dims), dim=1)]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        # Extract each discrete action and compute log prob for their respective distributions
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distribution, th.unbind(actions, dim=1))], dim=1
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distribution], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distribution], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distribution], dim=1)

    def actions_from_params(self, action_logits: th.Tensor, deterministic: bool = False) -> th.Tensor:
        # Update the proba distribution
        self.proba_distribution(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, action_logits: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob
        

def custom_make_proba_distribution(
    action_space: spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None
) -> Distribution:
    """
    Return an instance of Distribution for the correct type of action space

    :param action_space: the input action space
    :param use_sde: Force the use of StateDependentNoiseDistribution
        instead of DiagGaussianDistribution
    :param dist_kwargs: Keyword arguments to pass to the probability distribution
    :return: the appropriate Distribution object
    """
    if dist_kwargs is None:
        dist_kwargs = {}
    return TransformerMultiCategoricalDistribution(list(action_space.nvec), **dist_kwargs)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = GraphFeaturesExtractor2,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        # Disable orthogonal initialization
        ortho_init = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            [],
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )
        self.share_features_extractor = True
        self.action_dist = custom_make_proba_distribution(action_space, use_sde=use_sde, dist_kwargs=self.dist_kwargs)
        self._build(lr_schedule)
        
    def obs_to_tensor(self, observation: gym.spaces.GraphInstance):
            if isinstance(observation, list):
                vectorized_env = True
            else:
                vectorized_env = False
            if vectorized_env:
                torch_obs = list()
                for obs in observation:
                    x = th.tensor(obs.nodes).float()
                    #edge_index = th.tensor(obs.edge_links, dtype=th.long).t().contiguous().view(2, -1)
                    edge_index = th.tensor(obs.edge_links, dtype=th.long)
                    # edges = th.tensor(obs.edges, dtype=th.float)
                    torch_obs.append(thg.data.Data(x=x, edge_index=edge_index))
                if len(torch_obs) == 1:
                    torch_obs = torch_obs[0]
            else:
                x = th.tensor(observation.nodes).float()
                #edge_index = th.tensor(observation.edge_links, dtype=th.long).t().contiguous().view(2, -1)
                edge_index = th.tensor(observation.edge_links, dtype=th.long)
                # edges = th.tensor(observation.edges, dtype=th.float)
                torch_obs = thg.data.Data(x=x, edge_index=edge_index)
            return torch_obs, vectorized_env
           
    def _build_mlp_extractor(self) -> None:
         self.mlp_extractor = CustomTransformerExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
            num_heads = 4,
            num_layers = 4,
            max_position_embedding = 50)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            (latent_pi, latent_pi_score), latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, latent_pi_score = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_score)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]


    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, latent_pi_score: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        action_node = latent_pi
        mean_actions_score = self.action_net(latent_pi_score)
        mean_actions = torch.cat((action_node, mean_actions_score), dim=-1)

        if isinstance(self.action_dist, TransformerMultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:   
            raise ValueError("Invalid action distribution")

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            (latent_pi, latent_pi_score), latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi, latent_pi_score = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        distribution = self._get_action_dist_from_latent(latent_pi, latent_pi_score)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = super().extract_features(obs, self.pi_features_extractor)
        latent_pi, latent_pi_score = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi,latent_pi_score)
         