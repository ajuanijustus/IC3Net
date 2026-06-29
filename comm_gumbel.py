import torch
import torch.nn.functional as F
from torch import nn

from models import MLP
from action_utils import select_action, translate_action

class CommNetMLPGumbel(nn.Module):
    """
    Decoupled Gumbel-Softmax CommNet.
    Agents learn a discrete symbolic language (tokens) rather than 
    sharing raw continuous hidden states.
    """
    def __init__(self, args, num_inputs):
        super(CommNetMLPGumbel, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.continuous = args.continuous
        
        # --- NEW GUMBEL COMM ARGUMENTS ---
        # How many distinct words in the vocabulary (default: 10)
        self.vocab_size = getattr(args, 'vocab_size', 10)
        # How many words an agent can say per step (default: 3)
        self.num_tokens = getattr(args, 'num_tokens', 3)
        # Temperature for Gumbel Softmax (controls exploration of words)
        self.gumbel_tau = getattr(args, 'gumbel_tau', 1.0)
        
        # 1. SHARED TRUNK: Encodes raw environment observations
        self.encoder = nn.Linear(num_inputs, args.hid_size)
        if args.recurrent:
            self.hidd_encoder = nn.Linear(args.hid_size, args.hid_size)
            self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)
            self.init_hidden(args.batch_size)
        else:
            if args.share_weights:
                self.f_module = nn.Linear(args.hid_size, args.hid_size)
                self.f_modules = nn.ModuleList([self.f_module for _ in range(self.comm_passes)])
            else:
                self.f_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size) for _ in range(self.comm_passes)])

        # 2. COMMPOLICY BRANCH: Decide what to say
        # Create a separate linear head for EACH token in the message phrase
        self.comm_heads = nn.ModuleList([
            nn.Linear(args.hid_size, self.vocab_size) for _ in range(self.num_tokens)
        ])
        
        # Differentiable Embedding: Converts discrete tokens back to continuous vectors for message passing
        self.msg_embedding = nn.Embedding(self.vocab_size, args.hid_size)
        
        # Comm Value Critic: Evaluates the state BEFORE communication happens
        self.comm_value_head = nn.Linear(self.hid_size, 1)

        # 3. MESSAGE PASSING NETWORK
        if args.share_weights:
            self.C_module = nn.Linear(args.hid_size, args.hid_size)
            self.C_modules = nn.ModuleList([self.C_module for _ in range(self.comm_passes)])
        else:
            self.C_modules = nn.ModuleList([nn.Linear(args.hid_size, args.hid_size) for _ in range(self.comm_passes)])

        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()

        # 4. ENVPOLICY BRANCH: Decide how to act in the environment
        if self.continuous:
            self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
            self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
        else:
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
            
        # Env Value Critic: Evaluates the state AFTER receiving messages
        self.env_value_head = nn.Linear(self.hid_size, 1)
        
        self.tanh = nn.Tanh()

        # Mask for communication (preventing self-talk)
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) - torch.eye(self.nagents, self.nagents)

    def get_agent_mask(self, batch_size, info):
        n = self.nagents
        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, 1, n).expand(batch_size, n, n).unsqueeze(-1)
        return num_agents_alive, agent_mask

    def forward_state_encoder(self, x):
        hidden_state, cell_state = None, None
        if self.args.recurrent:
            x, extras = x
            x = self.encoder(x)
            if self.args.rnn_type == 'LSTM':
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
        else:
            x = self.tanh(self.encoder(x))
            hidden_state = x
        return x, hidden_state, cell_state

    def forward(self, x, info={}):
        x, hidden_state, cell_state = self.forward_state_encoder(x)
        batch_size = x.size(0)
        n = self.nagents
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)
        agent_mask_transpose = agent_mask.transpose(1, 2)

        # --- COMMPOLICY: GENERATE DISCRETE MESSAGES ---
        comm_state = hidden_state.view(batch_size * n, self.hid_size)
        
        # 1. Compute Comm Critic Value (Before hearing from others)
        comm_value = self.comm_value_head(comm_state).view(batch_size, n, 1)
        
        # 2. Generate Tokens via Gumbel-Softmax
        comm_action_logits = []
        comm_messages = []
        
        for head in self.comm_heads:
            # Logits for vocabulary distribution
            logits = head(comm_state) 
            comm_action_logits.append(logits)
            
            # Gumbel Softmax Trick: hard=True gives a discrete one-hot vector (e.g. [0, 0, 1, 0]),
            # but the backward pass flows through the soft continuous probabilities.
            one_hot_token = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
            
            # Differentiable Embedding lookup via matrix multiplication
            # This turns the discrete word back into a continuous representation for the channel
            token_embed = torch.matmul(one_hot_token, self.msg_embedding.weight)
            comm_messages.append(token_embed)
            
        # Sum the token embeddings to form the single outgoing message payload
        # Shape: (batch_size * n, hid_size)
        outgoing_message = sum(comm_messages) 

        # --- ENVPOLICY: MESSAGE PASSING ROUNDS ---
        for i in range(self.comm_passes):
            # Reshape message for broadcasting: Agent J to Agent I
            comm = outgoing_message.view(batch_size, n, self.hid_size)
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)

            # Apply structural masks (no self-talk, ignore dead agents)
            mask = self.comm_mask.view(1, n, n).expand(comm.shape[0], n, n).unsqueeze(-1)
            comm = comm * mask
            comm = comm * agent_mask
            comm = comm * agent_mask_transpose

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            # Sum received messages
            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)

            if self.args.recurrent:
                # Combine physical observation (x) with received messages (c)
                inp = x + c
                inp = inp.view(batch_size * n, self.hid_size)
                output = self.f_module(inp, (hidden_state, cell_state))
                hidden_state = output[0]
                cell_state = output[1]
            else:
                hidden_state = sum([x, self.f_modules[i](hidden_state), c])
                hidden_state = self.tanh(hidden_state)

        # --- ENVPOLICY: ACT IN ENVIRONMENT ---
        env_state = hidden_state.view(batch_size * n, self.hid_size)
        
        # Compute Env Critic Value (After hearing from others)
        env_value = self.env_value_head(env_state).view(batch_size, n, 1)

        h = hidden_state.view(batch_size, n, self.hid_size)

        if self.continuous:
            action_mean = self.action_mean(h)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
            env_action = (action_mean, action_log_std, action_std)
        else:
            env_action = [F.log_softmax(head(h), dim=-1) for head in self.heads]

        # Package the distinct outputs
        return_payload = (
            (env_action, env_value),
            (comm_action_logits, comm_value)
        )

        if self.args.recurrent:
            return return_payload, (hidden_state.clone(), cell_state.clone())
        else:
            return return_payload

    def init_hidden(self, batch_size):
        return tuple((torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                      torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))