import torch
import torch.nn.functional as F
from torch import nn

class CommNetMLP(nn.Module):
    def __init__(self, args, num_inputs):
        super(CommNetMLP, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.recurrent = args.recurrent
        self.continuous = args.continuous
        
        self.init_std = args.init_std if hasattr(args, 'comm_init_std') else 0.2

        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)

        # ==========================================
        # PHASE 4: SEPARATE TRUNKS INITIALIZATION
        # ==========================================
        if args.phase == 4:
            self.comm_encoder = nn.Linear(num_inputs, args.hid_size)
            self.env_encoder = nn.Linear(num_inputs, args.hid_size)
        else:
            # Shared trunk for Phases 1, 2, and 3
            self.encoder = nn.Linear(num_inputs, args.hid_size)

        # Branching Modules
        if args.phase == 1:
            if args.recurrent:
                self.init_hidden(args.batch_size)
                self.f_module = nn.LSTMCell(args.hid_size, args.hid_size)
            self.heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in args.naction_heads])
            self.value_head = nn.Linear(self.hid_size, 1)
        else:
            # Phases 2, 3, and 4
            if args.recurrent:
                self.init_hidden(args.batch_size)
                self.comm_f_module = nn.LSTMCell(args.hid_size, args.hid_size)
                self.env_f_module = nn.LSTMCell(args.hid_size, args.hid_size)
            
            if self.continuous:
                self.action_mean = nn.Linear(args.hid_size, args.dim_actions)
                self.action_log_std = nn.Parameter(torch.zeros(1, args.dim_actions))
            else:
                if args.phase == 2:
                    env_action_heads = args.naction_heads[:-1]
                else:  # Phases 3 and 4
                    env_action_heads = args.naction_heads[:-args.nagents]
                self.env_heads = nn.ModuleList([nn.Linear(args.hid_size, o) for o in env_action_heads])

            if args.phase == 2:
                self.comm_head = nn.Linear(args.hid_size, 2)
            else:  # Phases 3 and 4 Matrix Layout
                self.comm_head = nn.Linear(args.hid_size, args.nagents * 2)

            self.env_value_head = nn.Linear(self.hid_size, 1)
            self.comm_value_head = nn.Linear(self.hid_size, 1)

        self.C_modules = nn.ModuleList([
            nn.Linear(args.hid_size, args.hid_size) for _ in range(self.comm_passes)
        ])

        if args.comm_init == 'zeros':
            for i in range(self.comm_passes):
                self.C_modules[i].weight.data.zero_()
        self.tanh = nn.Tanh()

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

    def init_hidden(self, batch_size):
        if self.args.phase == 1:
            return tuple((torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                          torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))
        else:
            return tuple((torch.zeros(batch_size * self.nagents, self.hid_size * 2, requires_grad=True),
                          torch.zeros(batch_size * self.nagents, self.hid_size * 2, requires_grad=True)))

    def forward(self, x, info={}):
        # Phase 1 Backward Compatibility Block
        if self.args.phase == 1:
            x, extras = x
            hidden_state, cell_state = extras
            x = self.encoder(x)
            batch_size = x.size()[0]
            n = self.nagents
            num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

            if self.args.hard_attn:
                comm_action = torch.tensor(info['comm_action'])
                comm_action_mask = comm_action.expand(batch_size, n, n).unsqueeze(-1)
                agent_mask *= comm_action_mask.double()

            agent_mask_transpose = agent_mask.transpose(1, 2)

            for i in range(self.comm_passes):
                comm = hidden_state.view(batch_size, n, self.hid_size)
                comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)
                mask = self.comm_mask.view(1, n, n).expand(comm.shape[0], n, n).unsqueeze(-1)
                comm = comm * mask

                if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                    comm = comm / (num_agents_alive - 1)

                comm = comm * agent_mask * agent_mask_transpose
                comm_sum = comm.sum(dim=1)
                c = self.C_modules[i](comm_sum)

                inp_flat = (x + c).view(batch_size * n, self.hid_size)
                hidden_state, cell_state = self.f_module(inp_flat, (hidden_state, cell_state))

            value_head = self.value_head(hidden_state)
            h = hidden_state.view(batch_size, n, self.hid_size)
            action = [F.log_softmax(head(h), dim=-1) for head in self.heads]
            return action, value_head, (hidden_state.clone(), cell_state.clone())

        # ==========================================
        # PHASES 2, 3, & 4 MOVEMENT AND PIPELINES
        # ==========================================
        x, (prev_hidden, prev_cell) = x
        batch_size = x.size()[0]
        n = self.nagents
        
        # Unpack split hidden state dimensions
        comm_h, env_h = prev_hidden.chunk(2, dim=-1)
        comm_c, env_c = prev_cell.chunk(2, dim=-1)
        num_agents_alive, agent_mask = self.get_agent_mask(batch_size, info)

        # 1. ENCODING PIPELINE STEP (COMMUNICATION)
        if self.args.phase == 4:
            x_comm_encoded = self.comm_encoder(x)
        else:
            x_comm_encoded = self.encoder(x)

        # 2. GATING / ROUTING DECISION (Based on incoming time-step state)
        if self.args.phase == 2:
            comm_logits = F.log_softmax(self.comm_head(comm_h), dim=-1)
            rng_state = torch.get_rng_state()
            comm_action = torch.multinomial(torch.exp(comm_logits), 1).squeeze(-1)
            torch.set_rng_state(rng_state)

            if self.args.hard_attn:
                comm_action_mask = comm_action.view(batch_size, n, 1).expand(batch_size, n, n).unsqueeze(-1)
                agent_mask = agent_mask * comm_action_mask.double()
                
        elif self.args.phase in [3, 4]:
            comm_logits_raw = self.comm_head(comm_h).view(batch_size, n, n, 2)
            comm_logits = F.log_softmax(comm_logits_raw, dim=-1)
            
            rng_state = torch.get_rng_state()
            comm_probs_flat = torch.exp(comm_logits).view(-1, 2)
            comm_action_flat = torch.multinomial(comm_probs_flat, 1).squeeze(-1)
            comm_action = comm_action_flat.view(batch_size, n, n)
            torch.set_rng_state(rng_state)

            if self.args.hard_attn:
                comm_action_mask = comm_action.unsqueeze(-1)
                agent_mask = agent_mask * comm_action_mask.double()

        agent_mask_transpose = agent_mask.transpose(1, 2)
        
        # 3. DYNAMIC MULTI-PASS MESSAGE PROPAGATION LOOP
        # Both comm_h and comm_c evolve at each pass as messages flow
        for i in range(self.comm_passes):
            comm = comm_h.view(batch_size, n, self.hid_size)
            comm = comm.unsqueeze(-2).expand(-1, n, n, self.hid_size)
            mask = self.comm_mask.view(1, n, n).expand(comm.shape[0], n, n).unsqueeze(-1)
            comm = comm * mask

            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

            comm = comm * agent_mask * agent_mask_transpose
            comm_sum = comm.sum(dim=1)
            c = self.C_modules[i](comm_sum)

            # UPDATE STEP: Evolve the communication representations with the new messages
            comm_inp = x_comm_encoded + c
            comm_inp_flat = comm_inp.view(batch_size * n, self.hid_size)
            comm_h, comm_c = self.comm_f_module(comm_inp_flat, (comm_h, comm_c))

        # 4. ENVIRONMENT BRANCH PROCESSING
        if self.args.phase == 4:
            x_env_encoded = self.env_encoder(x)
        else:
            x_env_encoded = self.encoder(x)

        # The environment branch integrates the final message payload consensus 'c'
        env_inp = x_env_encoded + c
        env_inp_flat = env_inp.view(batch_size * n, self.hid_size)
        env_h, env_c = self.env_f_module(env_inp_flat, (env_h, env_c))
        
        h_env_reshaped = env_h.view(batch_size, n, self.hid_size)
        env_logits = [F.log_softmax(head(h_env_reshaped), dim=-1) for head in self.env_heads]
        
        if self.args.phase == 2:
            action = env_logits + [comm_logits.view(batch_size, n, 2)]
        elif self.args.phase in [3, 4]:
            comm_logits_heads = [comm_logits[:, :, r, :] for r in range(n)]
            action = env_logits + comm_logits_heads

        env_v = self.env_value_head(env_h)
        comm_v = self.comm_value_head(comm_h)
        value = (env_v, comm_v)
        
        next_hidden = torch.cat([comm_h, env_h], dim=-1)
        next_cell = torch.cat([comm_c, env_c], dim=-1)

        return action, value, (next_hidden, next_cell)