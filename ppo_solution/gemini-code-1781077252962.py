import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. LATENT DIFFUSION GENERATOR (Algorithm 1)
# ==========================================
class LatentDiffusionGenerator(nn.Module):
    def __init__(self, encoder, decoder, denoiser):
        super().__init__()
        self.encoder = encoder  # E(x)
        self.decoder = decoder  # D(z)
        self.denoiser = denoiser # e_theta(z_t, t, c, psi(x))
        
    def forward(self, x_t, action, c, alpha_cls=2.0, alpha_pres=0.1):
        """
        Executes a single structural edit mapped from the PPO action space.
        Maps to Algorithm 1 and Eq: x_{t+1} = G_phi(x_t, a_t, c)
        """
        r_t, u_t, alpha_t, tau_t = action
        
        # 1. Encode image: z_0 <- E(x)
        z_t = self.encoder(x_t)
        
        # In a real implementation, the action (r_t, u_t, alpha_t) would condition 
        # the diffusion timestep t or the noise mask to apply a localized edit.
        # Here we mock the denoising step guided by gradients (Eq in Algorithm 1, Line 8)
        
        # ... Reverse diffusion loop applied for a subset of steps controlled by alpha_t ...
        # z_{t-1} = mu_theta(...) - alpha_cls * grad(L_cls) - alpha_pres * grad(L_pres)
        
        # 10. Decode final provisional image
        x_t_plus_1 = self.decoder(z_t) # Mock decode
        return x_t_plus_1

# ==========================================
# 2. PPO ENVIRONMENT (Algorithm 2 Logic)
# ==========================================
class MedicalCounterfactualEnv:
    def __init__(self, query_image, classifier, target_c, generator, seg_net, obj_detector, budget_T=25):
        self.x_orig = query_image
        self.f = classifier
        self.c = target_c
        self.generator = generator
        
        # Frozen networks for preservation rewards
        self.S = seg_net
        self.O = obj_detector
        
        self.T = budget_T
        self.current_step = 0
        self.x_t = query_image
        
        # Precompute initial latent and distances
        with torch.no_grad():
            self.z_orig = self.generator.encoder(self.x_orig)
            self.p_orig = self.f(self.x_orig) # Target class index assumed handled internally
            self.d_t = torch.abs(self.p_orig - self.c)
            
    def get_observation(self):
        """Constructs state o_t = [z_orig, z_t, s_t, d_t, c, m_t]"""
        with torch.no_grad():
            z_t = self.generator.encoder(self.x_t)
        s_t = torch.tensor([self.current_step / self.T])
        m_t = self.S(self.x_t) # Anatomical feasibility mask
        
        # Flatten and concatenate into a single state vector for the PPO policy
        o_t = torch.cat([self.z_orig.flatten(), z_t.flatten(), s_t, self.d_t, torch.tensor([self.c]), m_t.flatten()])
        return o_t

    def step(self, action):
        """Executes action, computes rewards, returns next state"""
        self.current_step += 1
        
        # Unpack action: (region, edit_type, strength, stop_flag)
        r_t, u_t, alpha_t, tau_t = action
        
        # Generate next state
        x_next = self.generator(self.x_t, action, self.c)
        p_next = self.f(x_next)
        d_next = torch.abs(p_next - self.c)
        
        # --- Reward Computation ---
        # 1. Progress: d_t - d_{t+1} (Corrected from text typo)
        R_prog = self.d_t - d_next
        
        # 2. Validity: -|f(x_t+1) - c|
        R_val = -d_next
        
        # 3. Fidelity: -LPIPS(x_{t+1}, x_orig)
        R_fid = -F.mse_loss(x_next, self.x_orig) # Using MSE as mock for LPIPS
        
        # 4. Preservation: Context-aware preservation
        S_x = self.S(self.x_orig)
        l1_diff = torch.abs(x_next - self.x_orig)
        # S_j(x) * ||x_{t+1} - x||_1 / sum(S_j(x))
        R_pres = -torch.sum(S_x * l1_diff) / (torch.sum(S_x) + 1e-8)
        
        # 5. Budget Penalty
        R_bud = -1.0
        
        # 6. Stop Reward
        is_terminal_valid = (d_next < 0.05).float() # epsilon = 0.05
        R_stop = is_terminal_valid if tau_t > 0.5 else 0.0
        
        # Total Reward (using lambda weights instead of the typo 'A')
        lam_prog, lam_val, lam_fid, lam_pres, lam_bud, lam_stop = 1.0, 2.0, 0.5, 0.5, 0.1, 5.0
        total_reward = (lam_prog * R_prog + lam_val * R_val + lam_fid * R_fid + 
                        lam_pres * R_pres + lam_bud * R_bud + lam_stop * R_stop)
        
        # Update internal state
        self.x_t = x_next
        self.d_t = d_next
        
        # Check termination
        done = bool(is_terminal_valid > 0 or tau_t > 0.5 or self.current_step >= self.T)
        
        return self.get_observation(), total_reward, done, {"x_t": self.x_t}

# ==========================================
# 3. PPO POLICY NETWORK (With Auxiliary Heads)
# ==========================================
class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # PPO standard heads
        self.actor = nn.Linear(256, action_dim) # Outputs params for (r_t, u_t, alpha_t, tau_t)
        self.critic = nn.Linear(256, 1)         # Outputs Value V(s)
        
        # Auxiliary heads to improve exploration in hard image spaces
        self.feasibility_head = nn.Linear(256, 1) # Predicts if edit is valid under anatomical mask
        self.progress_head = nn.Linear(256, 1)    # Predicts expected reduction in target-class error
        self.stop_head = nn.Linear(256, 1)        # Predicts if trajectory should terminate
        
    def forward(self, obs):
        features = self.shared(obs)
        
        # Action logits/values
        action_logits = self.actor(features)
        value = self.critic(features)
        
        # Auxiliary predictions
        feasibility_pred = torch.sigmoid(self.feasibility_head(features))
        progress_pred = self.progress_head(features)
        stop_pred = torch.sigmoid(self.stop_head(features))
        
        return action_logits, value, feasibility_pred, progress_pred, stop_pred