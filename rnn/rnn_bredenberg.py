import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# # Set seeds
# nmb = np.random.randint(0, 1000)
nmb = 0
np.random.seed(nmb)
torch.manual_seed(nmb)
device = torch.device("cpu")

##############################################
# Task parameters
##############################################
coherence_values = [-0.512, -0.256, -0.128, -0.064, -0.032, 0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
k = 0.4 # sensitivity, scaling factor for coherence to determine mean of coherence distribution
input_std = 1.0
a = 0.025 # rescaling factor, "used to keep the integrated variable within the dynamic range of network output"
b = 0.5 # bound on integration 

num_train_trials = 25000
num_test_trials = 1500
lr = 2e-6 # learning rate
batch_size = 1

# "During training, the length of each trial was selected randomly from a
# mean-shifted and truncated exponential distribution, T∼100 +
# exprand(200), with a maximum duration of 500 time steps.""
def sample_train_length():
    val = 100 + np.random.exponential(200)
    length = int(min(500, round(val)))
    if length < 1:
        length = 1
    return length

def generate_trial(T):
    C = np.random.choice(coherence_values)
    mean = k * C
    s = np.random.normal(mean, input_std, T)
    dv = a * np.cumsum(s)
    dv = np.clip(dv, -b, b) # EQUATION 1
    # inp and tgt are (T,) arrays
    inp = torch.tensor(s, dtype=torch.float32)  # (T,)
    tgt = torch.tensor(dv, dtype=torch.float32) # (T,)
    return inp, tgt, C

# For testing fix T=500 
T_test = 500
def generate_fixed_trial(C, T=T_test):
    mean = k*C
    s = np.random.normal(mean, input_std, T)
    dv = a*np.cumsum(s)
    dv = np.clip(dv, -b, b) # EQUATION 1
    return s, dv

##############################################
# Network
##############################################
"""
"The first population (P1) receives the evidence input, s(t)
via a linear projection to its neurons, and a linear decoder (Eq. 4) reads
out the integrated input from the second population (P2)."
"""
N1 = 30
N2 = 60
N = N1 + N2
limit = 1.0/math.sqrt(N)

class HierarchicalRNN(nn.Module):
    def __init__(self, N1, N2, p_feedforward=0.3):
        super(HierarchicalRNN, self).__init__()
        self.N1 = N1
        self.N2 = N2
        self.N = N1 + N2

        # From alex rnn paper "the connection weights were randomly initialized from the uniform 
        # distribution over (−1N‾‾‾√, 1N‾‾‾√) which is the default initialization scheme in PyTorch.""
        self.W11 = nn.Parameter(torch.empty(N1, N1)) # recurrent 1
        self.W22 = nn.Parameter(torch.empty(N2, N2)) # recurrent 2
        nn.init.uniform_(self.W11, -limit, limit)
        nn.init.uniform_(self.W22, -limit, limit)

        self.W21 = nn.Parameter(torch.empty(N2, N1)) # input to 2 from 1
        nn.init.uniform_(self.W21, -limit, limit)

        # binary mask for W21
        mask = (torch.rand(N2, N1) < p_feedforward).float()
        self.register_buffer('W21_mask', mask)

        self.Win = nn.Parameter(torch.empty(N1, 1))
        nn.init.uniform_(self.Win, -limit, limit)

        self.Wout = nn.Parameter(torch.empty(1, N2))
        nn.init.uniform_(self.Wout, -limit, limit)

        self.tanh = nn.Tanh()

    def forward(self, inp):
        # inp: (T,B,1)
        T = inp.shape[0]
        B = inp.shape[1]
        # last dim is always 1 (feature)
        
        # Hidden states: (B,N1), (B,N2)
        r1 = inp.new_zeros(B, N1)
        r2 = inp.new_zeros(B, N2)

        outputs = []
        for t in range(T):
            # x_t:(B,1)
            x_t = inp[t,:]  
            # noise per step
            noise1 = torch.randn(B, N1, device=x_t.device)*0.1 # s ~ N(kC, 0.1) 
            noise2 = torch.randn(B, N2, device=x_t.device)*0.1

            # Input to P1
            r1_inp = x_t @ self.Win.t()
            r1_out = self.tanh(r1 @ self.W11.t() + r1_inp + noise1) # EQUATION 3

            # Feedforward to P2:
            ff = (self.W21 * self.W21_mask) @ r1_out.transpose(0,1)  # (N2,B) # WITHOUT MASK
            # ff = self.W21 @ r1_out.transpose(0,1)  # (N2,B) # WITH MASK
            ff = ff.transpose(0,1) # (B,N2)

            r2_out = self.tanh(r2 @ self.W22.t() + ff + noise2) # EQUATION 3

            r1 = r1_out
            r2 = r2_out

            # Output y=(B,1)
            y = r2 @ self.Wout.t()
            outputs.append(y.unsqueeze(0)) # (1,B,1)

        return torch.cat(outputs, dim=0) # (T,B,1)

    def record_hidden(self, inp):
        if inp.dim() == 3:
            inp = inp.squeeze(-1) 
        
        T = inp.shape[0]
        B = inp.shape[1] if inp.dim() > 1 else 1
        if B != 1:
            raise ValueError("record_hidden method is designed for single-trial only.")
        
        # Now inp:(T,1)
        r1 = inp.new_zeros(1, self.N1) # (1,N1)
        r2 = inp.new_zeros(1, self.N2) # (1,N2)

        r1_activity = []
        r2_activity = []

        for t in range(T):
            x_t = inp[t,:]  # (1,)
            x_t = x_t.unsqueeze(0) if x_t.dim()==0 else x_t # ensure (1,1) if needed
            # noise per step
            # noise1 = torch.randn(1, self.N1, device=inp.device)*0.1
            # noise2 = torch.randn(1, self.N2, device=inp.device)*0.1

            # Input to P1
            # x_t:(1,1), Win:(N1,1)
            r1_inp = x_t @ self.Win.t() # (1,N1)
            r1_out = self.tanh(r1 @ self.W11.t() + r1_inp) # + noise1)

            ff = (self.W21 * self.W21_mask) @ r1_out.transpose(0,1)  # (N2,1)
            # ff = self.W21 @ r1_out.transpose(0,1)  # (N2,1)
            ff = ff.transpose(0,1) # (1,N2)
            r2_out = self.tanh(r2 @ self.W22.t() + ff) # + noise2)

            r1 = r1_out
            r2 = r2_out

            r1_activity.append(r1.detach().cpu().numpy()) # (1,N1)
            r2_activity.append(r2.detach().cpu().numpy()) # (1,N2)

        r1_activity = np.concatenate(r1_activity, axis=0) # (T,N1)
        r2_activity = np.concatenate(r2_activity, axis=0) # (T,N2)

        return r1_activity, r2_activity

model = HierarchicalRNN(N1, N2).to(device)
criterion = nn.MSELoss() # EQUATION 2
optimizer = optim.Adam(model.parameters(), lr=lr)

##############################################
# Training
##############################################
model.train()
losses = []

num_batches = num_train_trials // batch_size
losses_batch = []
losses_idx = []
for batch_i in range(num_batches):
    T = sample_train_length()
    inputs = []
    targets = []
    for _ in range(batch_size):
        inp_i, tgt_i, C_i = generate_trial(T)
        # inp_i:(T,), tgt_i:(T,)
        inputs.append(inp_i)
        targets.append(tgt_i)

    inp_tensor = torch.stack(inputs, dim=1).unsqueeze(-1).to(device)  # (T,B,1)
    tgt_tensor = torch.stack(targets, dim=1).unsqueeze(-1).to(device) # (T,B,1)

    optimizer.zero_grad()
    out = model(inp_tensor) # (T,B,1)
    loss = criterion(out, tgt_tensor) # EQUATION 2
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
    optimizer.step()

    current_trial = (batch_i+1)*batch_size
    losses_batch.append(loss.item())
    if current_trial % 1000 == 0:
        print(f"Trial {current_trial}/{num_train_trials}, T={T}, Loss={loss.item():.6f}")
        losses.append(np.mean(losses_batch))
        midpoint = current_trial - batch_size//2
        losses_idx.append(midpoint)
        losses_batch = []

# (b) Plot training loss over time
plt.figure()
plt.plot(losses_idx, losses, 'o-')
plt.xlabel('Group Midpoint')
plt.ylabel('Avg MSE Loss (per 1000 bins)')
plt.title('Training Loss (Fig. 1b)')
plt.tight_layout()
# plt.show()
plt.savefig('Fig1b.png')

##############################################
# Testing
##############################################
model.eval()

def simulate_test_trials(n_trials=1500, T=500):
    coh_indices = {c: {"Tin": [], "Tout": []} for c in coherence_values}
    Cs = []
    outputs = []
    s_all = []
    dv_all = []
    for _ in range(n_trials):
        C = np.random.choice(coherence_values)
        s, dv = generate_fixed_trial(C, T)
        inp = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(-1) # (T,1)
        with torch.no_grad():
            out = model(inp).cpu().numpy().squeeze()  # (T,)
        Cs.append(C)
        outputs.append(out)
        s_all.append(s)
        dv_all.append(dv)
    return np.array(Cs), np.array(outputs), np.array(s_all), np.array(dv_all)

Cs_test, outs_test, s_test, dv_test = simulate_test_trials(num_test_trials, T_test)

# (c,e) Compute mean and variance by coherence
coh_unique = sorted(coherence_values)
colors = plt.cm.coolwarm(np.linspace(0,1,len(coh_unique)))
mean_out = {}
var_out = {}
for C_ in coh_unique:
    mask = (Cs_test==C_)
    out_C = outs_test[mask] # (nC, T)
    mean_out[C_] = out_C.mean(axis=0)
    var_out[C_] = out_C.var(axis=0)

# (c) Mean output vs time with linear fit
plt.figure()
c = 0
for C_ in coh_unique:
    x_ = np.arange(50)
    y_ = mean_out[C_][:50]
    plt.plot(x_, y_, 'o', color=colors[c], label=f'C={C_}') 
    c += 1
    
    slope, intercept = np.polyfit(x_, y_, 1)
    y_fit_line = intercept + slope*x_
    plt.plot(x_, y_fit_line, 'k-', alpha=0.7)

plt.xlabel('Time steps')
plt.ylabel('Mean Output')
plt.title('Mean Output vs Time (Fig. 1c)')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
# plt.show()
plt.savefig('Fig1c.png')

# (d) Slope vs coherence
slopes = []
for C_ in coh_unique:
    x_ = np.arange(50)
    y_ = mean_out[C_][:50]
    slope, intercept = np.polyfit(x_, y_, 1)
    slopes.append((C_, slope))
slopes = sorted(slopes, key=lambda x: x[0])

plt.figure()
# slopes as points
plt.plot([c for c,s in slopes],[s for c,s in slopes],'ko')
# fit line to slopes
slope, intercept = np.polyfit([c for c,s in slopes],[s for c,s in slopes],1)
x_fit = np.linspace(min(coh_unique), max(coh_unique),200)
y_fit = slope*x_fit + intercept
plt.plot(x_fit, y_fit, 'k-', alpha=0.7)

plt.xlabel('Coherence')
plt.ylabel('Slope (0-50 steps)')
plt.title('Slope vs Coherence (Fig. 1d)')
plt.tight_layout()
# plt.show()
plt.savefig('Fig1d.png')

# (e) Variance vs time
plt.figure()
c = 0
for C_ in coh_unique:
    plt.plot(var_out[C_], label=f'C={C_}', color=colors[c])
    c += 1
plt.xlabel('Time steps')
plt.ylabel('Variance')
plt.title('Variance vs Time (Fig. 1e)')
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
# plt.show()
plt.savefig('Fig1e.png')

# (f,g) Psychometric and Chronometric
def get_choice_rt(out, bound=0.5):
    idx_pos = np.where(out>=bound)[0]
    idx_neg = np.where(out<=-bound)[0]
    pos_rt = idx_pos[0] if len(idx_pos)>0 else T_test
    neg_rt = idx_neg[0] if len(idx_neg)>0 else T_test

    if pos_rt < neg_rt:
        choice = 1
        rt = pos_rt
    elif neg_rt < pos_rt:
        choice = -1
        rt = neg_rt
    else:
        choice = 1 if out[-1]>=0 else -1
        rt = T_test
    return choice, rt

choices = []
rts = []
for i in range(num_test_trials):
    c,r = get_choice_rt(outs_test[i]) # EQUATION 5 and 6
    choices.append(c)
    rts.append(r)
choices = np.array(choices)
rts = np.array(rts)

p_right = []
for C_ in coh_unique:
    mask = (Cs_test==C_)
    p = np.mean(choices[mask]==1)
    p_right.append(p)

# Psychometric 
def logistic(x,b0,b1):
    return 1/(1+np.exp(-(b0+b1*x)))
popt,_ = curve_fit(logistic, coh_unique, p_right) # EQUATION 7
x_fit = np.linspace(min(coh_unique), max(coh_unique),200)
y_fit = logistic(x_fit,*popt)

plt.figure()
# plt.plot(coh_unique,p_right,'ko',label='Data')
plt.plot(x_fit,y_fit,'k-',label='Fit')
for i in range(len(coh_unique)):
    plt.plot(coh_unique[i],p_right[i],'o',color=colors[i])
plt.xlabel('Coherence')
plt.ylabel('P(Right)')
plt.title('Psychometric (Fig. 1f)')
plt.axhline(0.5,color='gray',linestyle='--')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('Fig1f.png')

# Chronometric 
mean_rt = []
for C_ in coh_unique:
    mask=(Cs_test==C_)
    mean_rt.append(np.mean(rts[mask]))

def gauss_bell(x,b0,b1,b2):
    return b0+b1*np.exp(-((x/b2)**2))
popt_c,_ = curve_fit(gauss_bell, coh_unique, mean_rt) # EQUATION 8
y_fit_c = gauss_bell(x_fit,*popt_c)

plt.figure()
# plt.plot(coh_unique,mean_rt,'o',label='Data') 
for i in range(len(coh_unique)):
    plt.plot(coh_unique[i],mean_rt[i],'o',color=colors[i])
plt.plot(x_fit,y_fit_c,'-',label='Fit')
plt.xlabel('Coherence')
plt.ylabel('Mean Decision Time')
plt.title('Chronometric (Fig. 1g)')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('Fig1g.png')

# (h,i,j) Example neuron responses from P1 and P2
example_cohs = coh_unique
n_trials_example = 200
with torch.no_grad():
    P1_responses = {C:[] for C in example_cohs}
    P2_responses = {C:[] for C in example_cohs}

    for C_ in example_cohs:
        for _ in range(n_trials_example):
            s, dv = generate_fixed_trial(C_, T_test)
            inp = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(-1)
            r1_act, r2_act = model.record_hidden(inp)
            P1_responses[C_].append(r1_act) # (T,N1)
            P2_responses[C_].append(r2_act) # (T,N2)

    for C_ in example_cohs:
        P1_responses[C_] = np.mean(np.stack(P1_responses[C_], axis=0), axis=0) # (T,N1)
        P2_responses[C_] = np.mean(np.stack(P2_responses[C_], axis=0), axis=0) # (T,N2)

plt.figure()
neuron_examples = 1
for i in range(neuron_examples):
    plt.subplot(neuron_examples,1,i+1)
    c = 0
    for C_ in example_cohs:
        plt.plot(P1_responses[C_][:100,i], label=f'C={C_}', color=colors[c])
        c += 1
    if i==0:
        plt.legend()
    plt.ylabel(f'P1 Unit {i+1}')
plt.xlabel('Time steps')
plt.suptitle('Example P1 Neurons (Fig. 1h)')
plt.tight_layout()
# plt.show()
plt.savefig('Fig1h.png')

plt.figure()
for i in range(neuron_examples):
    plt.subplot(neuron_examples,1,i+1)
    c = 0
    for C_ in example_cohs:
        plt.plot(P2_responses[C_][:100,i], label=f'C={C_}', color=colors[c])
        c += 1
    if i==0:
        plt.legend()
    plt.ylabel(f'P2 Unit {i+1}')
plt.xlabel('Time steps')
plt.suptitle('Example P2 Neurons (Fig. 1i,j)')
plt.tight_layout()
# plt.show()
plt.savefig('Fig1hij.png')

##############################################
# Save hidden activity
##############################################

def collect_hidden_states(model, coherence_values, test_trials, Cs_test, save_path="./rnn/rnn_data/hidden_activity"):
    os.makedirs(save_path, exist_ok=True)
    hidden_states = {}
    for C in coherence_values:
        hidden_p1 = []
        hidden_p2 = []
        for _ in range(len(test_trials)):
            s = test_trials[np.where(Cs_test==C)[0][0]]
            inp = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(-1)
            r1_act, r2_act = model.record_hidden(inp)  
            hidden_p1.append(r1_act)
            hidden_p2.append(r2_act)
        hidden_p1 = np.stack(hidden_p1, axis=0) 
        hidden_p2 = np.stack(hidden_p2, axis=0) 
        hidden_states[C] = {'P1': hidden_p1, 'P2': hidden_p2}
        np.save(os.path.join(save_path, f"P1_hidden_C{C}.npy"), hidden_p1)
        np.save(os.path.join(save_path, f"P2_hidden_C{C}.npy"), hidden_p2)
    return hidden_states

def collect_hidden_states_by_bounds(model, coherence_values, test_trials, Cs_test, bound, save_path="./rnn/rnn_data/hidden_activity"):
    os.makedirs(save_path, exist_ok=True)
    
    hidden_states = {c: {"Tin": {"P1": [], "P2": []}, "Tout": {"P1": [], "P2": []}} for c in coherence_values}
    hidden_state_ramps = {c: {"Tin": {"P1": [], "P2": []}, "Tout": {"P1": [], "P2": []}} for c in coherence_values}
    
    for i, C in enumerate(coherence_values):
        trial_indices = np.where(Cs_test == C)[0]
        for idx in trial_indices:
            s = test_trials[idx]
            inp = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(-1)
            
            with torch.no_grad():
                out = model(inp).cpu().numpy().squeeze()  
                r1_act, r2_act = model.record_hidden(inp)  
            
            idx_pos = np.where(out >= bound)[0]
            idx_neg = np.where(out <= -bound)[0]
            pos_rt = idx_pos[0] if len(idx_pos) > 0 else len(out)
            neg_rt = idx_neg[0] if len(idx_neg) > 0 else len(out)
            
            if pos_rt < neg_rt:
                choice = "Tin"
                bound_idx = idx_pos[0]
            elif neg_rt < pos_rt:
                choice = "Tout"
                bound_idx = idx_neg[0]
            else:
                choice = "Tin" if out[-1] >= 0 else "Tout" 
                bound_idx = len(out)
            
            hidden_states[C][choice]["P1"].append(r1_act)
            hidden_states[C][choice]["P2"].append(r2_act)

            r1_ramp = np.arange(0, bound_idx) / bound_idx
            r1_ramp = np.hstack((r1_ramp, np.ones(len(out)-len(r1_ramp))))
            r2_ramp = np.arange(0, bound_idx) / bound_idx
            r2_ramp = np.hstack((r2_ramp, np.ones(len(out)-len(r2_ramp))))
            hidden_state_ramps[C][choice]["P1"].append(r1_ramp)
            hidden_state_ramps[C][choice]["P2"].append(r2_ramp)
        
        for choice in ["Tin", "Tout"]:
            hidden_states[C][choice]["P1"] = np.array(hidden_states[C][choice]["P1"])
            hidden_states[C][choice]["P2"] = np.array(hidden_states[C][choice]["P2"])
            hidden_state_ramps[C][choice]["P1"] = np.array(hidden_state_ramps[C][choice]["P1"])
            hidden_state_ramps[C][choice]["P2"] = np.array(hidden_state_ramps[C][choice]["P2"])
            
            np.save(os.path.join(save_path, f"P1_hidden_C{C}_{choice}.npy"), hidden_states[C][choice]["P1"])
            np.save(os.path.join(save_path, f"P2_hidden_C{C}_{choice}.npy"), hidden_states[C][choice]["P2"])
            np.save(os.path.join(save_path, f"P1_hidden_ramp_C{C}_{choice}.npy"), hidden_state_ramps[C][choice]["P1"])
            np.save(os.path.join(save_path, f"P2_hidden_ramp_C{C}_{choice}.npy"), hidden_state_ramps[C][choice]["P2"])
    
    return hidden_states, hidden_state_ramps

def collect_hidden_states_correct_trials(model, coherence_values, test_trials, Cs_test, save_path="./rnn/rnn_data/hidden_activity", bound=None):
    os.makedirs(save_path, exist_ok=True)
    hidden_states = {C: {"P1": [], "P2": []} for C in coherence_values}
    
    for C in coherence_values:
        trial_indices = np.where(Cs_test == C)[0]
        for idx in trial_indices:
            s = test_trials[idx]
            inp = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(-1)
            
            with torch.no_grad():
                out = model(inp).cpu().numpy().squeeze()  # Model output
                r1_act, r2_act = model.record_hidden(inp)  # Hidden states
            
            # Choice
            idx_pos = np.where(out >= bound)[0]
            idx_neg = np.where(out <= -bound)[0]
            pos_rt = idx_pos[0] if len(idx_pos) > 0 else len(out)
            neg_rt = idx_neg[0] if len(idx_neg) > 0 else len(out)

            if pos_rt < neg_rt:
                model_choice = "Tin"
            elif neg_rt < pos_rt:
                model_choice = "Tout"
            else:
                model_choice = "Tin" if out[-1] >= 0 else "Tout" 
            
            true_choice = "Tin" if C > 0 else "Tout"
            
            if model_choice == true_choice:  
                hidden_states[C]["P1"].append(r1_act)
                hidden_states[C]["P2"].append(r2_act)
        
        hidden_states[C]["P1"] = np.array(hidden_states[C]["P1"])
        hidden_states[C]["P2"] = np.array(hidden_states[C]["P2"])
        
        np.save(os.path.join(save_path, f"P1_hidden_C{C}_correct.npy"), hidden_states[C]["P1"])
        np.save(os.path.join(save_path, f"P2_hidden_C{C}_correct.npy"), hidden_states[C]["P2"])
    
    return hidden_states

# Hidden states for all coherence levels
# hidden_states = collect_hidden_states(model, coherence_values, s_test, Cs_test) 
hidden_states, hidden_state_ramps = collect_hidden_states_by_bounds(model, coherence_values, s_test, Cs_test)
# hidden_states = collect_hidden_states_correct_trials(model, coherence_values, s_test, Cs_test, 0.4) # Look at equation 5, paper uses b=0.5 but bound=0.4 in eq 5