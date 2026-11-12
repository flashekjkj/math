import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_logprob(model,input_ids,attention_mask):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:,:-1,:]
    target = input_ids[:,1:]
    logp = F.log_softmax(logits, dim=-1)
    logp_token = torch.gather(logp,dim=-1,index=target.unsqueeze(-1)).squeeze(-1)
    mask = attention_mask[:,1:]
    sum_logp = (logp_token * mask).sum(dim=-1)
    return sum_logp

class DPO(nn.Module):
    def __init__(self,policy_model,ref_model,beta=0.1):
        super().__init__()
        self.policy = policy_model
        self.ref = ref_model
        self.beta = beta
        for p in self.ref.parameters():
            p.requires_grad = False

    def forward(self,chosen_ids,chosen_mask,reject_ids,reject_mask):
        logp_p_chosen = compute_logprob(self.policy,chosen_ids,chosen_mask)
        logp_p_reject = compute_logprob(self.policy,reject_ids,reject_mask)

        with torch.no_grad():
            logp_r_chosen = compute_logprob(self.ref,chosen_ids,chosen_mask)
            logp_r_reject = compute_logprob(self.ref,reject_ids,reject_mask)

        log_ratio_chosen = logp_p_chosen - logp_r_chosen
        log_ratio_reject = logp_p_reject - logp_r_reject

        inside = self.beta * (log_ratio_chosen - log_ratio_reject)
        loss = -F.logsigmoid(inside).mean()
        return loss

class ActorCritic(nn.Module):
    def __init__(self,base_llm,hidden_dim):
        super().__init()
        self.base = base_llm
        self.critic_head = nn.Linear(hidden_dim,1)

    def forward(self,input_ids,attention_mask):
        out = self.base(input_ids=input_ids,attention_mask=attention_mask,output_hiddden_states=True)
        last_hidden = out.hidden_states[-1][:,-1,:]
        value = self.critic_head(last_hidden)
        logits = out.logits
        return logits,value

def ppo_loss(logp_new,logp_old,advantage,eps=0.2):
    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio,1-eps,1+eps) * advantage
    actor_loss = -torch.min(surr1,surr2).mean()
    return actor_loss

def critic_loss(value,value_target):
    return F.mse_loss(value,value_target)

def grpo_loss(logp_new,logp_old,group_rewards,eps=0.2,eps_std=1e-8):
    mean_r = group_rewards.mean()
    std_r = group_rewards.std()
    adv = (group_rewards - mean_r) / (std_r + eps_std)

    ratio = torch.exp(logp_new - logp_old)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio,1-eps,1+eps) * adv
    loss = -torch.min(surr1,surr2).mean()

    return loss
    