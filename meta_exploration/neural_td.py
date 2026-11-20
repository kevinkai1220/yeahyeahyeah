"""World-local feedforward Double DQN and observed-statistics meta-explorer.

The Q network has NO recurrent state/history encoder. Only parameter updates
adapt it to a world. Replay ownership is checked before any mutation.
"""
import copy
import time
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .learning import Transition, Controller


class QNetwork(nn.Module):
    def __init__(self, hidden=64, reward_input_scale=10.):
        super().__init__()
        self.reward_input_scale=reward_input_scale
        self.net=nn.Sequential(nn.Linear(11,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,3))

    def forward(self,x):
        # Fixed input units only; targets/rewards remain unscaled.
        scale=x.new_ones(11);scale[-1]=self.reward_input_scale
        return self.net(x/scale)


@torch.no_grad()
def double_dqn_targets(online,target,next_states,rewards,dones,gamma):
    selected=online(next_states).argmax(-1,keepdim=True)
    continuation=target(next_states).gather(1,selected).squeeze(1)
    return rewards+gamma*(1-dones.float())*continuation


class Replay:
    def __init__(self,world_id,capacity,seed):
        self.world_id=world_id;self.capacity=capacity;self.rng=np.random.default_rng(seed)
        self.data=[];self.cursor=0;self.inserted=0;self.sampled=0;self.visits={}

    def add(self,transition,world_id):
        if world_id!=self.world_id:raise ValueError('Cross-world replay insertion forbidden')
        t=transition
        row=(t.observation.vector().copy(),t.action,float(t.reward),t.next_observation.vector().copy(),bool(t.done),self.inserted)
        if len(self.data)<self.capacity:self.data.append(row)
        else:self.data[self.cursor]=row
        self.cursor=(self.cursor+1)%self.capacity;self.inserted+=1

    def sample(self,batch_size,device):
        if not self.data:raise ValueError('Empty replay')
        indices=self.rng.integers(len(self.data),size=batch_size)
        rows=[self.data[i] for i in indices]
        for row in rows:self.visits[row[5]]=self.visits.get(row[5],0)+1
        self.sampled+=batch_size
        states,actions,rewards,next_states,dones,_=zip(*rows)
        return (torch.as_tensor(np.array(states),device=device),torch.as_tensor(actions,device=device),
                torch.as_tensor(rewards,dtype=torch.float32,device=device),torch.as_tensor(np.array(next_states),device=device),
                torch.as_tensor(dones,dtype=torch.float32,device=device))


class NeuralTD:
    def __init__(self,initial,config,world_id,seed):
        self.model=copy.deepcopy(initial);self.target=copy.deepcopy(initial).eval()
        self.target.requires_grad_(False)
        self.config=config;self.world_id=world_id;self.device=next(self.model.parameters()).device
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        self.replay=Replay(world_id,config['capacity'],seed+1)
        self.rng=np.random.default_rng(seed);self.steps=0;self.updates=0;self.seconds=0.;self.gpu_ms=0.
        self.last_td_error=0.;self.update_log=[]
        self.action_counts=np.zeros(3);self.pair_counts=np.zeros(9)
        self.counts=np.zeros(12);self.sums=np.zeros(12);self.squares=np.zeros(12)

    @torch.no_grad()
    def values(self,observation):
        return self.model(torch.as_tensor(observation.vector(),device=self.device)[None])[0].cpu().numpy()

    def act(self,observation):
        q=self.values(observation);c=self.config
        f=min(1.,self.steps/max(1,c['epsilon_decay_steps']))
        eps=c['epsilon_start']+f*(c['epsilon_end']-c['epsilon_start'])
        best=np.flatnonzero(np.isclose(q,q.max(),rtol=0,atol=1e-8));p=np.full(3,eps/3)
        p[best]+=(1-eps)/len(best)
        return int(self.rng.choice(3,p=p)),dict(action_probs=p.tolist(),q_values=q.tolist(),epsilon=eps)

    def observe(self,t,world_id):
        self.replay.add(t,world_id)
        self.steps+=1;self.action_counts[t.action]+=1
        key=t.action
        if t.observation.mode:
            pair=3*t.observation.previous_own+t.action;self.pair_counts[pair]+=1;key=3+pair
        self.counts[key]+=1;self.sums[key]+=t.reward;self.squares[key]+=t.reward*t.reward

    def update(self,count=1):
        if len(self.replay.data)<self.config['learning_starts']:return 0
        for _ in range(count):
            start=time.perf_counter()
            if self.device.type=='cuda':
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            s,a,r,n,d=self.replay.sample(self.config['batch_size'],self.device)
            y=double_dqn_targets(self.model,self.target,n,r,d,self.config['gamma'])
            pred=self.model(s).gather(1,a[:,None]).squeeze(1)
            loss=F.smooth_l1_loss(pred,y)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TD loss')
            self.optimizer.zero_grad(set_to_none=True);loss.backward()
            norm=nn.utils.clip_grad_norm_(self.model.parameters(),self.config['max_grad_norm'],error_if_nonfinite=True)
            self.optimizer.step();self.updates+=1
            with torch.no_grad():
                if self.config['target_interval']:
                    if self.updates%self.config['target_interval']==0:self.target.load_state_dict(self.model.state_dict())
                else:
                    for target,online in zip(self.target.parameters(),self.model.parameters()):
                        target.lerp_(online,self.config['target_tau'])
            self.last_td_error=float((y-pred).abs().mean().detach())
            gpu_ms=0.
            if self.device.type=='cuda':
                end.record();end.synchronize();gpu_ms=begin.elapsed_time(end);self.gpu_ms+=gpu_ms
            elapsed=time.perf_counter()-start;self.seconds+=elapsed
            self.update_log.append(dict(update=self.updates,td_loss=float(loss.detach()),td_error=self.last_td_error,
                gradient_norm=float(norm),target_mean=float(y.mean()),q_mean=float(pred.mean().detach()),
                replay_size=len(self.replay.data),seconds=elapsed,gpu_ms=gpu_ms))
        return count

    def diagnostics(self):
        return dict(td_updates=self.updates,sampled_transitions=self.replay.sampled,
            replay_insertions=self.replay.inserted,replay_size=len(self.replay.data),
            distinct_sampled_transitions=len(self.replay.visits),
            replay_ratio=self.replay.sampled/max(1,self.replay.inserted),
            optimizer_seconds=self.seconds,gpu_compute_ms=self.gpu_ms,
            last_td_error=self.last_td_error,mean_td_loss=float(np.mean([r['td_loss'] for r in self.update_log])) if self.update_log else None)

    def features(self,observation):
        mean=self.sums/np.maximum(self.counts,1)
        variance=np.maximum(0,self.squares/np.maximum(self.counts,1)-mean**2)
        # Fixed, observed-only transforms; no hidden label or horizon/oracle input.
        obs=observation.vector().copy();obs[-1]/=10.
        return np.concatenate((obs,self.values(observation)/10.,np.log1p(self.action_counts),
            np.log1p(self.pair_counts),mean/10.,np.sqrt(variance)/10.,[self.last_td_error/10.])).astype(np.float32)


FEATURE_SIZE=51  # 11 observation + 3 Q + 3 action + 9 pair counts + 12 means + 12 std + TD error


class MetaExplorer(nn.Module):
    """Feedforward policy: observed summary state, no recurrent context necessary.

    Compatible with the corrected PPO sequence interface. Encoder is shared by
    actor/critic, but there is deliberately no GRU. Hidden output is always None.
    """
    def __init__(self,hidden=64):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(FEATURE_SIZE,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.actors=nn.ModuleList([nn.Linear(hidden,3)]);self.critics=nn.ModuleList([nn.Linear(hidden,1)])

    def forward(self,x,hidden=None):
        h=self.encoder(x)
        return self.actors[0](h).unsqueeze(-2),self.critics[0](h),None


@dataclass
class FeatureObservation:
    features: np.ndarray
    def vector(self):return self.features.copy()


def episode(game,initial,td_config,B,K,strategy,seed,explorer=None,existing=None,online_k=True):
    """Exactly ONE B-step prefix per method; no candidate selection.

    Timing variants use exactly B*prefix_updates steps (learning_starts=1), plus
    K*k_updates if online. Batch-after-B postpones the same scheduled count.
    """
    from .training import HistoryCollector
    world_id=game._world.world_id  # ownership only, never a policy feature
    learner=NeuralTD(initial,td_config,world_id,seed+10000)
    if strategy=='meta':actor=Controller(explorer,seed)
    elif strategy=='existing':actor=Controller(existing,seed)
    else:actor=HistoryCollector(seed,strategy)
    with torch.no_grad():initial_weights=torch.cat([p.flatten().clone() for p in learner.model.parameters()])
    history=[];outer=[];pending=0
    for _ in range(B):
        obs=game.observation();feature=FeatureObservation(learner.features(obs))
        action,info=actor.act(feature if strategy=='meta' else obs)
        nxt,reward,done,b=game.step(action);t=Transition(obs,action,reward,nxt,done,b,info)
        history.append(t);learner.observe(t,world_id)
        if len(learner.replay.data)>=td_config['learning_starts']:
            if td_config['prefix_timing']=='online':learner.update(td_config['prefix_updates'])
            else:pending+=td_config['prefix_updates']
        if strategy=='meta':outer.append(Transition(feature,action,reward,FeatureObservation(learner.features(nxt)),done,b,info))
    if td_config['prefix_timing']=='batch':learner.update(pending)
    prefix_diagnostics=learner.diagnostics();post_B=game.snapshot()
    with torch.no_grad():B_weights=torch.cat([p.flatten().clone() for p in learner.model.parameters()])
    for _ in range(K):
        obs=game.observation();action,info=learner.act(obs)
        nxt,reward,done,b=game.step(action);t=Transition(obs,action,reward,nxt,done,b,info);history.append(t)
        if online_k:
            learner.observe(t,world_id);learner.update(td_config['k_updates'])
        else:learner.steps+=1  # same epsilon schedule, NO replay/optimizer mutation
    with torch.no_grad():delta=float((torch.cat([p.flatten() for p in learner.model.parameters()])-B_weights).norm())
    return history,outer,learner,dict(prefix=prefix_diagnostics,total=learner.diagnostics(),
        b_parameter_delta=float((B_weights-initial_weights).norm()),k_parameter_delta=delta,k_updates=learner.updates-prefix_diagnostics['td_updates'],post_B=post_B)


def explorer_targets(prefix,suffix,config):
    """Complete observed reward-to-go; MC = gamma1/GAE-lambda1, train only."""
    if not suffix or not suffix[-1].done:raise ValueError('Complete suffix required for outer credit')
    if config['training']['gamma']!=1.:raise ValueError('Main outer objective requires gamma=1')
    tail=sum(t.reward for t in suffix);returns=[]
    for t in reversed(prefix):
        tail=config['training']['exploration_lambda']*t.reward+tail;returns.append(tail)
    returns=np.asarray(returns[::-1],dtype=np.float32)
    return returns-np.array([t.info['value'] for t in prefix],dtype=np.float32),returns


def leave_one_out(episodes,group_ids):
    """Optional same-world sibling baseline, independent of this rollout's actions.

    Critic targets remain unnormalized MC. All siblings' interactions are charged.
    """
    result=[]
    for i,(h,a,r,prefix) in enumerate(episodes):
        others=[float(e[2][0]) for j,e in enumerate(episodes) if group_ids[j]==group_ids[i] and j!=i]
        if not others:raise ValueError('LOO needs >=2 independent rollouts per world in each batch')
        result.append((h,r-np.mean(others),r,prefix))
    return result
