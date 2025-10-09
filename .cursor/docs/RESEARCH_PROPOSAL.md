## 1 Motivation & Introduction

Current code‑space search frameworks such as *AFLOW* and *MermaidFlow* achieve notable accuracy but are intrinsically *static*: their optimiser, their sampling policy, and their credit‑assignment scheme are frozen at design time. As a result they cannot profit from experience, and their fixed heuristics waste compute exploring low‑value branches.

We propose **Agent‑Workflow Contrastive RL**, a *self‑improving contrastive reinforcement‑learning system* that co‑optimises *both* the search agent and the workflows it produces. By factorising the policy into an **outer Agent** that proposes search operators and an **inner Workflow** policy that instantiates executable graphs, we can (i) *decouple* optimiser design from topology generation while training them jointly; (ii) drive exploration with **contrastive exemplars** **scored by an LLM‑as‑Judge** task performance or certain evaluation metrics **[Comment: Why not only the performance on sub-tasks? Or maybe during the training the workflows are not compliable, then workflow need to be judged by LLM?]**, and converts those scores into **dense pairwise rewards** that update every token in both the Agent and Workflow streams via a single Grouped‑Reinforcement Policy Optimisation (GRPO).

This tight RL loop enables the optimiser itself to *evolve*, automatically adjusts sampling towards promising regions, and furnishes fine‑grained credit that accelerates learning. We therefore expect to deliver higher‑quality workflows across code reasoning, mathematical reasoning, and other complex domains, surpassing static baselines on benchmarks such as HumanEval, MBPP, GSM8K and MATH. **[Comment: As highlight. The difference is that our method generates different workflow for each question of a domain, while ADAS generates one fixed workflow for a domain, c.f., MAS-Zero]**

## 2. Background

Large-language-model (LLM) agents have evolved from single-prompt chains into **multi-step, multi-agent “workflows”** that decompose a complex task into reusable sub-skills. **AFLOW** recasts workflow discovery as a search problem over *Python-encoded graphs*.  It exploits a Monte-Carlo Tree Search (MCTS) loop that iteratively mutates code snippets, executes the resulting workflow, and back-propagates the score to guide future expansions. **MermaidFlow** introduces a declarative, *graph-typed* representation written in the Mermaid language. Static typing and flow verification guarantee that every candidate is executable and semantically valid, which sharply reduces failure cases and aids interpretability.  
In AFLOW, MermaidFlow, etc., these policies are **hard‑coded**—they never update their parameters, sampling temperatures, or credit‑assignment rules from experience.  Once an algorithmic search policy (e.g. MCTS, heuristic list search, or evolution schedule) is chosen, it does not adapt from the successes and failures it encounters. 

More recent work such as *FlowReasoner* takes a step toward adaptability, training a meta-agent with reinforcement learning to assemble bespoke workflows per query. However, it still optimises **the workflow alone** rather than ****the search policy that generates it,  and it does not push reward back to every generation token. **However, it still optimises *only the workflow* itself; the outer search strategy is not a learnable object, and rewards are injected only at the end of generation rather than at every decision token.** As a result, none of AFLOW, MermaidFlow, or FlowReasoner **separate the optimisation of (i) the agent that chooses search operators from (ii) the workflow generator that realises those operators**, leaving valuable gradient signal untapped.

Recent successes such as CUDA-L1 show that **contrastive RL** can outperform evolutionary LLM search on code-optimization tasks, motivating its adoption here.

## 3. Problem Statement

We seek a single parameter-shared policy $π_\theta$ that, for any query $q$, **jointly** emits

- an **outer search-agent trace** $y_{\text{agt}}$, and
- an **inner workflow** $y_{\text{wfl}}$ that realises those decisions,

while receiving dense, **token-level** **[Comment: Good point, but consider during refinement. Many recent works on step level (e.g., one agent generation corresponds to one reasoning step), e.g., GSPO. Could add as back-up or refined algorithm]** credit for both.  The policy factorises as

$π_\theta(y_{\text{agt}},y_{\text{wfl}}\mid q)=π^A_\theta(y_{\text{agt}}\mid q)\;
π^W_\theta(y_{\text{wfl}}\mid q,y_{\text{agt}})$ ,

and is trained to maximise

$\max_\theta \; \mathbb E_{q, y_{\text{agt}}, y_{\text{wfl}}\sim π_\theta}\!\left[r(q,y_{\text{agt}},y_{\text{wfl}})\right],$

where $r$ rewards both solution quality and **search efficiency [Comment: Good point! How to define the efficiency of the workflow, e.g., graph complexity or agent representation complexity.]**.  A unified, token-wise update lets the agent strategy and workflow topology **co-evolve**, enabling higher accuracy than static baselines on code- and mathematics- reasoning benchmarks.

## 4 Proposed Method

### 4.1 Symbol Representation

| Symbol | Meaning |
| --- | --- |
| $q$ | Natural‑language **query / task description** |
| $y_A\!\in\!\mathcal Y_A$ | Token sequence of **Agent operators** (outer trace) |
| $y_W\!\in\!\mathcal Y_W$ | Token sequence of **Workflow graph** (inner topology) |
| $\theta$ | **Shared parameters** of one autoregressive LLM |
| $\pi_\theta^A(y_A\!\mid q)$ | **Outer policy** over agent traces |
| $\pi_\theta^W(y_W\!\mid q,y_A)$ | **Inner policy** over workflows conditioned on the agent |
| $N_A$ | # new agent candidates drawn per query |
| $N_W$ | # new workflows drawn per agent |
| $r_{ij}\in\mathbb R$ | Raw **quality score** from LLM‑as-a-Judge for agent $i$, workflow $j$ |
| $r_i = \frac1M\sum_{j} r_{ij}$ | **Agent‑level reward** (mean of its workflows) |
| $\bar r,\;\sigma$ | Global mean & std‑dev of all $r_{ij}$ in the batch |
| $A_A^{(i)},$  $A_W^{(i,j)}$ | **Standardised advantages**  |
| $\rho_t$ | Importance ratio $\pi_\theta/\pi_{\text{old}}$ for a token |
| $\varepsilon$ | PPO clip parameter |
| $\beta$ | KL‑penalty coefficient |
| $B^A_{F/M/S}$ | *Agent‑bucket, $k=Fast/Medium/Slow$* |
| $B^{W,(i)}_\ell$ | *Workflow‑bucket* $\ell=Fast/Medium/Slow$ inside agent $i$  |
| $\tilde q^A_k,\;\tilde q^{W,(i)}_\ell$ | Mean scores of the respective buckets **[Comment: What is upper index (i)]** predefined upper index **** |
| $\mathcal D_A$ | Replay buffer storing agents |
| $\mathcal D_W[i]$ | Replay buffer storing workflows *for agent $i$* |
| $\tau_A,\;\tau_W$ | Softmax temperatures (agent / workflow) |
| $\varepsilon,\;\beta$ | PPO clip & KL coefficients |
| $ε_{adv}$ | **Denominator jitter** for $σ$ |

MermaidFlow, and quality score from LLM-as-a-Judge $s(·)$

### 4.2 Hierarchical policy factorisation

$\;
\pi_\theta(y_A,\,y_W\mid q)\;=\;\pi_\theta^A(y_A\mid q)\;\cdot\;\pi_\theta^W(y_W\mid q,y_A)
\;$

*Implementation trick*: first sample $y_A,$ then append it to the prompt and sample $y_W$. This keeps memory constant while yielding log‑probs for both streams separately.

### 4.3 **Two‑level Contrastive‑RL exploration**

1. calculate the dividing point 
    1. ***Agents:***  $T_A^{F}=split(s(A_i),2/3),\;T_A^{S}=split(s(A_i),1/3)$
    2. ***Workflows, for each agent $i$:***  $T^{F}_{W,(i)}=split(s(W_{ij}),2/3),\;T^{S}_{W,(i)}=split(s(W_{ij}),1/3)$
2. **Assign indices.**
    
    $B^A_F=\{i\mid A_i\ge T_A^{F}\},\;
    B^A_M=\{i\mid T_A^{S}<A_i<T_A^{F}\},\;
    B^A_S=\{i\mid A_i\le T_A^{S}\},$
    

and analogously for $B^{W,(i)}_{\cdot}$.

1. **Bucket‑level sampling weights**

we soft‑weight the *three* buckets with a centred softmax (temperatures $\tau_{A},\tau_{W}$).

$P^A_t(B^A_\kappa)=\frac{\exp\!\bigl[(\tilde q^A_\kappa-\mu^A_t)/\tau_A\bigr]}
                       {\sum_{\lambda\in\{F,M,S\}}\exp\!\bigl[(\tilde q^A_\lambda-\mu^A_t)/\tau_A\bigr]},
\quad
\kappa\in\{F,M,S\},$ where $\tilde q^A_\kappa=\frac1{|B^A_\kappa|}\sum_{d\in B^A_\kappa} s(A_d)$

The same formula (replace superscript $A$ by $W,(i)$ is applied inside every agent.

1. **Contrastive Exemplar selection rule**
    1. **Agent stage.** Draw **two distinct buckets** $\kappa_1\neq\kappa_2$ according to $P^A_t(B^A_\kappa)$, then sample **one agent** from each: $y_A^{(1)}\sim\text{Unif}(B^A_{\kappa_1}),\;y_A^{(2)}\sim\text{Unif}(B^A_{\kappa_2})$
    2. **Workflow stage.** repeat inside every selected agent with its own $P^{W,(i)}_t$.
2. **new candidate generation: Embed those exemplars + their scores into the prompt** that conditions the next generation pass; for workflow, the generation prompt also need to involve the current agent list.
    1. **Outer sampling:**  $y_A^{(i)}\sim\pi_\theta^A(\,\cdot\mid q),\; i=1..N_A$
    2. **Inner sampling:** $y_W^{(i,j)}\sim\pi_\theta^W(\,\cdot\mid q,y_A^{(i)}),\; j=1..N_W$
    
    step 1: Given two Agent lists (from different agent buckets) + scores + prompt → new Agent list candidates
    
    step2：Given two Workflows under the current agent list (from different workflow buckets) + score + prompt → new workflow candidates under the current agent list
    
3. **reward scoring:** $r_{ij}= s(\,y_W^{(i,j)})$
4. **Reward aggregation**: $r_i=\tfrac1{N_W}\sum_{j} r_{ij}$ 
5. **Advantage computation:**  $A^A_i=\frac{r_i-\bar r}{\sigma+\varepsilon_{adv}},\quad
A^W_{i,j}= \frac{r_{ij}-\bar r}{\sigma+\varepsilon_{adv}}$

### 4.3 GRPO update objective

1. **Per-token importance ratios**:
    1. ***Agent tokens**:*  $\rho^{A}_{i,t}\;=\;
    \frac{\;\pi_{\theta}\!\bigl(y^{(i)}_{A,t}\;\bigm|\;q,\;y^{(i)}_{A,\;<t}\bigr)}
         {\pi_{\text{old}}\!\bigl(y^{(i)}_{A,t}\;\bigm|\;q,\;y^{(i)}_{A,\;<t}\bigr)}$
    2. ***Workflow tokens**:* $\rho^{W}_{i,j,t}\;=\;
    \frac{\;\pi_{\theta}\!\bigl(y^{(i,j)}_{W,t}\;\bigm|\;q,\;y^{(i)}_{A},\;y^{(i,j)}_{W,\;<t}\bigr)}
         {\pi_{\text{old}}\!\bigl(y^{(i,j)}_{W,t}\;\bigm|\;q,\;y^{(i)}_{A},\;y^{(i,j)}_{W,\;<t}\bigr)}$

where

- $q$ = task prompt,
- $y^{(i)}_{A}\!=y^{(i)}_{A,1:T_A^{(i)}}$ is the $i^{\text{th}}$ sampled **agent** (length $T_A^{(i)}$),
- $y^{(i,j)}_{W}\!=y^{(i,j)}_{W,1:T_W^{(i,j)}}$ is the $j^{\text{th}}$ **workflow** under agent $i$ (length $T_W^{(i,j)}$).
1. **GRPO Loss**:$\begin{aligned} \mathcal L_{GRPO}(\theta)= -\Bigg[
&\frac1{N_A}\sum_{i=1}^{N_A}\frac1{T_A^{(i)}}\sum_{t\in y_A^{(i)}}\min\!\bigl(\rho_t {A_i^A},\ \text{clip}(\rho^{A}_{i,t},1-\varepsilon,1+\varepsilon){A_i^A}\bigr) \\
&+\frac1{N_AN_W}\sum_{i=1}^{N_A}\sum_{j=1}^{N_W}\frac1{T_W^{(i,j)}}\sum_{t\in y_W^{(i,j)}}\min\!\bigl(\rho_t {A_{i,j}^W},\ \text{clip}(\rho^{W}_{i,j,t},1-\varepsilon,1+\varepsilon){A_{i,j}^W}\bigr)\Bigg] \\
&\;+\;\beta\,\text{KL}\bigl[\pi_\theta \,\Vert\, \pi_{\text{ref}}\bigr]
\end{aligned}$

*Key property*: **all tokens that belong to the same agent share $A_i^{A}$**, while tokens in a specific workflow share $A_{i,j}^W$. This “grouping” yields lower‑variance gradients than per‑token returns but is more informative than episode‑level rewards.

1. **Parameter update:**  $\theta \;\leftarrow\; \theta - \eta\,\nabla_{\theta}\mathcal L_{\text{GRPO}}(\theta)$

### 4.4 **buffers Update:**

1. **Agent buffer** $D_A\leftarrow\mathcal D_A\cup\{y_A^{(i)}\mid r_i\ge r^A_{\min}\}$
2. **Workflow buffer**  $D_W[i]\leftarrow\mathcal D_W[i]\cup\{y_W^{(i,j)}\mid r_{ij}\ge r^W_{\min}\}$

The two buffers are **independent**: workflows are stored *keyed by the agent that produced them.*

## 5. Pseudo-Code

```python
# --------------------------- 1. Optimisation step ---------------------------
for step in range(T_total):
    batch = sample_queries(BATCH_SIZE)          # q₁ … q_B
    traj = []                                   # store rollouts for gradient

    # 1‑A —— iterate over queries ------------------------------------------------
    for q in batch:

        # ---------- (a) Agent exemplar selection ----------
        B_F, B_M, B_S = partition_fixed_percent(buffer_A_all_scores(), 
                                                pA_F,M,S)
        E_A = pick_two_distinct_exemplars({F:B_F, M:B_M, S:B_S},
                                          K=K_A, weights=softmax_means(tau_A))

        # ---------- (b) Build outer prompt & sample agents ----------
        prompt_A = build_agent_prompt(q, E_A)
        agents   = sample_from_policy(pi_A, prompt_A, NA)  # y_A^(1..NA)

        # ---------- (c) Workflow stage per agent ----------
        for i, yA in enumerate(agents):

            # exemplar pool for this agent
            if yA.id not in buffer_W:     # init per‑agent workflow buffer
                buffer_W[yA.id] = {F:[], M:[], S:[]}

            Bf, Bm, Bs = partition_fixed_percent(
                             workflow_scores(buffer_W[yA.id]),
                             pW_F,M,S)
            E_W = pick_two_distinct_exemplars(
                      {F:Bf, M:Bm, S:Bs},
                      K=K_W,
                      weights=softmax_means_agent(tau_W, yA.id))

            prompt_W = build_workflow_prompt(q, yA, E_W)
            workflows = sample_from_policy(pi_W, prompt_W, NW)  # y_W^(i,1..NW)

            # ---------- (d) Execute & score ----------
            for j, yW in enumerate(workflows):
                result = run_workflow(yA, yW)                 # python / mermaid exec
                r_ij   = judge_score(q, yA, yW, result)       # LLM‑as‑Judge
                # store logprobs at rollout time
                logp_A = yA.logprobs      # tokenwise logprobs from pi_old
                logp_W = yW.logprobs
                traj.append((q, yA, yW, r_ij))

    # 1‑B —— compute statistics --------------------------------------------------
    r_all   = [r for (_,_,_,r) in traj]
    r_mean  = mean(r_all)
    r_std   = stdev(r_all) + eps_adv

    # build per‑agent and per‑workflow advantages
    A_A, A_W = build_advantages(traj, r_mean, r_std)

    # 1‑C —— single backward pass (GRPO) -----------------------------------------
    loss = 0
    for (q, yA, yW, r) in traj:
        loss += grpo_tokenwise_loss(
            q, yA, yW,
            A_A[yA.id], A_W[(yA.id, yW.id)],
            eps_clip,
        )

    loss += beta_init * kl_divergence(pi_theta, pi_ref)
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

    # 1‑D —— Update buckets & buffers -------------------------------------------
    update_agent_buckets(buffer_A, traj, pA_F,M,S)
    update_workflow_buckets(buffer_W, traj, pW_F,M,S)

    # 1‑E —— optional: anneal τ, adjust β, prune buffers -------------------------
    anneal_temperatures(step)
    beta_init = adjust_beta(beta_init, kl_current)

# -------------------------- 2. Inference path ---------------------------------
def answer_query(q):
    # same exemplar‑pair construction but without back‑prop
    prompt_A, agents = sample_best_agents(q)
    yA_best = select_top_by_score(agents)
    prompt_W, workflows = sample_best_workflows(q, yA_best)
    yW_best = select_top_by_score(workflows)
    return execute_and_return(yA_best, yW_best)

```

---