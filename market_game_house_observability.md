# HELICS Market Game: House-Agent Observability Bedrock

## Assumptions

1. All honest houses share the same base demand profile.
2. A house observes only its own state plus market price/history.
3. The pricing rule and battery rules are known.
4. Some agents may be adversarial or volatility-seeking.

## Observability / Inference Legend

| Class | Confidence | Meaning |
|---|---:|---|
| Direct | 100% | Given directly to the house or stored from its own actions. |
| Rule-known | 95-100% | Known from game rules, code, or confirmed organizer assumptions. |
| Algebraic inference | 75-95% | Recoverable from price, timing, pricing function, and own action history. |
| Shared-demand inference | 65-90% | Depends on honest houses sharing the same demand profile. |
| Belief estimate | 35-75% | Useful internal model, but not guaranteed true. |
| Anomaly signal | 15-60% | Suggestive evidence of adversarial or unusual behavior, not proof. |
| Not identifiable | 0-15% | Cannot be recovered from legal observations alone. |

## Final Table

| Element | Class | Confidence | Description | Methodology | Why it matters |
|---|---:|---:|---|---|---|
| Current hour | Direct | 100% | Current simulation hour. | Read from `hour`. | Anchors demand lookup, planning, and history. |
| Current market price | Direct | 100% | Price faced by the house this hour. | Read from `price`. | Main signal for charge/discharge decisions. |
| Price history | Direct | 100% | Sequence of observed prices so far. | Use `price_history` or store locally. | Enables trend, volatility, and aggregate-load inference. |
| Own base demand profile | Direct | 100% | Your house's 24-hour physical demand. | Read from `demand`. | Defines your required load and battery scheduling problem. |
| Own current battery charge | Direct | 100% | Your state of charge at the current hour. | Read from `battery_charge`. | Determines feasible charge/discharge action. |
| Own action history | Direct | 100% | Your past submitted market demands. | Store every returned demand value. | Needed to separate your effect from the crowd's effect. |
| Own cumulative cost | Direct / computed | 100% | Total cost paid so far. | Sum hourly `price * own_purchase`. | Objective being minimized by a normal house. |
| Battery constraints | Rule-known | 100% | Battery capacity, charge limit, discharge limit, initial charge. | Use game rules/code constants. | Defines feasible actions for you and honest competitors. |
| Pricing function | Rule-known | 100% if rules unchanged | Mapping from aggregate average demand to next price. | Encode the market maker's price rule. | Enables inversion from price back to average load. |
| Price-signal lag | Rule-known | 100% | Price at hour `t` reflects aggregate demand from hour `t-1`. | Account for one-hour delay in inference. | Prevents wrong attribution of current price to current actions. |
| Number of houses | Rule/config-known | High if visible | Number of participating house agents. | Read config or use organizer-provided value. | Needed to estimate total load and your marginal impact. |
| Shared honest-house demand profile | Rule-known assumption | ~95-100% if confirmed | Honest agents have the same physical demand profile as you. | Treat your `demand` vector as the common honest baseline. | Makes aggregate battery-behavior inference much stronger. |
| Future shared base demand | Rule-known assumption | ~95-100% if confirmed | Future physical load for honest houses. | Look ahead in your own `demand` profile. | Lets agent prepare battery before high-demand hours. |
| Previous-hour average market purchase | Algebraic inference | High | Average submitted demand across all houses at hour `t-1`. | Invert observed `price[t]` through pricing function. | Core aggregate market-state estimate. |
| Previous-hour others-only average purchase | Algebraic inference | High if `N` known | Average purchase of all houses except yours. | `(N * avg_market_purchase - own_purchase) / (N - 1)`. | Separates self-caused from crowd-caused behavior. |
| Previous-hour price tier | Algebraic inference | High | Whether prior average load sat in low, moderate, steep, or extreme pricing region. | Map price to pricing bracket. | Helps avoid triggering expensive price cliffs. |
| Tier-boundary risk | Algebraic/model inference | Medium-high | Risk that a small extra purchase pushes the next price into a worse bracket. | Compare estimated average load to pricing thresholds. | Important for deciding whether charging now is worth it. |
| Price inversion ambiguity | Algebraic caveat | Medium-high | Some prices reveal exact-ish load; others only reveal a range. | Track whether price lies in flat/lossy region or invertible region. | Prevents overconfidence in average-load estimates. |
| Average above-base purchase | Shared-demand inference | High under shared demand | Whether the average house bought more than physical demand. | Compare inferred average purchase to shared base demand. | Indicates net charging by the crowd. |
| Average below-base purchase | Shared-demand inference | High under shared demand | Whether the average house bought less than physical demand. | Compare inferred average purchase to shared base demand. | Indicates net discharge, underbuying, or adversarial behavior. |
| Average crowd net battery movement | Shared-demand inference | Medium-high | Estimated average charge/discharge of the population. | `avg_purchase - shared_base_demand`, with battery-rate limits considered. | Lets agent model crowd energy reserves. |
| Average crowd battery belief | Belief estimate | Medium | Approximate state of charge of the average other house. | Integrate estimated crowd net battery movement over time; clamp to feasible battery range. | Predicts whether others can keep suppressing demand later. |
| Crowd battery feasibility | Belief / anomaly signal | Medium | Whether inferred aggregate behavior is plausible under battery constraints. | Check inferred average charge/discharge against capacity and rate limits. | Flags suspicious or nonphysical aggregate behavior. |
| Feasible honest-action envelope | Model estimate | Medium-high | Range of market purchases an honest battery agent could plausibly submit. | Use shared demand, battery limits, current/future prices, and rational scheduling assumptions. | Baseline for normal vs suspicious behavior. |
| Deviation from honest envelope | Anomaly signal | Medium | Amount by which observed aggregate behavior differs from plausible honest behavior. | Compare inferred average or others-only purchase against honest-action envelope. | Best general signal for adversarial or volatility-seeking agents. |
| Market volatility metric | Direct / computed | High | Magnitude of price jumps, variance, and tier crossings. | Track `abs(price[t] - price[t-1])`, rolling variance, and threshold crossings. | Directly relevant if malicious agents try to maximize volatility. |
| Self-induced vs other-induced price movement | Algebraic/model inference | Medium-high | Whether a price change was mostly caused by your action or by others. | Compare own previous purchase to inferred others-only average purchase. | Prevents your agent from misreading its own market impact as crowd behavior. |
| Crowd behavior class | Belief estimate | Medium | Broad class: passive, price-reactive, flattening, aggressive arbitrage, adversarial. | Pattern-match aggregate behavior against price, demand, and battery feasibility. | Supports adaptive strategy selection. |
| Future price pressure | Belief estimate | Medium | Expected risk of upcoming price spike or softening. | Combine future shared demand, crowd battery belief, current tier, and volatility. | Helps decide whether to save battery or spend it now. |
| Adversarial behavior likelihood | Anomaly signal | Low-medium early; improves over time | Probability that some agents are increasing volatility rather than minimizing cost. | Look for repeated deviations from feasible honest behavior, irrational charging/discharging, or volatility-amplifying moves. | Triggers robust or defensive strategy mode. |
| Aggregate distribution ambiguity | Structural limitation | Low identifiability | Same average can come from many distributions of house actions or batteries. | Recognize that price reveals aggregate average, not variance or individual values. | Prevents overfitting to a fake average-house story. |
| Individual attribution | Not identifiable | 0-15% | Exact action, battery, cost, strategy, or malicious identity of a specific house. | Not recoverable from legal price-only observations. | Do not design strategy around knowing who did what. |

## Compact Conceptual Stack

| Layer | What the house agent can do |
|---|---|
| 1. Local control | Use own demand, own battery, price, and hour to choose feasible market demand. |
| 2. Market reconstruction | Use price history and pricing function to infer previous aggregate average demand. |
| 3. Others-only inference | Subtract own contribution to estimate average behavior of the rest of the field. |
| 4. Shared-demand reasoning | Compare inferred average purchase against the common demand profile to estimate crowd charging/discharging. |
| 5. Belief-state modeling | Maintain a running estimate of average crowd battery posture and likely future pressure. |
| 6. Robust/adversarial detection | Look for behavior inconsistent with feasible honest battery scheduling or unusually volatility-amplifying patterns. |
| 7. Hard limits | Accept that individual identities, individual batteries, exact strategies, and true distributions are not legally observable. |

## Core Framing

A house agent is not blind, but it is **aggregate-observant rather than individually observant**. It can reconstruct a delayed average market state, compare that state against the shared demand profile, and maintain a belief over crowd battery behavior. What it cannot do is identify which specific house caused what.
