# Terminal-reward proposal training feasibility v1

Primary condition20D/600. Frozen online operator pool, surrogate/ranking rules and
slot allocation; no calibration or residual. Initialize three proposal models
from complementary_proposal_v1 selected weights. Preserve architecture58,504
parameters. Train only these proposal parameters through full closed-loop search.

Reward U(F)-U(O), where U=g/(1+g),g=max(0,(initial_best-final_best)/initial_std).
Use antithetic ES with8 directions per update, Gaussian noise sigma=.02 in
normalized parameter coordinates. Layer scale=max(initial parameter RMS,.01),
frozen. theta=theta0+scale*phi,phi0=0. Gradient estimate
mean((Rplus-Rminus)*epsilon)/(2*.02). Common task/population/policy RNG for +/-/O.
O baseline cancels in the paired difference; it still defines reward/validation,
not an extra claimed variance reduction. Adam on phi lr=.01,gradient norm clip1;
no normalization by observed reward std. All58,504 parameters eligible.

Three seeds,30 updates maximum. Each update uses6 of36 configurations, rotating
through a fixed shuffled36 order every6 updates; instance=update, two populations.
Fresh terminal_v1_train tasks. Each update/seed has96 full rollouts of batches2
(8 directions x2 signs x6 tasks). Same six O batches shared across seeds.
No objective derivatives/teacher labels. This uses CPU full inference,16 workers;
GPU is not useful merely to optimize a58k vector after CPU rollouts.

Validation terminal_v1_validation: all36 configurations, one instance,4 populations,
at updates0,5,10,15,20,25,30. O shared and cached once. Select maximum mean final
reward, include initial checkpoint. Do not select by training reward or single-step
metrics. If all three seeds show no validation gain>1e-5 across two consecutive
checks after update10, stop. All-zero/nonfinite ES signal stops with diagnosis.
A paired-noise measurement and actual timing precede costly work; no hyperparameter
sweep if poor signals. Store centers, RNG roles, optimizer/resume state and costs.

Maximum training objective calls:30*(3*96*1200 +6*1200)=10,584,000.
Validation max:(3*7*36+36)*2400=1,900,800.
Contracts/timing are separate. No overlap with earlier diagnostic instances;
shared function families remain development evidence.

Freeze selected weights. Validation feasibility gate: selected versus O mean>=.005
and recipe-cluster95%lower>0; selected versus initial mean>=.001 and95%lower>0;
all three seed effects positive for both. If it fails STOP, no confirmation set.
If it passes automatically run terminal_v1_confirmation,36configs x2instances x4
populations, A(same standalone long-trained anchor), O(shared), Initial(current
proposal fusion) and Terminal(new fusion), three learned seeds:2880 trajectories,
1,728,000 true evaluations. Three primary contrasts Terminal-A/O/Initial;
Bonferroni98.3333% intervals, mean>=.005,lower>0,all seeds positive required.
No external/general necessity claim from shared-family confirmation. No further
training variant, larger-network trial, gate/residual or parameter sweep.
