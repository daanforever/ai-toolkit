import torch
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from toolkit.train_tools import apply_snr_weight, get_all_snr

sched = CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

def test_flowmatch_v_prediction(gamma):
  print(f'\ngamma={gamma:2.1f}')
  for ts in [200, 300, 400, 500, 600]:
    loss = torch.ones(1)
    t = torch.tensor([float(ts)])
    w_old = apply_snr_weight(loss, t, sched, gamma, prediction_type='v_prediction')
    w_new = apply_snr_weight(loss, t, sched, gamma, prediction_type='flow_match')
    snr = get_all_snr(sched, 'cpu')[int(ts)-1].item()
    print(f't={ts:3d} snr={snr:10.4f}  v_pred={w_old.item():.6f}  flow_match={w_new.item():.6f}  ratio={w_new.item()/w_old.item():.3f}')

test_flowmatch_v_prediction(gamma=1.0)
test_flowmatch_v_prediction(gamma=2.0)
test_flowmatch_v_prediction(gamma=3.0)
test_flowmatch_v_prediction(gamma=4.0)
test_flowmatch_v_prediction(gamma=5.0)
