import time
import server_simulator as sim
from pipeline import PredictionPipeline

sim.start()
p = PredictionPipeline()
p.start()

print('Waiting 90s for warmup...')
time.sleep(90)

preds = p.get_predictions()
for sid, pred in preds.items():
    name = pred['server_name']
    curr = pred['current_cpu']
    pr = pred['predicted_cpu']
    model = pred['model_used']
    print(name, 'CURR=', curr, 'PRED=', pr, 'MODEL=', model)

