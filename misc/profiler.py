
import torch
import numpy as np

from whale_classifier.cnn0 import WhaleClassifier

model = WhaleClassifier(len(np.zeros(4251))).cuda()
inputs = torch.randn(32, 3, 224, 224).cuda()

# warmup
model(inputs)
model(inputs)
model(inputs)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
    with torch.profiler.record_function("model_inference"):
        model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))