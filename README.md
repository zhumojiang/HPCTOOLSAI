Original code reference link: https://github.com/jia-zhuang/pytorch-multi-gpu-training

I have made the following improvements to the original author's code:
1. Introduced a random seed setting to ensure the reproducibility of results.
2. Increased the number of training epochs. Setting `epochs` to 50 means the entire training process will execute 50 complete cycles.
3. Added training time statistics.
4. Introduced a learning rate variable.

Common code:
- model.py
- data.py

Code 1 (baseline):
- single_gpu_train.py

Script:
- baseline.sh

Output:
- single_gpu_train.out

Runtime: 4m29s

Code 2 (DataParallel):
- data_parallel_train.py

Script:
- data_parallel_train.sh

Output:
- data_parallel_train.out

Runtime: 5m51s

Code 3 (LightningDDP):
- lightningDDP.py

Script:
- lightningDDP.sh

Output:
- lightningDDP.out

Runtime: 1m32s

Results Analysis:
The runtime of the three codes is 2 > 1 > 3. The reasons are as follows:

First code: Uses a single GPU without parallel processing.

Second code: Utilizes two GPUs with `nn.DataParallel` for data parallelism. However, this method may not efficiently use multiple GPUs, especially when communication overhead between each GPU is high.

Third code: Uses PyTorch Lightning, which encapsulates and optimizes the training process, providing many built-in functions and optimization strategies such as more efficient GPU utilization, better logging, and easier support for distributed training. It employs the `ddp` strategy for distributed training, leveraging multiple nodes and GPUs.
`ddp` is typically more efficient than `DataParallel` because it parallelizes processes and reduces communication overhead between GPUs.
