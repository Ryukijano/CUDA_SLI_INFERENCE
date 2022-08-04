{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-8e2d348b5b75>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m     37\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     38\u001b[0m \u001b[1;32mif\u001b[0m \u001b[0m__name__\u001b[0m \u001b[1;33m==\u001b[0m \u001b[1;34m\"__main__\"\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 39\u001b[1;33m     \u001b[0mdemo_basic\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m<ipython-input-1-8e2d348b5b75>\u001b[0m in \u001b[0;36mdemo_basic\u001b[1;34m()\u001b[0m\n\u001b[0;32m     18\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     19\u001b[0m \u001b[1;32mdef\u001b[0m \u001b[0mdemo_basic\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 20\u001b[1;33m     \u001b[0mdist\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0minit_process_group\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"nccl\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     21\u001b[0m     \u001b[0mrank\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdist\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget_rank\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     22\u001b[0m     \u001b[0mprint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34mf\"Start running basic DDP example on rank {rank}.\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\torch\\distributed\\distributed_c10d.py\u001b[0m in \u001b[0;36minit_process_group\u001b[1;34m(backend, init_method, timeout, world_size, rank, store, group_name, pg_options)\u001b[0m\n\u001b[0;32m    593\u001b[0m                 \u001b[0minit_method\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrank\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mworld_size\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtimeout\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    594\u001b[0m             )\n\u001b[1;32m--> 595\u001b[1;33m             \u001b[0mstore\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mrank\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mworld_size\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mnext\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mrendezvous_iterator\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    596\u001b[0m             \u001b[0mstore\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset_timeout\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtimeout\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    597\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\torch\\distributed\\rendezvous.py\u001b[0m in \u001b[0;36m_env_rendezvous_handler\u001b[1;34m(url, timeout, **kwargs)\u001b[0m\n\u001b[0;32m    245\u001b[0m         \u001b[0mrank\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mquery_dict\u001b[0m\u001b[1;33m[\u001b[0m\u001b[1;34m\"rank\"\u001b[0m\u001b[1;33m]\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    246\u001b[0m     \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 247\u001b[1;33m         \u001b[0mrank\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mint\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m_get_env_or_raise\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m\"RANK\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    248\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    249\u001b[0m     \u001b[1;32mif\u001b[0m \u001b[1;34m\"world_size\"\u001b[0m \u001b[1;32min\u001b[0m \u001b[0mquery_dict\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\torch\\distributed\\rendezvous.py\u001b[0m in \u001b[0;36m_get_env_or_raise\u001b[1;34m(env_var)\u001b[0m\n\u001b[0;32m    230\u001b[0m         \u001b[0menv_val\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mos\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0menviron\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mget\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0menv_var\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    231\u001b[0m         \u001b[1;32mif\u001b[0m \u001b[1;32mnot\u001b[0m \u001b[0menv_val\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 232\u001b[1;33m             \u001b[1;32mraise\u001b[0m \u001b[0m_env_error\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0menv_var\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    233\u001b[0m         \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    234\u001b[0m             \u001b[1;32mreturn\u001b[0m \u001b[0menv_val\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: Error initializing torch.distributed using env:// rendezvous: environment variable RANK expected, but not set"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.distributed as dist\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "\n",
    "from torch.nn.parallel import DistributedDataParallel as DDP\n",
    "\n",
    "class ToyModel(nn.Module):\n",
    "    def __init__(self):\n",
    "        super(ToyModel, self).__init__()\n",
    "        self.net1 = nn.Linear(10, 10)\n",
    "        self.relu = nn.ReLU()\n",
    "        self.net2 = nn.Linear(10, 5)\n",
    "\n",
    "    def forward(self, x):\n",
    "        return self.net2(self.relu(self.net1(x)))\n",
    "\n",
    "\n",
    "def demo_basic():\n",
    "    dist.init_process_group(\"nccl\")\n",
    "    rank = dist.get_rank()\n",
    "    print(f\"Start running basic DDP example on rank {rank}.\")\n",
    "\n",
    "    # create model and move it to GPU with id rank\n",
    "    device_id = rank % torch.cuda.device_count()\n",
    "    model = ToyModel().to(device_id)\n",
    "    ddp_model = DDP(model, device_ids=[device_id])\n",
    "\n",
    "    loss_fn = nn.MSELoss()\n",
    "    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)\n",
    "\n",
    "    optimizer.zero_grad()\n",
    "    outputs = ddp_model(torch.randn(20, 10))\n",
    "    labels = torch.randn(20, 5).to(device_id)\n",
    "    loss_fn(outputs, labels).backward()\n",
    "    optimizer.step()\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    demo_basic()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
