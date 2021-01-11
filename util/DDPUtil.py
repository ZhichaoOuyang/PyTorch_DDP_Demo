import random

import torch
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

device = None
USE_GPU = False
SEED = 2020
local_rank = -1
visible_device_count = 0


def is_master():
    global local_rank
    return local_rank == -1 or local_rank == 0


def master_print(info):
    global local_rank

    if local_rank == 0 or local_rank == -1:
        print(info)


def move_to_device(data):
    global device
    global USE_GPU

    if USE_GPU:
        return data.to(device, non_blocking=True)
    else:
        return data


def move_model_to_device(model):
    global device
    global USE_GPU

    if USE_GPU:
        model.to(device)


def use_multi_GPUs():
    global USE_GPU
    global visible_device_count

    return USE_GPU and visible_device_count > 1


def get_device():
    global device
    return device


def set_seed():
    """
        Make results reproduceable and give the same initialization to each node/gpu.
        https://github.com/pytorch/pytorch/issues/7068#issuecomment-484918113
    :param seed:
    """
    global SEED
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    master_print("[Setting] SEED: %d" % SEED)


def worker_init_fn(worker_id):
    global SEED
    np.random.seed(SEED)


def set_DDP_device(gpu_id_str):
    global device
    global USE_GPU
    global local_rank
    global visible_device_count

    local_rank = -1

    if gpu_id_str == "-1":  # Use CPU
        device = torch.device("cpu")
    else:

        visible_device_count = len(gpu_id_str.split(","))

        if visible_device_count > 1:  # Use multiple GPUs

            USE_GPU = True
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda:%d" % local_rank)

            assert dist.get_world_size() == visible_device_count, "visible_device_count %d must be the same as process num (--nproc_per_node) %d" % (visible_device_count, dist.get_world_size())

        else:
            if torch.cuda.is_available():   # Use single GPU
                USE_GPU = True
                device = torch.device("cuda")
            else:                           # Use CPU
                USE_GPU = False
                print("Cannot find GPU id(s) %s! Use CPU." % gpu_id_str)
                device = torch.device("cpu")
    # set random seed
    set_seed()
    if visible_device_count > 1:
        dist.barrier()

    print("[Setting] device: %s" % (device))

    if visible_device_count > 1:
        dist.barrier()

    return local_rank


def DDP_prepare(train_dataset, test_dataset, num_data_processes, global_batch_size, collate_fn, model):
    """
        Prepare data loaders for datasets. The core of this function is the setting of the DistributedSampler.
    :param train_dataset:
    :param test_dataset:
    :param num_data_processes:
    :param global_batch_size:
    :param collate_fn:
    :param model:
    :return:
    """
    global USE_GPU
    global local_rank
    global visible_device_count

    train_data_sampler = None
    test_data_sampler = None

    if USE_GPU and visible_device_count > 1:

        assert global_batch_size % visible_device_count == 0, "batch_size %d must be divisible by number of GPUs %d" % (global_batch_size, visible_device_count)

        local_batch_size = int(global_batch_size / visible_device_count)

        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[local_rank],
                                                          output_device=local_rank, find_unused_parameters=True)

        train_data_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
        test_data_sampler = DistributedSampler(dataset=test_dataset, shuffle=False)
        train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=local_batch_size,
                                          sampler=train_data_sampler,
                                          pin_memory=True, num_workers=num_data_processes,
                                          collate_fn=collate_fn, worker_init_fn=worker_init_fn)
        test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=local_batch_size,
                                         sampler=test_data_sampler,
                                         pin_memory=True, num_workers=num_data_processes,
                                         collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    else:
        # use CPU or single GPU
        train_dataset_loader = DataLoader(dataset=train_dataset, batch_size=global_batch_size,
                                          shuffle=True,
                                          pin_memory=True, num_workers=num_data_processes,
                                          collate_fn=collate_fn, worker_init_fn=worker_init_fn)

        test_dataset_loader = DataLoader(dataset=test_dataset, batch_size=global_batch_size,
                                         shuffle=False,
                                         pin_memory=True, num_workers=num_data_processes,
                                         collate_fn=collate_fn, worker_init_fn=worker_init_fn)

    return model, train_dataset_loader, test_dataset_loader, train_data_sampler, test_data_sampler