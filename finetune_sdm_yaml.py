# --------------------------------------------------------
# DisCo - Disentangled Control for Referring Human Dance Generation in Real World
# Licensed under The Apache-2.0 license License [see LICENSE for details]
# Tan Wang (TAN317@e.ntu.edu.sg)
# Work done during internship at Microsoft
# --------------------------------------------------------

from utils.wutils_ldm import *
from agent import Agent_LDM, WarmupLinearLR, WarmupLinearConstantLR
import os
import torch
from utils.lib import *
from utils.dist import dist_init
from dataset.tsv_dataset import make_data_sampler, make_batch_data_sampler
torch.multiprocessing.set_sharing_strategy('file_system')
from dataloader import omniDataLoader


def get_loader_info(args, size_batch, dataset):
    is_train = dataset.split == 'train'
    if is_train:
        images_per_gpu = min(
            size_batch * max(1, (args.max_video_len // dataset.max_video_len)),
            128)
        print(f'images_per_gpu: {images_per_gpu}')
        images_per_batch = images_per_gpu * args.world_size
        iter_per_ep = len(dataset) // images_per_batch
        print(f'iter_per_ep: {iter_per_ep}')
        if args.epochs == -1: # try to add iters into args
            assert args.ft_iters > 0
            num_iters = args.ft_iters
            args.epochs = (num_iters * images_per_batch) // len(dataset) + 1
        else:
            num_iters = iter_per_ep  * args.epochs
    else:
        images_per_gpu = size_batch * (
            args.max_video_len // dataset.max_video_len)
        images_per_batch = images_per_gpu * args.world_size
        iter_per_ep = None
        num_iters = None
    loader_info = (images_per_gpu, images_per_batch, iter_per_ep, num_iters)
    print(f'loader info in getter: {loader_info}', flush=True)
    return loader_info


def make_data_loader(
        args, size_batch, dataset, start_iter=0, loader_info=None):
    is_train = dataset.split == 'train'
    collate_fn = None #dataset.collate_batch
    is_distributed = args.distributed
    if is_train:
        shuffle = True
        start_iter = start_iter
    else:
        shuffle = False
        start_iter = 0
    if loader_info is None:
        loader_info = get_loader_info(args, size_batch, dataset)
    images_per_gpu, images_per_batch, iter_per_ep, num_iters = loader_info

    if hasattr(args, 'limited_samples'):
        limited_samples = args.limited_samples // args.local_size
    else:
        limited_samples = -1
    random_seed = args.seed
    sampler = make_data_sampler(
        dataset, shuffle, is_distributed, limited_samples=limited_samples,
        random_seed=random_seed)
    batch_sampler = make_batch_data_sampler(
        sampler, images_per_gpu, num_iters, start_iter
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, num_workers=args.num_workers, batch_sampler=batch_sampler,
        pin_memory=True, collate_fn=collate_fn
    )
    # for items in data_loader:
    #     print(f'first items: {items.keys()}, {len(items["img_key"]), len(items["input_text"]), items["label_imgs"].shape}', flush=True)
    #     break
    meta_info = (images_per_batch, iter_per_ep, num_iters)
    
    return data_loader, meta_info


def main_worker(args):
    """

    """
    #print(args)
    #print(args.cf)
    #exit()
    sorted_args = dict(sorted(args.items()))
    print(sorted_args)
    
    cf = import_filename(args.cf)
    #print(f'cf: {cf}')
    Net, inner_collect_fn = cf.Net, cf.inner_collect_fn
    #print(f'net: {Net}')
    

    dataset_cf = import_filename(args.dataset_cf)
    BaseDataset = dataset_cf.BaseDataset

    # args = update_args(parsed_args, args)

    # init models
    logger.info('Building models...')
    model = Net(args)
    #print(f"Args: {edict(vars(args))}")
    if args.do_train:
        logger.warning("Do training...")
        # Prepare Dataset.
        if getattr(args, 'refer_clip_preprocess', None):
            train_dataset = BaseDataset(args, args.train_yaml, split='train', preprocesser=model.feature_extractor)
            eval_dataset = BaseDataset(args, args.val_yaml, split='val', preprocesser=model.feature_extractor)
        else:
            print(f'refer clip preprocess is false', flush=True)
            train_dataset = BaseDataset(args, args.train_yaml, split='train')
            eval_dataset = BaseDataset(args, args.val_yaml, split='val')

        train_info = get_loader_info(args, args.local_train_batch_size, 
            train_dataset)
        print(f'train info: {train_info}', flush=True)
        _, images_per_batch, args.iter_per_ep, args.num_iters = train_info


        ######################################## Custom code to load NTU dataset #############################################################################
        train_dataloader_gen = omniDataLoader('train')
        eval_dataloader_gen = omniDataLoader('test')
        train_dataloader = DataLoader(train_dataloader_gen, batch_size=args.local_train_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        eval_dataloader = DataLoader(eval_dataloader_gen, batch_size=args.local_eval_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)

        images_per_batch = args.train_batch_size
        args.iter_per_ep = 283586 // images_per_batch
        args.num_iters = args.iter_per_ep * args.epochs
        print(f'new train info: {images_per_batch, args.iter_per_ep, args.num_iters}, {len(train_dataloader)}')
        ######################################## Custom code to load NTU dataset #############################################################################

        if args.eval_step <= 5.0:
            args.eval_step =  args.eval_step * args.iter_per_ep
        if args.save_step <= 5.0:
            args.save_step = args.save_step * args.iter_per_ep
        
        args.eval_step = int(max(10, args.eval_step))
        args.save_step = int(max(10, args.save_step))
        # if args.deepspeed:
        #     # from deepspeed.ops.adm import FusedAdam as Adam
        #     from deepspeed.ops.adam import DeepSpeedCPUAdam as Adam
        #     optimizer = Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=1e-3)
        # else:
        from torch.optim import AdamW 
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), weight_decay=args.decay)
        optimizer = getattr(model, 'optimizer', optimizer)

        if args.constant_lr:
            scheduler = WarmupLinearConstantLR(
                optimizer,
                max_iter=(args.num_iters // args.gradient_accumulate_steps) + 1,
                warmup_ratio=getattr(args, 'warmup_ratio', 0.05))
        else:
            scheduler = WarmupLinearLR(
                optimizer,
                max_iter=(
                    args.num_iters//args.gradient_accumulate_steps)+1,
                warmup_ratio=getattr(args, 'warmup_ratio', 0.05))
        scheduler = getattr(model, 'scheduler', scheduler)

        trainer = Agent_LDM(args, model, optimizer, scheduler)
        trainer.setup_model_for_training()
    

        # train_dataloader, train_info = make_data_loader(
        #     args, args.local_train_batch_size, 
        #     train_dataset, start_iter=trainer.global_step+1, loader_info=train_info)

        # eval_dataloader, eval_info = make_data_loader(
        #     args, args.local_eval_batch_size, 
        #     eval_dataset)


            
        logger.info(
            f"Video Length {train_dataset.size_frame}")
        logger.info(
            f"Total batch size {images_per_batch}")
        logger.info(
            f"Total training steps {args.num_iters}")
        logger.info(f"Starting train iter: {trainer.global_step+1}")
        logger.info(
            f"Training steps per epoch (accumulated) {args.iter_per_ep}")
        logger.info(
            f"Training dataloader length {len(train_dataloader)}")
        logger.info(
            f"Evaluation happens every {args.eval_step} steps")
        logger.info(
            f"Checkpoint saves every {args.save_step} steps")
        
        trainer.train_eval_by_iter(train_loader=train_dataloader, eval_loader=eval_dataloader,  inner_collect_fn=inner_collect_fn)
        
    if args.eval_visu:
        logger.warning("Do eval_visu...")
        eval_dataloader_gen = omniDataLoader('test')
        eval_dataloader = DataLoader(eval_dataloader_gen, batch_size=args.local_eval_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
        trainer = Agent_LDM(args=args, model=model)
        trainer.eval(eval_dataloader, inner_collect_fn=inner_collect_fn,
                        enc_dec_only='enc_dec_only' in args.eval_save_filename)

        

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser = add_custom_arguments(parser)
    # parsed_args = parser.parse_args()
    # main_worker(parsed_args)
    from utils.args import sharedArgs
    parsed_args = sharedArgs.parse_args()
    #print(f'parsed: {parsed_args}')
    main_worker(parsed_args)
