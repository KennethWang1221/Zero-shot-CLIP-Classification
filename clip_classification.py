import os
import time
import numpy as np
import datetime
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.datasets import CLIPDataset
from utils.utils import * 
from model import *
import itertools

def main(**args):

    save_path = 'Train_E{}_B{}_'.format(args['epochs'], args['batch_size'])
    if args['time_stamp']:
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        save_path = save_path+timestamp
        save_path = os.path.join(args['opts_dir'],save_path) # ./train_opts/E10_B32_09_26_23_27

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    log_dir = os.path.join(save_path,'log_dir')
    ckpt_dir = os.path.join(save_path,'ckpt_dir')
    res_dir = os.path.join(save_path,'res_dir')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    
    results_csv_path = os.path.join(args['data_path'], "captions.txt")
    image_path = os.path.join(args['data_path'], "Images")
    print('results_csv_path:',results_csv_path)
    generate_captions(results_csv_path, save_path = "captions.csv")

    train_df, valid_df = split_data(args['captions_path'])

    tokenizer = DistilBertTokenizer.from_pretrained(args['text_tokenizer'])
    
    train_transform = create_transform(args['target_img_size'], transform_mode = 'train')
    valid_transform = create_transform(args['target_img_size'], transform_mode = 'validation')

    train_dataset = CLIPDataset(
        image_path,
        train_df["image"].values,
        train_df["caption"].values,
        tokenizer=tokenizer,
        transforms=train_transform,
        max_length = args['max_length']
    )

    val_dataset = CLIPDataset(
        image_path,
        valid_df["image"].values,
        valid_df["caption"].values,
        tokenizer=tokenizer,
        transforms=valid_transform,
        max_length = args['max_length']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=True
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        shuffle=False,
    )

    model = CLIPModel(args['temperature'], args['image_embedding'], args['text_embedding'], args['image_encoder_model_name'], args['text_encoder_model_name'], args['pretrained'], args['trainable'], args['projection_dim'], args['dropout']).to(args['device'])
    params = [
        {"params": model.image_encoder.parameters(), "lr": args['image_encoder_lr']},
        {"params": model.text_encoder.parameters(), "lr": args['text_encoder_lr']},
        {"params": itertools.chain(
            model.image_projection.parameters(), model.text_projection.parameters()
        ), "lr": args['head_lr'], "weight_decay": args['weight_decay']}
    ]
    optimizer = torch.optim.AdamW(params, weight_decay=0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args['patience'], factor=args['factor']
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(args['epochs']):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step, args['device'])
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader, args['device'])
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            best = os.path.join(ckpt_dir,'best.pt')
            torch.save(model.state_dict(), best)
            print("Saved Best Model!")
        
        lr_scheduler.step(valid_loss.avg)

    eval_model = CLIPModel(args['temperature'], args['image_embedding'], args['text_embedding'], args['image_encoder_model_name'], args['text_encoder_model_name'], args['pretrained'], args['trainable'], args['projection_dim'], args['dropout']).to(args['device'])
    eval_model.load_state_dict(torch.load(best, map_location=args['device']))
    eval_model.eval()

    image_embeddings = get_image_embeddings(valid_loader, eval_model, args['text_tokenizer'], args['device'])

    query_list = ["one dog sitting on the grass", "cars running on the road."]
    
    for i in range(len(query_list)):
        res_path = os.path.join(res_dir, f"{i+1:04d}.png")
        find_matches(eval_model, 
                    image_embeddings,
                    query=query_list[i],
                    image_path = image_path,
                    image_filenames=valid_df['image'].values,
                    text_tokenizer=args['text_tokenizer'],
                    device=args['device'],
                    n=9,
                    res_path = res_path
                    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train the VQE")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="Enable debug mode")
    parser.add_argument("--data_path", type=str, default="./flickr30k",
                        help="Path to the image directory")
    parser.add_argument("--captions_path", type=str, default=".",
                        help="Path to the captions directory")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--head_lr", type=float, default=1e-3,
                        help="Learning rate for the head")
    parser.add_argument("--image_encoder_lr", type=float, default=1e-4,
                        help="Learning rate for the image encoder")
    parser.add_argument("--text_encoder_lr", type=float, default=1e-5,
                        help="Learning rate for the text encoder")
    parser.add_argument("--weight_decay", type=float, default=1e-3,
                        help="Weight decay for the optimizer")
    parser.add_argument("--patience", type=int, default=1,
                        help="Patience for learning rate scheduler")
    parser.add_argument("--factor", type=float, default=0.8,
                        help="Factor for learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")

    parser.add_argument("--image_encoder_model_name", type=str, default='resnet50',
                        help="Model name for the image encoder")
    parser.add_argument("--image_embedding", type=int, default=2048,
                        help="Dimension of the image embedding")
    parser.add_argument("--text_encoder_model_name", type=str, default="distilbert-base-uncased",
                        help="Model name for the text encoder")
    parser.add_argument("--text_embedding", type=int, default=768,
                        help="Dimension of the text embedding")
    parser.add_argument("--text_tokenizer", type=str, default="distilbert-base-uncased",
                        help="Tokenizer for the text encoder")
    parser.add_argument("--max_length", type=int, default=200,
                        help="Maximum length for text tokenization")

    parser.add_argument("--pretrained", action='store_true', default=True,
                        help="Use pretrained models for both encoders")
    parser.add_argument("--trainable", action='store_true', default=True,
                        help="Make both encoders trainable")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature parameter for contrastive loss")

    parser.add_argument("--size", type=int, default=224,
                        help="Image size for input")
    parser.add_argument("--eval_examples", type=int, default=3,
                            help="Image size for input")

    parser.add_argument("--num_projection_layers", type=int, default=1,
                        help="Number of layers in the projection head")
    parser.add_argument("--projection_dim", type=int, default=256,
                        help="Dimension of the projection head")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate for the projection head")
    parser.add_argument("--mean", nargs='+', type=float,default=[0, 0, 0], 
                        help='value of mean for model')
    parser.add_argument("--scale", nargs='+', type=float,default=[255.], 
                        help='value of scale for model (mobilenetv2)')
    parser.add_argument("--target_img_size", nargs='+', type=int, default=[224,224,3], 
                        help='size of target image default=[256,256,3]')
    parser.add_argument("--opts_dir", type=str, default="./train_opts/", \
                        help='path of outputs files')
    parser.add_argument('--time_stamp', action='store_false', default=True,\
                        help='don\'t append date timestamp to folder' )
    print("\n### Training VQE model ###")
    print("> Parameters:")
    argspar = parser.parse_args()    
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main(**vars(argspar))