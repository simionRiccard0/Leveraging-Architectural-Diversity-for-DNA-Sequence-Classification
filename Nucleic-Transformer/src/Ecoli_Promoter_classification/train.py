import os
import torch
import torch.nn as nn
import time
import argparse

from Functions import *
from Dataset import *
from Network import *
from LrScheduler import *
import Metrics
from Logger import CSVLogger


# =========================
# AMP (PyTorch modern)
# =========================
from torch.cuda.amp import autocast, GradScaler


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--path', type=str, default='v9d3.csv')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--ntoken', type=int, default=4)
    parser.add_argument('--nclass', type=int, default=2)
    parser.add_argument('--ninp', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nhid', type=int, default=2048)
    parser.add_argument('--nlayers', type=int, default=6)

    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=.1)

    parser.add_argument('--warmup_steps', type=int, default=3200)
    parser.add_argument('--lr_scale', type=float, default=0.1)

    parser.add_argument('--nmute', type=int, default=18)
    parser.add_argument('--kmers', type=int, nargs='+', default=[2,3,4,5,6])

    parser.add_argument('--kmer_aggregation', dest='kmer_aggregation', action='store_true')
    parser.add_argument('--no_kmer_aggregation', dest='kmer_aggregation', action='store_false')
    parser.set_defaults(kmer_aggregation=True)

    parser.add_argument('--nfolds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)

    return parser.parse_args()


def train_fold():

    opts = get_args()

    seed_everything(20)

    os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = pd.read_csv(opts.path)

    sequences = np.asarray(df.seqs)
    labels = np.asarray(df.labels)

    train_indices, val_indices = iter_split(sequences, labels, opts.fold)

    dataset = PromoterDataset(sequences[train_indices], labels[train_indices])
    val_dataset = PromoterDataset(sequences[val_indices], labels[val_indices])

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=opts.batch_size*2, shuffle=False)

    checkpoints_folder = f'checkpoints_fold{opts.fold}'
    csv_file = f'log_fold{opts.fold}.csv'

    columns = ['epoch','train_loss','train_acc','val_loss','val_acc','val_sens','val_spec']
    logger = CSVLogger(columns, csv_file)

    model = NucleicTransformer(
        opts.ntoken, opts.nclass, opts.ninp, opts.nhead,
        opts.nhid, opts.nlayers, opts.kmer_aggregation,
        kmers=opts.kmers, dropout=opts.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    lr_schedule = lr_AIAYN(optimizer, opts.ninp, opts.warmup_steps, opts.lr_scale)

    # =========================
    # AMP SCALER (IMPORTANT)
    # =========================
    scaler = GradScaler()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Training fold {opts.fold}/{opts.nfolds}")

    for epoch in range(opts.epochs):

        model.train()
        t0 = time.time()

        total_loss = 0
        train_preds = []
        ground_truths = []

        total_steps = len(dataloader)

        for step, data in enumerate(dataloader):

            lr = lr_schedule.step()

            src = data['data'].long()
            labels_batch = data['labels'].to(device).long()

            mutated = mutate_dna_sequence(src).to(device).long()

            optimizer.zero_grad()

            # =========================
            # AMP forward pass
            # =========================
            with autocast():
                output = model(mutated)
                loss = torch.mean(criterion(output, labels_batch))

            # backward scaled
            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            preds = torch.argmax(output, dim=1).detach().cpu().numpy()
            train_preds.append(preds)
            ground_truths.append(labels_batch.cpu().numpy())

            print(
                f"Epoch [{epoch+1}/{opts.epochs}] "
                f"Step [{step+1}/{total_steps}] "
                f"Loss: {total_loss/(step+1):.4f} "
                f"LR: {lr:.6f} "
                f"Time: {time.time()-t0:.1f}",
                end="\r",
                flush=True
            )

        print()

        train_preds = np.concatenate(train_preds)
        ground_truths = np.concatenate(ground_truths)

        train_acc = Metrics.accuracy(train_preds, ground_truths)
        train_loss = total_loss / len(dataloader)

        val_loss, val_acc, val_sens, val_spec = validate(
            model, device, val_dataloader,
            batch_size=opts.batch_size*2
        )

        print(f"Epoch {epoch+1} train acc: {train_acc}")

        logger.log([
            epoch+1,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_sens,
            val_spec
        ])

        if (epoch+1) % opts.save_freq == 0:
            save_weights(model, optimizer, epoch, checkpoints_folder)

    # get_best_weights_from_fold(opts.fold)


if __name__ == "__main__":
    train_fold()
