import argparse
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
import tqdm

from dataloader import get_examples_and_labels
from torch.utils.data import DataLoader, RandomSampler, WeightedRandomSampler, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_cosine_schedule_with_warmup
from tensorboardX import SummaryWriter
from tokenizers import BertWordPieceTokenizer
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('root')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model_dir', type=str, default=None, help='Directory of model to train')
    parser.add_argument('--dataset', type=str, default='mr', help='Dataset for training (e.g., \'yelp\')')
    parser.add_argument('--logging_steps', type=int, default=500, help='Log model after this many steps')
    parser.add_argument('--checkpoint_steps', type=int, default=5000, help='Save model after this many steps')
    parser.add_argument('--checkpoint_every_epoch', action='store_true', help='Save model checkpoint every epoch')
    parser.add_argument('--output_prefix', type=str, default='', help='Prefix for model saved output')
    parser.add_argument('--cased', action='store_true', default=False, help='If true, BERT is cased')
    parser.add_argument('--num_train_epochs', type=int, default=10, help='Total number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum length of a sequence')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for optimization')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Warmup proportion for scheduling')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Steps to accumulate gradients before optimizing')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed to set')

    args = parser.parse_args()
    if not args.model_dir:
        args.model_dir = 'bert-base-cased' if args.cased else 'bert-base-uncased'

    if args.output_prefix:
        args.output_prefix += '-'

    cased_str = '-' + ('cased' if args.cased else 'uncased')
    date_now = datetime.datetime.now().strftime("%Y-%m-%d")
    root_output_dir = 'outputs'
    args.output_dir = os.path.join(root_output_dir, f'{args.output_prefix}{args.dataset}{cased_str}-{date_now}')

    args.num_gpus = torch.cuda.device_count()
    set_seed(args.random_seed)
    return args


def get_class_weights(labels):
    """Compute class weights for imbalanced datasets."""
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def train(args):
    args = parse_args()
    make_directories(args.output_dir)

    tb_writer = SummaryWriter(args.output_dir)
    train_text, train_labels, eval_text, eval_labels = get_examples_and_labels(args.dataset)

    # Tokenization
    tokenizer = BertWordPieceTokenizer('bert-base-uncased-vocab.txt', lowercase=True)
    tokenizer.enable_padding(pad_to_multiple_of=args.max_seq_len)
    tokenizer.enable_truncation(max_length=args.max_seq_len)

    train_text_ids = [encoding.ids for encoding in tokenizer.encode_batch(train_text)]
    eval_text_ids = [encoding.ids for encoding in tokenizer.encode_batch(eval_text)]

    train_input_ids = torch.tensor(train_text_ids, dtype=torch.long)
    train_label_ids = torch.tensor(train_labels, dtype=torch.long)
    class_weights = get_class_weights(train_labels)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Model setup
    model = BertForSequenceClassification.from_pretrained(args.model_dir, num_labels=len(set(train_labels)))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = len(train_text_ids) // args.batch_size * args.num_train_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * args.warmup_proportion),
                                                num_training_steps=num_training_steps)

    # DataLoader
    train_data = TensorDataset(train_input_ids, train_label_ids)
    train_sampler = WeightedRandomSampler(weights=class_weights, num_samples=len(train_label_ids), replacement=True)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # Tracking accuracy
    epoch_accuracies = []

    model.train()
    global_step = 0
    for epoch in range(args.num_train_epochs):
        epoch_correct = 0
        epoch_total = 0
        for step, batch in enumerate(tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
            input_ids, labels = batch[0].to(device), batch[1].to(device)

            logits = model(input_ids)[0]
            loss = loss_function(logits, labels)
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar('loss', loss.item(), global_step)

            # Track accuracy
            predictions = torch.argmax(logits, dim=-1)
            epoch_correct += (predictions == labels).sum().item()
            epoch_total += labels.size(0)

        # Compute and save epoch accuracy
        epoch_accuracy = epoch_correct / epoch_total
        epoch_accuracies.append(epoch_accuracy)

        # Save model checkpoint
        model.save_pretrained(os.path.join(args.output_dir, f'checkpoint-epoch-{epoch + 1}'))

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_accuracies) + 1), [acc * 100 for acc in epoch_accuracies], marker='o', linestyle='-',
             label='Training Accuracy')
    plt.title('Epoch-wise Training Accuracy', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(range(1, len(epoch_accuracies) + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(args.output_dir, 'accuracy_plot_improved.png'))
    plt.show()

    tb_writer.close()
    logger.info("Training completed.")


if __name__ == '__main__':
    train(parse_args())
