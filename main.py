import torch
import argparse

from dataset import Dataset
from utils import compute_F1, compute_exact_match
from transformers import BertForQuestionAnswering
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from trainer import train, valid
from transformers import BertTokenizerFast
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

parser = argparse.ArgumentParser()

parser.add_argument('--patience' ,  type = int, default=3)
parser.add_argument('--batch_size' , type = int, default=8)
parser.add_argument('--max_epoch' ,  type = int, default=10)
parser.add_argument('--pretrained_model' , type = str,  help = 'pretrainned model')
parser.add_argument('--dataset_name' , required= True, type = str,  help = 'mrqa|squad|coqa')
parser.add_argument('--gpu_number' , type = int,  default = 0, help = 'which GPU will you use?')



args = parser.parse_args()

if __name__ =="__main__":

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    train_dataset = Dataset(args.dataset_name, tokenizer, "train")
    val_dataset = Dataset(args.dataset_name, tokenizer,  "validation") 
    model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
    device = torch.device(f'cuda:{args.gpu_number}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device) # change allocation of current GPU

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    dev_loader = DataLoader(val_dataset, args.batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)


    if args.pretrained_model:
        model.load_state_dict(torch.load(args.pretrained_model))

    model.to(device)
    penalty = 0
    max_F1 = 0
    for epoch in range(args.max_epoch):
        train(model, train_loader, optimizer, device)

        EM = 0
        F1 = 0 
        pred_texts, ans_texts = valid(model, dev_loader, device, tokenizer)
        for iter, (pred_text, ans_text) in enumerate(zip(pred_texts, ans_texts)):
            EM += compute_exact_match(pred_text, ans_text)
            F1 += compute_F1(pred_text, ans_text)
        
        print("EM=%.04f " % (EM/iter))
        print("F1 = %.04f " % (F1/iter))

        writer.add_scalar("EM/train", EM/iter, epoch)
        writer.add_scalar("F1/train", F1/iter, epoch)

        if (F1/iter) > max_F1:
            max_F1 = F1/iter
            penalty = 0
            torch.save(model.state_dict(), "model/qa_extraction.pt")
        else:
            penalty +=1
            if penalty>args.patience:
                print("early stopping")
                break
        writer.close()


