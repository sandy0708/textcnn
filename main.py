import argparse
from pytorch.train import Trainer

parser = argparse.ArgumentParser(description='TextCNN implementation')
parser.add_argument('--model_ver', type=str, default='CNN-static', help= 'CNN-rand, CNN-static, CNN-non-static, CNN-multichannel')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--lrate', type=float, default=0.1)

args = parser.parse_args()

trainer = Trainer(model_ver=args.model_ver, 
                batch_size=args.batch_size,
                learning_rate=args.lrate,
                epochs=args.epoch)

trainer.train()