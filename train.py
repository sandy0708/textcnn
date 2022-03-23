import os
import sys
sys.path.append('../../04_word_embedding')
import numpy as np
import torch
from dataloader import Dataloader
from model import TextCNN
import pdb
import time

class Trainer:
    def __init__(self, model_ver='CNN-static', 
                batch_size=50,
                learning_rate=0.1,
                epochs=20):
        ## device confiuration
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        torch.cuda.empty_cache()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using', self.device)

        self.model_ver = model_ver
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs


    def evaluate(self, model, val_dataloader, loss_fn):
        # Put the model into the evaluation mode. The dropout layers are disabled
        # during the test time.
        model.eval()

        # Tracking variables
        val_accuracy = []
        val_loss = []

        # For each batch in our validation set...
        for batch in val_dataloader:
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(self.device) for t in batch)

            # Compute logits
            with torch.no_grad():
                logits = model(b_input_ids)

            # Compute loss
            loss = loss_fn(logits, b_labels)
            val_loss.append(loss.item())

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)

        # Compute the average accuracy and loss over the validation set.
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy


    def train(self, pretrained_embedding=True,
                    freeze_embedding=True,
                    multichannel=False,):
        ## dataloader class
        dl = Dataloader(batch_size=self.batch_size)
        vocab_size = dl.vocab_size
        train_dataloader = dl.train_dataloader
        val_dataloader = dl.val_dataloader

        ## model initialize
        model_ver = self.model_ver
        if model_ver == 'CNN-rand':
            pretrained_embedding=False
        elif model_ver == 'CNN-static':
            pretrained_embedding=True
            freeze_embedding=True
        elif model_ver == 'CNN-non-static':
            pretrained_embedding=True
            freeze_embedding=False
        elif model_ver == 'CNN-multichannel':
            pretrained_embedding=True
            multichannel=True

        model = TextCNN(pretrained_embedding=pretrained_embedding,
                        freeze_embedding=freeze_embedding,
                        multichannel=multichannel,
                        num_classes=2, vocab_size=vocab_size)
        model.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer = torch.optim.Adadelta(model.parameters(), lr=self.learning_rate, rho=0.95)
        """Train the CNN model."""

        # Tracking best validation accuracy
        best_accuracy = 0

        # Start training loop
        print("Start training...\n")
        print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*60)

        epochs = self.epochs
        for epoch_i in range(epochs):
            # =======================================
            #               Training
            # =======================================

            # Tracking time and loss
            t0_epoch = time.time()
            total_loss = 0

            # Put the model into the training mode
            model.train()

            for step, batch in enumerate(train_dataloader):
                # Load batch to GPU
                b_input_ids, b_labels = tuple(t.to(self.device) for t in batch)

                # Zero out any previously calculated gradients
                model.zero_grad()

                # Perform a forward pass. This will return logits.
                logits = model(b_input_ids)

                # Compute loss and accumulate the loss values
                loss = loss_fn(logits, b_labels)
                total_loss += loss.item()

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Update parameters
                optimizer.step()

                # if (step+1) % 100 == 0:
                #     print(f"Epoch [{epoch_i}/{epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")


            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)

            # =======================================
            #               Evaluation
            # =======================================
            if val_dataloader is not None:
                # After the completion of each training epoch, measure the model's
                # performance on our validation set.
                val_loss, val_accuracy = self.evaluate(model, val_dataloader, loss_fn)

                # Track the best accuracy
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy

                # Print performance over the entire training data
                time_elapsed = time.time() - t0_epoch
                print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
                
        print("\n")
        print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

        ## training loop
        # epochs = 20
        # total_loss = 0
        # steps = 0
        # for epoch in range(1,epochs+1):
        #     for i, (data, target) in enumerate(train_dataloader):
        #         data = data.to(device)
        #         target = target.to(device) ## target은 1차원으로 되어있음 
        #         optimizer.zero_grad()

        #         output = model(data) ## (8,2) 소프트 맥스 형태
        #         loss = loss_fn(output, target)
        #         total_loss += loss.item()
        #         steps += 1
        #         # print('OUTPUT')
        #         # print(output)
        #         # print('TARGET')
        #         # print(target)
                
        #         # backward and optimize
        #         loss.backward()
        #         optimizer.step()
        #         torch.cuda.empty_cache()

        #         if (i+1) % 100 == 0:
        #             print(f"Epoch [{epoch}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
        #             print(f'Average train loss : {total_loss / steps}')
        # ## test 
        # model.eval()
        # test_accuracy = []
        # test_loss = []
        # for i, (data, target) in enumerate(test_dataloader):
        #     data = data.to(device)
        #     target = target.to(device) ## target은 1차원으로 되어있음 
        #     optimizer.zero_grad()

        #     with torch.no_grad():
        #         output = model(data) ## (8,2) 소프트 맥스 형태
            
        #     loss = loss_fn(output, target)
        #     test_loss.append(loss.item())

        #     preds = torch.argmax(output, dim=1).flatten()
        #     accuracy = (preds == target).cpu().numpy().mean() * 100
        #     test_accuracy.append(accuracy)

        # test_loss = np.mean(test_loss)
        # test_accuracy = np.mean(test_accuracy)

        # print('Average loss :', test_loss)
        # print('Average accuracy :', test_accuracy)