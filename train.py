import os
import torch
import numpy as np

from dataset import custom_collate_fn, custom_collate_fn_2, wdcnn_collate_fn
from torch.utils.data import DataLoader

class siamese_trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, lr, num_epochs, device):
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn_2)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.BCELoss()
        self.num_epochs = num_epochs
        self.device = device

        self.save_path = 'models'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.best_val_loss = float('inf')  # initialize with a large value


    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for (x1, x2), y in self.train_loader:
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x1, x2)
                loss = self.criterion(output.squeeze(1), y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                

            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for (x1, x2), y in self.val_loader:
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                    output = self.model(x1, x2)
                    val_loss += self.criterion(output.squeeze(1), y)
                    predicted = np.where(output.cpu().numpy().squeeze() > 0.5, 1, 0)
                    correct += np.sum(predicted == y.cpu().numpy())
                    total += len(y)

                val_loss /= len(self.val_loader)
                accuracy = 100.0 * correct / total


            if epoch % 10 == 0:
                print(correct, total, correct/total)
                print("Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Accuracy: {:.2f}%".format(
                    epoch+1, self.num_epochs, running_loss/len(self.train_loader), val_loss, accuracy))
            
            # Save the model if validation loss is the lowest seen so far
            if val_loss < self.best_val_loss:
                if epoch % 10 == 0:
                    print("Validation loss decreased from {:.4f} to {:.4f}. Saving model...".format(self.best_val_loss, val_loss))
                torch.save(self.model.state_dict(), self.save_path + '/weights-best-cwru-data.pth')
                self.best_val_loss = val_loss


class wdcnn_trainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, lr, num_epochs, device):
        self.model = model.to(device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=wdcnn_collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=wdcnn_collate_fn)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.device = device

        self.save_path = 'models'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.best_val_loss = float('inf')  # initialize with a large value


    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x, y in self.val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output = self.model(x)
                    val_loss += self.criterion(output, y)
                    _, predicted = torch.max(output.data, 1)
                    val_total += y.size(0)
                    val_correct += (predicted == y).sum().item()

                val_loss /= len(self.val_loader)
                val_accuracy = 100.0 * val_correct / val_total


            if epoch % 10 == 0:
                print("Epoch [{}/{}], Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.2f}%".format(
                    epoch+1, self.num_epochs, running_loss/len(self.train_loader), val_loss, val_accuracy))
            
            # Save the model if validation loss is the lowest seen so far
            if val_loss < self.best_val_loss:
                if epoch % 10 == 0:
                    print("Validation loss decreased from {:.4f} to {:.4f}. Saving model...".format(self.best_val_loss, val_loss))
                torch.save(self.model.state_dict(), self.save_path + '/weights-best-cwru-data-wcdnn.pth')
                self.best_val_loss = val_loss