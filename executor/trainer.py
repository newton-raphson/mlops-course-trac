import torch 
from model.model import LogisticRegression
from torchvision import datasets, transforms
import wandb
import numpy as np

class Trainer:
    def __init__(self,config):
        self.config = config
        self.model = LogisticRegression(config.input_dim,config.output_dim,config.hidden_layers)

        print(self.model.hidden_layers)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train(self,train_loader):
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
        return loss.item()  
    def test(self, test_loader, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        logged_images = []  # List to store images and their predictions for wandb logging

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                # Save 5 images and predictions
                if batch_idx < 5:  # Save the first 5 images in the test set
                    for i in range(data.size(0)):  # Iterate over batch
                        if len(logged_images) >= 5:
                            break
                        img = data[i].cpu().numpy().squeeze()  # Remove channel dimension
                        pred_label = pred[i].item()
                        true_label = target[i].item()

                        img = wandb.Image(img, caption=f"Pred: {pred_label}, True: {true_label}")
                        logged_images.append(img)
       
        # Log images to wandb every 'log_interval' epochs
        if epoch % self.config.log_interval == 0 and logged_images:
            wandb.log({"test_predictions": logged_images}, step=epoch)
            


        # Compute average loss
        test_loss /= len(test_loader.dataset)

        return test_loss, correct / len(test_loader.dataset)
    def run(self):
        # Load and preprocess data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.batchsize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config.batchsize, shuffle=False)

        for epoch in range(self.config.epochs):
            train_loss = self.train(train_loader)
            test_loss,correct= self.test(test_loader,epoch)
            # save the model with wandb
            wandb.log({"epoch":epoch,"train_loss":train_loss,"test_loss":test_loss,"accuracy":correct})
            if epoch % self.config.log_interval == 0:
                checkpoint_path = f"model_epoch_{epoch}.pth"
                torch.save(self.model.state_dict(), checkpoint_path)

                # Log the checkpoint
                artifact = wandb.Artifact(name=f"model_epoch_{epoch}", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)
