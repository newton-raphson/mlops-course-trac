from configs.config import Configuration
from executor.trainer import Trainer
import sys
import wandb
import os
from dotenv import load_dotenv
if __name__ == '__main__':
    # pass the config file path to the function
    config_file_path = sys.argv[1]
    print("Running with config file: ",config_file_path)

    # Load the .env file
    load_dotenv()

    wandb.init(project="pytorch-wandb-integration")
    # Read the API key
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY not found in .env file. Please add it!")

    # Set the API key and initialize wandb
    os.environ["WANDB_API_KEY"] = wandb_api_key
    config = Configuration(config_file_path)
    wandb.init(
    project="mlops_trac_mnist", 
    config=config.to_dict())  
    trainer = Trainer(config)
    trainer.run()
    wandb.finish()