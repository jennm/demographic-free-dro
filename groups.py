
import argparse
import gc
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from data.data import dataset_attributes
from classifier import LogisticRegressionModel
from models import model_attributes
from get_celeba_model_embeddings import get_model_and_dataset_celeba
# from utils import get_model


class FindGroups():

    def __init__(self, args):
        self.groups_from_classifiers = dict()
        self.group_counts = list()
        self.args = args
        
        if type(self.args.layer_nums) is list:
            layer_num = self.args.layer_nums[0]
        else:
            layer_num = self.args.layer_nums

        if args.dataset == 'celebA':
            self.dataloaders, self.old_model, self.dataset, self.shared_dl_args, self.num_classes = get_model_and_dataset_celeba(self.args, layer_num)
        else:
            self.dataloaders, self.old_model, self.dataset, self.shared_dl_args, self.num_classes = get_model_and_dataset_celeba(self.args, layer_num) # to be changed for other datasets

        torch.manual_seed(42)
        mp.set_start_method('spawn')
        count = 0
        for data_type in list(self.dataloaders.keys()):
            for batch in self.dataloaders[data_type]:
                print('embedding shape:', batch['embeddings'].shape)
                idxs = batch['idxs']
                for idx in idxs:
                    self.groups_from_classifiers[idx] = [0]
                    count += 1
        
        self.group_num = 1
        self.group_counts.append(count)
        count = 0

    def find_groups(self):
        if self.args.test_celebA:
            self.get_groups_default()
        
        example_idxs = list(self.groups_from_classifiers.keys())
        example_idxs.sort()
        groups_from_classifiers_list = list()
        for idx in example_idxs:
            groups_from_classifiers_list.append(self.groups_from_classifiers[idx])
        groups_from_classifiers_tensor = torch.tensor(groups_from_classifiers_list)

        # torch.save(data_to_save, "classifiers.pt")
        torch.save({'group_array': groups_from_classifiers_tensor, 'group_counts': torch.tensor(self.group_counts)}, "groups_from_classifiers_info.pt")

        return self.group_counts

    def get_groups_default(self): # to be changed to get_group code
        self.get_classifer_default()
        if type(arg.layer_nums) == list:
            for i in args.layer_nums[1:]:
                self.dataloaders = create_dataloader(self.old_model, self.datasets, self.shared_dl_args, i - 1) # assuming that if layer given is 1 we want i to be 0
                self.get_classifer_default()

        

    def get_classifer_default(self):
        # Set random seed for reproducibility
        # torch.manual_seed(42)
        # mp.set_start_method('spawn')
        count = 0

        # dataloaders = create_dataloader(old_model, datasets, shared_dl_args, 3)
        # count = 0
        # for data_type in ['train', 'val']:
        #     for batch in dataloaders[data_type]:
        #         idxs = batch['idxs']
        #         for idx in idxs:
        #             groups_from_classifiers[idx] = [0]
        #             count += 1
        
        # group_num = 1
        # group_counts.append(count)
        # count = 0

        
        # for i in range(5):
        #     print(f'Layer {i}')
        #     dataloaders = create_dataloader(old_model, datasets, shared_dl_args, i)
        first_batch_embeddings = next(iter(self.dataloaders['train']))['embeddings']
        first_batch_embeddings = first_batch_embeddings.view(first_batch_embeddings.size(0), -1)


        # Initialize the model, loss function, and optimizer
        input_size = first_batch_embeddings.shape[-1]  # MNIST images are 28x28 pixels
        num_classes = 2  # use a variable
        log_model = LogisticRegressionModel(input_size, num_classes)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.01,1], device=torch.cuda.current_device()), reduction='none')
        optimizer = optim.SGD(log_model.parameters(), lr=0.01)

        # Training loop
        num_epochs = 5

        for epoch in range(num_epochs):
            log_model.train()
            for batch in self.dataloaders['train']:
                device = torch.cuda.current_device()
                log_model.to(device)
                embeddings = batch['embeddings']
                # loss = batch['loss']
                class_labels = batch['class_label']

                # Flatten the images
                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                class_labels = class_labels.to(device)

                # Forward pass
                outputs = log_model(embeddings)

                # Changing torch type
                class_labels = class_labels.to(torch.long)

                # Calculate loss
                loss =  criterion(outputs, class_labels)
                loss_mean = torch.mean(loss)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss_mean.backward()
                optimizer.step()

            # Print training loss after each epoch
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_mean.item():.4f}')
            gc.collect()
            torch.cuda.empty_cache()


            # Evaluation on the validation set
            log_model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for batch in dataloaders['val']:
                    embeddings = batch['embeddings']
                    loss = batch['loss']
                    class_labels = batch['class_label']
                    class_labels = class_labels.to(device)
                    class_labels = class_labels.to(torch.long)

                    embeddings = embeddings.view(embeddings.size(0), -1)
                    outputs = log_model(embeddings)

                    _, predicted = torch.max(outputs, 1)
                    total += class_labels.size(0)
                    correct += (predicted == class_labels).sum().item()

                accuracy = correct / total
                print(f'Val Accuracy: {accuracy:.4f}')

                gc.collect()
                torch.cuda.empty_cache()

                count = 0
                for data_type in ['train', 'val']:
                    print(group_num, data_type)
                    for batch in dataloaders[data_type]:
                        embeddings = batch['embeddings']
                        idxs = batch['idxs']
                        embeddings = embeddings.view(embeddings.size(0), -1)
                        outputs = log_model(embeddings)
                        _, predicted = torch.max(outputs, 1)
                        for i in range(len(idxs)):
                            idx = idxs[i]
                            if predicted[i].item() == 1:
                                self.groups_from_classifiers[idx].append(self.group_num)
                                count += 1
                            else:
                                self.groups_from_classifiers[idx].append(-1)
                    gc.collect()
                    torch.cuda.empty_cache()
                self.group_counts.append(count)
                self.group_num += 1

                gc.collect()
                torch.cuda.empty_cache()





            # data_to_save['input_size'].append(input_size)
            # data_to_save['num_classes'].append(num_classes)
            # data_to_save['classifiers'].append(log_model.state_dict())

        example_idxs = list(groups_from_classifiers.keys())
        example_idxs.sort()
        groups_from_classifiers_list = list()
        for idx in example_idxs:
            groups_from_classifiers_list.append(groups_from_classifiers[idx])
        groups_from_classifiers_tensor = torch.tensor(groups_from_classifiers_list)

        # torch.save(data_to_save, "classifiers.pt")
        torch.save({'group_array': groups_from_classifiers_tensor, 'group_counts': torch.tensor(group_counts)}, "groups_from_classifiers_info.pt")

        return log_model, accuracy

def parse_list(s):
    if '[' in s:
        try:
            return [int(x) for x in s.strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid list format. Must be a comma-separated list of integers.")
    else:
        try:
            return int(s)
        except ValueError:
            raise argparse.ArgumentTypeError("Invalid format. Must be an integer or comma-separated list of integers of form [1,2,3].")


def main():
    parser = argparse.ArgumentParser(description='tests that poor performinc class is separable by embeddings')
    parser.add_argument("-d",
                        "--dataset",
                        choices=dataset_attributes.keys(),
                        default="celebA")
    parser.add_argument("--model",
                        choices=model_attributes.keys(),
                        default="resnet50")
    parser.add_argument("--train_from_scratch",
                        action="store_true",
                        default=False)
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_celebA", action="store_true", default=False)
    parser.add_argument('--layer_nums', type=parse_list, help='Input list of integers', required=True)
    args = parser.parse_args()

    find_groups = FindGroups(args) # need to add an argument for this
    group_counts = find_groups.find_groups()
    print(group_counts)
    # find_groups(args)
    # log_model, acc = train_test_classifier(args)
    # print('Accuracy:', acc)
    

    



if __name__ == '__main__':
    main()
