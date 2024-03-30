
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
from get_mnist_model_embeddings import get_model_and_dataset_mnist
# from utils import get_model


class FindGroups():

    def __init__(self, args):
        if args.groups_exist:
            old_group_info = torch.load(args.group_info_path)
            old_group_counts = old_group_info['group_counts'].tolist()
            self.group_counts = old_group_counts
            self.group_num = len(old_group_counts)
            self.old_group_array = old_group_info['group_array']
        else:
            self.group_counts = list()
            self.group_num = 1
        
        self.groups_from_classifiers = dict()
        self.args = args
        
        if type(self.args.layer_nums) is list:
            layer_num = self.args.layer_nums[0]
        else:
            layer_num = self.args.layer_nums

        if args.dataset == 'celebA':
            self.dataloaders, self.old_model, self.dataset, self.shared_dl_args, self.num_classes = get_model_and_dataset_celeba(self.args, layer_num)
        elif args.dataset == 'ColoredMNIST':
            self.dataloaders, self.old_model, self.dataset, self.shared_dl_args, self.num_classes = get_model_and_dataset_mnist(self.args, layer_num)
        else:
            self.dataloaders, self.old_model, self.dataset, self.shared_dl_args, self.num_classes = get_model_and_dataset_celeba(self.args, layer_num) # to be changed for other datasets

        torch.manual_seed(42)
        mp.set_start_method('spawn')
        count = 0
        for data_type in list(self.dataloaders.keys()):
            for batch in self.dataloaders[data_type]:
                idxs = batch['idxs']
                for idx in idxs:
                    self.groups_from_classifiers[idx] = [0]
                    count += 1
        
        if not args.groups_exist:
            self.group_counts.append(count)
        count = 0

    def find_groups(self, exp=False):
        first_batch_embeddings = next(iter(self.dataloaders['train']))['embeddings']
        first_batch_embeddings = first_batch_embeddings.view(first_batch_embeddings.size(0), -1)

        if exp:
            self.exp_num_data_points_classifer_default()
            return [0]
        if self.args.test_celebA:
            self.get_groups_default()
        else:
            self.get_groups_lr(input_size=first_batch_embeddings.shape[-1])
        
        self.save_group_info() # this line will be removed after refactoring

        return self.group_counts

    def save_group_info(self):
        example_idxs = list(self.groups_from_classifiers.keys())
        example_idxs.sort()
        groups_from_classifiers_list = list()
        if self.args.groups_exist:
            start_index = 1
        else:
            start_index = 0

        for idx in example_idxs:
            groups_from_classifiers_list.append(self.groups_from_classifiers[idx][start_index:])
        groups_from_classifiers_tensor = torch.tensor(groups_from_classifiers_list)

        if self.args.groups_exist:
            groups_from_classifiers_tensor = torch.concat([self.old_group_array, groups_from_classifiers_tensor], dim=1)

        # torch.save(data_to_save, "classifiers.pt")
        torch.save({'group_array': groups_from_classifiers_tensor, 'group_counts': torch.tensor(self.group_counts)}, self.args.group_info_path) #"groups_from_classifiers_info.pt")


    def identify_misclassified(self, model, old_misclassified_indices=None, num_batches=None, threshold=1.5):
        device = torch.cuda.current_device()
        with torch.no_grad():
            cur_batch = 0
            misclassified_idxes = set()
            for batch in self.dataloaders['train']:
                if num_batches is not None and cur_batch >= num_batches:
                    break
                cur_batch +=1

                embeddings = batch['embeddings']
                idxs = batch['idxs']
                labels = batch['actual_label']
                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                outputs = model(embeddings)
                predicted = (outputs[:, 1] > 0.5).long()
                if old_misclassified_indices:
                    labels = labels.tolist()
                    # print('miscalssified_indices')
                    for i in range(len(idxs)):
                        idx = idxs[i]
                        if idx in old_misclassified_indices:
                            labels[i] = (labels[i] + 1) % 2
                            print('diff')
                    labels = torch.tensor(labels, dtype=torch.long)
                predicted = predicted.to(device)
                labels = labels.to(device)
                misclassified_indices = torch.where(predicted != labels)[0]
                distances = torch.abs(outputs - 0.5)
                for idx in misclassified_indices:
                    if distances[idx, 0] < threshold:
                        misclassified_idxes.add(idx)

        return misclassified_idxes


    # Train the model
    def train_lr_model(self, log_model, criterion, optimizer, misclassified_indices=None, num_epochs=5):
        log_model.train()
        device = torch.cuda.current_device()
        for epoch in range(num_epochs):
            log_model.train()
            for batch in self.dataloaders['train']:
                embeddings = batch['embeddings']
                idxs = batch['idxs']
                labels = batch['actual_label']
                if misclassified_indices:
                    # print(misclassified_indices)
                    labels = labels.tolist()
                    for i in range(len(idxs)):
                        idx = idxs[i]
                        if idx in misclassified_indices:
                            # print('diff')
                            labels[i] = (labels[i] + 1) % 2
                    labels = torch.tensor(labels, dtype=torch.long)
                else:
                    labels = labels.to(torch.long)

                criterion.weight = torch.tensor([(labels == 0).sum() / labels.shape[0], (labels == 1).sum() / labels.shape[0]], device=device)
                optimizer.zero_grad()

                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                labels = labels.to(device)
                log_model = log_model.to(device)

                outputs = log_model(embeddings)
                loss = criterion(outputs, labels)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                # if epoch % 100 == 0:
                #   print(f'Loss epoch: {loss}')
    
    def get_groups_lr(self, input_size, k=10, num_classes=2):
        # first_batch_embeddings = next(iter(self.dataloaders['train']))['embeddings']
        # first_batch_embeddings = first_batch_embeddings.view(first_batch_embeddings.size(0), -1)


        # Initialize the model, loss function, and optimizer
        # input_size = first_batch_embeddings.shape[-1] 
        device = torch.cuda.current_device()

        
        model = LogisticRegressionModel(input_size, num_classes)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)


        # Train the model
        self.train_lr_model(model, criterion, optimizer)
        self.get_performance_metrics(model, device)
        # return

        # visualize_model(model, X, y, colors)
        misclassified_indices = None
        for i in range(k):
            misclassified_indices = self.identify_misclassified(model, misclassified_indices)
            print(f'misclassified length: {len(misclassified_indices)}')
            if len(misclassified_indices) == 0: return

            # Define model, criterion, and optimizer
            # does this have to be a new model?
            model = LogisticRegressionModel(input_size, num_classes)
            model = model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            # Train the model
            self.train_lr_model(model, criterion, optimizer, misclassified_indices)
            #   visualize_model(model, X, y, colors)
            self.get_performance_metrics(model, device)
        
        count = 0
        for data_type in ['train', 'val']:
            for batch in self.dataloaders[data_type]:
                embeddings = batch['embeddings']
                idxs = batch['idxs']
                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                outputs = model(embeddings)
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

            # self.remove_points_from_canidacy()
    

    def get_groups_default(self): # to be changed to get_group code
        self.get_classifer_default()
        if type(self.args.layer_nums) == list:
            for i in self.args.layer_nums[1:]:
                self.dataloaders = create_dataloader(self.old_model, self.datasets, self.shared_dl_args, i - 1) # assuming that if layer given is 1 we want i to be 0
                self.get_classifer_default()

    def get_performance_metrics(self, log_model, device):
        log_model.eval()
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in self.dataloaders['val']:
                embeddings = batch['embeddings']
                loss = batch['loss']
                erm_predicted_labels = batch['predicted_label']
                actual_labels = batch['actual_label']
                all_ones = torch.ones(actual_labels.size(0), device=device)
                all_zeros = torch.zeros(actual_labels.size(0), device=device)

                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                outputs = log_model(embeddings)

                _, predicted = torch.max(outputs, 1)
                tp += ((predicted == 1) & (erm_predicted_labels != actual_labels) & (predicted == all_ones)).sum().item()
                fp += ((predicted != 1) & (erm_predicted_labels != actual_labels) & (predicted == all_ones)).sum().item()
                tn += ((predicted == 0) & (erm_predicted_labels == actual_labels) & (predicted == all_zeros)).sum().item()
                fn += ((predicted != 0) & (erm_predicted_labels == actual_labels) & (predicted == all_zeros)).sum().item()
                total += predicted.size(0)
                correct += tp + tn

            accuracy = correct / total
            print(tp, fp, tn, fn)
            print(f'Val Accuracy: {accuracy:.4f}')
            ppv = tp/(max(1, tp+fp))
            print(f'TPR: {tp/(max(1, tp+fn))}\tFPR: {fp/(max(1, tn+fp))}\tTNR: {tn/(max(1, tn+fp))}\tFNR: {fn/(max(1, tp+fn))}\tPPV: {ppv}\t1 - PPV: {1 - ppv}')

            gc.collect()
            torch.cuda.empty_cache()

    def get_performance_metrics_group_labels(self, log_model, device):
        log_model.eval()
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in self.dataloaders['val']:
                embeddings = batch['embeddings']
                loss = batch['loss']
                class_labels = batch['class_label']
                class_labels = class_labels.to(device)
                class_labels = class_labels.to(torch.long)
                all_ones = torch.ones(class_labels.size(0), device=device)
                all_zeros = torch.zeros(class_labels.size(0), device=device)

                embeddings = embeddings.view(embeddings.size(0), -1)
                embeddings = embeddings.to(device)
                outputs = log_model(embeddings)

                _, predicted = torch.max(outputs, 1)
                tp += ((predicted == class_labels) & (predicted == all_ones)).sum().item()
                fp += ((predicted != class_labels) & (predicted == all_ones)).sum().item()
                tn += ((predicted == class_labels) & (predicted == all_zeros)).sum().item()
                fn += ((predicted != class_labels) & (predicted == all_zeros)).sum().item()
                total += class_labels.size(0)
                correct += (predicted == class_labels).sum().item()

            accuracy = correct / total
            print(tp, fp, tn, fn)
            print(f'Val Accuracy: {accuracy:.4f}')
            ppv = tp/(max(1, tp+fp))
            print(f'TPR: {tp/(max(1, tp+fn))}\tFPR: {fp/(max(1, tn+fp))}\tTNR: {tn/(max(1, tn+fp))}\tFNR: {fn/(max(1, tp+fn))}\tPPV: {ppv}\t1 - PPV: {1 - ppv}')

            gc.collect()
            torch.cuda.empty_cache()

    def exp_num_data_points_classifer_default(self):
        # looks at how accuracy is affected when different sample amounts are used

        # Set random seed for reproducibility
        torch.manual_seed(42)
        # mp.set_start_method('spawn')
        count = 0

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

        num_batches = 5
        will_break = True

        while will_break:
            log_model = LogisticRegressionModel(input_size, num_classes)
            criterion = nn.CrossEntropyLoss(weight=torch.tensor([.01,1], device=torch.cuda.current_device()), reduction='none')
            optimizer = optim.SGD(log_model.parameters(), lr=0.01)

            will_break = False
            for epoch in range(num_epochs):
                log_model.train()
                batch_num = 0
                for batch in self.dataloaders['train']:
                    if batch_num >= num_batches:
                        will_break = True
                    batch_num += 1
                    device = torch.cuda.current_device()
                    log_model = log_model.to(device)
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
                # self.get_performance_metrics(log_model)
                log_model.eval()
                with torch.no_grad():
                    correct = 0
                    total = 0
                    batch_num = 0
                    for batch in self.dataloaders['val']:
                        if batch_num >= num_batches:
                            break
                        batch_num += 1
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
                    print(f'Val Accuracy Epoch {epoch + 1} {num_batches} batches: {accuracy:.4f}')

                    gc.collect()
                    torch.cuda.empty_cache()
            num_batches += 5


        gc.collect()
        torch.cuda.empty_cache()


        

    def get_classifer_default(self):
        # Set random seed for reproducibility
        torch.manual_seed(42)
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
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([.001,1], device=torch.cuda.current_device()), reduction='none')
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
                print('outputs shape',outputs.shape)
                print('class_labels shape', class_labels.shape)
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
            self.get_performance_metrics(log_model, device)
            
        # set group information according to classifier    
        count = 0
        for data_type in ['train', 'val']:
            for batch in self.dataloaders[data_type]:
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

        example_idxs = list(self.groups_from_classifiers.keys())
        example_idxs.sort()
        groups_from_classifiers_list = list()
        for idx in example_idxs:
            groups_from_classifiers_list.append(self.groups_from_classifiers[idx])
        groups_from_classifiers_tensor = torch.tensor(groups_from_classifiers_list)

        # torch.save(data_to_save, "classifiers.pt")
        torch.save({'group_array': groups_from_classifiers_tensor, 'group_counts': torch.tensor(self.group_counts)}, "groups_from_classifiers_info.pt")

        return log_model

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
                        default="CelebA")
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
    parser.add_argument("--target_name")
    parser.add_argument("--confounder_name")
    parser.add_argument("--groups_exist", action="store_true", default=False)
    parser.add_argument("--group_info_path")
    args = parser.parse_args()

    if args.dataset == "CUB":
        args.root_dir = "./cub"
        args.target = "waterbird_complete95"
        args.confounder_name = "forest2water2"
        args.model = "resnet50"
    elif args.dataset == "CelebA":
        args.root_dir = "./"
        args.target = "Blond_Hair"
        args.confounder_name = "Male"
        args.model = "resnet50"
    elif args.dataset == "MultiNLI":
        args.root_dir = "./"
        args.target = "gold_label_random"
        args.confounder_name = "sentence2_has_negation"
        args.model = "bert"
    elif args.dataset == "jigsaw":
        args.root_dir = "./jigsaw"
        args.target = "toxicity"
        args.confounder_name = "identity_any"
        args.model = "bert-base-uncased"
    elif args.dataset == "ColoredMNIST":
        args.root_dir = "./"
        args.target = "target"
        args.confounder_name = "confounder"
        args.model = "cnn"
    else:
        assert False, f"{args.dataset} is not a known dataset."


    print("model", args.model)

    find_groups = FindGroups(args) # need to add an argument for this
    group_counts = find_groups.find_groups()
    print('group_counts:', group_counts)
    # find_groups(args)
    # log_model, acc = train_test_classifier(args)
    # print('Accuracy:', acc)
    

# def test():
#     find_groups = FindGroups(None)
#     find_groups.



if __name__ == '__main__':
    main()
