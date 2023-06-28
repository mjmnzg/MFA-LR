import torch
import numpy as np
from modules import z_score
from datasets_ import Dataset
from modules import PseudoLabeledData

class PairedData(object):
    def __init__(self, data_loader_src, data_loader_trg, max_dataset_size, num_domains_src):
        self.data_loader_src = data_loader_src
        self.data_loader_trg = data_loader_trg
        self.stop_src = [False]*num_domains_src
        self.stop_trg = False
        self.max_dataset_size = max_dataset_size
        self.num_domains_src = num_domains_src

    def __iter__(self):
        self.data_loader_src_iter = []
        for i in range(self.num_domains_src):
            self.stop_src[i] = False
            self.data_loader_src_iter.append(iter(self.data_loader_src[i]))

        self.stop_trg = False
        self.data_loader_trg_iter = iter(self.data_loader_trg)
        self.iter = 0
        return self

    def __next__(self):
        # initialize
        src_x = []
        src_y = []

        stop = True

        for i in range(self.num_domains_src):
            src_x.append(None)
            src_y.append(None)
            try:
                src_x[i], src_y[i] = next(self.data_loader_src_iter[i])
            except StopIteration:
                if src_x[i] is None or src_y[i] is None:
                    self.stop_src[i] = True
                    self.data_loader_src_iter[i] = iter(self.data_loader_src[i])
                    src_x[i], src_y[i] = next(self.data_loader_src_iter[i])

            if not self.stop_src[i]:
                stop = False

        trg_x, trg_y = None, None
        try:
            trg_x, trg_y = next(self.data_loader_trg_iter)
        except StopIteration:
            if trg_x is None or trg_y is None:
                self.stop_trg = True
                self.data_loader_trg_iter = iter(self.data_loader_trg)
                trg_x, trg_y = next(self.data_loader_trg_iter)

        if (stop and self.stop_trg) or self.iter > self.max_dataset_size:
            for i in range(self.num_domains_src):
                self.stop_src[i] = False
            self.stop_trg = False
            raise StopIteration()

        else:
            self.iter += 1
            data = {}
            # add source data
            for i in range(self.num_domains_src):
                data["Sx" + str(i + 1)] = src_x[i]
                data["Sy" + str(i + 1)] = src_y[i]
            # add target data
            data["Tx"] = trg_x
            data["Ty"] = trg_y

            return data




class UnalignedDataLoader():
    def initialize(self, num_domains, Sx, Sy, Tx, Ty, trg_subject, batch_size_src, batch_size_trg, drop_last_testing, shuffle_testing):

        # source domain
        self.dataset_src = []
        data_loader_src = []

        #####################################
        ###### MULTIPLES DOMINIOS ###########
        #####################################
        num_domains_src = num_domains - 1
        print("[*] Target subject", trg_subject)
        # Store SOURCE DOMAINS
        for i in range(num_domains):
            if i != trg_subject:
                # obtain data from subject 's'
                x_tr = np.array(Sx[i])
                y_tr = np.array(Sy[i])

                # Standardize training data
                x_tr, m, std = z_score(x_tr)

                a = len(np.where(y_tr == 0)[0])
                b = len(np.where(y_tr == 1)[0])
                c = len(np.where(y_tr == 2)[0])
                minimum = min(list([a, b, c]))
                num_classes = len(np.unique(y_tr))
                print("Subject", str(i + 1), " Total:", len(y_tr), "  Partial:", a, b, c, " Minimum:", minimum, " # classes:", num_classes)
                dataset = Dataset(x_tr, y_tr)
                self.dataset_src.append(dataset)
                data_loader_src.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size_src, shuffle=True, num_workers=1, drop_last=True))
        #################################################


        # Store TARGET DOMAIN
        dataset_target = Dataset(Tx, Ty)
        data_loader_trg = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size_trg, shuffle=shuffle_testing, num_workers=1, drop_last=drop_last_testing)

        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_src, data_loader_trg, float("inf"), num_domains_src)
        self.num_domains_src = num_domains_src

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        maxim = -1
        for i in range(self.num_domains_src):
            m = len(self.dataset_src[i])
            if m > maxim:
                maxim = m

        return min(max(maxim, len(self.dataset_t)), float("inf"))


class PairedDataModified(object):
    def __init__(self, data_loader_src, data_loader_trg, max_dataset_size, num_domains_src):
        self.data_loader_src = data_loader_src
        self.data_loader_trg = data_loader_trg
        self.stop_src = [False]*num_domains_src
        self.stop_trg = False
        self.max_dataset_size = max_dataset_size
        self.num_domains_src = num_domains_src

    def __iter__(self):
        self.data_loader_src_iter = []
        for i in range(self.num_domains_src):
            self.stop_src[i] = False
            self.data_loader_src_iter.append(iter(self.data_loader_src[i]))

        self.stop_trg = False
        self.data_loader_trg_iter = iter(self.data_loader_trg)
        self.iter = 0
        return self

    def __next__(self):
        # initialize
        src_x = []
        src_y = []

        stop = True

        for i in range(self.num_domains_src):
            src_x.append(None)
            src_y.append(None)
            try:
                src_x[i], src_y[i] = next(self.data_loader_src_iter[i])
            except StopIteration:
                if src_x[i] is None or src_y[i] is None:
                    self.stop_src[i] = True
                    self.data_loader_src_iter[i] = iter(self.data_loader_src[i])
                    src_x[i], src_y[i] = next(self.data_loader_src_iter[i])

            if not self.stop_src[i]:
                stop = False

        # inputs_target, pseudo_labels_target, weights
        trg_x, trg_y, trg_w = None, None, None

        try:
            trg_x, trg_y, trg_w = next(self.data_loader_trg_iter)
        except StopIteration:
            if trg_x is None or trg_y is None or trg_w is None:
                self.stop_trg = True
                self.data_loader_trg_iter = iter(self.data_loader_trg)
                trg_x, trg_y, trg_w = next(self.data_loader_trg_iter)

        if (stop and self.stop_trg) or self.iter > self.max_dataset_size:
            for i in range(self.num_domains_src):
                self.stop_src[i] = False
            self.stop_trg = False
            raise StopIteration()

        else:
            self.iter += 1
            data = {}
            # add source data
            for i in range(self.num_domains_src):
                data["Sx" + str(i + 1)] = src_x[i]
                data["Sy" + str(i + 1)] = src_y[i]
            # add target data
            data["Tx"] = trg_x
            data["Ty"] = trg_y
            data["Tw"] = trg_w

            return data


class UnalignedDataLoaderModified():
    def initialize(self, num_domains, Sx, Sy, target_tr, trg_subject, batch_size_src, batch_size_trg):

        # source domain
        self.dataset_src = []
        data_loader_src = []

        num_domains_src = num_domains - 1
        print("[*] Target subject", trg_subject)
        # Store SOURCE DOMAINS
        for i in range(num_domains):
            if i != trg_subject:
                # obtain data from subject 's'
                x_tr = np.array(Sx[i])
                y_tr = np.array(Sy[i])

                # Standardize training data
                x_tr, m, std = z_score(x_tr)

                dataset = Dataset(x_tr, y_tr)
                self.dataset_src.append(dataset)
                data_loader_src.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size_src, shuffle=True, num_workers=1, drop_last=True))

        # Store TARGET DOMAIN
        data_loader_trg = torch.utils.data.DataLoader(target_tr, batch_size=batch_size_trg, shuffle=True, num_workers=1, drop_last=True)

        self.dataset_t = target_tr
        self.paired_data = PairedDataModified(data_loader_src, data_loader_trg, float("inf"), num_domains_src)
        self.num_domains_src = num_domains_src

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        maxim = -1
        for i in range(self.num_domains_src):
            m = len(self.dataset_src[i])
            if m > maxim:
                maxim = m

        return min(max(maxim, len(self.dataset_t)), float("inf"))

class UnalignedDataLoaderTesting():
    def initialize(self, Tx, Ty, batch_size_trg, drop_last_testing, shuffle_testing):

        # Store TARGET DOMAIN
        dataset_target = Dataset(Tx, Ty)
        data_loader_trg = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size_trg, shuffle=shuffle_testing, num_workers=1, drop_last=drop_last_testing)

        self.dataset_t = dataset_target
        self.paired_data = PairedDataTesting(data_loader_trg, float("inf"))

    def name(self):
        return 'UnalignedDataLoaderTesting'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_t)



class PairedDataTesting(object):
    def __init__(self, data_loader_trg, max_dataset_size):
        self.data_loader_trg = data_loader_trg
        self.stop_trg = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_trg = False
        self.data_loader_trg_iter = iter(self.data_loader_trg)
        self.iter = 0
        return self

    def __next__(self):
        trg_x, trg_y = None, None
        try:
            trg_x, trg_y = next(self.data_loader_trg_iter)
        except StopIteration:
            if trg_x is None or trg_y is None:
                self.stop_trg = True
                self.data_loader_trg_iter = iter(self.data_loader_trg)
                trg_x, trg_y = next(self.data_loader_trg_iter)

        if (self.stop_trg) or self.iter > self.max_dataset_size:
            self.stop_trg = False
            raise StopIteration()

        else:
            self.iter += 1
            data = {}
            # add target data
            data["Tx"] = trg_x
            data["Ty"] = trg_y

            return data