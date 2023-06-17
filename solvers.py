import network
from dataloader import *
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import lr_schedule
import utils
import torch.nn.functional as F
from modules import PseudoLabeledData, load_seed, load_seed_iv, split_data,z_score, normalize
import numpy as np
import adversarial
from vat import ConditionalEntropyLoss
from models import EMA
from cmd import CMD
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize
from utils import plot_tsne

def test_suda(loader, model):
    start_test = True
    with torch.no_grad():
        # get iterate data
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            # get sample and label
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            # load in gpu
            inputs = inputs.type(torch.FloatTensor).cuda()
            labels = labels
            # obtain predictions
            _, outputs = model(inputs)
            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # obtain labels
    _, predictions = torch.max(all_output, 1)
    # calculate accuracy for all examples
    accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

    y_true = all_label.cpu().data.numpy()
    y_pred = predictions.cpu().data.numpy()
    labels = np.unique(y_true)

    # Binarize ytest with shape (n_samples, n_classes)
    ytest = label_binarize(y_true, classes=labels)
    ypreds = label_binarize(y_pred, classes=labels)

    f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
    matrix = confusion_matrix(y_true, y_pred)

    return accuracy, f1, auc, matrix


def test_muda(dataset_test, model):
    start_test = True
    features = None
    with torch.no_grad():

        for batch_idx, data in enumerate(dataset_test):
            Tx = data['Tx']
            Ty = data['Ty']
            Tx = Tx.float().cuda()

            # obtain predictions
            feats, outputs = model(Tx)

            # concatenate predictions
            if start_test:
                all_output = outputs.float().cpu()
                all_label = Ty.float()
                features = feats.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, Ty.float()), 0)
                features = np.concatenate((features, feats.float().cpu()), 0)

            # obtain labels
        _, predictions = torch.max(all_output, 1)
        # calculate accuracy for all examples
        accuracy = torch.sum(torch.squeeze(predictions).float() == all_label).item() / float(all_label.size()[0])

        y_true = all_label.cpu().data.numpy()
        y_pred = predictions.cpu().data.numpy()
        labels = np.unique(y_true)

        # Binarize ytest with shape (n_samples, n_classes)
        ytest = label_binarize(y_true, classes=labels)
        ypreds = label_binarize(y_pred, classes=labels)

        f1 = f1_score(y_true, y_pred, average='macro')
        auc = roc_auc_score(ytest, ypreds, average='macro', multi_class='ovr')
        matrix = confusion_matrix(y_true, y_pred)

        return accuracy, f1, auc, matrix, features, y_pred


def MFA_LR(args):
    """
    Parameters:
        @args: arguments
    """
    # --------------------------
    # Prepare data
    # --------------------------

    # Load SEED and SEED-IV data
    if args.dataset in ["seed", "seed-iv"]:
        print("DATA:", args.dataset, " SESSION:", args.session)
        # Load imagined speech data
        if args.dataset == "seed":
            X, Y = load_seed(args.file_path, session=args.session, feature="de_LDS")
        else:
            # [1 session]
            if args.mixed_sessions == 'per_session':
                X, Y = load_seed_iv(args.file_path, session=args.session)
            # [3 sessions]
            elif args.mixed_sessions == 'mixed':
                X1, Y1 = load_seed_iv(args.file_path, session=1)
                X2, Y2 = load_seed_iv(args.file_path, session=2)
                X3, Y3 = load_seed_iv(args.file_path, session=3)

                X = {}
                Y = {}
                for key in X1.keys():
                    X1[key], _, _ = z_score(X1[key])
                    X2[key], _, _ = z_score(X2[key])
                    X3[key], _, _ = z_score(X3[key])

                    X[key] = np.concatenate((X1[key], X2[key], X3[key]), axis=0)
                    Y[key] = np.concatenate((Y1[key], Y2[key], Y3[key]), axis=0)
            else:
                print("Option [mixed_sessions] is not valid.")
                exit(-1)

        # select target subject
        trg_subj = args.target - 1
        # Target data
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # subjects
        subject_ids = X.keys()
        num_domains = len(subject_ids)

        # [Option 1]: Evaluation over all target domain
        # Vx = Tx
        # Vy = Ty

        # [Option 2]: Evaluation over test data from Target domain
        # Split target data for testing
        Tx, Ty, Vx, Vy = split_data(Tx, Ty, args.seed, test_size=0.2)
        # Standardize target data
        Tx, m, std = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=std)


        print("Target subject:", trg_subj)
        print("Tx:", Tx.shape, " Ty:", Ty.shape)
        print("Vx:", Vx.shape, " Vy:", Vy.shape)
        print("Num. domains:", num_domains)

        print("Data were succesfully loaded")

        # Train dataset
        train_loader = UnalignedDataLoader()
        train_loader.initialize(num_domains, X, Y, Tx, Ty, trg_subj, args.batch_size, args.batch_size, shuffle_testing=True, drop_last_testing=True)
        datasets = train_loader.load_data()

        #classes = np.unique(Ty)

        # t-SNE
        #Tx = np.reshape(Tx, (Tx.shape[0], Tx.shape[1]*Tx.shape[2]*Tx.shape[3]))
        #plot_tsne(Tx, Ty, "original_features.png", classes)


        # Test dataset
        test_loader = UnalignedDataLoaderTesting()
        test_loader.initialize(Vx, Vy, 200, shuffle_testing=False, drop_last_testing=False)
        dataset_test = test_loader.load_data()

    else:
        print("This dataset does not exist.")
        exit(-1)


    # --------------------------
    # Create Deep Neural Network
    # --------------------------
    # For synthetic dataset
    if args.dataset in ["seed", "seed-iv"]:
        # Define Neural Network
        # 2790 for SEED
        # 620 for SEED-IV
        input_size = 2790 if args.dataset == "seed" else 620   # windows_size=9
        hidden_size = 512

        model = network.DFN(input_size=input_size, hidden_size=hidden_size, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=args.radius).cuda()


    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    #
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2]

    optimizer_classifier = torch.optim.SGD(parameter_classifier, lr=args.lr_a, momentum=0.9, weight_decay=0.005)
    optimizer_feature = torch.optim.SGD(parameter_feature, lr=args.lr_a, momentum=0.9, weight_decay=0.005)

    # if gpus are availables
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        #adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    # ------------------------
    # Model training
    # ------------------------

    # Number of centroids for semantic loss
    if args.dataset in ["seed", "seed-iv"]:
        Cs_memory = []
        for d in range(num_domains):
            Cs_memory.append(torch.zeros(args.num_class, args.bottleneck_dim).cuda())
        Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

    else:
        print("SETTING number of centroids: The dataset does not exist.")
        exit()

    cent = ConditionalEntropyLoss().cuda()

    ''' Exponential moving average (simulating teacher model) '''
    ema = EMA(0.998)
    ema.register(model)

    # for weighting loss
    weights_d = torch.zeros(num_domains - 1).cuda()
    weights_d += 1

    weights_s = torch.zeros(num_domains - 1).cuda()
    weights_s += 1

    alpha = 0.90

    # [CMD]
    cmd = CMD(n_moments=2)
    log_total_loss = []

    for i in range(args.max_iter1):

        for batch_idx, data in enumerate(datasets):
            # get the source batches
            x_src = list()
            y_src = list()

            for domain_idx in range(num_domains - 1):
                tmp_x = data['Sx' + str(domain_idx + 1)].float().cuda()
                tmp_y = data['Sy' + str(domain_idx + 1)].long().cuda()
                x_src.append(tmp_x)
                y_src.append(tmp_y)

            # get the target batch
            x_trg = data['Tx'].float().cuda()

            # Enable model to train
            model.train(True)

            # obtain schedule for learning rate
            optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_a)
            optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_a)

            # Get features target
            features_target, outputs_target = model(x_trg)
            # pseudo-labels
            pseu_labels_target = torch.argmax(outputs_target, dim=1)


            sm_loss = []
            dom_loss = []
            pred_src = []

            for domain_idx in range(num_domains - 1):
                # get features and predictions
                features_source, outputs_source = model(x_src[domain_idx])
                pred_src.append(outputs_source)

                # [CMD loss]
                loss_domain = cmd.forward(features_source, features_target)

                # calculate semantic loss
                semantic_loss, Cs_memory[domain_idx], Ct_memory = utils.SM(features_source, features_target, y_src[domain_idx], pseu_labels_target, Cs_memory[domain_idx], Ct_memory, decay=0.9)

                # get loss value for domainidx
                sm_loss.append(semantic_loss)
                dom_loss.append(loss_domain)

            # Stack/Concat data from each source domain
            pred_source = torch.cat(pred_src, dim=0)
            labels_source = torch.cat(y_src, dim=0)

            # [COARSE-grained training loss]
            classifier_loss = nn.CrossEntropyLoss()(pred_source, labels_source)


            #with torch.no_grad():
            # [weighting MDAN loss]
            weights_dom = torch.stack(dom_loss)
            weights_dom = weights_dom.detach()  # to avoid inplace gradient since it modifies original gradient
            weights_dom = 1 / weights_dom
            weights_dom = torch.nn.functional.softmax(weights_dom, dim=0)
            # moving average [opt.1]
            weights_d = (1 - alpha) * weights_d + alpha * weights_dom

            # [weighting SEM loss]
            weights_sem = torch.stack(sm_loss)
            weights_sem = weights_sem.detach()  # to avoid inplace gradient since it modifies original gradient
            weights_sem = 1 / weights_sem
            weights_sem = torch.nn.functional.softmax(weights_sem, dim=0)
            # moving average [opt.1]
            weights_s = (1 - alpha) * weights_s + alpha * weights_sem

            align_loss = 0
            for domain_idx in range(num_domains - 1):
                align_loss += weights_s[domain_idx] * sm_loss[domain_idx] + weights_d[domain_idx] * dom_loss[domain_idx]


            # [Conditional Entropy loss]
            loss_trg_cent = cent(outputs_target)

            # [TOTAL LOSS]
            # [1] total_loss = classifier_loss + align_loss + 0.1 * loss_trg_cent
            total_loss = classifier_loss + 0.5 * align_loss + 0.1 * loss_trg_cent

            # Reset gradients
            optimizer_classifier.zero_grad()
            optimizer_feature.zero_grad()

            # Compute gradients
            # [normal]
            total_loss.backward()

            # [Update weights]
            # classifier
            optimizer_classifier.step()
            optimizer_feature.step()

            # Polyak averaging.
            ema(model)  # TODO: move ema into the optimizer step fn.

            # free variables
            for d in range(num_domains):
                Cs_memory[d].detach_()
            Ct_memory.detach_()

        # set model to test
        model.train(False)

        # calculate accuracy performance
        best_acc, best_f1, best_auc, best_mat, features, labels = test_muda(dataset_test, model)
        log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc, best_f1, best_auc)
        args.log_file.write(log_str)
        args.log_file.flush()
        print(log_str)
        log_total_loss.append(total_loss.data)

    return X, Y, best_acc, best_f1, best_auc, best_mat, model, log_total_loss



def RSDA(X, Y, args, samples, weighted_pseu_label, weights):

    # prepare data
    dset_loaders = {}

    if args.dataset in ["seed", "seed-iv"]:

        print("DATA:", args.dataset, " SESSION:", args.session)

        # get dictionary keys
        subjects = X.keys()

        print(subjects)

        # build Source dataset
        Sx = Sy = None
        i = 0
        flag = False
        selected_subject = args.target - 1
        trg_subj = -1

        for s in subjects:
            # if subject is not the selected for target
            if i != selected_subject:

                tr_x = np.array(X[s])
                tr_y = np.array(Y[s])

                # global-wise standardization
                tr_x, m, std = z_score(tr_x)

                if not flag:
                    Sx = tr_x
                    Sy = tr_y
                    flag = True
                else:
                    Sx = np.concatenate((Sx, tr_x), axis=0)
                    Sy = np.concatenate((Sy, tr_y), axis=0)
            else:
                # store ID
                trg_subj = s
            i += 1

        print("[+] Target subject:", trg_subj)

        # Target dataset
        Tx = np.array(X[trg_subj])
        Ty = np.array(Y[trg_subj])

        # Split target data for testing
        Tx, Ty, Vx, Vy = split_data(Tx, Ty, args.seed, test_size=0.2)

        # Global-wise standardization
        Tx, m, sd = z_score(Tx)
        Vx = normalize(Vx, mean=m, std=sd)

        print("Sx_train:", Sx.shape, "Sy_train:", Sy.shape)
        print("Tx_train:", Tx.shape, "Ty_train:", Ty.shape)
        print("Tx_test:", Vx.shape, "Ty_test:", Vy.shape)

        # to tensor
        Sx_tensor = torch.tensor(Sx)
        Sy_tensor = torch.tensor(Sy)

        # create containers for source data
        source_tr = TensorDataset(Sx_tensor, Sy_tensor)

        # create container for target data
        target_tr = PseudoLabeledData(samples.numpy(), weighted_pseu_label, weights)

        # create container for test data
        Vx_tensor = torch.tensor(Vx)
        Vy_tensor = torch.tensor(Vy)
        target_ts = TensorDataset(Vx_tensor, Vy_tensor)

        # data loader
        dset_loaders["source"] = DataLoader(source_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["target"] = DataLoader(target_tr, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        dset_loaders["test"] = DataLoader(target_ts, batch_size=200, shuffle=False, num_workers=4)

        print("Data were succesfully loaded")

    else:
        print("This dataset does not exist.")
        exit()

    # Create model
    if args.dataset in ["seed", "seed-iv"]:

        # [Window]
        input_size = 2790 if args.dataset == "seed" else 620   # windows_size=9

        model = network.DFN(input_size=input_size, hidden_size=512, bottleneck_dim=args.bottleneck_dim, class_num=args.num_class, radius=10.0).cuda()

        # setting Adversarial net
        adv_net = network.DiscriminatorDANN(in_feature=model.output_num(), radius=10.0, hidden_size=args.bottleneck_dim, max_iter=1000).cuda()

    else:
        print("A neural network for this dataset has not been selected yet.")
        exit(-1)

    # Ger trainable weights
    parameter_classifier = [model.get_parameters()[2]]
    parameter_feature = model.get_parameters()[0:2] + adv_net.get_parameters()

    # gradient reversal layer
    my_grl = adversarial.AdversarialLayer()

    optimizer_classifier = torch.optim.Adam(parameter_classifier, lr=args.lr_b, weight_decay=0.005)
    optimizer_feature = torch.optim.Adam(parameter_feature, lr=args.lr_b, weight_decay=0.005)

    # if number of GPUS is greater 1
    gpus = args.gpu_id.split(',')
    if len(gpus) > 1:
        adv_net = nn.DataParallel(adv_net, device_ids=[int(i) for i in gpus])
        model = nn.DataParallel(model, device_ids=[int(i) for i in gpus])

    ## Train MODEL

    # lenght of data
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    # auxiliar variables
    best_acc = 0.0

    # centroids for each cluster
    if args.dataset in ["seed", "seed-iv"]:
        Cs_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()
        Ct_memory = torch.zeros(args.num_class, args.bottleneck_dim).cuda()

    else:
        print("The number of centroids for this dataset has not been selected yet.")
        exit()

    ''' Exponential moving average (simulating teacher model) '''
    ema = EMA(0.998)
    ema.register(model)

    # iterate over
    for i in range(args.max_iter2):

        # Testing phase
        if i % args.test_interval == args.test_interval - 1:
            # set model training to False
            model.train(False)
            # calculate accuracy on test set
            best_acc, best_f1, best_auc, best_mat = test_suda(dset_loaders, model)

            log_str = "iter: {:05d}, \t accuracy: {:.4f} \t f1: {:.4f} \t auc: {:.4f}".format(i, best_acc, best_f1, best_auc)
            args.log_file.write(log_str)
            args.log_file.flush()
            print(log_str)

        # Enable model for training
        model.train(True)
        adv_net.train(True)

        # obtain schedule for learning rate
        optimizer_classifier = lr_schedule.inv_lr_scheduler(optimizer_classifier, i, lr=args.lr_b)
        optimizer_feature = lr_schedule.inv_lr_scheduler(optimizer_feature, i, lr=args.lr_b)

        # get data
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        # Get batch for source and target domains
        inputs_source_, labels_source = iter_source.next()
        inputs_target, pseudo_labels_target, weights = iter_target.next()
        # Cast
        inputs_source_ = inputs_source_.type(torch.FloatTensor)
        labels_source = labels_source.type(torch.LongTensor)
        # to cuda
        inputs_source, labels_source = inputs_source_.cuda(), labels_source.cuda()
        inputs_target, pseudo_labels_target = inputs_target.cuda(), pseudo_labels_target.cuda()
        weights = weights.type(torch.Tensor).cuda()

        weights[weights < 0.5] = 0.0

        # get features and labels for source and target domain
        features_source, outputs_source = model(inputs_source)
        features_target, outputs_target = model(inputs_target)

        # concatenate features
        features = torch.cat((features_source, features_target), dim=0)
        # concatenate logits
        logits = torch.cat((outputs_source, outputs_target), dim=0)

        # cross-entropy loss
        source_class_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        # adversarial loss
        adv_loss = utils.loss_adv(my_grl.apply(features), adv_net, logits=torch.nn.Softmax(dim=1)(logits).detach())

        # [Conditional entropy]
        ce_loss = torch.mean(utils.Entropy(F.softmax(outputs_target, dim=1)))

        # function robust loss
        target_robust_loss = utils.robust_pseudo_loss(outputs_target, pseudo_labels_target, weights)

        # classifier loss
        classifier_loss = source_class_loss + target_robust_loss

        # obtain pseudo labels
        pseu_labels_target = torch.argmax(outputs_target, dim=1)
        # semantic loss
        loss_sm, Cs_memory, Ct_memory = utils.SM(features_source, features_target, labels_source, pseu_labels_target, Cs_memory, Ct_memory, decay=0.9)

        # [FINAL LOSS]
        # [original]
        #total_loss = classifier_loss + 0.1 * adv_loss + 0.1 * loss_sm + 0.1 * ce_loss
        # [set]
        total_loss = classifier_loss + 1.0 * adv_loss + 0.1 * loss_sm + 0.1 * ce_loss

        # reset gradients
        optimizer_classifier.zero_grad()
        optimizer_feature.zero_grad()

        # compute gradients
        total_loss.backward()

        # update weights
        optimizer_feature.step()
        optimizer_classifier.step()

        # Polyak averaging.
        ema(model)  # TODO: move ema into the optimizer step fn.

        Cs_memory.detach_()
        Ct_memory.detach_()

    return best_acc, best_f1, best_auc, best_mat, model



