"""
Neural Network tools developed for UCSD ECE285 MLIP.

Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.
"""

import os
import time
import torch
from torch import nn
import torch.utils.data as td
from abc import ABC, abstractmethod

class NeuralNetwork(nn.Module, ABC):
    """An abstract class representing a neural network.

    All other neural network should subclass it. All subclasses should override
    ``forward``, that makes a prediction for its input argument, and
    ``criterion``, that evaluates the fit between a prediction and a desired
    output. This class inherits from ``nn.Module`` and overloads the method
    ``named_parameters`` such that only parameters that require gradient
    computation are returned. Unlike ``nn.Module``, it also provides a property
    ``device`` that returns the current device in which the network is stored
    (assuming all network parameters are stored on the same device).
    """

    def __init__(self):
        super(NeuralNetwork, self).__init__()

    @property
    def device(self):
        # This is important that this is a property and not an attribute as the
        # device may change anytime if the user do ``net.to(newdevice)``.
        return next(self.parameters()).device

    def named_parameters(self, recurse=True):
        nps = nn.Module.named_parameters(self)
        for name, param in nps:
            if not param.requires_grad:
                continue
            yield name, param

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def criterion(self, y, d):
        pass


class StatsManager(object):
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss_G, loss_D_A, loss_D_B):
        """Accumulate statistics

        Arguments:
            loss_G (float): the loss_G obtained during the last update.
            loss_G_A (float): the loss_D_A obtained during the last update.
            loss_d_B (float): the loss_D_B obtained during the last update.
        """
        loss = loss_G + loss_D_A + loss_D_B
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / self.number_update


class Experiment(object):
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """
    def __init__(self, net, train_set, val_set, optimizer_G, optimizer_D_A, optimizer_D_B, stats_manager,
                 output_dir=None, batch_size=16, perform_validation_during_training=False):
        
        self.real_target = torch.autograd.Variable(torch.Tensor(batch_size).fill_(1.0),
                                                   requires_grad=False).to(net.device)
        self.fake_target = torch.autograd.Variable(torch.Tensor(batch_size).fill_(0.0),
                                                   requires_grad=False).to(net.device)
        
        # Define data loaders
        train_loader = td.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                     drop_last=True, pin_memory=True)
        val_loader = td.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True)

        # Initialize history
        history = []

        # Define checkpoint paths
        if output_dir is None:
            output_dir = 'experiment_{}'.format(time.time())
        os.makedirs(output_dir, exist_ok=True)
        checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
        config_path = os.path.join(output_dir, "config.txt")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k is not 'self'}
        self.__dict__.update(locs)

        # Load checkpoint and check compatibility
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                if f.read()[:-1] != repr(self):
                    raise ValueError(
                        "Cannot create this experiment: "
                        "I found a checkpoint conflicting with the current setting.")
            self.load()
        else:
            self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def setting(self):
        """Returns the setting of the experiment."""
        return {'Net': self.net,
#                 'TrainSet': self.train_set,
#                 'ValSet': self.val_set,
                'Optimizer_G': self.optimizer_G,
                'Optimizer_D_A': self.optimizer_D_A,
                'Optimizer_D_B': self.optimizer_D_B,
                'StatsManager': self.stats_manager,
                'BatchSize': self.batch_size,
                'PerformValidationDuringTraining': self.perform_validation_during_training}

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.setting().items():
            string += '{}({})\n'.format(key, val)
        return string

    def state_dict(self):
        """Returns the current state of the experiment."""
        return {'Net': self.net.state_dict(),
                'Optimizer_G': self.optimizer_G.state_dict(),
                'Optimizer_D_A': self.optimizer_D_A.state_dict(),
                'Optimizer_D_B': self.optimizer_D_B.state_dict(),
                'History': self.history}

    def load_state_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer_G.load_state_dict(checkpoint['Optimizer_G'])
        self.optimizer_D_A.load_state_dict(checkpoint['Optimizer_D_A'])
        self.optimizer_D_B.load_state_dict(checkpoint['Optimizer_D_B'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        for state in self.optimizer_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.net.device)
        for state in self.optimizer_D_A.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.net.device)
        for state in self.optimizer_D_B.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.net.device)

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        torch.save(self.state_dict(), self.checkpoint_path)
        with open(self.config_path, 'w') as f:
            print(self, file=f)

    def load(self):
        """Loads the experiment from the last checkpoint saved on disk."""
        checkpoint = torch.load(self.checkpoint_path,
                                map_location=self.net.device)
        self.load_state_dict(checkpoint)
        del checkpoint

    def run(self, num_epochs, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """
        self.net.train()
        self.stats_manager.init()
        start_epoch = self.epoch
        print("Start/Continue training from epoch {}".format(start_epoch))
        if plot is not None:
            plot(self)
        for epoch in range(start_epoch, num_epochs):
            s = time.time()
            self.stats_manager.init()
            for real_a,real_b in self.train_loader:
                real_a, real_b = real_a.to(self.net.device), real_b.to(self.net.device)

                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(b) should equal b if real b is fed
                same_b = self.net.G_A2B(real_b)
                loss_Idt_B = self.net.criterion_identity(same_b, real_b)
                # G_B2A(a) should equal a if real a is fed
                same_a = self.net.G_B2A(real_a)
                loss_Idt_A = self.net.criterion_identity(same_a, real_a)

                # GAN loss
                fake_b = self.net.G_A2B(real_a)
                fake_pred = self.net.D_B(fake_b)
                loss_GAN_A2B = self.net.criterion_GAN(fake_pred, self.real_target)

                fake_a = self.net.G_B2A(real_b)
                fake_pred = self.net.D_A(fake_a)
                loss_GAN_B2A = self.net.criterion_GAN(fake_pred, self.real_target)

                # Cycle loss
                recovered_a = self.net.G_B2A(fake_b)
                loss_cycle_ABA = self.net.criterion_cycle(recovered_a, real_a)

                recovered_b = self.net.G_A2B(fake_a)
                loss_cycle_BAB = self.net.criterion_cycle(recovered_b, real_b)

                # Total loss
                loss_G = loss_Idt_A + loss_Idt_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()
                
                self.optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()

                # Real loss
                real_pred = self.net.D_A(real_a)
                loss_D_real = self.net.criterion_GAN(real_pred, self.real_target)

                # Fake loss
                fake_a = self.net.fake_a_buffer.push_and_pop(fake_a)
                fake_pred = self.net.D_A(fake_a.detach())
                loss_D_fake = self.net.criterion_GAN(fake_pred, self.fake_target)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                real_pred = self.net.D_B(real_b)
                loss_D_real = self.net.criterion_GAN(real_pred, self.real_target)
                
                # Fake loss
                fake_b = self.net.fake_b_buffer.push_and_pop(fake_b)
                fake_pred = self.net.D_B(fake_b.detach())
                loss_D_fake = self.net.criterion_GAN(fake_pred, self.fake_target)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5
                loss_D_B.backward()

                self.optimizer_D_B.step()
                ###################################
                with torch.no_grad():
                    self.stats_manager.accumulate(loss_G.item(), loss_D_A.item(), loss_D_B.item())
            
            LOSS=self.stats_manager.summarize()
            
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                self.history.append(
                    (self.stats_manager.summarize(), self.evaluate()))
            print("Epoch {} [G loss: {:.4f}, D_A loss: {:.4f}, D_B loss: {:.4f}] (Time: {:.2f}s)".format(
                self.epoch, LOSS['G loss'], LOSS['D_A loss'], LOSS['D_B loss'], time.time() - s))
            self.save()
            if plot is not None:
                plot(self)
        print("Finish training for {} epochs".format(num_epochs))

    def evaluate(self):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.net.eval()
        with torch.no_grad():
            for real_a,real_b in self.val_loader:
                real_a, real_b = real_a.to(self.net.device), real_b.to(self.net.device)

                ###### Generators A2B and B2A ######
                self.optimizer_G.zero_grad()

                # Identity loss
                # G_A2B(b) should equal b if real b is fed
                same_b = self.net.G_A2B(real_b)
                loss_Idt_B = self.net.criterion_identity(same_b, real_b)
                # G_B2A(a) should equal a if real a is fed
                same_a = self.net.G_B2A(a)
                loss_Idt_A = self.net.criterion_identity(same_a, real_real_a)

                # GAN loss
                fake_b = self.net.G_A2B(real_a)
                fake_pred = self.net.D_B(fake_b)
                loss_GAN_A2B = self.net.criterion_GAN(fake_pred, self.real_target)

                fake_a = self.net.G_B2A(real_b)
                fake_pred = self.net.D_A(fake_a)
                loss_GAN_B2A = self.net.criterion_GAN(fake_pred, self.real_target)

                # Cycle loss
                recovered_a = self.net.G_B2A(fake_b)
                loss_cycle_ABA = self.net.criterion_cycle(recovered_a, real_a)

                recovered_b = self.net.G_A2B(fake_a)
                loss_cycle_BAB = self.net.criterion_cycle(recovered_b, real_b)

                # Total loss
                loss_G = loss_Idt_A + loss_Idt_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
                loss_G.backward()
                
                self.optimizer_G.step()
                ###################################

                ###### Discriminator A ######
                self.optimizer_D_A.zero_grad()

                # Real loss
                real_pred = self.net.D_A(real_a)
                loss_D_real = self.net.criterion_GAN(real_pred, self.real_target)

                # Fake loss
                fake_a = self.net.fake_a_buffer.push_and_pop(fake_a)
                fake_pred = self.net.D_A(fake_a.detach())
                loss_D_fake = self.net.criterion_GAN(fake_pred, self.fake_target)

                # Total loss
                loss_D_A = (loss_D_real + loss_D_fake)*0.5
                loss_D_A.backward()

                self.optimizer_D_A.step()
                ###################################

                ###### Discriminator B ######
                self.optimizer_D_B.zero_grad()

                # Real loss
                real_pred = self.net.D_B(real_b)
                loss_D_real = self.net.criterion_GAN(real_pred, self.real_target)
                
                # Fake loss
                fake_b = self.net.fake_b_buffer.push_and_pop(fake_b)
                fake_pred = self.net.D_B(fake_b.detach())
                loss_D_fake = self.net.criterion_GAN(fake_pred, self.fake_target)

                # Total loss
                loss_D_B = (loss_D_real + loss_D_fake)*0.5

                self.stats_manager.accumulate(loss_G.item(), loss_D_A.item(), loss_D_B.item())
        self.net.train()
        return self.stats_manager.summarize()
