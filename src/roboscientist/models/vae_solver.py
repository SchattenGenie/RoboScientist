from roboscientist.datasets import equations_generation, Dataset, equations_utils, equations_base
from .solver_base import BaseSolver
from .vae_solver_lib import config, model, train, optimize_constants

from sklearn.metrics import mean_squared_error

import torch

from collections import deque, namedtuple
import numpy as np


VAESolverParams = namedtuple(
    'VAESolverParams', [
        # model parameters
        'model_params',
        # formula parameters
        'max_formula_length',
        'max_degree',
        'functions',
        'constants',
        # training parameters
        'n_pretrain_steps',
        'batch_size',
        'n_pretrain_formulas',
        'create_pretrain_dataset',
        'kl_coef',
        'device',
        'learning_rate',
        'betas',
        'use_n_last_steps',
        'percentile',
        'n_formulas_to_sample',
        'add_noise_to_model_params',
        'noise_coef',
        'add_noise_every_n_steps',
        # files
        'file_to_sample',
        'pretrain_train_file',
        'pretrain_val_file',
        # specific settings
        'no_retrain',
        'continue_training_on_pretrain_dataset',
        # data
        'xs',
        'ys',
    ])
VAESolverParams.__new__.__defaults__ = (
    {'token_embedding_dim': 128, 'hidden_dim': 128, 'encoder_layers_cnt': 1,
     'decoder_layers_cnt': 1, 'latent_dim':  8, 'x_dim': 1},  # model_params
    15,  # max_formula_length
    2,  # max_degree
    ['sin', 'cos', 'Add', 'Mul'],  # functions
    [],  # constants
    50,  # n_pretrain_steps
    256,  # batch_size
    20000,  # n_pretrain_formulas
    False,  # create_pretrain_dataset
    0.2,  # kl_coef
    torch.device("cuda:0"),  # device
    0.0005,  # learning_rate
    (0.5, 0.999),  # betas
    5,  # use_n_last_steps
    20,  # percentile
    2000,  # n_formulas_to_sample
    False,  # add_noise_to_model_params
    0.01,  # noise_coef
    5,  # add_noise_every_n_steps
    'sample',  # file_to_sample
    'train',  # pretrain_train_file
    'val',  # pretrain_val_file
    False,  # no_retrain
    False,  # continue_training_on_pretrain_dataset
    np.linspace(0.1, 1, 100),  # xs
    np.zeros(100),  # ys
)


class VAESolver(BaseSolver):
    def __init__(self, logger, solver_params=None):
        super().__init__(logger)

        if solver_params is None:
            solver_params = VAESolverParams()
        self.params = solver_params

        # TODO(julia): "Symbol('x0')" -> a better way to do this + adapt for multiple variables
        self._ind2token = list(self.params.functions) + [str(c) for c in self.params.constants] + \
                         [config.START_OF_SEQUENCE, config.END_OF_SEQUENCE, config.PADDING] + \
                         ["Symbol('x0')"]
        self._token2ind = {t: i for i, t in enumerate(self._ind2token)}

        if self.params.create_pretrain_dataset:
            self._create_pretrain_dataset()

        self.stats = FormulaStatistics(use_n_last_steps=self.params.use_n_last_steps,
                                       percentile=self.params.percentile)

        model_params = model.ModelParams(vocab_size=len(self._ind2token), device=self.params.device,
                                         **self.params.model_params)
        self.model = model.FormulaVARE(model_params, self._ind2token, self._token2ind)
        self.model.to(self.params.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate,
                                          betas=self.params.betas)

        self.xs = self.params.xs
        self.ys = self.params.ys

        cond_x = np.copy(self.params.xs)
        cond_y = np.copy(self.params.ys)
        if len(self.params.xs.shape) == 1:
            cond_x = cond_x.reshape(-1, 1)
        cond_y = cond_y.reshape(-1, 1)
        self.cond_x = np.repeat(cond_x.reshape(1, -1, 1), self.params.n_formulas_to_sample, axis=0)
        self.cond_y = np.repeat(cond_y.reshape(1, -1, 1), self.params.n_formulas_to_sample, axis=0)

        self.pretrain_batches, _ = train.build_ordered_batches(formula_file='train', batch_size=self.params.batch_size,
                                                       device=self.params.device, real_X=self.params.xs,
                                                       real_y=self.params.ys,
                                                       token2ind=self._token2ind)
        self.valid_batches, _ = train.build_ordered_batches(formula_file='val', batch_size=self.params.batch_size,
                                                       device=self.params.device, real_X=self.params.xs,
                                                       real_y=self.params.ys,
                                                       token2ind=self._token2ind)
        train.pretrain(n_pretrain_steps=self.params.n_pretrain_steps, model=self.model, optimizer=self.optimizer,
                       pretrain_batches=self.pretrain_batches, pretrain_val_batches=self.valid_batches,
                       kl_coef=self.params.kl_coef)

    def _training_step(self, equation, epoch) -> Dataset:
        self.stats.clear_the_oldest_step()

        noises = self._maybe_add_noise_to_model_params(epoch)

        sample_res = self.model.sample(self.params.n_formulas_to_sample, self.params.max_formula_length,
                                       self.params.file_to_sample,
                                       Xs=self.cond_x,
                                       ys=self.cond_y, ensure_valid=False)

        self._maybe_remove_noise_from_model_params(epoch, noises)

        valid_formulas = []
        valid_equations = []
        valid_mses = []
        with open(self.params.file_to_sample) as f:
            for line in f:
                try:
                    f_to_eval = equations_utils.infix_to_expr(line.strip().split())
                    f_to_eval = equations_base.Equation(f_to_eval)
                    # constants = optimize_constants.optimize_constants(f_to_eval, self.xs, self.ys)
                    # TODO(julia): fix optimize constants here
                    constants = None
                    # TODO(julia): now it only works for variables, no constants
                    y = f_to_eval.func(self.xs.reshape(-1, 1), constants)
                    valid_formulas.append(line.strip())
                    valid_mses.append(mean_squared_error(y, self.ys))
                    valid_equations.append(f_to_eval)
                except:
                    continue

        self.stats.save_best_samples(sampled_mses=valid_mses, sampled_formulas=valid_formulas)

        self.stats.write_last_n_to_file(self.params.file_to_sample)

        train_batches, _ = train.build_ordered_batches(self.params.file_to_sample, self.params.batch_size,
                                                       device=self.params.device,
                                                       real_X=self.xs, real_y=self.ys, token2ind=self._token2ind)

        if not self.params.no_retrain:
            train.run_epoch(self.model, self.optimizer, train_batches, train_batches, kl_coef=self.params.kl_coef)
        if self.params.continue_training_on_pretrain_dataset:
            train.pretrain(n_pretrain_steps=1, model=self.model, optimizer=self.optimizer,
                           pretrain_batches=train_batches, pretrain_val_batches=self.valid_batches,
                           kl_coef=self.params.kl_coef)

        # TODO(julia) add active learning
        return Dataset(valid_equations)

    def _create_pretrain_dataset(self):
        self._pretrain_formulas = [
            equations_generation.generate_random_equation_from_settings({
                'functions': self.params.functions, 'constants': self.params.constants},
            max_degree=self.params.max_degree, return_graph_infix=True) for _ in range(self.params.n_pretrain_formulas)]

        self._pretrain_formulas_val = [
            equations_generation.generate_random_equation_from_settings({
                'functions': self.params.functions, 'constants': self.params.constants},
                max_degree=self.params.max_degree, return_graph_infix=True) for _ in range(
                self.params.n_pretrain_formulas)]

        with open(self.params.pretrain_train_file, 'w') as ff:
            for i, D in enumerate(self._pretrain_formulas):
                ff.write(D)
                if i != len(self._pretrain_formulas) - 1:
                    ff.write('\n')

        with open(self.params.pretrain_val_file, 'w') as ff:
            for i, D in enumerate(self._pretrain_formulas_val):
                ff.write(D)
                if i != len(self._pretrain_formulas_val) - 1:
                    ff.write('\n')

    def _maybe_add_noise_to_model_params(self, epoch):
        noises = []
        if self.params.add_noise_to_model_params and epoch % self.params.add_noise_every_n_steps == 1:
            with torch.no_grad():
                for param in self.model.parameters():
                    noise = torch.randn(
                        param.size()).to(self.params.device) * self.params.noise_coef * torch.norm(param).to(
                        self.params.device)
                    param.add_(noise)
                    noises.append(noise)
        return noises

    def _maybe_remove_noise_from_model_params(self, epoch, noises):
        noises = noises[::-1]
        if self.params.add_noise_to_model_params and epoch % self.params.add_noise_every_n_steps == 1:
            with torch.no_grad():
                for param in self.model.parameters():
                    noise = noises.pop()
                    param.add_(-noise)


class FormulaStatistics:
    def __init__(self, use_n_last_steps, percentile):
        self.reconstructed_formulas = []
        self.last_n_best_formulas = []
        self.last_n_best_mses = []
        self.last_n_best_sizes = deque([0] * use_n_last_steps, maxlen=use_n_last_steps)
        self.the_best_formulas = []
        self.the_best_mses = []
        self.percentile = percentile

    def clear_the_oldest_step(self):
        s = self.last_n_best_sizes.popleft()
        self.last_n_best_formulas = self.last_n_best_formulas[s:]
        self.last_n_best_mses = self.last_n_best_mses[s:]

    def save_best_samples(self, sampled_mses, sampled_formulas):
        mse_threshold = np.nanpercentile(sampled_mses + self.last_n_best_mses, self.percentile)
        epoch_best_mses = [x for x in sampled_mses if x < mse_threshold]
        epoch_best_formulas = [
            sampled_formulas[i] for i in range(len(sampled_formulas)) if sampled_mses[i] < mse_threshold]
        assert len(epoch_best_mses) == len(epoch_best_formulas)

        self.last_n_best_sizes.append(len(epoch_best_formulas))
        self.last_n_best_mses += epoch_best_mses
        self.last_n_best_formulas += epoch_best_formulas
        self._update_the_best_formulas(epoch_best_formulas=epoch_best_formulas, epoch_best_mses=epoch_best_mses)

    def _update_the_best_formulas(self, epoch_best_formulas, epoch_best_mses):
        self.the_best_formulas += epoch_best_formulas
        self.the_best_mses += epoch_best_mses

        the_best_pairs = sorted(zip(self.the_best_mses, self.the_best_formulas))[:200]
        used_formulas = set()
        self.the_best_formulas = []
        self.the_best_mses = []
        for i in range(len(the_best_pairs)):
            if the_best_pairs[i][1] not in used_formulas:
                self.the_best_formulas.append(the_best_pairs[i][1])
                self.the_best_mses.append(the_best_pairs[i][0])
            used_formulas.add(the_best_pairs[i][1])

    def write_last_n_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write('\n'.join(self.last_n_best_formulas))
