from roboscientist.logger.logger import BaseLogger

from heapq import merge
import numpy as np
from sklearn.metrics import mean_squared_error
import wandb


class SingleFormulaLogger(BaseLogger):
    """
    This logger should be used in experiments in which a model runs several epochs to learn a single formula.
    Before the learning process starts, initialize the logger:

        logger = SingleFormulaLogger(project_name='project_name',
                                     experiment_name='experiment_name',
                                     experiment_config={})

    Then each time new candidate equations are generated, add them to the logger:

        logger.log_metrics(reference_problem, equations)

    At the end of each epoch commit the metrics:

        logger.commit_metrics()

        If you want, you can add your own metrics:

            logger.commit_metrics(custom_log)

    An example of using this logger can be found in examples/logger/single_formula_logger_example.ipynb
    """
    def __init__(self, project_name, experiment_name, experiment_config,
                 n_best_equations_to_store=500, evaluation_dataset_size=1000):
        """
        :param project_name: str. Project name. Will be used as wandb project. One project can have multiple
        experiments.
        :param experiment_name: str. Experiment name. Will be used as wandb name. Each experiment should have a unique
        experiment name.
        :param experiment_config: Dict[str, Any]. A dictionary containing useful information about the experiment,
        e.g. model, correct formula.
        :param n_best_equations_to_store: int. Number of best equations to store. Default 500.
        :param evaluation_dataset_size: int. Size of X - a dataset sampled from a reference problem domain to
        evaluate candidate equations. Default 1000.
        """
        super().__init__()
        self._project = project_name
        self._experiment_name = experiment_name
        wandb.init(project=self._project, name=experiment_name)

        config_table = wandb.Table(columns=[*sorted(experiment_config.keys())])
        config_table.add_data(*[experiment_config[k] for k in sorted(experiment_config.keys())])
        wandb.log({'experiment config': config_table})

        self._best_formulas_table = []
        self._epoch_best_formulas_table = []

        self._ordered_best_formulas = []
        self._ordered_best_mses = []
        self._ordered_current_epoch_best_formulas = []
        self._ordered_current_epoch_best_mses = []

        self._n_best_equations_to_store = n_best_equations_to_store
        self._evaluation_dataset_size = evaluation_dataset_size
        self._current_epoch = 1

    def log_metrics(self, reference_problem, equations):
        """
        :param reference_problem: BaseProblem. Desired Solution.
        :param equations: list. New candidate equations generated by the model
        """
        super().log_metrics(reference_problem, equations)

        X, y_true = reference_problem.dataset

        mses = [mean_squared_error(y_true, eq.func(X)) for eq in equations]
        str_equations = [str(eq) for eq in equations]

        # sort in terms of mse
        ordered_equation_mse_pairs = sorted(zip(str_equations, mses), key=lambda x: x[1])

        # update epoch best formulas/mses
        current_epoch_ordered_equation_mse_pairs = list(merge(
            zip(self._ordered_current_epoch_best_formulas, self._ordered_current_epoch_best_mses),
            ordered_equation_mse_pairs,
            key=lambda x: x[1]))
        self._ordered_current_epoch_best_mses = [x[1] for x in current_epoch_ordered_equation_mse_pairs]
        self._ordered_current_epoch_best_formulas = [x[0] for x in current_epoch_ordered_equation_mse_pairs]

        # update best formulas/mses
        best_ordered_equation_mse_pairs = list(merge(
            zip(self._ordered_best_formulas, self._ordered_best_mses),
            ordered_equation_mse_pairs,
            key=lambda x: x[1]))[:self._n_best_equations_to_store]
        self._ordered_best_mses = [x[1] for x in best_ordered_equation_mse_pairs]
        self._ordered_best_formulas = [x[0] for x in best_ordered_equation_mse_pairs]

    def commit_metrics(self, custom_log=None):
        """
        :param custom_log: dict. Default None. Specify if some custom metrics must be logged.
        :return:
        """
        super().commit_metrics()

        wandb_log = {
            'epoch': self._current_epoch,
        }

        for count in [1, 10, 25, 50, 100, 250, 500]:
            wandb_log[f'epoch_mean_mse_top_{count}'] = np.mean(self._ordered_current_epoch_best_mses[:count])
            wandb_log[f'best_mean_mse_top_{count}'] = np.mean(self._ordered_best_mses[:count])
            if np.mean(self._ordered_current_epoch_best_mses[:count]) != 0:
                wandb_log[f'epoch_log_mean_mse_top_{count}'] = np.log(
                    np.mean(self._ordered_current_epoch_best_mses[:count]))
            else:
                wandb_log[f'epoch_log_mean_mse_top_{count}'] = -100
            if np.mean(self._ordered_best_mses[:count]) != 0:
                wandb_log[f'best_log_mean_mse_top_{count}'] = np.log(
                    np.mean(self._ordered_best_mses[:count]))
            else:
                wandb_log[f'best_log_mean_mse_top_{count}'] = -100

        n_formulas_to_show = 10
        for r, (f, m) in enumerate(zip(self._ordered_best_formulas[:n_formulas_to_show],
                                       self._ordered_best_mses[:n_formulas_to_show])):
            self._best_formulas_table.append([self._current_epoch, r + 1, f, m])
        wandb_log[f'Best formulas'] = wandb.Table(data=self._best_formulas_table,
                                                  columns=['epoch', 'rank', 'formula', 'mse'])

        for r, (f, m) in enumerate(zip(self._ordered_current_epoch_best_formulas[:n_formulas_to_show],
                                       self._ordered_current_epoch_best_mses[:n_formulas_to_show])):
            self._epoch_best_formulas_table.append([self._current_epoch, r + 1, f, m])
        wandb_log[f'Epoch best formulas'] = wandb.Table(data=self._epoch_best_formulas_table,
                                                        columns=['epoch', 'rank', 'formula', 'mse'])

        if custom_log is not None:
            wandb_log.update(custom_log)

        wandb.log(wandb_log)

        self._current_epoch += 1

        self._ordered_current_epoch_best_formulas = []
        self._ordered_current_epoch_best_mses = []
