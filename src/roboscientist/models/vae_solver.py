from roboscientist.models.vae_solver_utils import experiments, generate_pretrain_dataset


class VAESolver():
    # TODO(Julia): adapt for BaseSolver format
    # TODO(Julia): use common dataset, instead of formula_config
    def __init__(self, logger, max_time=1, *args, **kwargs):
        pass

    def solve(self, xs, ys, formula, epochs=100):
        model_params = {'token_embedding_dim': 128, 'hidden_dim': 128,
                        'encoder_layers_cnt': 1, 'decoder_layers_cnt': 1, 'latent_dim': 8}

        generate_pretrain_dataset.generate_pretrain_dataset(20000, 13, 'train')
        generate_pretrain_dataset.generate_pretrain_dataset(10000, 13, 'val')

        experiments.exp_generative_train(xs=xs, ys=ys, formula=formula, train_file='train', val_file='val',
                                         percentile=20,
                                         max_len=13, epochs=epochs, model_conf_params=model_params, n_pretrain_steps=50,
                                         use_n_last_steps=5, n_formulas_to_sample=6000, add_noise_to_model_params=True,
                                         noise_to_model_params_weight=0.005, add_noise_every_n_steps=3,
                                         no_retrain=False, continue_training_on_train_dataset=False, kl_coef=0.2)

        return []
