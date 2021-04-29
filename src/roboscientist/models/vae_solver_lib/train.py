from . import config, optimize_constants
from roboscientist.datasets import equations_utils, equations_base

import numpy as np


import torch
import random
import torch.nn.functional as F


def build_single_batch_from_formulas_list(formulas_list, solver, batch_Xs, batch_ys):
    batch_in, batch_out = [], []
    max_len = max([len(f) for f in formulas_list])
    t_c = 0
    for i, f in enumerate(formulas_list):
        try:
            f_idx = [solver._token2ind[t] for t in f]
            padding = [solver._token2ind[config.PADDING]] * (max_len - len(f_idx))
            batch_in.append([solver._token2ind[config.START_OF_SEQUENCE]] + f_idx + padding)
            batch_out.append(f_idx + [solver._token2ind[config.END_OF_SEQUENCE]] + padding)
        except:
            t_c +=1
            batch_Xs.pop(i)
            batch_ys.pop(i)
    print(f'Failed to add formula to single batch {t_c}/{len(formulas_list)}')
    # we transpose here to make it compatible with LSTM input
    return (torch.LongTensor(batch_in).T.contiguous().to(solver.params.device), \
           torch.LongTensor(batch_out).T.contiguous().to(solver.params.device)), \
           np.array(batch_Xs), np.array(batch_ys)


def build_ordered_batches(formula_file, solver):
    formulas = []
    Xs = []
    ys = []
    t_c = 0
    total_count = 0
    with open(formula_file) as f:
        for line in f:
            total_count += 1
            try:
                f_to_eval = line.split()
                formulas.append(line.split())
                f_to_eval = [float(x) if x in solver.params.float_constants else x for x in f_to_eval]
                f_to_eval = equations_utils.infix_to_expr_with_arities(f_to_eval, solver.params.arities)
                f_to_eval = equations_base.Equation(f_to_eval)
                constants = optimize_constants.optimize_constants(f_to_eval, solver.xs, solver.ys)
                y = f_to_eval.func(solver.xs.reshape(-1, 1), constants)
                Xs.append(solver.xs.reshape(-1, 1))
                ys.append(y.reshape(-1, 1))
            except:
                t_c += 1
    print(f'Failed to add formula to batch {t_c}/{total_count}')

    batches = []
    order = range(len(formulas))  # This will be necessary for reconstruction
    sorted_formulas, sorted_Xs, sorted_ys, order = zip(*sorted(zip(formulas, Xs, ys, order), key=lambda x: len(x[0])))
    for batch_ind in range((len(sorted_formulas) + solver.params.batch_size - 1) // solver.params.batch_size):
        batch_formulas = sorted_formulas[batch_ind * solver.params.batch_size:(batch_ind + 1) * solver.params.batch_size]
        batch_Xs = sorted_Xs[batch_ind * solver.params.batch_size:(batch_ind + 1) * solver.params.batch_size]
        batch_ys = sorted_ys[batch_ind * solver.params.batch_size:(batch_ind + 1) * solver.params.batch_size]
        batches.append((build_single_batch_from_formulas_list(batch_formulas, solver, batch_Xs, batch_ys)))
    return batches, order


# Reconstruction error + KL divergence
def _loss_function(logits, targets, mu, logsigma, model):
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=model._token2ind[config.PADDING], reduction='none').view(targets.size())
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) / len(mu)
    # reconstruction_loss: (formula_dim, batch_size), so we take sum over all tokens and mean over formulas in batch
    return reconstruction_loss.sum(dim=0).mean(), KLD


def _evaluate(model, batches, kl_coef):
    model.eval()
    kl_losses, rec_losses, losses = [], [], []
    with torch.no_grad():
        for (inputs, targets), Xs, ys in batches:
            logits, mu, logsigma, z = model(inputs, Xs, ys)
            rec, kl = _loss_function(logits, targets, mu, logsigma, model)
            kl_losses.append(kl.item())
            rec_losses.append(rec.item())
            losses.append(rec.item() + kl_coef * kl.item())
    return np.mean(losses), np.mean(rec_losses), np.mean(kl_losses)


def run_epoch(model, optimizer, train_batches, valid_batches, kl_coef=0.01):
    kl_losses, rec_losses, losses = [], [], []
    model.train()
    indices = list(range(len(train_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        optimizer.zero_grad()
        (inputs, targets), Xs, ys = train_batches[idx]
        logits, mu, logsigma, z = model(inputs, Xs, ys)
        rec, kl = _loss_function(logits, targets, mu, logsigma, model)
        loss = rec + kl_coef * kl
        loss.backward()
        optimizer.step()
        rec_losses.append(rec.item())
        losses.append(loss.item())
        kl_losses.append(kl.item())

    print('\t[training] batches count: %d' % len(indices))
    print('\t[training] loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % (
        np.mean(losses), np.mean(rec_losses), np.mean(kl_losses)))

    valid_losses = _evaluate(model, valid_batches, kl_coef)
    print('\t[validation] loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % valid_losses)
    train_losses = (np.mean(losses), np.mean(rec_losses), np.mean(kl_losses))
    return train_losses, valid_losses


def pretrain(n_pretrain_steps, model, optimizer, pretrain_batches, pretrain_val_batches, kl_coef):
    for step in range(n_pretrain_steps):
        run_epoch(model, optimizer, pretrain_batches, pretrain_val_batches, kl_coef)
