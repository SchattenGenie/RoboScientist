from . import config, optimize_constants
from roboscientist.datasets import equations_utils, equations_base

import numpy as np


import torch
import random
import torch.nn.functional as F


def build_single_batch_from_formulas_list(formulas_list, solver):
    batch_in, batch_out = [], []
    max_len = max([len(f) for f in formulas_list])
    for f in formulas_list:
        f_idx = [solver._token2ind[t] for t in f]
        padding = [solver._token2ind[config.PADDING]] * (max_len - len(f_idx))
        batch_in.append([solver._token2ind[config.START_OF_SEQUENCE]] + f_idx + padding)
        batch_out.append(f_idx + [solver._token2ind[config.END_OF_SEQUENCE]] + padding)
    # we transpose here to make it compatible with LSTM input
    return torch.LongTensor(batch_in).T.contiguous().to(solver.params.device), torch.LongTensor(batch_out).T.contiguous(

    ).to(solver.params.device)


def build_ordered_batches(formula_file, solver):
    formulas = []
    Xs = []
    ys = []
    with open(formula_file) as f:
        for line in f:
            f_to_eval = line.split()
            formulas.append(line.split())
            # TODO(julia): add formula evaluation here to add correct Xs and ys after ensuring formulas correctness
            f_to_eval = [float(x) if x in solver.params.float_constants else x for x in f_to_eval]
            f_to_eval = equations_utils.infix_to_expr(f_to_eval)
            f_to_eval = equations_base.Equation(f_to_eval)
            constants = optimize_constants.optimize_constants(f_to_eval, solver.xs, solver.ys)
            y = f_to_eval.func(solver.xs.reshape(-1, 1), constants)
            Xs.append(solver.xs.reshape(-1, 1))
            ys.append(y.reshape(-1, 1))

    batches = []
    order = range(len(formulas))  # This will be necessary for reconstruction
    sorted_formulas, sorted_Xs, sorted_ys, order = zip(*sorted(zip(formulas, Xs, ys, order), key=lambda x: len(x[0])))
    for batch_ind in range((len(sorted_formulas) + solver.params.batch_size - 1) // solver.params.batch_size):
        batch_formulas = sorted_formulas[batch_ind * solver.params.batch_size:(batch_ind + 1) * solver.params.batch_size]
        batch_Xs = sorted_Xs[batch_ind * solver.params.batch_size:(batch_ind + 1) * solver.params.batch_size]
        batch_ys = sorted_ys[batch_ind * solver.params.batch_size:(batch_ind + 1) * solver.params.batch_size]
        batches.append((build_single_batch_from_formulas_list(batch_formulas, solver),
                        np.array(batch_Xs), np.array(batch_ys)))
    return batches, order


# Reconstruction error + KL divergence
def _loss_function(logits, targets, mu, logsigma, model):
    reconstruction_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=model._token2ind[config.PADDING], reduction='none').view(targets.size())
    KLD = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp()) / len(mu)
    # reconstruction_loss: (formula_dim, batch_size), so we take sum over all tokens and mean over formulas in batch
    return reconstruction_loss.sum(dim=0).mean(), KLD


def _evaluate(model, batches):
    model.eval()
    kl_losses, rec_losses = [], []
    with torch.no_grad():
        for (inputs, targets), Xs, ys in batches:
            logits, mu, logsigma, z = model(inputs, Xs, ys)
            rec, kl = _loss_function(logits, targets, mu, logsigma, model)
            kl_losses.append(kl.item())
            rec_losses.append(rec.item())
    loss = np.mean(rec_losses)
    return loss, np.mean(rec_losses), np.mean(kl_losses)


def run_epoch(model, optimizer, train_batches, valid_batches, kl_coef=0.01):
    kl_losses, rec_losses, losses = [], [], []
    model.train()
    indices = list(range(len(train_batches)))
    random.shuffle(indices)
    for i, idx in enumerate(indices):
        optimizer.zero_grad()
        # TODO(julia): sample Xs, ys from formula
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

    valid_losses = _evaluate(model, valid_batches)
    print('\t[validation] loss: %0.3f, rec loss: %0.3f, kl: %0.3f' % valid_losses)


def pretrain(n_pretrain_steps, model, optimizer, pretrain_batches, pretrain_val_batches, kl_coef):
    for step in range(n_pretrain_steps):
        run_epoch(model, optimizer, pretrain_batches, pretrain_val_batches, kl_coef)
