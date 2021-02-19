import comet_ml
import os
import numpy as np
import torch

from roboscientist.models import brute_force_solver
from roboscientist.datasets import PolynomialDataset

from tqdm import tqdm

def train(
        dataset_name, model_name, device, seed,
        n_epoches, batch_size, data_root, parameters_path,
        output_root, logger, project, workspace, quiet=False
):
  log = get_logger(logger, output_root, project, workspace)
  if parameters_path is None:
    model_root, = ensure_directories(output_root, 'models/')
    parameters_path = os.path.join(
      model_root,
      '{dataset}-{model}.pt'.format(dataset=dataset_name, model=model_name)
    )

  device = torch.device(device)
  torch.manual_seed(seed)

  dataset = available.datasets[dataset_name].Dataset(data_root)
  trainloader = torch.utils.data.DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True, )

  model = available.models[model_name].Model(dataset.train_set[0][0].shape).to(device)
  clf = DigitClassifier(model, device=device)

  if not quiet:
    arguments = dict(seed=seed, n_epoches=n_epoches, batch_size=batch_size)
    print('Training {model}({arguments}) on {dataset}'.format(
      model=model_name,
      dataset=dataset_name,
      arguments=', '.join([ '%s = %s' % (k, v) for k, v in arguments.items() ]),
    ))

  losses = clf.fit(trainloader, n_epoches=n_epoches, progress=None if quiet else tqdm)

  torch.save(clf.classifier.state_dict(), parameters_path)
  if not quiet:
    print('  saving to {parameters_path}'.format(parameters_path=parameters_path))
  log.log_losses(dataset_name, model_name, losses)

def test(
  dataset_name, model_name,
  device, batch_size,
  data_root, parameters_path, output_root,
  logger, project, workspace,
):
  log = get_logger(logger, output_root, project, workspace)
  if parameters_path is None:
    model_root, = ensure_directories(output_root, 'models/')
    parameters_path = os.path.join(
      model_root,
      '{dataset}-{model}.pt'.format(dataset=dataset_name, model=model_name)
    )

  device = torch.device(device)

  dataset = available.datasets[dataset_name].Dataset(data_root)
  trainloader = torch.utils.data.DataLoader(dataset.train_set, batch_size=batch_size, shuffle=True)
  testloader = torch.utils.data.DataLoader(dataset.test_set, batch_size=batch_size, shuffle=True)

  model = available.models[model_name].Model(dataset.train_set[0][0].shape).to(device)
  clf = DigitClassifier(model, device=device)

  state_dict = torch.load(parameters_path)
  clf.classifier.load_state_dict(state_dict)

  predictions_train, true_train = clf.predict(trainloader)
  predictions_test, true_test = clf.predict(testloader)

  accuracy_train = np.mean(np.argmax(predictions_train, axis=1) == true_train)
  accuracy_test = np.mean(np.argmax(predictions_test, axis=1) == true_test)

  log.log_metrics(dataset_name, model_name, accuracy_train=accuracy_train, accuracy_test=accuracy_test)

def download_data(data_root):
  for dataset_name in available.datasets:
    available.datasets[dataset_name].Dataset(data_root)


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(prog='digits')
  subparsers = parser.add_subparsers(title='actions', description='action to perform', dest='action')
  parser.add_argument('dataset', type=str, choices=available.datasets, default='mnist')
  parser.add_argument('model', type=str, choices=available.models, default='logreg')
  parser.add_argument(
    '--dataroot', type=str, default='data/',
    help='directory from which data is read or to which data will be downloaded if absent, '
         'by default, output/models/<dataset>-<model>.pt'
  )
  parser.add_argument('--output-root', type=str, default='output/', help='root directory to write various statistics to')
  parser.add_argument('--device', type=str, default='cpu', help='device in torch format')
  parser.add_argument('--batch-size', type=int, default=32)
  parser.add_argument('--logger', type=str, choices=['local', 'comet'], default='local')
  parser.add_argument('--project', type=str, default=None, help='project name for comet logger, None by default')
  parser.add_argument('--workspace', type=str, default=None, help='workspace for comet logger, None by default')
  parser.add_argument('--quiet', type=bool, default=False)

  train_parser = subparsers.add_parser('train')
  train_parser.add_argument('--seed', type=int, default=111222333)
  train_parser.add_argument('--epoches', type=int, default=16)
  train_parser.add_argument(
    '--parameters-path', type=str, default=None,
    help='path to store network parameters after training'
  )

  test_parser = subparsers.add_parser('test')
  test_parser.add_argument(
    '--parameters-path', type=str, default=None,
    help='path that contains network parameters to apply, '
         'by default, output/models/<dataset>-<model>.pt'
  )

  test_parser = subparsers.add_parser('download')

  args = parser.parse_args()

  if args.action == 'train':
    train(
      model_name=args.model, dataset_name=args.dataset,
      device=args.device, seed=args.seed,
      n_epoches=args.epoches, batch_size=args.batch_size,
      data_root=args.dataroot, parameters_path=args.parameters_path, output_root=args.output_root,
      logger=args.logger, project=args.project, workspace=args.workspace,
      quiet=args.quiet
    )
  elif args.action == 'test':
    test(
      model_name=args.model, dataset_name=args.dataset,
      device=args.device,
      batch_size=args.batch_size,
      data_root=args.dataroot, parameters_path=args.parameters_path, output_root=args.output_root,
      logger=args.logger, project=args.project, workspace=args.workspace,
    )
  elif args.action == 'download':
    download_data(data_root=args.dataroot)
  else:
    raise ValueError("Unknown option {}".format(args.action))
