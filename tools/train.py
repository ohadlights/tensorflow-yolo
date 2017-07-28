import sys
from optparse import OptionParser
from yolo.dataset.text_dataset import TextDataSet
from yolo.net.yolo_tiny_net import YoloTinyNet
from yolo.solver.yolo_solver import YoloSolver

sys.path.append('./')

import yolo
from yolo.utils.process_config import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
  conf_file = str(options.configure)
else:
  print('please sspecify --conf configure filename')
  exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)
dataset = TextDataSet(common_params, dataset_params) #eval(dataset_params['name'])(common_params, dataset_params)
net = YoloTinyNet(common_params, net_params) #eval(net_params['name'])(common_params, net_params)
solver = YoloSolver(dataset, net, common_params, solver_params) #eval(solver_params['name'])(dataset, net, common_params, solver_params)
solver.solve()