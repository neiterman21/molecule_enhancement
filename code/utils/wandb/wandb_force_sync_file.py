# because wandb works fine without problems
# a utility file to sync your hard-earned weights
import argparse
from config_parser import ConfigParser
from wandb_straitjacket import StraitjakcetDetached
from shutil import copyfile
import os


parser = argparse.ArgumentParser()
parser.add_argument("--file-to-sync")
parser.add_argument("--config-file")
parser.add_argument("--config-section")

args = parser.parse_args()


configs = ConfigParser.Parse(args.config_file, args.config_section)


straitJacket = StraitjakcetDetached(configs.project, configs.owner, configs.id)
fileName = os.path.basename(args.file_to_sync)
runDir = straitJacket.runDir
copyfile(args.file_to_sync, runDir + "/" + fileName)
straitJacket.Save(args.file_to_sync)
