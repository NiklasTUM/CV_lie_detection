import os
from dotenv import load_dotenv

this_folder = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(this_folder, "..", "..")
env_file = ROOT_DIR + "/.env"
load_dotenv(env_file)
