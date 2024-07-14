import torch
import sys, os, time
import gin
gin.enter_interactive_mode()
cwd = os.getcwd()
os.makedirs(cwd+'/model_files', exist_ok=True)
sys.path.append(cwd)
sys.path.append(cwd+'/inference_env')
sys.path.append(cwd+'/inference_env/dependencies')
from inference_env.dependencies.spiralnet import instantiate_spiralnet
from inference_env.models.decoders import DummyDecoder, LSTMClassifier
from inference_env.models.encoders import DummyAudioEncoder
dummy_audio_enc = DummyAudioEncoder()
dummy_dec = LSTMClassifier()
spiralnet,_, _  = instantiate_spiralnet(nr_layers=8, output_dim = 128, motion = 'hands')
dummy_audio_enc = torch.jit.script(dummy_audio_enc)
dummy_dec = torch.jit.script(dummy_dec)
torch.jit.save(dummy_audio_enc, cwd+'/model_files/dummy_audio_enc.pt')
torch.jit.save(dummy_dec, cwd+'/model_files/dummy_dec.pt')
scripted_model = torch.jit.script(spiralnet)
# Now you can save this scripted model to a file
torch.jit.save(scripted_model, cwd+'/model_files/spiralnet_test.pt')
