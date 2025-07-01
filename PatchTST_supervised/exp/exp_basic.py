import os
import torch

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def train(self, setting):
        """
        Implement training loop here.
        The 'setting' param can be used for logging or checkpoint naming.
        """
        pass

    def vali(self, setting):
        """
        Validation step if needed.
        """
        pass

    def test(self, setting, test=0):
        """
        Testing step.
        """
        pass

    def predict(self, setting, load=False):
        """
        Prediction step.
        """
        pass
