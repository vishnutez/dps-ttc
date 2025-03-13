from abc import ABC, abstractmethod
import torch

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
        self.l1 = kwargs.get('l1', 0.0)
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):

        if self.noiser.__name__ == 'gaussian': 
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            difference_reshaped = difference.reshape(difference.shape[0], -1)  # Reshape to flatten the image dimensions
            norm = torch.linalg.norm(difference_reshaped, dim=-1)

            norm_exp = kwargs.get('norm_exp', 1)
            if norm_exp == 2:
                print('Doing norm^2')
                norm_power = norm**2  
            else:
                print('Doing norm')
                norm_power = norm 
            norm_grad = torch.autograd.grad(outputs=norm_power.sum(), inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)  
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 0.3)
        self.operator_name = operator.name

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        # Net scaling is computed with respect to the norm^2
        net_scaling = self.scale / 2 / norm
        return x_t, norm, net_scaling
    
@register_conditioning_method(name='ps_semantic')
class PosteriorSamplingSemanticGuid(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.operator_name = operator.name

        self.scale = kwargs.get('scale', 0.3)
        self.sem_guid_scale = kwargs.get('sem_guid_scale', 0.5)

        guid_image = kwargs.get('guid_image', None)  # Pillow image
        self.guid_image = guid_image

        # Load the MTCNN and InceptionResnetV1 models
        self.mtcnn = MTCNN(image_size=256, margin=10, min_face_size=20, device='cuda:0')  # Keep the default values
        self.resnet = InceptionResnetV1(pretrained='vggface2', device='cuda:0').eval()

        # Get cropped and prewhitened image tensor
        with torch.no_grad():
            guid_image_cropped = self.mtcnn(self.guid_image).unsqueeze(0).to('cuda:0')
            print('guid_image_cropped shape:', guid_image_cropped.shape)
            self.guid_image_emb = self.resnet(guid_image_cropped).detach()  # Used for semantic guidance
            print('guid_image_emb shape:', self.guid_image_emb.shape)
    

    def measurement_semantic_guidance(self, x_prev, x_0_hat, measurement, **kwargs):

        print('Adding semantic guidance')
        # print('x_0_hat_cropped shape:', x_0_hat_cropped.shape)
        x_0_hat_emb = self.resnet(x_0_hat)  # Get the embedding of the face
        sem_diff = x_0_hat_emb - self.guid_image_emb  # Reshape to flatten the image dimensions (batch_size, emb_dim)
        # Compute the semantic guidance
        sem_guid_norm = torch.norm(sem_diff, dim=-1) # Compute the norm of the difference

        if self.noiser.__name__ == 'gaussian': 
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement - Ax
            difference_reshaped = difference.reshape(difference.shape[0], -1)  # Reshape to flatten the image dimensions
            measurement_guid_norm = torch.linalg.norm(difference_reshaped, dim=-1)
            norm_exp = kwargs.get('norm_exp', 1)
            if norm_exp == 2:
                print('Doing norm^2')
                norm_power = measurement_guid_norm ** 2  
            else:
                print('Doing norm')
                print('Doing norm2 for semantic guidance')
                norm_power = measurement_guid_norm 

            overall_norm = self.scale * norm_power + self.sem_guid_scale * sem_guid_norm**2

        norm_grad = torch.autograd.grad(outputs=overall_norm.sum(), inputs=x_prev)[0]

        return norm_grad, measurement_guid_norm, sem_guid_norm


    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        # norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, retain_graph=True, **kwargs)
        norm_grad, measurement_error, semantic_error = self.measurement_semantic_guidance(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad
        return x_t, measurement_error, semantic_error 
    
    
@register_conditioning_method(name='ps_anneal')
class PosterorSamplingAnnealing(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.noise_sigma = max(noiser.sigma, 0.05)
        self.scale = kwargs.get('scale', 0.3)
        self.operator_name = operator.name

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        beta_scale = kwargs.get('beta_scale', self.scale)
        anneal = kwargs.get('anneal', 1.0)
        net_scaling =  torch.tensor(beta_scale / (anneal * self.noise_sigma**2)).to(x_t.device)
        norm_power_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, norm_exp=2, **kwargs)  # norm_exp=2 for ps_anneal
        x_t -= net_scaling * norm_power_grad
        return x_t, norm, net_scaling
    
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling

        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
    


