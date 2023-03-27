import numpy as np
import math
import torch
import matplotlib.pyplot as plt

c=10
eps_0 = 1/c
mu_0 = 1/c

### functions to generate points ###
def generate_xz_grid_tensor(max_coord, grid_length: int):
    '''Generate a grid of x,z co-ordinated in range (-max_coord, max_coord). Number of points is grid_length^2'''

    x = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)
    z = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)

    return torch.meshgrid(x, z, indexing='xy')

def generate_xzt_grid_tensor(max_coord, time_range, grid_length: int):

    x = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)
    z = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)
    t = torch.linspace(0, time_range, steps=grid_length, requires_grad=True)

    return torch.meshgrid(x,z,t, indexing='xy')

def generate_xyz_grid_tensor(max_coord, grid_length: int):

    x = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)
    y = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)
    z = torch.linspace(-max_coord, max_coord, steps=grid_length, requires_grad=True)

    return torch.meshgrid(x,y,z, indexing='xy')

def xz_to_rtheta(x,z):
    '''Converts xz points into a plane of constant phi r theta co-ords'''

    r = torch.sqrt(x**2 + z**2)
    theta = torch.atan2(x, z)
    theta = torch.where(theta>0, theta, 2*math.pi - torch.abs(theta)) #converts the (-pi, pi) interval to a (0,2*pi) interval

    return r,theta

def rtheta_to_xz(r,theta):
    
    x = r*torch.sin(theta)
    z = r*torch.cos(theta)

    return x,z

def generate_rtheta_tensor(min_radius, max_radius, grid_length: int):
    '''Generates r values on (min_radius, max_radius) interval and theta on (0, 2*pi). Will not produce a nice grid in xz plane,
       but will meshgrid these two intervals together.'''

    r = torch.linspace(min_radius, max_radius, steps=grid_length, requires_grad=True)
    theta = torch.linspace(0.01, 2*math.pi - 0.01, steps=grid_length, requires_grad=True) #little incremement at boundary to stop pole

    return torch.meshgrid(r, theta, indexing='xy')

def generate_rtheta_t_tensor(min_radius, max_radius, time_range, grid_length: int):
    '''Generates r values on (min_radius, max_radius) interval and theta on (0, 2*pi), and t on (0,time_range). 
       Will not produce a nice grid in xz plane, but will meshgrid these two intervals together.'''

    r = torch.linspace(min_radius, max_radius, steps=grid_length, requires_grad=True)
    theta = torch.linspace(0.01, 2*math.pi - 0.01, steps=grid_length, requires_grad=True)
    t = torch.linspace(0, time_range, steps=grid_length, requires_grad=True)

    return torch.meshgrid(r,theta,t, indexing='xy')

def generate_rtheta_grid_tensor(max_coord, grid_length: int):
    '''Generate an xz plane grid from -max_coord to max_coord, expressed in r and theta co-ordinates'''

    if grid_length%2!=0:
        print('Warning: odd number grid length leads to r=0, thus divergences in electric field')

    x,z = generate_xz_grid_tensor(max_coord, grid_length)

    r = torch.sqrt(x**2 + z**2)
    theta = torch.atan2(x, z)
    theta = torch.where(theta>0, theta, 2*math.pi - torch.abs(theta)) #converts the (-pi, pi) interval to a (0,2*pi) interval

    ### how to implement tolerance here without messing up dimensions??
    ### I don't need to, just when plotting avoid plotting points near origin

    return r, theta

def generate_rtheta_t_grid_tensor(max_coord, time_range, grid_length: int):

    if grid_length%2!=0:
        print('Warning: odd number grid length leads to r=0, thus divergences in electric field')

    x,z,t = generate_xzt_grid_tensor(max_coord, time_range, grid_length)

    r = torch.sqrt(x**2 + z**2)
    theta = torch.atan2(x, z)
    theta = torch.where(theta>0, theta, 2*math.pi - torch.abs(theta)) #converts the (-pi, pi) interval to a (0,2*pi) interval

    return r,theta,t

### functions to compute gradients of fields here ###

def compute_rtheta_gradients(E_r, E_theta, r, theta):
    '''Computes dE_r/dr and dE_theta/dtheta gradients'''
    
    dE_r_dr = torch.autograd.grad(outputs=E_r, inputs=r, grad_outputs=torch.ones_like(E_r), retain_graph=True, create_graph=True)[0]
    dE_theta_dtheta = torch.autograd.grad(outputs=E_theta, inputs=theta, grad_outputs=torch.ones_like(E_theta), retain_graph=True, create_graph=True)[0]

    return dE_r_dr, dE_theta_dtheta

def compute_polar_divergence(E_r, E_theta, r, theta):
    '''Computes the divergence of the electric field'''

    dE_r_dr, dE_theta_dtheta = compute_rtheta_gradients(E_r, E_theta, r, theta)

    div = ( (2/r)*E_r + dE_r_dr ) + (1/r)*( (1/torch.tan(theta))*E_theta + dE_theta_dtheta )

    return div

def compute_spherical_divergence(E_r, dE_r_dr, r, E_theta, dE_theta_dtheta, theta):

    div = ( (2/r)*E_r + dE_r_dr ) + (1/r)*( (1/torch.tan(theta))*E_theta + dE_theta_dtheta )

    return div


### functions to generate field here ###
def E_field_HD_tensor(x,y,z,t,p_0,omega, noise=False):
   '''Calculates the real electric field due to a Hertzian dipole at the origin, aligned with z-axis, for a harmonically oscillating charge on the dipole.
      Time input should be either single valued, or with the same dimensions as x,y,z'''
   r = torch.sqrt(x**2 + y**2 + z**2) #the distance from origin at (x,y,z)
   ret_p = p_0 * torch.cos(omega*(t - r/c)) #the dipole moment evaluated at retarded time for c=10
   ret_p_dot = -omega * p_0 * torch.sin(omega*(t-r/c)) #first time deriv
   ret_p_ddot = -omega**2 * p_0 * torch.cos(omega*(t - r/c)) #second time deriv

   radial_bracket = ret_p/(r**3) + ret_p_dot/(c*r**2) #bracket in radial electric field term
   polar_bracket = ret_p/(r**3) + ret_p_dot/(c*r**2) + ret_p_ddot/(c**2*r) #bracket in polar electric field term

   E_x = (2*x*z)/(4*math.pi*eps_0*r**2) * radial_bracket + (x*z)/(4*math.pi*eps_0*r**2) * polar_bracket
   E_y = (2*y*z)/(4*math.pi*eps_0*r**2) * radial_bracket + (y*z)/(4*math.pi*eps_0*r**2) * polar_bracket
   E_z = (2*z**2)/(4*math.pi*eps_0*r**2) * radial_bracket - (x**2 + y**2)/(4*math.pi*eps_0*r**2) * polar_bracket

   if noise:
      E_x = E_x + torch.normal(0.0, 0.1*torch.mean(torch.abs(E_x)))
      E_y = E_y + torch.normal(0.0, 0.1*torch.mean(torch.abs(E_y)))
      E_z = E_z + torch.normal(0.0, 0.1*torch.mean(torch.abs(E_z)))

   return torch.stack((E_x, E_y, E_z), axis=-1)

def E_rtheta_field_HD_tensor(r,theta,t, p_0, omega, noise=False):
   '''Calculates the real electric field's r and theta components'''

   ret_p = p_0 * torch.cos(omega*(t - r/c)) #the dipole moment evaluated at retarded time for c=10
   ret_p_dot = -omega * p_0 * torch.sin(omega*(t-r/c)) #first time deriv
   ret_p_ddot = -omega**2 * p_0 * torch.cos(omega*(t - r/c)) #second time deriv

   radial_bracket = ret_p/(r**3) + ret_p_dot/(c*r**2) #bracket in radial electric field term
   polar_bracket = ret_p/(r**3) + ret_p_dot/(c*r**2) + ret_p_ddot/(c**2*r) #bracket in polar electric field term

   E_r = (2*torch.cos(theta)/(4*math.pi*eps_0))*radial_bracket
   E_theta = (torch.sin(theta)/(4*math.pi*eps_0))*polar_bracket

   if noise:
      E_r = E_r + torch.normal(0.0, 0.1*torch.mean(torch.abs(E_r)))
      E_theta = E_theta + torch.normal(0.0, 0.1*torch.mean(torch.abs(E_theta)))

   return torch.stack((E_r, E_theta), axis=-1)

def E_rtheta_in_xz_field_HD_tensor(r,theta,t, p_0, omega, noise=False, phi=0.0):

   E_rtheta = E_rtheta_field_HD_tensor(r,theta,t, p_0, omega, noise=noise)
   phi = torch.tensor(phi)
   E_x = torch.sin(theta)*torch.cos(phi)*E_rtheta[...,0] + torch.cos(theta)*torch.cos(phi)*E_rtheta[...,1]
   E_z = torch.cos(theta)*E_rtheta[...,0] - torch.sin(theta)*E_rtheta[...,1]

   return torch.stack((E_x, E_z), axis=-1)

def B_field_HD_tensor(x,y,z,t,p_0,omega, noise=False):
   '''Calculates the real magnetic field due to a Hertzian dipole at the origin, aligned with z-axis, for a harmonically oscillating charge on the dipole.
      Time input should be a fixed time.'''
   r = torch.sqrt(x**2 + y**2 + z**2) #the distance from origin at (x,y,z)
   ret_p_dot = -omega * p_0 * torch.sin(omega*(t-r/c)) #first time deriv
   ret_p_ddot = -omega**2 * p_0 * torch.cos(omega*(t - r/c)) #second time deriv

   azimuth_bracket = ret_p_dot/(r**2) + ret_p_ddot/(c*r)

   B_x = -(mu_0*y)/(4*math.pi*r) * azimuth_bracket
   B_y = (mu_0*x)/(4*math.pi*r) * azimuth_bracket
   B_z = torch.zeros_like(B_x)

   if noise:
      B_x = B_x + torch.normal(0.0, 0.1*torch.mean(torch.abs(B_x)))
      B_y = B_y + torch.normal(0.0, 0.1*torch.mean(torch.abs(B_y)))
      B_z = B_z + torch.normal(0.0, 0.1*torch.mean(torch.abs(B_z)))

   return torch.stack((B_x, B_y, B_z), axis=-1)


### build the NN models here ###

class MLP_Model(torch.nn.Module):

    def __init__(self, input_size, N1, N2, output_size):
        super(MLP_Model, self).__init__() #inherits the nn.module __init__()
        self.linear1 = torch.nn.Linear(input_size, N1) #first linear layer, with N1 hidden neurons
        self.act1 = torch.nn.ReLU() #activation layer using a sigmoid function since this is smooth and differentiable
        self.linear2 = torch.nn.Linear(N1, N2) #second linear layer, and output
        self.act2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(N2, output_size)
        
    def forward(self, x): #defines the forward propagation of input, x
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x) 
        x = self.act2(x) 
        yhat = self.linear3(x) #yhat is the approximate solution that is outputted
        return yhat


### here we will build a class containing everything needed to train the network ###
### WIP ###
class MLP_network:
    def __init__(self, input, training_data):
        #device = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
        device = torch.device("cpu")

        self.model = MLP_Model(3, 200, 200, 2).to(device)

        self.X = input
        self.X = self.X.to(device)
        self.X_train = input.detach()
        self.X_train = self.X_train.to(device)

        self.Y = training_data
        self.Y = self.Y.to(device)
        self.Y_train = training_data.detach()
        self.Y_train = self.Y_train.to(device)

        self.iter = 1 #used to print loss at intervals

        self.ls_criterion = torch.nn.MSELoss()
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.losses_MLP = []

    def loss_func(self): #for now this is just simple mse loss, but will implement soft constraints

        yhat = self.model(self.X_train)
        loss_data = self.ls_criterion(yhat, self.Y_train)
        loss = loss_data

        return loss

    def train(self, epochs):
        for epoch in range(epochs):
            self.adam.zero_grad()
            train_loss = self.loss_func()
            #print(train_loss)
            self.losses_MLP.append(train_loss.item())
            train_loss.backward()
            self.adam.step()

            fifth_progress = epochs // 5
            if epoch % fifth_progress == 0:
                print(epoch, train_loss.item())

        plt.figure('losses')
        plt.plot(np.linspace(0,len(self.losses_MLP),len(self.losses_MLP)), np.log(self.losses_MLP))
        plt.xlabel('Number of epochs')
        plt.ylabel('log MSE Loss')
        plt.show()

        return self.losses_MLP[-1]

### data and divergence loss function ###
class supervised_soft_network:
    def __init__(self, input, training_data, alpha, beta):
        #device = torch.device("mps") if torch.backends.mps.is_available() and torch.backends.mps.is_built() else torch.device("cpu")
        device = torch.device("cpu")

        self.model = MLP_Model(3, 200, 200, 2).to(device) #(r, theta, t) input, two outputs, (E_r, E_theta)

        # make sure input is in (grid_size, grid_size, grid_size, 3) format
        self.X = input
        self.X = self.X.to(device)
        self.X_train = input.detach()
        self.X_train = self.X_train.to(device)

        self.Y = training_data
        self.Y = self.Y.to(device)
        self.Y_train = training_data.detach()
        self.Y_train = self.Y_train.to(device)

        self.iter = 1 #used to print loss at intervals

        self.alpha = alpha
        self.beta = beta

        self.ls_criterion = torch.nn.MSELoss()
        self.adam = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.losses_network = []

    def loss_func(self):
        
        yhat = self.model(self.X_train)
        loss_data = self.ls_criterion(yhat, self.Y_train)

        u = self.model(self.X)
        du_dX = torch.autograd.grad(outputs=u, inputs=self.X, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        du_r_dr = du_dX[...,0]
        du_theta_dtheta = du_dX[...,1]
        u_div = compute_spherical_divergence(u[...,0], du_r_dr, self.X[...,0], u[...,1], du_theta_dtheta, self.X[...,1]) #minimise this as divergence should be 0
        loss_div = self.ls_criterion(u_div, torch.zeros_like(u_div, requires_grad=False))
        loss = self.alpha * loss_data + self.beta * loss_div
        #print(loss_data, loss_div)

        return loss

    def train(self, epochs):
        for epoch in range(epochs):
            self.adam.zero_grad()
            train_loss = self.loss_func()
            #print(train_loss)
            self.losses_network.append(train_loss.item())
            train_loss.backward()
            self.adam.step()

            fifth_progress = epochs // 5
            if epoch % fifth_progress == 0:
                print(epoch, train_loss.item())

        plt.figure('losses')
        plt.plot(np.linspace(0,len(self.losses_network),len(self.losses_network)), np.log(self.losses_network))
        plt.xlabel('Number of epochs')
        plt.ylabel('log MSE Loss')
        plt.show()

        return(self.losses_network[-1])
