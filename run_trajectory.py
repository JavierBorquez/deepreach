import pickle
import numpy as np
import torch
from dynamics.dynamics import Dubins3D
from utils import modules
import matplotlib.pyplot as plt


def plot_value_function(orig_opt, tMax, theta, sys, model):
    #override resolution for value function
    orig_opt.val_x_resolution = 1000
    orig_opt.val_y_resolution = 1000

    #define matrix of coordinates to pass to the DNN
    coords = torch.zeros(orig_opt.val_x_resolution*orig_opt.val_y_resolution, 4)
    coords[:, 0] = tMax
    coords[:, 3] = theta - np.pi/2
    x = np.linspace(-1, 1, orig_opt.val_x_resolution)
    y = np.linspace(-1, 1, orig_opt.val_y_resolution)
    xv, yv = np.meshgrid(x, y)
    coords[:, 1] = torch.from_numpy(xv.flatten())
    coords[:, 2] = torch.from_numpy(yv.flatten())

    #query the DNN for value function
    model_in = sys.coord_to_input(coords.cuda())
    results = model({'coords': model_in})
    model_in, model_out = results['model_in'].detach(), results['model_out'].squeeze(dim=-1).detach()
    values = sys.io_to_value(input=model_in, output=model_out)

    #reshape and plot
    values = values.reshape((orig_opt.val_x_resolution, orig_opt.val_y_resolution))
    plt.imshow(1*(values.detach().cpu().numpy().reshape(orig_opt.val_x_resolution, orig_opt.val_y_resolution).T <= 0), cmap='Blues',alpha=0.4, origin='lower', extent=(-1., 1., -1., 1.))
    plt.gca().set_aspect('equal', adjustable='box')
    return plt


# load original experiment settings
with open('C:\\Users\\javie\\Documents\\deepreach_feb24rel\\deepreach\\runs\\dubins3d_tutorial_run\\orig_opt.pickle', 'rb') as opt_file:
    orig_opt = pickle.load(opt_file)

#crete instance of the dynamics system
sys=Dubins3D(goalR=orig_opt.goalR, velocity=orig_opt.velocity, omega_max=orig_opt.omega_max, 
            angle_alpha_factor=orig_opt.angle_alpha_factor, set_mode=orig_opt.set_mode, freeze_model=orig_opt.freeze_model)

#create torch model and load it into the neural network, set it to evaluation mode
model = modules.SingleBVPNet(in_features=sys.input_dim, out_features=1, type=orig_opt.model, mode=orig_opt.model_mode,
                             final_layer_factor=1., hidden_features=orig_opt.num_nl, num_hidden_layers=orig_opt.num_hl)
model.cuda()
model.load_state_dict(torch.load('C:\\Users\\javie\\Documents\\deepreach_feb24rel\\deepreach\\runs\\dubins3d_tutorial_run\\training\\checkpoints\\model_current.pth'))
model.eval()
print('Model loaded successfully!') 


# example of how to query the value for a single state
s0 = torch.tensor([0., 0., 0., 0.])
results = model({'coords': sys.coord_to_input(s0[None].cuda())})
model_in, model_out = results['model_in'], results['model_out']
val0 = sys.io_to_value(input=model_in, output=model_out)[0]
dval0 = sys.io_to_dv(input=model_in, output=model_out)[0]
print('Value:', val0) #as its the center of the target, the value should be close to -goalR
print('Value Gradient:', dval0)

tMax = orig_opt.tMax
dt = 0.01
num_steps = 150
e = 0.01

s0 = torch.tensor([0.0, -0.8, np.pi/2]).cuda()
s_hist = [s0]
val_hist = []


for i in range(num_steps):
    s = s_hist[-1]
    s = s[None].cuda() #batch size 1
    #query the DNN for value function and its gradient
    model_in = torch.cat([torch.tensor([[tMax]]).cuda(),s], dim=-1)
    results = model({'coords': sys.coord_to_input(model_in)})
    model_in, model_out = results['model_in'], results['model_out']
    val0 = sys.io_to_value(input=model_in, output=model_out)[0]
    dval = sys.io_to_dv(input=model_in, output=model_out).squeeze()
    # print('val:', val0)
    # print('dval:', dval)
    dvds = dval[1:]
    dvds = dvds[None].cuda() #batch size 1
    # print('dvds:', dvds)
    #if the value is below treshold we engage the optimal control
    if val0 < e:
        u = sys.optimal_control(state=s, dvds=dvds)
    else:
        u = torch.tensor([[0.0]], device='cuda:0')
    # print('control:', u)
    d = 0 
    dsdt = sys.dsdt(s, u, d).squeeze()
    # print('dsdt:', dsdt)
    s = s + dt*dsdt
    s = s.squeeze()
    # print('s:', s)
    s_hist.append(s)
    val_hist.append(val0.squeeze())


# Create a figure
plt.figure(figsize=(10, 5))

# Create the first subplot
plt.subplot(1, 2, 1)
#plot trajectory
s_hist = torch.stack(s_hist)
plt.plot(s_hist[:,0].cpu().detach().numpy(), s_hist[:,1].cpu().detach().numpy())
#plot converged (tMax) BRT over xy plane at a given angle 
plt = plot_value_function(orig_opt=orig_opt, tMax=tMax, theta=np.pi/2, sys=sys, model=model)
#plot obstacle set
plt = plot_value_function(orig_opt=orig_opt, tMax=0, theta=np.pi/2, sys=sys, model=model)

#find first value below treshold in the value history
idx = 0
for i in range(len(val_hist)):
    if val_hist[i] < e:
        idx = i
        break
#grab the state at the first negative value and plot it
s_boundary = s_hist[idx]
plt.plot(s_boundary[0].cpu().detach().numpy(), s_boundary[1].cpu().detach().numpy(), 'ro', markersize=3)
plt.xlabel('x')
plt.ylabel('y')
plt.yticks([])
plt.xticks([])


# Create the second subplot
plt.subplot(1, 2, 2)
#define matrix of coordinates to pass to the DNN
th_points = 1000
coords = torch.zeros(th_points, 4)
coords[:, 0] = tMax
coords[:, 1] = s_boundary[0]
coords[:, 2] = s_boundary[1]
theta = np.linspace(-0.05, 0.05, th_points)
coords[:, 3] = torch.from_numpy(theta) + s_boundary[2].cpu()
#query the DNN for value function
model_in = sys.coord_to_input(coords.cuda())
results = model({'coords': model_in})
model_in, model_out = results['model_in'].detach(), results['model_out'].squeeze(dim=-1).detach()
values = sys.io_to_value(input=model_in, output=model_out)

#plot the value function along the theta axis
plt.plot(theta, values.cpu().detach().numpy())
plt.axhline(0, color='black', linestyle='dotted', linewidth=0.5)
plt.xlabel('Theta')
plt.ylabel('Value')
plt.yticks([0])
plt.xticks([])
plt.title('Value function along the theta axis')


#show the plot
plt.show()





