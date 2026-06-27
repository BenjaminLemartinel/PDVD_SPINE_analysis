import numpy as np

import os, sys

#SOFTWARE_DIR = './miniconda3/envs/py312/.local/bin/spine' 
SOFTWARE_DIR = 'C:/Users/Benjamin Lemartinel/AppData/Local/Programs/Python/Python313/Lib/site-packages/spine'

sys.path.insert(0, SOFTWARE_DIR)

from scipy.spatial.distance import cdist
from spine.vis.trace.point import scatter_points
from spine.vis.layout import PLOTLY_COLORS_WGRAY, layout3d
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

from sklearn.decomposition import PCA

import matplotlib
import seaborn

seaborn.set(rc={
    'figure.figsize':(15, 10),
})
seaborn.set_style("ticks")

seaborn.set_context('talk')



#beam information
#beam_source = [94.8,142.6,0.7]
beam_source = [85,160,0.7]
thetaXZ = 169.175*np.pi/180
thetaYZ = 135.514 * np.pi /180
beam_direction = [np.tan(thetaXZ),np.tan(thetaYZ),1]
beam_direction = beam_direction/np.linalg.norm(beam_direction)

beam_points = np.array([beam_source + beam_direction * d for d in np.linspace(0,424,500)]) #beam exits the detection volume at d=424

def get_primaries(particles):
    """Selects only the particles labelled as primary by spine"""
    primary_particles = []
    for p in particles:
        if p.is_primary:
            primary_particles.append(p)
    return np.array(primary_particles)


def get_starting_positions(particles):
    """Returns the lists of the starting positions for x y and z"""
    starting_positions_x,starting_positions_y,starting_positions_z = [],[],[]
    
    for p in particles:
        starting_positions_x.append(p.start_point[0])
        starting_positions_y.append(p.start_point[1])
        starting_positions_z.append(p.start_point[2])
    return starting_positions_x,starting_positions_y,starting_positions_z


def start_pos_histogram(particles,binsYZ,binsYX,binsZX,show_beam):
    """Draws YZ,YX,ZX 2D histograms of the starting positions of particles
        bins1-2-3 : tuples of the number of bins for the histograms
        If show_beam is true, the trajectory of the beam is drawn in red"""
    
    starting_positions_x,starting_positions_y,starting_positions_z = get_starting_positions(particles)


    plt.hist2d(starting_positions_y,starting_positions_z, cmap="viridis",bins = binsYZ)
    if show_beam:
        plt.plot(beam_points[:,1], beam_points[:,2], color='red', linewidth=5)
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("z")
    plt.show()

    plt.hist2d(starting_positions_y, starting_positions_x, cmap="viridis",bins = binsYX)
    if show_beam:
        plt.plot( beam_points[:,1],beam_points[:,0], color='red', linewidth=5)
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("x")
    plt.show()

    plt.hist2d(starting_positions_z,starting_positions_x, cmap="viridis",bins = binsYZ)
    if show_beam:
        plt.plot(beam_points[:,2],beam_points[:,0],  color='red', linewidth=5)
    plt.colorbar()
    plt.xlabel("z")
    plt.ylabel("x")

    
def start_pos_histogram_log(particles,binsYZ,binsYX,binsZX,show_beam):
    """Same as start_pos_histogram but in log color scale"""
    
    starting_positions_x,starting_positions_y,starting_positions_z = get_starting_positions(particles)


    plt.hist2d(starting_positions_y,starting_positions_z, cmap="viridis",bins = binsYZ,norm=colors.LogNorm())
    if show_beam:
        plt.plot(beam_points[:,1], beam_points[:,2], color='red', linewidth=5)
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("z")
    plt.show()

    plt.hist2d(starting_positions_y, starting_positions_x, cmap="viridis",bins = binsYX,norm=colors.LogNorm())
    if show_beam:
        plt.plot( beam_points[:,1],beam_points[:,0], color='red', linewidth=5)
    plt.colorbar()
    plt.xlabel("y")
    plt.ylabel("x")
    plt.show()

    plt.hist2d(starting_positions_z,starting_positions_x, cmap="viridis",bins = binsYZ,norm=colors.LogNorm())
    if show_beam:
        plt.plot(beam_points[:,2],beam_points[:,0],  color='red', linewidth=5)
    plt.colorbar()
    plt.xlabel("z")
    plt.ylabel("x")
    plt.show()


def select_type(particles,pid):
    """pid : bool array of size 6 
        Selects the particles that are of one of the desired types"""
    mask = np.zeros(len(particles),dtype = bool)
    for i in range(len(particles)):
        mask[i] = pid[particles[i].pid]
    return particles[mask]


def get_beam_alignement(particles):
    """Returns the array of the cosines of the angles between the particles directions and the beam_direction"""
    beam_alignement= np.zeros(len(particles))
    for i in range(len(particles)):
        beam_alignement[i] =  np.dot(beam_direction,particles[i].start_dir)
    return beam_alignement

    
def get_particles_aligned_with_beam(particles,threshold):
    """Selects the particles that have a beam_alignement (cosine of the angle between the particle direction and the beam direction) higher 
        than  the threshold"""
    beam_alignement = get_beam_alignement(particles)
    mask = (beam_alignement > threshold) 
    return particles[mask]

def get_particles_opposed_to_beam(particles,threshold):
    """Selects the particles that have a beam_alignement (cosine of the angle between the particle direction and the beam direction) lower 
        than  the threshold"""
    beam_alignement = get_beam_alignement(particles)
    mask = (beam_alignement < threshold) 
    return particles[mask]


    
def get_particles_close_to_beam(particles,threshold,max_beam_depth):
    """Selects particles close to the beam trajectory
    threshold: max distance to the trajectory
    max_beam_depth: selected length of the beam trajectory """
    beam_points = np.array([beam_source + beam_direction * d for d in np.linspace(0,max_beam_depth,500)])
    distances = np.zeros(len(particles))
    for i in range(len(particles)):
        distances[i] = cdist([particles[i].start_point],beam_points).min()
    mask = (distances < threshold)
    return particles[mask]

def get_contained_particles(particles):
    """Selects particles that are contained in the detector"""
    mask = np.zeros(len(particles),dtype=bool)
    for i,p in enumerate(particles):
        mask[i] = p.is_contained
    return particles[mask]


def get_energy(particles):
    """Returns the list of the energies of particles"""
    energies = np.zeros(len(particles))
    for i in range(len(particles)):
        energies[i] = particles[i].calo_ke
    return energies

def get_energy_ke(particles):
    """Returns the list of the energies of particles with the attribute ke"""
    energies = np.zeros(len(particles))
    for i in range(len(particles)):
        energies[i] = particles[i].ke
    return energies



def energy_filter(particles,minE,maxE):
    """Selects particles that have their energy minE < E < maxE"""
    mask = (get_energy(particles) > minE) & (get_energy(particles) < maxE)
    return particles[mask]

    
def plot_particles_start_pos_3D(particles,opacity):
    """Point scatter of the starting positions of particles, with the point opacity set to opacity"""
    trace = scatter_points(np.array([p.start_point for p in particles]),
                            markersize=1,
                            color='blue',opacity=0.3)
    fig = go.Figure(data = trace)
    iplot(fig)   

    
def get_momentum(particles):
    """Returns 3 arrays containing each component of the momentum of particles"""
    momentum_x = np.zeros(len(particles))
    momentum_y = np.zeros(len(particles))
    momentum_z = np.zeros(len(particles))
    
    for i,p in enumerate(particles):
        momentum_x[i] = p.momentum[0]
        momentum_y[i] = p.momentum[1]
        momentum_z[i] = p.momentum[2]
    return momentum_x,momentum_y,momentum_z

def get_momentum_norm(particles):
    """Return the array of the norm of the momentum of particles"""
    momentum = np.zeros(len(particles))
    for i,p in enumerate(particles):
        momentum[i] = np.linalg.norm(p.momentum)
    return momentum
    
from matplotlib import colors
def momentum_histogram(particles,bins1,bins2,bins3,range):
    """Plots the 3D histogram of the momentum of particles
    bins1,bins2,bins3 : selects the number of bins for each view
    range: selects the range for all 3 views"""
    momentum_x,momentum_y,momentum_z = get_momentum(particles)
    #momentum_x = np.sign(momentum_x) * (np.log(np.absolute(momentum_x))/np.log(10) +1)
    #momentum_y = np.sign(momentum_y) * (np.log(np.absolute(momentum_y))/np.log(10) +1)
    #momentum_z = np.sign(momentum_z) * (np.log(np.absolute(momentum_z))/np.log(10) +1)
    beam_direction_points = np.array([beam_direction * p for p in np.linspace(0,range,500)])

    plt.hist2d(momentum_y,momentum_z, cmap="viridis",bins = bins1, norm=colors.LogNorm(),range=[[-range,range],[-range,range]])
    plt.plot(beam_direction_points[:,1],beam_direction_points[:,2],color='r')
    plt.colorbar()
    plt.xlabel("Py")
    plt.ylabel("Pz")
    plt.show()

    plt.hist2d(momentum_y, momentum_x, cmap="viridis",bins = bins2, norm=colors.LogNorm(),range=[[-range,range],[-range,range]])
    plt.plot(beam_direction_points[:,1],beam_direction_points[:,0],color='r')
    plt.colorbar()
    plt.xlabel("Py")
    plt.ylabel("Px")
    plt.show()

    plt.hist2d(momentum_z,momentum_x , cmap="viridis",bins = bins3, norm=colors.LogNorm(),range=[[-range,range],[-range,range]])
    plt.plot(beam_direction_points[:,2],beam_direction_points[:,0],color='r')
    plt.colorbar()
    plt.xlabel("Pz")
    plt.ylabel("Px")
    plt.show()

def plot_hist_ratio(x,y,bins,range,title,xlabel,ylabel):
    """Plots the ratio of the histograms of x and y (count_y/count_x)"""
    counts_x,edges,_= plt.hist(x, bins=bins, range=range,density = True)
    counts_y,_,_= plt.hist(y, bins=bins,range=range,density = True)
    plt.clf()
    _= plt.plot(edges[:-1],np.array(counts_y)/np.array(counts_x))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

def get_track_length(particles):
    """Returns the list of the lengths of particles using the norm of end_point - start_point"""
    lengths = np.zeros(len(particles))
    for i,p in enumerate(particles):
        lengths[i]= np.linalg.norm(p.start_point - p.end_point)
    return lengths

def get_track_length_spine(particles):
    """Returns the list of the lengths of particles using the SPINE length attribute"""
    lengths = np.zeros(len(particles))
    for i,p in enumerate(particles):
        lengths[i]= p.length#np.linalg.norm(p.start_point - p.end_point)
    return lengths

def interacting_particles(reco_particles,reco_interactions): 
    """ Returns an bool array that is true on the index p.id if p is part of an interaction"""
    interacts = np.zeros(len(reco_particles),dtype = bool)
    for i in reco_interactions:
        if i.num_particles >1:
            for p in i.particles:
                interacts[p.id] = True
    return interacts

def plot_interactions(interactions):
    """Plots all the particles inside the interactions"""
    
    cmap = plt.colormaps['Set1']
    trace = []
    
    for i in interactions:
       for np,p in enumerate(i.particles):
           trace += scatter_points(p.points,
                            markersize=1,
                            hovertext ='ParticleId : ' + str(p.id) + '\n' + 'Type : ' + str(p.pid),
                            color='rgb%s' % str(tuple(int(x*255) for x in cmap(np/len(i.particles))[:-1])))
    fig = go.Figure(data = trace)
    iplot(fig)   

def plot_particles_3D(particles):
    """Plots all the particles in 3D"""
    cmap = plt.cm.get_cmap('Set1')
    trace = []
    
    for i,p in enumerate(particles):
        trace += scatter_points(p.points,
                            markersize=1,
                            hovertext ='ParticleId : ' + str(p.id) + '\n' + 'Type : ' + str(p.pid),
                            color='rgb%s' % str(tuple(int(x*255) for x in cmap(i/len(particles))[:-1])))
    fig = go.Figure(data = trace)
    iplot(fig)   



def get_PCA(p):
    pca = PCA(n_components=1)
    coords_pca = pca.fit_transform(p.points)[:]
    return coords_pca.flatten(), pca.components_[0]


def get_particle_dQdx(p,bin_size):
    """Gets the dQ/dx of a single particles, used for multiprocessing in the next function"""
    coords_pca,_ = get_PCA(p)

    
                      
    end_point = len(p.points)-1
    
    List_dQ = []
    List_dx = []
    List_residual_range = []

    pca = PCA(n_components=1)
    
    coords = p.points

    # Make sure where the end vs start is
    # if end == 0 we have the right bin ordering, otherwise might need to flip when we record the residual range
    distances_endpoints = [((coords[coords_pca.argmin(), :] - coords[end_point, :])**2).sum(), ((coords[coords_pca.argmax(), :] - coords[end_point, :])**2).sum()]
    end = np.argmin(distances_endpoints)

    # Split into segments and compute local dQ/dx
    if not end: # no need to flip
        bins = np.arange(coords_pca.min(), coords_pca.max(), bin_size)
        bin_inds = np.digitize(coords_pca, bins)

    else: # must flip
        bins = np.arange(-coords_pca.max(), -coords_pca.min(), bin_size)
        bin_inds = np.digitize(-coords_pca, bins)

    for i in np.unique(bin_inds):
        mask = bin_inds == i
        if np.count_nonzero(mask) < 2: continue

        # Repeat PCA locally for better measurement of dx
        pca_axis = pca.fit_transform(p.points[mask])

        dx = pca_axis[:, 0].max() - pca_axis[:, 0].min()
        dQ = p.depositions[mask].sum()
        residual_range = ((i - 0.5) * bin_size)

        List_dx.append(dx)
        List_dQ.append(dQ)
        List_residual_range.append(residual_range)


    return List_dQ,List_dx,List_residual_range


def get_dQdx(particles,bin_size):
    """Returns the list of dQ, dx and residual range for segments of bin_size points all the particles
    Residual range is taken with reference to the particle's end point"""
    N = len(particles)  

    process_iteration =  partial(get_particle_dQdx,bin_size=bin_size)
    
    with Pool(processes=15) as pool:  
        try:
            results = list(tqdm(
            pool.imap(process_iteration, particles,chunksize = 20),
            total=N
            ))
            pool.close()  
            pool.join()    
        except Exception as e:
            pool.terminate()  
            pool.join()
            raise

    results_dQ,results_dx,results_residual_range =  zip(*results)

    dQ = np.hstack(results_dQ)
    dx = np.hstack(results_dx)
    residual_range = np.hstack(results_residual_range)

    return dQ,dx,residual_range

"""
def data_set_particles(particles,particle_indices,index):
    mask = np.zeros(len(particles),dtype=bool)
    for i in range(len(particles)):
        if particles[i].id < particle_indices[index+1] and particles[i].id >= particle_indices[index]:
            mask[i] = True
    return particles[mask]

def particle_data_set(p,particle_indices):
    i = 0
    while p.id >= particle_indices[i+1]:
        i += 1
    return i

def plot_particle_event(p,particles,particle_indices):
    event = particle_data_set(p,particle_indices)
    particles = data_set_particles(particles,particle_indices,event)
    plot_particles_3D(particles)"""

from tabulate import tabulate

def print_composition(particle_sets,sets_names):
    """Prints a table with the number of particles of each type for each particle set in particle_sets
    sets_names: name for each particle set in particle_sets"""
    tab  =[]
    for i,particles in enumerate(particle_sets):
        counts = [0,0,0,0,0]
        for p in particles:
            counts[p.pid] +=1

        tab.append([sets_names[i] + '\n \n --- \n'] + [str(counts[j]) + '\n' + str(round(100*counts[j]/len(particles),2)) + ' %\n ----\n'  for j in range(5)])

    print(tabulate(tab, headers=['', 'Photons','Electrons','Muons','Pions','Protons']))


def event_info(event,beam_instrumentation):
    #binary search for the event
    left  = 0
    right = len(beam_instrumentation)-1

    while left <= right:
        mid = (left + right) // 2

        if beam_instrumentation[mid]['Event'] == event:
            return beam_instrumentation[mid]

        if beam_instrumentation[mid]['Event'] < event:
            left = mid + 1
        else:
            right = mid - 1


    return None
