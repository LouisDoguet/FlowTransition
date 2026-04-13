import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy

class ComPlot:
    def __init__(self,X=[],Y=[],Z=[]):
        self.Z = Z
        self.X = X
        self.Y = Y

    def fromSpace(self,space):
        self.Z = space.Z
        self.Y = space.Y
        self.X = space.X

    def __plot__(self,title='',xlabel='x',ylabel='y',levels=50):

        fig, ax = plt.subplots(figsize=(12, 6))

        phase = np.angle(self.Z)

        module = np.abs(self.Z)
        module /= module.max()
        module = module**(1/10)

        ext = (self.X.min(), self.X.max(), self.Y.min(), self.Y.max())
        im = ax.imshow(phase, extent=ext, origin='lower', cmap='hsv', alpha=module, vmin=-np.pi, vmax=np.pi)
        cont = ax.contour(self.X, self.Y, module, levels=levels, colors='black', linewidths=0.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = fig.colorbar(im, ticks=[-3,0,3])
        cbar.ax.set_yticklabels([r'-$\pi$','0',r'$\pi$'])

        return fig, ax, im, cont
    
    def plot(self,title='',xlabel='x',ylabel='y',levels=50):
        self.__plot__(title=title,xlabel=xlabel,ylabel=ylabel,levels=levels)
        plt.show()

    def animate(self,f,var_arg,title=''):

        fig, ax, im, cont = self.__plot__()
        ax.set_title(title)

        def update(frame):

            nonlocal cont
            nonlocal im
            nonlocal ax

            for c in ax.collections:
                c.remove()

            # recompute field for parameter frame
            tmpZ = f(self.Z,var_arg[int(frame)])

            phase = np.angle(tmpZ)
            module = np.abs(tmpZ)
            module /= module.max()
            module = module**(1/2)

            im.set_data(phase)
            im.set_alpha(module)

            cont = ax.contour(self.X, self.Y, np.imag(tmpZ),levels=50, colors='black', linewidths=0.5)

            return ax
        
        ani = FuncAnimation(fig, update, frames=np.linspace(0,len(var_arg)-1), blit=False, repeat=False, interval=1)

        plt.show()

    def animate_fields(self, fields, title='', interval=100, dynamic_scaling=True):
        """
        Animate a sequence of precomputed fields.
        
        @param fields: list of objects containing 'Z' attribute (like Space objects) or list of 2D numpy arrays.
        @param title: title of the animation
        @param interval: delay between frames in milliseconds
        @param dynamic_scaling: If True, normalize color intensity per frame (good for shape). 
                                If False, normalize against the global max (good for seeing growth/decay).
        """
        
        # Initialize with the first frame
        self.Z = fields[0].Z if hasattr(fields[0], 'Z') else fields[0]
        
        # Determine normalization factor
        if dynamic_scaling:
            max_val = 1.0 # Will be recomputed per frame
        else:
            # Find global max across all time steps
            all_max = 0
            for f in fields:
                data = f.Z if hasattr(f, 'Z') else f
                m = np.max(np.abs(data))
                if m > all_max: all_max = m
            max_val = all_max if all_max > 0 else 1.0

        fig, ax, im, cont = self.__plot__(title=title)
        
        def update(frame_idx):
            nonlocal cont
            nonlocal im
            nonlocal ax

            for c in ax.collections:
                c.remove()

            # Get current Z
            field_data = fields[frame_idx]
            tmpZ = field_data.Z if hasattr(field_data, 'Z') else field_data

            phase = np.angle(tmpZ)
            module = np.abs(tmpZ)
            current_max = module.max()

            # Normalize
            if dynamic_scaling:
                norm_factor = current_max if current_max > 0 else 1.0
            else:
                norm_factor = max_val
            
            # Apply normalization for alpha/transparency
            module_norm = module / norm_factor
            # Gamma correction for visibility
            module_vis = module_norm**(1/2) 
            
            # Update data
            im.set_data(phase)
            im.set_alpha(module_vis)

            # Update contours
            cont = ax.contour(self.X, self.Y, np.real(tmpZ), levels=5, colors='black', linewidths=0.5)
            
            # Update title with amplitude info
            ax.set_title(f"{title}\nAmplitude: {current_max:.3e}")

            return ax
        
        ani = FuncAnimation(fig, update, frames=len(fields), blit=False, repeat=False, interval=interval)
        plt.show()

    def animate_fields_3d(self, fields, title='', interval=30, dynamic_scaling=True):
        """
        Animate a sequence of precomputed fields in 3D.
        
        @param fields: list of objects containing 'Z' attribute (like Space objects) or list of 2D numpy arrays.
        @param title: title of the animation
        @param interval: delay between frames in milliseconds
        @param dynamic_scaling: If True, z-axis scales to fit current frame.
                                If False, z-axis is fixed to global max (good for growth/decay).
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Calculate global max
        global_max = 0
        for f in fields:
            data = f.Z if hasattr(f, 'Z') else f
            m = np.max(np.abs(data))
            if m > global_max: global_max = m
        
        global_max = global_max if global_max > 0 else 1.0

        # Set labels once
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(r'Re($v^\prime$)')

        # Set initial limits if not dynamic
        if not dynamic_scaling:
            ax.set_zlim(-global_max, global_max)

        # Container for the surface plot to update it
        plot_ref = {'surf': None}

        def update(frame_idx):
            # Remove previous surface
            if plot_ref['surf'] is not None:
                plot_ref['surf'].remove()
            
            field_data = fields[frame_idx]
            tmpZ = field_data.Z if hasattr(field_data, 'Z') else field_data
            
            # Real part of the wave as height
            Z_real = np.real(tmpZ)
            amplitude = np.abs(tmpZ)
            current_max = amplitude.max()
            
            # Plot surface
            surf = ax.plot_surface(self.X, self.Y, Z_real, cmap='viridis', linewidth=0, antialiased=False)
            plot_ref['surf'] = surf
            
            # Set axis limits only if dynamic
            if dynamic_scaling:
                lim = max(current_max, 1e-6)
                ax.set_zlim(-lim, lim)
                
            ax.set_title(f"{title}\nAmplitude: {current_max:.3e}")
            
            return surf,

        ani = FuncAnimation(fig, update, frames=len(fields), blit=False, repeat=False, interval=interval)
        plt.show()

