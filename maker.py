import astropy.units as u
import copy
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sunpy.map

from astropy.time import Time

from dataclasses import dataclass
from typing import Iterable


def make_time_slices(
    start: Time,
    end: Time,
    time_step: u.Quantity,
    exposure: u.Quantity
) -> list[tuple[Time, Time]]:
    """
    Makes a list of Time pairs that chunk the interval
    [start, end]. The length of a chunk is equal to exposure,
    and the difference between the starts of neighboring
    chunks is equal to time_step.
    
    Useful for chunking a photon list into frames.
    """

    starts = Time( np.arange(start, end-exposure, time_step) )
    ends = starts + exposure
    slices = list(zip(starts, ends))

    # Account for misalignment between the [start,end] interval
    # and the specified cadences.
    if end not in ends:
        slices.append( (starts[-1] + time_step, end) )

    return slices


@dataclass
class MapSet():
    """
    Container for managing map movie data.
    init_func is called once to initialize the map axis, and
    plot_func is called for each frame update. 
    init_func and plot_func must have the following signature:
    func(fig: plt.Figure, ax: plt.Axes, map_ sunpy.map.Map)
    """
    
    maps: list[sunpy.map.Map]
    init_func: callable = lambda a, b, c : None
    plot_func: callable = lambda a, b, c : None
    ax: plt.Axes = None


class SunpyMovieMaker():
    """
    Makes a movie out of the provided Sunpy maps.
    MapSet is a dataclass for managing the various groups of maps provided.
    This allows one to plot several sets of maps at the same time.
    All sets must have the same number of maps.
    """

    def __init__(
        self,
        map_sets: Iterable[MapSet],
        fig: plt.Figure,
        make_axes: bool = True
    ):
        self.map_sets = copy.deepcopy(map_sets)
        self.fig = fig
        if make_axes:
            for i, map_set in enumerate(self.map_sets, start=1):
                ax = fig.add_subplot(1, len(map_sets), i, projection=map_set.maps[0])
                map_set.ax = ax


    def make_movie(
        self,
        fps: float = 30,
        dpi: int = 100,
        out_path: str = './movie.gif'
    ):
        """
        Generates a movie using the list of MapSet and saves it to a file.
        """

        def update(frame, ims):
            for im, mset, map_ in zip(ims, self.map_sets, frame):
                im.set_array(map_.data)
                # mset.ax.set_title(map_.plot_settings['title'])
                mset.plot_func(self.fig, mset.ax, map_)
            return ims
        
        # Initialize artists.
        ims = []
        for i, mset in enumerate(self.map_sets):
            map_ = (mset.maps).pop(0)
            im = map_.plot(axes=mset.ax)
            ims.append(im)
            if mset.init_func:
                mset.init_func(self.fig, mset.ax, map_)

        # Sort maps into frames.
        num_frames = len((self.map_sets[0]).maps)
        frames = []
        for i in range(num_frames):
            frames.append( [self.map_sets[j].maps[i] for j in range(len(self.map_sets))] )

        # Animate.
        anim = animation.FuncAnimation(
            self.fig,
            update,
            fargs=(ims,),
            frames=frames,
            blit=True
        )
        FFwriter = animation.FFMpegWriter(fps=fps, codec='mpeg4')
        anim.save(out_path, dpi=dpi, writer=FFwriter)
